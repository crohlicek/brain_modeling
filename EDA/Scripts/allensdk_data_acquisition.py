# Imports relating to data interface with AllenSDK
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
# datetime module for saving files with unique filenames
from datetime import datetime


from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import (
    EcephysSession, 
    removed_unused_stimulus_presentation_columns
)
from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
from allensdk.brain_observatory.visualization import plot_running_speed

def remove_invalid_trials(session, stim_pres_ids, times):
    '''
    A session may contain invalid time intervals in which there was a recording failure
    that renders the data invalid. This method takes in a session object, a list of
    stimulus presentation ids, and a dataframe of spike times, and returns a list
    of presentation ids that represent the stimulus presentation trials that have
    no overlap with the session's invalid intervals.
    '''
    # creating a list that we'll fill with invalid intervals
    invalid_intervals = []
    for _, row in session.invalid_times.iterrows():
        invalid_intervals.append((row.start_time,row.stop_time))

    # a list of trials containing overlap with invalid intervals
    corrupted_trials = []
    for pres in stim_pres_ids:
        # get the list of spike times for a given trial
        trial_spikes = times[times['stimulus_presentation_id'] == pres].index
        
        # then we loop through the invalid intervals and check if any of them are contained
        for time in trial_spikes:
            # have to go through by hand because we can't make two comparisons using an any() command
            for a,b in invalid_intervals:
                if time > a and time < b:
                    # and if any of the intervals are contained then we'll add this trial to the corrupted list
                    corrupted_trials.append(pres)
                    break
            break

    # now we remove the corrupted stim pres ids from our list and return the new one
    stim_pres_ids_cleaned = np.array([pres for pres in stim_pres_ids if pres not in corrupted_trials])
    
    return stim_pres_ids_cleaned

def get_session_data(session_dir, manifest_file, session_id, all_trials=True):
    '''
    Function to retrieve a single session's spike time data. With the 'all_trials' flag set to False, this function
    will return the spike times (in dataframe form, along with columns giving 'stimulus_presentation_id',
    'unit_id', and 'time_since_stimulus_presentation_onset') corresponding to the first presentation
    of the drifting gradient stimulus. The type of gradient and number of presentation can be changed below.
    '''
    # Here we give the path to the manifest file describing the experiments we're considering
    manifest_path = os.path.join(session_dir, manifest_file)
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # setting the session corresponding to our data
    session = cache.get_session_data(session_id)

    # Using SNR to select just the least noisy channels for our analysis:
    # storing the units that we'll want to select
    units_with_very_high_snr = session.units[session.units['snr'] > 4]
    # ... getting the indices of the high SNR units
    high_snr_unit_ids = units_with_very_high_snr.index.values

    # get spike times from the first block of drifting gratings presentations 
    drifting_gratings_presentation_ids = session.stimulus_presentations.loc[
        (session.stimulus_presentations['stimulus_name'] == 'drifting_gratings')
    ].index.values

    times = session.presentationwise_spike_times(
        stimulus_presentation_ids=drifting_gratings_presentation_ids,
        unit_ids=high_snr_unit_ids
    )

    # IMPORTANT STEP: create mapping from `unit_id` value with index of occurrence
    # REASON: different subsets of units are present in different trials, so we'll build
    # our spike matrices with respect to the original list of units present in the session
    # (disregarding the noisy ones) and use that list to determine the row in which a spike
    # gets places in the matrices (that was all matrices will have the same standardized
    # set of channels)
    unit_ids = times['unit_id'].unique()

    id_to_index = {}

    for i in range(len(unit_ids)):
        id_to_index[unit_ids[i]] = i

    # creating a separate column to avoid issues with the original column
    # being overwritten
    times['unit_id_indx'] = times['unit_id'].apply(lambda x: id_to_index[x])

    # storing a list of the different stimulus presentation ids
    # for the drifting gradients (just to easily check multiple different ones)
    stim_pres_ids = times['stimulus_presentation_id'].unique()

    # IMPORTANT STEP: there are invalid time intervals in this session in which certain units
    # weren't recorded properly. We're going to go through our trials and check for overlap with
    # the invalid intervals and remove the trials that do.
    stim_pres_ids = remove_invalid_trials(session, stim_pres_ids, times)

    if not all_trials:
        # all_trials=False means we just want the first trial's spike times
        first_drifting_grating_presentation_id = stim_pres_ids[0]
        plot_times = times[times['stimulus_presentation_id'] == first_drifting_grating_presentation_id]

        # need to return all the objects we need to make the binary spike matrix
        return plot_times, session, first_drifting_grating_presentation_id
    else:
        # all_trials=True means we want all drifting gradient trial data, and will return
        # lists of plot times along with the matching list of presentation IDs
        plot_times_list = []
        for pres_id in stim_pres_ids:
            plot_times_list.append(times[times['stimulus_presentation_id'] == pres_id])

        # ... and we also want to return the unit it -> index mapping so we can refer to the number of entries
        return plot_times_list, session, stim_pres_ids, id_to_index

def create_spike_matrix(session, plot_times, presentation_id, unit_masterlist):
    '''
    Function to process our dataframe of plot times and produce a binary matrix of spikes.
    In the session metadata we find that the electrodes measuring the different units in this session
    have different exact sampling frequencies (30 kHz +/- ~50 Hz). In order to make the transformation
    into binary spike matrix, we calculate the index of a spike based on its given 'time_since_stimulus_presentation_onset'
    using a hardcoded sampling frequency of 30000. Calculating these indices in this way effective resamples to
    30000 kHz exactly, and aligns the discrepantly sampled groups of units.
    
    We also pass in the unit_masterlist, which gives the mapping from all units in the session (with
    high SNR as we selected earlier) to index that we will use to determine which row to place a spike in
    in the matrix
    '''
    # 1) get the number of samples
    start = session.stimulus_presentations.loc[presentation_id]['start_time']
    stop = session.stimulus_presentations.loc[presentation_id]['stop_time']
    # find a way to go in and grab the sampling frequency automatically
    # from the session metadata
    sample_freq = 30000

    # take the ceiling to make sure we get a whole number
    n_samples = int(np.ceil((stop-start)*sample_freq))

    # 2) initialize empty spike matrix
    ## ===> and in order to standardize the shape of the data across all trials we'll
    ## ===> use the number of units recorded in the session masterlist
    n_units = len(unit_masterlist)
    spikes = np.zeros([n_units, n_samples])
    
    # 4) Traverse through the rows adding spikes in the right place
    for row in plot_times.itertuples():
        # store the index that'll correspond to the target row in the spike mtx
        unit = row.unit_id_indx
        # find the sample this spike belongs in
        # --> want to round, not floor (what happens when you cast to int)
        sample = int(row.time_since_stimulus_presentation_onset*sample_freq)
        # flip the right value in the spike matrix
        spikes[unit][sample] = 1.0
        
    return spikes

def bin_spike_matrix(spike_matrix):
    '''
    Function to perform the additional preprocessing step of binning our a given spike
    matrix down to 1 ms resolution. This is a common data processing step when dealing with
    neural spiking data. It'll reduce the sparsity of the data, but also the volume of data
    so this step will be used when training models on full sets of all trials for a given
    session/stimulus combination.
    '''
    num_samples = len(spike_matrix[0])
    # the data is sampled at 30 kHz so we divide by 30 to get down to 1 ms bins
    num_bins = num_samples // 30
    # initialize the array we'll populate with the binned samples from the original matrix
    binned_spikes = np.zeros((spike_matrix.shape[0], num_bins))
    # iterate through the columns of the original matrix constructing the binned columns
    for i in range(num_bins):
        # binned_col = spike_matrix[:,i*30:min((i+1)*30, num_samples)].sum(axis=1)
        binned_col = np.amax(spike_matrix[:,i*30:min((i+1)*30, num_samples)], axis=1)
        binned_spikes[:,i] = binned_col

    return binned_spikes

def save_spikes(spike_data, session_id):
    '''
    save method designed for saving a single set of trial data
    '''
    # Adding a timestamp to the file name to prevent overwriting
    now = datetime.now()
    dt_string = now.strftime("_%H_%M_%S")

    with open('binary_spike_matrix_data/spike_mtx_'+str(session_id)+dt_string+'.npy', 'wb') as f:
        np.save(f, spike_data)

def select_most_common_length_trial(list_of_trials):
    '''
    Helper function to process a list of trial spike matrices and remove all matrices with a
    different number of samples than the most common number in the set (i.e. in the case of the
    drifting gradient, all binned trials have 2001 samples, except for one which has 1984. This
    method gets rid of the 1984-long matrix), thus allowing for saving in a numpy 3d array of the
    form (num_trials, num_units, num_samples)

    It is assumed that the list of trials being fed in at this point was created with the same
    number of channels in each matrix.
    '''
    trial_lengths = [trial.shape[1] for trial in list_of_trials]
    lens, cts = np.unique(trial_lengths, return_counts=True)
    out = [mtx for mtx in list_of_trials if mtx.shape[1] == lens[np.argmax(cts)]]
    return np.array(out)

def save_all_trail_spikes(session_dir, manifest_file, session_id, save_binned=True):
    '''
    Method to create a set of 1ms-resolution binned spike matrices to be stored in a np.array.

    We only take in the session_id here because we'll collect all the data for specific stimulus
    in the specified session. For consistency with our earlier experiments we'll collect the
    spikes related to the drifting gradient stimulus, but this can be changed below.
    '''
    # we'll use the all_trials flag to get the list of all spike times dfs and we'll iterate through those lists
    plot_times_list, session, stim_pres_ids, id_to_index = get_session_data(session_dir, manifest_file, session_id, all_trials=True)

    # going to iterate through the plot_times and stim_pres lists and populate this list of data matrices
    # and then save it to file
    spike_matrix_list = []

    # date-time tag for unique file name
    now = datetime.now()
    dt_string = now.strftime("_%H_%M_%S")

    for i in range(len(plot_times_list)):
        spike_matrix_list.append(create_spike_matrix(session, plot_times_list[i],stim_pres_ids[i],id_to_index))

    if save_binned:
        # if the save_binned flag is turned on then we'll apply our binning method before saving
        spike_matrix_list_binned = [bin_spike_matrix(spike_matrix) for spike_matrix in spike_matrix_list]
        output = select_most_common_length_trial(spike_matrix_list_binned)

        # saving with filename that tags it with 'BINNED' or 'NOTBINNED'
        with open('binary_spike_matrix_data/all_trial_data_BINNED_'+str(session_id)+dt_string+'.npy', 'wb') as f:
            np.save(f, output)
    else:
        # otherwise we'll just save the data at 30 kHz resolution
        output = select_most_common_length_trial(spike_matrix_list)
        with open('binary_spike_matrix_data/all_trial_data_NOTBINNED'+str(session_id)+dt_string+'.npy', 'wb') as f:
            np.save(f, output)

    # ... and returning the output as well so it can be inspected
    return output
    

def main():
    '''
    Running the main method of this file will save a copy of the list of binned spike matrices that
    correspond to all presentations of the drifting gradient in our specified session
    '''

    # setting the information that defines the location and provenance
    # of our data - these can be changed to reflect different AllenSDK
    # datasets that may be used
    session_dir = 'allensdk_session_data'
    manifest_file = 'manifest.json'
    session_id = 715093703

    # get our dataframe of spike times and our session object
    # plot_times_list, session, stim_pres_ids, id_to_index = get_session_data(session_dir, manifest_file, session_id)

    # ... and use those to get our binary spike matrix
    save_all_trail_spikes(session_dir, manifest_file, session_id)


if __name__ == '__main__':
    main()
