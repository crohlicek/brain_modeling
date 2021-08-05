import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.ndimage import gaussian_filter1d
import sys

class WindowedTensorDataset(TensorDataset):
    '''
    Dataset wrapping data tensors. This object can be used in both the memoryless/
    sample-to-sample case, and the windowed sample case. The logic for the tau window
    parameter holds if tau is given to be 0.

    Each sample will be retrieved by indexing tensors along the second
    dimension (a single sample is a time slice across all channels)

    Arguments:
        data_tensor (Tensor): contains sample data.
    '''
    # want to initialize the dataset with the range of times
    # to include on either side of the sample t (given by tau)
    def __init__(self, data_tensor, tau):
        self.data_tensor = data_tensor
        self.tau = tau
        # check that tau is an integer less than half the width of the input matrix
        if (not isinstance(tau, int)) or (tau > len(self.data_tensor[0]) // 2):
            raise ValueError('tau must be an integer value less than half the \
                                number of samples given in the data matrix')
        
    def __len__(self):
        # we new have to account for the window size when taking the length
        return len(self.data_tensor[0]) - 2*self.tau

    def __getitem__(self, index):
        # the data sample will be the subset across all unit channels
        # and the tau-radius on either side of the index time sample
        
        # This is the indexing logic for selecting the tau-wide windows centered on
        # a given index, but without going off the edges – this will work for indexes 
        # from 0 to self.__len__
        data = self.data_tensor[:, index:index+2*self.tau+1]
        label = self.data_tensor[:, index+self.tau]
        return data, label

class MultitrialDataset(TensorDataset):
    '''Dataset wrapping multi-trial spike data.

    The indexing logic will allow for the traversal of our 3d numpy array
    storing data in the form (num_trials, num_units, num_samples) with the
    application of a tau-radius delay window to be placed on either side
    of a given target sample
    '''
    # want to initialize the dataset with the range of times
    # to include on either side of the sample t (given by tau)
    def __init__(self, data_tensor, tau):
        self.data_tensor = data_tensor
        # storing the number of trials included
        self.trials = data_tensor.shape[0]
        # storing the length of each trial
        self.T = data_tensor.shape[2]
        self.tau = tau
        # check that tau is an integer less than half the width of the input matrix
        if (not isinstance(tau, int)) or (tau > self.T // 2):
            raise ValueError('tau must be an integer value less than half the \
                                number of samples given in the data matrix')
        
    def __len__(self):
        # we new have to account for the window size when taking the length
        return (self.T - 2*self.tau) * self.trials

    def __getitem__(self, index):
        # we'll use integer division to index into the right trial
        # and we'll use the remainder to index into the right sample
        # --> T - 2τ is the number of samples in a given trial
        trial = index // (self.T - 2*self.tau)
        ind = index % (self.T - 2*self.tau)
        
        data = self.data_tensor[trial, :, ind:ind+2*self.tau+1]
        label = self.data_tensor[trial, :, ind+self.tau]
        return data, label


def create_rates(st_dev, spikes):
    '''
    function to take in a standard deviation and spikes ndarray
    and return the continuous "rates" approximation – normalized convolution
    of input binary spikes with a Gaussian kernel
    '''
    # choice of sigma value -- in terms of time step
    # scaling by st_dev value to keep magnitude similar to original
    cts_spikes = st_dev*gaussian_filter1d(spikes,st_dev)
    # now we need to make sure that none of the data fed into our models is
    # outside [0,1], so we'll take the minimum of cst_spikes and the array of 1s
    ones = np.ones(cts_spikes.shape)
    capped_cts_spikes = np.minimum(cts_spikes, ones)

    return capped_cts_spikes

def get_binary_dataloader(data_path, batch_size, tau, multitrial=True):
    '''
    Funtion to generate the dataloader object storing binary spike data given by
    the specified path. The window size of the samples is given by tau (the time
    radius to be added on either side of a given target time sample).

    multitrial flag specifying whether we want to create a multitrial dataset, and
    calls the appropriate dataset object
    '''
    with open(data_path, 'rb') as f:
        spikes = np.load(f, allow_pickle=True)

    if multitrial:
        spike_dataset = MultitrialDataset(torch.from_numpy(spikes).float(), tau)
    else:
        spike_dataset = WindowedTensorDataset(torch.from_numpy(spikes).float(), tau)

    spike_loader= DataLoader(spike_dataset, batch_size=batch_size, shuffle=True)

    return spike_loader


def get_smoothed_dataloader(data_path, tau):
    '''
    Funtion to generate the dataloader object storing continuous spike data (rate
    approximations) given by the specified path and smoothed by convolution with a
    Gaussian filter. Window size again given by tau

    This method can only be called on single-trial datasets – I don't thing we'll
    want to ever call it on binned data but if we do then we should add a multi-trial
    capacity to the create_rates function. Here we throw an error if multitrial data is given
    '''
    with open(data_path, 'rb') as f:
        spikes = np.load(f, allow_pickle=True)

    if len(spikes.shape) > 2:
        raise ValueError('This method only accepts single-trial data but multi-trial \
            data was given')

    # we give 20 as the standard deviation of our Gaussian filter based on the
    # 30 kHz sampling rate
    filtered_spikes = create_rates(20, spikes)
    filtered_spike_dataset = WindowedTensorDataset(torch.from_numpy(filtered_spikes).float(), tau)
    fitlered_spike_loader= DataLoader(filtered_spike_dataset, batch_size=1)

    return fitlered_spike_loader


# ========================================================================

## The path to a specific sample; this will be given in the script that imports and calls
## the functions given here, but storing a copy of an example path here for reference
# data_path = 'binary_spike_matrix_data/spike_mtx_715093703_11_19_33.npy'
