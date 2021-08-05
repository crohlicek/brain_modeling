'''
File containing execution instructions for sets of experiments involved in
data collection, hyperparameter tuning, etc.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import sys
from datetime import datetime
from time import process_time
# importing from our other modules
from create_dataloaders import get_binary_dataloader
from models import FFW_Autoencoder, FFW_Autoencoder_v2, FFW_Autoencoder_v3, train

def execute_sanity_check(data_path):
    '''
    Method to carry out the sanity check experiment of training our FFW
    encoder-decoder model built without a smaller bottleneck to check
    that the model learns or approximate the identity function
    '''
    # 1) load in the data from the path; we're going to plug in the data path
    # 'binary_spike_matrix_data/all_trial_data_BINNED_715093703_15_20_01.npy'
    with open(data_path, 'rb') as f:
        spikes = np.load(f, allow_pickle=True)

    # 2) generate our data tensor object, and use this to create our dataset
    # and dataloader objects
    tnsr_spikes = torch.from_numpy(spikes).float()
    # spikes_dataset = MultitrialDataset(tnsr_spikes, tau=0) #<- in the sanity check we use no context
    spikes_dataloader = get_binary_dataloader(data_path, batch_size=100, tau=0, multitrial=True) #<- ... and this arbitrary batch size

    # 3) Get the dimensionality of the data and initialize and train the model
    N = tnsr_spikes.shape[1]
    M = N #<- sanity check: no bottleneck
    model = FFW_Autoencoder(N,M,tau=0)

    output = train(model, bce_weight=1, custom_dataloader=spikes_dataloader)

    # return the trained model and our output log
    return model, output

def run_experiment(data_path, num_layers, bottleneck_dim, bce_weight, tau, epochs, learning_rate=0.0025, batch_size=100, store_output=True):
    '''
    Generic experiment method that will execute the training for a model
    given by a specific set of hyperparameters given as input.
    '''

    with open(data_path, 'rb') as f:
        spikes = np.load(f, allow_pickle=True)

    # 2) generate our data tensor object, and use this to create our dataset
    # and dataloader objects
    tnsr_spikes = torch.from_numpy(spikes).float()
    # spikes_dataset = MultitrialDataset(tnsr_spikes, tau=0) #<- in the sanity check we use no context
    spikes_dataloader = get_binary_dataloader(data_path, batch_size=batch_size, tau=tau, multitrial=True)

    # 3) Get the dimensionality of the data and initialize and train the model
    N = tnsr_spikes.shape[1]
    M = bottleneck_dim

    # 4) choose model version with the specified number of layers
    if num_layers == 1:
        model = FFW_Autoencoder(N, M, tau=tau)
    elif num_layers == 2:
        model = FFW_Autoencoder_v2(N, M, tau=tau)
    elif num_layers == 3:
        model = FFW_Autoencoder_v3(N, M, tau=tau)
    else:
        raise ValueError('The number of layers parameter must be the integer 1, 2, or 3')

    # if store_output is true, this this will return the output log, but if it's set to false this will just
    # be an empty list
    print('STARTING TRAINING AT ',process_time(), datetime.now(), flush=True)
    output = train(model, bce_weight=bce_weight, custom_dataloader=spikes_dataloader, num_epochs=epochs, learning_rate=learning_rate, store_output=store_output)

    # return the trained model, and if store_output=True also our output log
    if store_output:
        return model, output
    else:
        return model

def calculate_avg_pred_prob(model, data, tau):
    '''
    Function to take in a trained model and our data tensor, and calculate the average
    predicted probability the model gives under the input conditions of spike and no spike.
    This is calculated by going through all data and preparing our predictions over all samples,
    and then separating out the positive (spike) samples (i.e. individual entries in the spike matrices)
    and negative (no spike) samples, and calculating the BCE over these two populations. This effectively
    separates the two terms of BCE and lets us calculate the model's probabilty of firing or not firing.
    '''
    #### Step 1: Iterate through our input (sample by sample and trial by trial) constructing 
    #### output matrices of corresponding arrangement (trial, unit, sample)

    # initialize output array
    # --> getting the dimensions of our output from the input and the window size
    num_trials = data.shape[0]
    num_units = data.shape[1]
    num_recon_samples = data.shape[2] - 2*tau
    reconstructed_output = torch.from_numpy((np.zeros((num_trials, num_units, num_recon_samples)))).float()

    # the loop to get the reconstructed outputs and populate the output array
    for trial in range(num_trials):
        for ind in range(num_recon_samples):
            # we get our indexing logic from our dataset object
            input_sample = data[trial,:,ind:ind+2*tau+1]
            # ... and recall that this is the windowed input for an output indexed by
            # data_tensor[trial, :, ind+self.tau]
            
            # now we create a single-sample batch
            single_sample_batch = torch.unsqueeze(input_sample, 0)

            # ... and feed it through our model:
            # (we have to use no_grad() because we don't want a gradient to get stored
            # with these outputs, we just want the plain numbers)
            with torch.no_grad():
                sample_recon = model.forward(single_sample_batch)
            
            reconstructed_output[trial,:,ind] = sample_recon

    #### Step 2: cut the tau samples off either side of the input so we have input 
    #### and output arrays of the same shape (with correspinding indexing

    # have to use separate delete statements for the two margins we want to take off
    # (looks like you can't do lists of intervals this way)
    if tau > 0:
        trimmed_input = np.delete(data, np.s_[:tau],axis=2)
        trimmed_input = np.delete(trimmed_input, np.s_[-tau:],axis=2)
    else:
        trimmed_input = data

    #### Step 3: Get the indices of all input cells with 1s and 0s separately

    # keep a separate list of the coord lists for the two cases
    trial_zero_coords = []
    trial_one_coords = []

    for trial in range(num_trials):
        us1, cs1 = np.argwhere(trimmed_input[trial])
        trial_one_coords.append([us1.tolist(), cs1.tolist()])
        
        us0, cs0 = np.argwhere(trimmed_input[trial] == 0)
        trial_zero_coords.append([us0.tolist(), cs0.tolist()])

    #### Step 4: Index into the trials with the zero and one coords and calculate the loss (summing
    #### over all trials and the dividing by the respective number of individual class 
    #### (spike/nospike) samples that were fed)

    bce_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')

    # accumulators for bce under spike and no spike, and for number of spike and no spike samples
    total_bce_nospike = 0
    total_bce_spike = 0
    nospike_samples = 0
    spike_samples = 0

    # iterating over all of the trials
    for trial in range(num_trials):
        # accumulating the total bce variables
        total_bce_nospike += bce_sum(reconstructed_output[trial][trial_zero_coords[trial]], trimmed_input[trial][trial_zero_coords[trial]])
        total_bce_spike += bce_sum(reconstructed_output[trial][trial_one_coords[trial]], trimmed_input[trial][trial_one_coords[trial]])
        # accumulating the number of samples
        nospike_samples += len(trial_zero_coords[trial][0])
        spike_samples += len(trial_one_coords[trial][0])
        
    avg_bce_nospike = total_bce_nospike/nospike_samples
    avg_bce_spike = total_bce_spike/spike_samples
        
    nospike_pred_prob = 1 - np.e**(-1*avg_bce_nospike)
    spike_pred_prob = np.e**(-1*avg_bce_spike)

    return nospike_pred_prob.item(), spike_pred_prob.item()

def execute_set_of_experiments(data_path):
    '''
    Function to execute the set of experiments given by a set of hyperparameters we
    specify here. The hyperparameters and the results of the results of the training
    will be streamed to a csv file, and the trained models will be saved to a directory
    created for a specific experimental session.

    Reproducability for the models is set in the training method, which initializes
    the torch random seed to 42 at the start of training.
    '''
    # specifying the hyperparameters we will test
    layers = [1,2,3]
    windows = [0,10,20,50]
    bottlenecks = [10,50,100]
    bce_weights = [25,50,100]

    # IMPT: When an experiment sweep crashes, we can rerun the sweep with an if statement
    # this will skip the trials we've already done:
    already_completed_combinations = [(1,0,10,25), (1,0,10,50), (1,0,10,100),
                                    (1,0,50,25), (1,0,50,50), (1,0,50,100),
                                    (1,0,100,25), (1,0,100,50), (1,0,100,100),
                                    (1,10,10,25), (1,10,10,50), (1,10,10,100),
                                    (1,10,50,25)]

    # setting a constant number of epochs for all experiments
    epochs = 5
    # learning rate set in the training method
    # also loading the data into memory so we can feed this into our calc prob method
    with open(data_path, 'rb') as f:
        spikes = np.load(f, allow_pickle=True)

    # datetime to create timestamped experiment directory name
    now = datetime.now()
    day_hours_mins_string = now.strftime("_%m_%d_%H_%M")
    experiment_dir = 'experiments'+day_hours_mins_string
    os.mkdir(experiment_dir)

    # create log file (csv) and instantiate it with the column headers
    # for our hyperparameters and performance metrics
    f = open(experiment_dir+'/log_file.csv', 'w')
    writer = csv.writer(f)
    header = ['layers', 'tau', 'bottleneck', 'BCE weight', 'avg. prob | spike', 'avg. prob | no spike']
    writer.writerow(header)
    # flush() updates the file without closing it
    f.flush()

    # our nested loops for all of our HP combinations
    for num_layers in layers:
        for tau in windows:
            for M in bottlenecks:
                for w in bce_weights:
                    # IMPT: if statement to skip the experiments that worked before the last run crashed
                    if (num_layers, tau, M, w) not in already_completed_combinations:    
                        print('----------------------------------------------------------------------')
                        print('Running experiment with the hyperparameters:')
                        print('Layers: ',str(num_layers),' Tau: ',str(tau),' M: ',str(M),' w: ',str(w))
                        # train the model with the given set of HPs
                        # --> we need to turn store_output off here because I think storing the output logs in memory is clobbering the CPU capacity
                        model = run_experiment(data_path=data_path, num_layers=num_layers, bottleneck_dim=M, bce_weight=w, tau=tau, epochs=epochs, store_output=False)
                        # calculating the avg pred probs for the trained model
                        nospike_pred_prob, spike_pred_prob = calculate_avg_pred_prob(model, torch.from_numpy(spikes).float(), tau)
                        # push the row to the log file
                        row = [str(num_layers), str(tau), str(M), str(w), str(spike_pred_prob), str(nospike_pred_prob)]
                        writer.writerow(row)
                        f.flush()
                        # save the model and the state dict
                        # --> we'll use filenames that list the hyperparameters in order and whether its a state dict or not
                        filename = 'model_%s_%s_%s_%s'%tuple(row[:4])
                        torch.save(model, experiment_dir+'/'+filename+'.pt')
                        torch.save(model.state_dict(), experiment_dir+'/'+filename+'_SD.pt')

    f.close()
    
def run_exp_sweep():
    '''
    wrapper function to run the experiment sweep from the command line
    '''
    dp = 'binary_spike_matrix_data/all_trial_data_BINNED_715093703_11_30_50.npy'
    execute_set_of_experiments(dp)

def run_exp_from_job_array(tau, layers, bottleneck, bce_weight, output_dir):
    '''
    Function to make a single experiment execution for a given set of hyperparameters read in
    from an input file that's fed in via the Slurm job array protocol.

    Writes output to a single-row output csv (with parameter headers)
    '''
    # right now we're running this batch job with 4 cores per task, so we set the number
    # of torch threads to 4
    torch.set_num_threads(4)

    # ADDING PRINT STATEMENTS TO SHOW CPU TIME AND ABSOLUTE TIME AT EACH MAJOR STEP
    print('ENTERING FUNCTION AT ',process_time(), datetime.now(), flush=True)

#    # MOVING THE LOG FILE BUSINESS TO AFTER THE MODEL TRAINING AND EVALUATION
#    # (I'm getting errors that the flush statements are being made to a closed file
#    # create mini-log file (csv) and instantiate it with the column headers
#    # for our hyperparameters and performance metrics
#    # --> create log file name using the hyperparameters (in the order we had them before - see above ^)
#    filename = 'log_file_%s_%s_%s_%s.csv'%(str(layers),str(tau),str(bottleneck),str(bce_weight))
#    f = open(output_dir+filename, 'w')
#    writer = csv.writer(f)
#    header = ['layers', 'tau', 'bottleneck', 'BCE weight', 'avg. prob | spike', 'avg. prob | no spike']
#    writer.writerow(header)
#    # flush() updates the file without closing it
#    f.flush()

    # run experiment for given set of HPs
    epochs = 5 #<- set the number of epochs beforehand
    # defining the fixed data path that will provide all experiments' data
    data_path = 'binary_spike_matrix_data/all_trial_data_BINNED_715093703_11_30_50.npy'
    # still have to load the spikes into memory to feed the object into the calc. avg. pred. prob. method
    print('LOADING DATA AT ',process_time(), datetime.now(), flush=True)
    with open(data_path, 'rb') as f:
        spikes = np.load(f, allow_pickle=True)
   
    print('RUNNING EXPERIMENT AT ',process_time(), datetime.now(), flush=True) 
    model = run_experiment(data_path=data_path, num_layers=layers, bottleneck_dim=bottleneck, bce_weight=bce_weight, tau=tau, epochs=epochs, store_output=False)
    print('CALCULATING MODEL PERFORMANCE AT ',process_time(), datetime.now(), flush=True)
    nospike_pred_prob, spike_pred_prob = calculate_avg_pred_prob(model, torch.from_numpy(spikes).float(), tau)

    print('LOGGING DATA AT ',process_time(), datetime.now(), flush=True)
    # CREATING AND POPULATING THE LOG FILE
    # create mini-log file (csv) and instantiate it with the column headers
    # for our hyperparameters and performance metrics
    # --> create log file name using the hyperparameters (in the order we had them before - see above ^)
    filename = 'log_file_%s_%s_%s_%s.csv'%(str(layers),str(tau),str(bottleneck),str(bce_weight))
    f = open(output_dir+filename, 'w')
    writer = csv.writer(f)
    header = ['layers', 'tau', 'bottleneck', 'BCE weight', 'avg. prob | spike', 'avg. prob | no spike']
    writer.writerow(header)
    # flush() updates the file without closing it
    # f.flush()
    

    # push the row to the log file
    row = [str(layers), str(tau), str(bottleneck), str(bce_weight), str(spike_pred_prob), str(nospike_pred_prob)]
    writer.writerow(row)
    # f.flush()
    f.close()


def main():
    '''
    main method for quick experimentation and debugging
    '''
    from time import process_time
    from datetime import datetime

    torch.set_num_threads(4)
    #print(torch.__config__.parallel_info(), flush=True)

    dp = 'binary_spike_matrix_data/all_trial_data_BINNED_715093703_11_30_50.npy'
    m = 5
    w = 100
    tau = 10
    epochs = 1
    learning_rate = 0.0025
    batch_size = 100

    print('starting experiment for m=%s, w=%s, tau=%s, epochs=%s, learning_rate=%s, batch_size=%s'%(m,w,tau,epochs,learning_rate,batch_size), flush=True)

    with open(dp, 'rb') as f:
        spikes = np.load(f, allow_pickle=True)

    print('starting model training at ', process_time(), flush=True)
    print("now =", datetime.now(), flush=True)
    model = run_experiment(data_path=dp, num_layers=3, bottleneck_dim=m, bce_weight=w, tau=tau, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, store_output=False)

    nospike_pred_prob, spike_pred_prob = calculate_avg_pred_prob(model, torch.from_numpy(spikes).float(), tau)

    print('model avg. probability for no input spike',str(nospike_pred_prob),str(type(nospike_pred_prob)))
    print('model avg. probability for input spike',str(spike_pred_prob),str(type(spike_pred_prob)))
    print('finished experiment at ', process_time(), flush=True)
    print('now =', datetime.now(), flush=True)


# ==== REMOVE THIS IF INTENDING TO CALL THIS FILE WITHOUT INPUT FILE ====
# check if an input file was given...
if len(sys.argv) < 2:
    # and if there wasn't one given, print the error message and exit
    print('need to provide input file with specified hyperparameters')
    sys.exit(1)

input_file = open(sys.argv[1])
for line in input_file:
    exec(line)

output_dir = sys.argv[2] 
# =======================================================================


if __name__ == '__main__':
    #run_exp_sweep()
    #main()
    run_exp_from_job_array(tau, layers, bottleneck, bce_weight, output_dir)
