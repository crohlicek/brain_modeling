'''
This file will have the constructors for the different kinds of neural
network models we will use for the various experiments. A given architecture
constructor can be imported and used in another file that will manage the
hyperparameter tuning processes.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# super basic baseline model
# Want to learn a nonlinear sample-to-sample embedding function
# (given by the encoder network – trained with a decoder to
# optimize for reconstruction ability)
class FFW_Autoencoder(nn.Module):
    '''
    Class for the generic feed forward autoencoder. "Autoencoder" is used loosely
    here, just to refer generally to the bottleneck structure and quasi-self-supervision
    being used. The tau parameter in the constructor tells the network whether or not the
    input data is windowed, and to what extent. The N and M parameters give the dimensionality
    of the number of recording channels and the number of time samples respectively.
    '''

    def __init__(self, N, M, tau):
        super(FFW_Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # logic for the number of total input size for a given window
            nn.Linear(N * (2 * tau + 1), M, bias=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(M, N, bias=True)
        )

    # ... and we specify the forward pass with our nonlinear activations
    def forward(self, x):
        # specifying a start_dim of 1 so we preserve the batch dimension
        x = torch.flatten(x,start_dim=1)
        x = self.encoder(x)
        # ReLU might not make sense as the output of the encoder
#         x = F.relu(x)
        x = torch.tanh(x)

        x = self.decoder(x)
        # removing the sigmoid because we're using BCE loss with the sigmoid incorporated
#         x = torch.sigmoid(x)
        return x

class FFW_Autoencoder_v2(FFW_Autoencoder):
    '''
    For experimentation we want to try versions of this architecture with different numbers
    of layers. This subclass will give a version of the FFW autoencoder with a 2-layer 
    encoder and decoder. We inherit from the FFW_Autoencoder class because we use the same
    forward pass method.
    '''
    def __init__(self, N, M, tau):
        super(FFW_Autoencoder, self).__init__()
        # in order to implement a funnel-shaped network we'll define the intermediate
        # dimensions here (calculated as the int-rounded averages of the dimension
        # on either side)
        input_size = N * (2 * tau + 1)
        enc_hidden_size = (input_size + M) // 2
        dec_hidden_size = (N + M) // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_size, enc_hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(enc_hidden_size, M, bias=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(M, dec_hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(dec_hidden_size, N, bias=True)
        )

class FFW_Autoencoder_v3(FFW_Autoencoder):
    '''
    Subclass for FFW autoencoder with a 3-layer encoder and decoder.
    '''
    def __init__(self, N, M, tau):
        super(FFW_Autoencoder, self).__init__()
        # in order to implement a funnel-shaped network we'll define the intermediate
        # dimensions here
        # For input dim I and output dim O, I - (I-M)/3 = (2/3)*I + (1/3)*M,
        # and I - 2*(I-M)/3 = (1/3)*I + (2/3)*M – these don't require negative number handling
        input_size = N * (2 * tau + 1)
        enc_hidden_size_1 = int((2/3)*input_size + (1/3)*M)
        enc_hidden_size_2 = int((1/3)*input_size + (2/3)*M)
        dec_hidden_size_1 = int((1/3)*N + (2/3)*M)
        dec_hidden_size_2 = int((2/3)*N + (1/3)*M)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, enc_hidden_size_1, bias=True),
            nn.Tanh(),
            nn.Linear(enc_hidden_size_1, enc_hidden_size_2, bias=True),
            nn.Tanh(),
            nn.Linear(enc_hidden_size_2, M, bias=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(M, dec_hidden_size_1, bias=True),
            nn.Tanh(),
            nn.Linear(dec_hidden_size_1, dec_hidden_size_2, bias=True),
            nn.Tanh(),
            nn.Linear(dec_hidden_size_2, N, bias=True)
        )

def train(model, bce_weight, custom_dataloader, num_epochs=1, learning_rate=0.0025, store_output=True):
    '''
    Generic training function that should be used with all of our models
    '''
    # torch.manual_seed(42)
    # ====================================================
    # proportion of spikes is given for the BCE weighting
    # --> the weighting we'll give to instances of spikes will be
    # --> (# zeros)/(# ones) across the whole input data matrix
    # ==> my first attempt (see above cell) at formulating the weight seemed too high
    # ==> so trying out a few hardcoded numbers
    pos_weight = torch.tensor(bce_weight)
    # ====================================================
    
    # criterion = nn.MSELoss() # mean square error loss
    # trying weighted BCE loss with incorporated sigmoid for stability
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # ===> trying just the BCELoss, I'll put in the sigmoid myself
#     criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)
    # going to input our custom dataloader here
    train_loader = custom_dataloader
    outputs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        # going to extract data and label because while in the memoryless
        # case these will be the same, they'll be different in the windowed case
        for i, (source, target) in enumerate(train_loader):
            ## ===============
            ## ==> don't want to squeeze when using batch_size>1, getting
            ## ==> rid of the batch dimension was okay when we had batch_size=1
            ## ==> but doesn't make sense in general
            # source = torch.squeeze(source)
            # target = torch.squeeze(target)
            ## ===============
            img = source
            # forward pass
            recon = model(img)
            # compute loss
            loss = criterion(recon, target)
            # zero out the gradient
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # parameter update
            optimizer.step()
            
            
            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000), flush=True)
                running_loss = 0.0
                # have to add the sigmoid manually becuase
                # we have it removed while using the BCEwithlogitsloss
            
            # store_output flag toggles whether or not we want to accumulate this data
            # --> for our experiments we may just need to model performance and not these specific items
            if store_output:
                outputs.append((epoch, loss.item(), img, torch.sigmoid(recon)),)
                
    return outputs
