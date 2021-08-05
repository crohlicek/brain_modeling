# Dynamical systems reduction in experimental brain modeling
Files related to Brown University Data Science ScM research practicum investigating machine learning methods of dynamical system identification in experimental neural spike data. Models and experiments were implemented in Pytorch and data was taken from the Allen Institute Brain Atlas. A full write-up of methods and results can be found in Practicum_Report.pdf

## Files and Contents
***Files in this repository are organized based on the section of the report with which they're associated.***

- EDA
    - Scripts:
        - allensdk_data_acquisition.py
            - Code used to query the AllenSDK and construct the empirical spike dataset used through the project
    - Notebooks:
        - latent_space_analysis.ipynb
            - Analyses of the embedding/latent data space induced by a given embedding function
        - allensdk_data_inspection.ipynb
            - Analyses of the data characteristics of the data directly from the AllenSDK
        - allensdk_pca.ipynb
            - Code to query and inspect the data available through the AllenSDK, includes an example application of PCA to a selection of data
- Models/Results
    - Scripts:
        - models.py
            - Model definitions for the feedforward networks tested as candidate embedding functions/initial states for the encoder network in our next-step predictor model
        - experiments.py
            - Included testing code used to perform parameter sweeps and calculate model performance
        - create_dataloaders.py
            - Code to create the relevant dataloader and dataset objects required by Pytorch models
    - Notebooks:
        - RNN_phase2_onestep.ipynb
            - Notebook containing development and exploration of the next-step predictor model
- Diagnostics
    - simulated_nondynamical_dataset.ipynb
        - Creation of nondynamical dataset and analyses of dataset characteristics
    - simulated_dynamical_dataset.ipynb
        - Creation of dynamical dataset and analyses of dataset characteristics
- Environments:
    - allensdk_env_requirements.txt
        - Requirements file for conda environment required for the files which query from the AllenSDK (allensdk_pca.ipynb and allensdk_data_acquisition.py)
    - pytorch_env_requirements.txt
        - Requirements file for all other files

***Notes:***
 - All files are meant to be run using the `pytorch_env` environment except for `allensdk_pca.ipynb` and `allensdk_data_acquisition.py`, which need to be run from `allensdk_env`. The requirements for those environemnts are both given in the Environments directory
 - The paths in these files do not reflect the hierarchy in which they are stored in this repository. The files are stored here in a way that is meant to align logically with the organization of the report
