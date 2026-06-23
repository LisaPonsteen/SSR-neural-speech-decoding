# SingleWordProductionDutch

Neural speech decoding pipeline using intracranial EEG data. This project implements State-Space Phase Estimation (SSPE) for extracting neural features and reconstructing speech spectrograms from brain activity.

## Dataset
The project uses intracranial EEG data from [here](https://osf.io/nrgx6/) described 
in this [article](https://www.nature.com/articles/s41597-022-01542-9).

## Dependencies
The scripts require Python >= 3.6 and the following packages
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/scipylib/index.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/) 
* [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)
* [somata](https://github.com/ckyrkou/somata) - Oscillator detection

## Repository content
To recreate the experiments, run the following scripts.
* __notebookfunctions.py__: Utility functions including `run_pipeline()` to execute the complete analysis pipeline
* __extract_features.py__: Reads in the iBIDS dataset and extracts features which are then saved to './features'
* __phaseEM.py__: Causal phase estimation using state-space oscillator models
* __reconstruction_minimal.py__: Reconstructs the spectrogram from the neural features in a 10-fold cross-validation and synthesizes the audio using the Method described by Griffin and Lim.
* __reconstuctWave.py__: Synthesizes an audio waveform using the method described by Griffin-Lim
* __MelFilterBank.py__: Applies mel filter banks to spectrograms.

## Jupyter Notebooks
The jupyter notebook `report_figures.ipynb` can be used to reproduce the results from the paper. (The rest of the notebooks were for exploratory analysis, and are a bit chaotic)

## Usage
Make sure to set the correct paths for 
`path_bids` (location to dataset), `path_somata` (folder in which you want to store somata results) and `path_result` (folder in which you want to store results)
