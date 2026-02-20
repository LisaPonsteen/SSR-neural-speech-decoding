import numpy as np
import os

import pandas as pd
import numpy as np 
from extract_features import *
import config as cfg

def highest_hg_mel_correlations(pt):
    """
    which 5 channels have the highest correlation to the
    melspectogram (HG features - mean spectogram)?

    hg: (T, n_channels) high-gamma features
    melSpec: (T, n_mels) mel spectrogram
    """
    #Load data
    io = NWBHDF5IO(os.path.join(cfg.PATH_BIDS,pt,'ieeg',f'{pt}_task-wordProduction_ieeg.nwb'), 'r')
    nwbfile = io.read()
    eeg = nwbfile.acquisition['iEEG'].data[:]
    eeg_sr = 1024

    feat_path = r'./features'
    hg = extractHG(eeg,eeg_sr) #compute unstacked features
    melSpec = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))

    # Reduce mel spectrogram to 1D speech envelope
    audio_ref = np.mean(melSpec, axis=1)
    audio_ref = (audio_ref - audio_ref.mean()) / audio_ref.std()

    n_channels = hg.shape[1]
    correlations = np.zeros(n_channels)

    for ch_idx in range(n_channels):
        x = hg[:, ch_idx]
        x = (x - x.mean()) / x.std()

        T = min(len(x), len(audio_ref)) #make sure they are the same size
        correlations[ch_idx] = np.corrcoef(x[:T], audio_ref[:T])[0, 1]

    top_channels = np.argsort(np.abs(correlations))[-5:][::-1]

    return top_channels, correlations

def get_channel_region(participant_id, ch_idx):
    # Load channel information

    channels_df = pd.read_csv(
        os.path.join(cfg.PATH_BIDS, participant_id, 'ieeg',
                f'{participant_id}_task-wordProduction_channels.tsv'),
                delimiter='\t'
            )
    region = channels_df.iloc[ch_idx]['description']
    return region

