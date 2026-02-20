import os

import pandas as pd
import numpy as np 
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack

from pynwb import NWBHDF5IO
import MelFilterBank as mel
import phaseEM
import config as cfg
from joblib import Parallel, delayed

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractSSPE(data, initParams_list, use_channels=None, sr=1024, windowLength=0.05, frameshift=0.01):
    """
    alternative to extract HG, has the same output shapes.

    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    initParams_list: list of dictionaries
        List of dictionaries containing the initialization parameters for the phaseEM algorithm
    use_channels: list of int
        List of channels to use. if none, it uses all
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    
    #remove (harmonics of) line noise
    for i in range (0,5):
        sos = scipy.signal.iirfilter(4, [(i*100+49)/(sr/2),(i*100+51)/(sr/2)],btype='bandstop',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

    #for data.shape == (samples, channels):
    if use_channels != None:
        n_channels = len(use_channels)
    else:
        n_channels = data.shape[1]

    '''
    for ch in range(n_channels):
        #for ch in use_channels: #can do this to speed things up for now we only do a few channels
        d = data[:, ch]
        # causal phase estimation
        phase, allX_full, returnParams = phaseEM.causalPhaseEM_MKmdl_noSeg(
            d, initParams, flagNoFit=True
        )
        # amplitude from oscillator states
        #amp_ch = np.sqrt(allX_full[:, 0]**2 + allX_full[:, 1]**2)
        #amp[:len(amp_ch), ch] = amp_ch
        amp_ch = np.sqrt(
            allX_full[:, :, 0]**2 +
            allX_full[:, :, 1]**2
        )
        amp[:len(amp_ch), ch, :] = amp_ch
    '''
    # --- Parallel Processing Block ---
    # We wrap the phaseEM call in a helper or call it directly.
    # n_jobs=-1 uses all available CPU cores.
    print(f"Starting parallel SSPE extraction for {n_channels} channels...")
    
    # Find max number of frequencies across all channels for padding
    #max_freq = max(len(params["freqs"]) for params in initParams_list)
    #max_freq = 7
    #amp = np.zeros((data.shape[0], n_channels, max_freq)) #amp.shape = (time, channels, max_freq)


    results = Parallel(n_jobs=-1)(
        delayed(phaseEM.causalPhaseEM_MKmdl_noSeg)(
            data[:, ch], 
            initParams_list[ch], 
            flagNoFit=True
        ) for ch in range(n_channels)
    )
    # results is now a list of tuples: [(phase, allX_full, returnParams), ...]
    
    # 1. Determine the total number of oscillators across all channels to avoid padding
    total_oscillators = sum(allX_full.shape[1] for _, allX_full, _ in results)
    
    # 2. Pre-allocate the feature matrix with the EXACT size needed
    feat = np.zeros((numWindows, total_oscillators))
    
    # 3. Iterate through channels and fill columns sequentially
    current_col = 0

    for ch_idx, (phase, allX_full, returnParams) in enumerate(results):
        # allX_full shape is (time, freq, state_dim) where state_dim is usually 2 (sine/cosine)
        amp_ch = np.sqrt(
            allX_full[:, :, 0]**2 +
            allX_full[:, :, 1]**2
        )
        # Only fill the frequencies that this channel actually has
        n_freq_ch = amp_ch.shape[1]
        # Process each oscillator for this specific channel
        for osc_idx in range(n_freq_ch):
            osc_data = amp_ch[:, osc_idx]
            
            # Apply windowing to this specific oscillator
            for win in range(numWindows):
                start = int(np.floor((win * frameshift) * sr))
                stop = int(np.floor(start + windowLength * sr))
                
                # Take the mean of the amplitude within the window
                feat[win, current_col] = np.mean(osc_data[start:stop])
            
            # Move to the next column in the global feature matrix
            current_col += 1
        #amp[:len(amp_ch), ch_idx, :n_freq_ch] = amp_ch

    '''
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        #feat[win,:] = np.mean(amp[start:stop,:],axis=0)
        window_amp = np.mean(amp[start:stop, :, :], axis=0)  # (channels, freqs)
        feat[win, :] = window_amp.reshape(-1)
    '''
    
    return feat



def extractSSPE_sequential(data, initParams, use_channels=None, sr=1024, windowLength=0.05, frameshift=0.01):
    """
    alternative to extract HG, has the same output shapes.


    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    eeg_sr=1024
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    
    #remove (harmonics of) line noise
    for i in range (0,5):
        sos = scipy.signal.iirfilter(4, [(i*100+49)/(eeg_sr/2),(i*100+51)/(eeg_sr/2)],btype='bandstop',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

    #for data.shape == (samples, channels):
    if use_channels != None:
        n_channels = len(use_channels)
    else:
        n_channels = data.shape[1]
    n_freq = len(initParams["freqs"])
    amp = np.zeros((data.shape[0], n_channels, n_freq)) #amp.shape = (time, channels, freqs)
    
    for ch in range(n_channels):
        #for ch in use_channels: #can do this to speed things up for now we only do a few channels
        d = data[:, ch]
        # causal phase estimation
        phase, allX_full, returnParams = phaseEM.causalPhaseEM_MKmdl_noSeg(
            d, initParams, flagNoFit=True
        )
        # amplitude from oscillator states
        #amp_ch = np.sqrt(allX_full[:, 0]**2 + allX_full[:, 1]**2)
        #amp[:len(amp_ch), ch] = amp_ch
        amp_ch = np.sqrt(
            allX_full[:, :, 0]**2 +
            allX_full[:, :, 1]**2
        )
        amp[:len(amp_ch), ch, :] = amp_ch
        

    #Create feature space
    feat = np.zeros((numWindows, n_channels * n_freq)) #feat.shape = (windows, channels * freqs)

    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        #feat[win,:] = np.mean(amp[start:stop,:],axis=0)
        window_amp = np.mean(amp[start:stop, :, :], axis=0)  # (channels, freqs)
        feat[win, :] = window_amp.reshape(-1)
    return feat

def extractHG(data, sr, windowLength=0.05, frameshift=0.01):
    """
    Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Attenuate second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    #Create feature space
    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() #Add 'F' if stacked the same as matlab
    return featStacked

def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """
    Downsamples non-numerical data by using the mode
    
    Parameters
    ----------
    labels: array of str
        Label time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    numWindows=int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        vals, counts = np.unique(labels[start:stop], return_counts=True)
        newLabels[w] = vals[np.argmax(counts)]
        #newLabels[w]=scipy.stats.mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
    return newLabels

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram: array (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = scipy.signal.windows.hann(int(np.floor(windowLength*sr + 1)))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w*frameshift)*sr))
        stop_audio = int(np.floor(start_audio+windowLength*sr))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 23, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def nameVector(elecs, modelOrder=4):
    """
    Creates list of electrode names
    
    Parameters
    ----------
    elecs: array of str
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names
    Returns
    ----------
    names: array of str
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  #Add 'F' if stacked the same as matlab


def run_extract_features(pt, initParams_list, SSPE=True):
    '''
    extracts eeg features, processes audio and alligns and saves everything for a single participant
    
    Parameters
    ----------
    pts: array
        array of strings with the participants you want to analyze. e.g. pts = ['sub-01']
    Fs: 2Darray
        array with per participant, the frequencies you want to track if using the SSPE method. e.g. Fs = [[63, 78, 106]]
    SSPE: boolean
        True=use SSPE feature extraction. False=use HG extraction
    '''
    winL = 0.05
    frameshift = 0.01
    modelOrder = 4
    stepSize = 5
    path_bids = '/Users/lisa/Documents/DSAI_year2/Marble/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS'
    path_output = r'./features'
        
    #Load data
    io = NWBHDF5IO(os.path.join(path_bids,pt,'ieeg',f'{pt}_task-wordProduction_ieeg.nwb'), 'r')
    nwbfile = io.read()
    #sEEG
    eeg = nwbfile.acquisition['iEEG'].data[:]
    eeg_sr = 1024
    #audio
    audio = nwbfile.acquisition['Audio'].data[:]
    audio_sr = 48000
    #words (markers)
    words = nwbfile.acquisition['Stimulus'].data[:]
    words = np.array(words, dtype=str)
    io.close()
    #channels
    channels = pd.read_csv(os.path.join(path_bids,pt,'ieeg',f'{pt}_task-wordProduction_channels.tsv'), delimiter='\t')
    channels = np.array(channels['name'])

    #Extract features
    if not SSPE:
        #Extract HG features
        feat = extractHG(eeg, eeg_sr, windowLength=winL, frameshift=frameshift)
    else:
        #Extract SSPE features (all channels at once with channel-specific initParams)
        feat = extractSSPE(eeg, initParams_list)
        print("extracted features")
    
    #Stack features
    feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)
    
    #Process Audio
    target_SR = 16000
    audio = scipy.signal.decimate(audio,int(audio_sr / target_SR))
    audio_sr = target_SR
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    os.makedirs(os.path.join(path_output), exist_ok=True)
    scipy.io.wavfile.write(os.path.join(path_output,f'{pt}_orig_audio.wav'),audio_sr,scaled)   

    #Extract spectrogram
    melSpec = extractMelSpecs(scaled,audio_sr,windowLength=winL,frameshift=frameshift)
    
    #Align to EEG features
    words = downsampleLabels(words,eeg_sr,windowLength=winL,frameshift=frameshift)
    words = words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]
    melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
    #adjust length (differences might occur due to rounding in the number of windows)
    if melSpec.shape[0]!=feat.shape[0]:
        tLen = np.min([melSpec.shape[0],feat.shape[0]])
        melSpec = melSpec[:tLen,:]
        feat = feat[:tLen,:]
    
    #Create feature names by appending the temporal shift 
    feature_names = nameVector(channels[:,None], modelOrder=modelOrder)

    #Save everything
    if SSPE:
        np.save(os.path.join(path_output,f'{pt}_SSPEfeat.npy'), feat)
    else:
        np.save(os.path.join(path_output,f'{pt}_feat.npy'), feat)
    np.save(os.path.join(path_output,f'{pt}_procWords.npy'), words)
    np.save(os.path.join(path_output,f'{pt}_spec.npy'), melSpec)
    np.save(os.path.join(path_output,f'{pt}_feat_names.npy'), feature_names)



if __name__=="__main__":
    pts = ['sub-06']
    Fs = [[1, 6, 26, 54, 73]]
    ampVec = [1.0, 0.999, 0.991, 0.983, 0.970]
    sigma = [6.56, 4.29, 3.55, 0.79, 0.35]
    run_extract_features(pts,Fs, ampVec, sigma)
