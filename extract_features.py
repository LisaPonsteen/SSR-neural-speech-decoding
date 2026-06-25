import os

import pandas as pd
import numpy as np 
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.io.wavfile
import scipy.fftpack

from pynwb import NWBHDF5IO
import MelFilterBank as mel
import phaseEM
import notebookfunctions
from joblib import Parallel, delayed

winL = 0.05 #windowlength 
frameshift = 0.01
modelOrder = 4
stepSize = 5
path_bids = '/Users/lisa/Documents/DSAI_year2/Marble/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS'
path_output = r'./features'
sr=1024

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]


def extractSSPE(data, initParams_list, include_phase = True, include_pac = True):
    """
    Extract State-Space Phase Estimation (SSPE) features from EEG data.
    
    This is an alternative to extractHG with the same output shapes. It tracks multiple oscillators per channel and extract
    amplitude, phase, and phase-amplitude coupling (PAC) features.

    Parameters
    ----------
    data : array (samples, channels)
        EEG time series data
    initParams_list : list of dict
        List of dictionaries containing initialization parameters for the phaseEM
        algorithm for each channel. Each dict should contain: freqs, Fs, ampVec,
        sigmaFreqs, sigmaObs, windowSize, lowFreqBand
    include_phase : bool, optional
        Whether to include phase features for low-frequency oscillators (<20 Hz).
        Phase features are represented as sine and cosine components. Default is True.
    include_pac : bool, optional
        Whether to include phase-amplitude coupling features between low-frequency
        oscillators and high-gamma oscillators. Default is True.
        
    Returns
    -------
    feat : array (windows, oscillators)
        Feature matrix where each column represents an oscillator feature.
        Features include: amplitude for all oscillators, sine/cosine phase for
        low-frequency oscillators (if include_phase=True), and PAC features for
        low-frequency oscillators (if include_pac=True).
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-winL*sr)/(frameshift*sr)))
    
    #remove (harmonics of) line noise
    for i in range (0,4):
        sos = scipy.signal.iirfilter(4, [(i*100+49)/(sr/2),(i*100+51)/(sr/2)],btype='bandstop',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)

    #note: I saw that with extractHG(), line noise is removed around 100 and 150 hz
    #we should probably filter [(i*50+49)/(sr/2),(i*50+51)/(sr/2)] to also get rid of the small peak in psd sometimes occuring at 100 hz

    #for data.shape == (samples, channels):
    n_channels = data.shape[1]

    # n_jobs=-1 uses all available CPU cores.
    print(f"Starting parallel SSPE extraction for {n_channels} channels...")
    results = Parallel(n_jobs=4)(
        delayed(phaseEM.causalPhaseEM_MKmdl_noSeg)(
            data[:, ch], 
            initParams_list[ch], 
            flagNoFit=True
        ) for ch in range(n_channels)
    )
    #results is a list of tuples: [(phase, allX_full, returnParams), ...]
    
    # the threshold to devine low frequency oscillators
    LOW_FREQ_THRESHOLD = 20.0

    #determine how big the future space must be
    total_oscillators = sum(allX_full.shape[1] for _, allX_full, _ in results)

    if include_phase or include_pac:
        total_low_freq_oscillators = sum(
            sum(1 for f in params["freqs"] if f < LOW_FREQ_THRESHOLD) 
            for params in initParams_list
        )
        if include_phase:
            total_oscillators +=2*total_low_freq_oscillators #cos and sin phase features for low freq oscs
        if include_pac:
            total_oscillators +=total_low_freq_oscillators #each low freq oscillators interaction with 1 hg osc

    # pre-allocate the feature matrix with the size needed
    feat = np.zeros((numWindows, total_oscillators))
    
    # iterate through channels and fill columns of the matrix sequentially
    current_col = 0

    for ch_idx, (phase, allX_full, returnParams) in enumerate(results):
        # allX_full shape is (time, freq, state_dim) where state_dim is 2 (sine/cosine)
        amp_ch = np.sqrt(
            allX_full[:, :, 0]**2 +
            allX_full[:, :, 1]**2
        )
        phase_ch = np.angle(allX_full[:, :, 0] + 1j * allX_full[:, :, 1])

        n_freq_ch = amp_ch.shape[1]
        ch_freqs = initParams_list[ch_idx]['freqs']
        assert len(ch_freqs) == n_freq_ch, "smth wrong with nr of oscillators and features"

        high_osc_idx = None
        if include_pac:
            #get next high_osc_idx from initparams_list, which is the one most prominant in the signal since it is the first one found by somata, or else added from the template  (since init params is by that order)
            for idx, freq in enumerate(ch_freqs):
                if 70 < freq < 170:
                    high_osc_idx = idx
                    break
            assert high_osc_idx is not None, f"Channel {ch_idx} appears to have no HG oscillator"
        
        # Process each oscillator for this specific channel
        for osc_idx in range(n_freq_ch):
            osc_data = amp_ch[:, osc_idx]
            
            # Apply windowing to this specific oscillator
            for win in range(numWindows):
                start = int(np.floor((win * frameshift) * sr))
                stop = int(np.floor(start + winL * sr))
                
                # Take the mean of the amplitude within the window
                feat[win, current_col] = np.mean(osc_data[start:stop])

                #get this osc's freq from initparams_list:

                if ch_freqs[osc_idx] < LOW_FREQ_THRESHOLD:
                    p_low = phase_ch[start:stop, osc_idx]

                    if include_phase:
                        #pure phase feature:
                        complex_mean = np.mean(np.exp(1j * p_low))
                        feat_sin = np.imag(complex_mean) # give both sin and cos feats since decoder cannot process imaginary numbers
                        feat_cos = np.real(complex_mean)
                        feat[win, current_col+1] = feat_sin
                        feat[win, current_col+2] = feat_cos

                    if include_pac:
                        
                        a_high = amp_ch[start:stop, high_osc_idx]

                        #pac feature: use mean vector length, abs mean of amp hg freq * angle low freq
                        pac_feature = np.abs(np.mean(a_high * np.exp(1j * p_low)))
                        if include_phase:
                            feat[win, current_col+3] = pac_feature
                        else:
                            feat[win, current_col+1] = pac_feature


            # Move to the next empty column in the global feature matrix 
            current_col += 1
            if ch_freqs[osc_idx] < LOW_FREQ_THRESHOLD:
                if include_phase:
                    current_col += 2
                if include_pac:
                    current_col += 1
    return feat

def extractHG(data, low_pass = False):
    """
    Extract high-gamma band envelope features from EEG data using Hilbert transform. If low-pass=True, also frequencies <20 Hz are included.
    
    This function windows the data, applies bandpass filtering in the high-gamma
    range (70-170 Hz) and low-pass filtering (<20 Hz) if low_pass=True, removes line noise harmonics,
    and extracts the envelope using the Hilbert transform.

    
    Parameters
    ----------
    data : array (samples, channels)
        EEG time series
    low_pass : bool, optional
        If True, apply low-pass filtering before high-gamma extraction. Default is False.
        
    Returns
    -------
    feat : array (windows, channels) or (windows, channels * 2) if low_pass=True
        Envelope feature matrix. Each row corresponds to a time window
        and each column to a channel.
        If low_pass=ture, the matrix has interleaved low-pass and high-gamma feature columns:
        [ch0_low, ch0_hg, ch1_low, ch1_hg, ...]
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-winL*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    #sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    #data = scipy.signal.sosfiltfilt(sos,data,axis=0) (original uncausal method)
    #data = scipy.signal.sosfilt(sos,data,axis=0) (original causal method, produces roughly the same result as uncausal method)
    
    # Design FIR filter with 400ms memory for causal filtering
    filter_memory_samples = int(0.4 * sr)
    
    # Process high-gamma band
    data_hg = data.copy()
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data_hg = scipy.signal.sosfilt(sos,data_hg,axis=0)
    
    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data_hg = scipy.signal.sosfiltfilt(sos,data_hg,axis=0)
    #Attenuate second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data_hg = scipy.signal.sosfiltfilt(sos,data_hg,axis=0)
    data_hg = np.abs(hilbert3(data_hg))

    if low_pass:
        # Process low-pass band
        data_lp = data.copy()
        # Low-pass FIR filter (<20 Hz) using a Hamming window
        lp_filter = scipy.signal.firwin(filter_memory_samples, 20/(sr/2), 
                                        pass_zero='lowpass', window='hamming')
        data_lp = scipy.signal.lfilter(lp_filter, [1.0], data_lp, axis=0)
        data_lp = np.abs(hilbert3(data_lp))

    #Create feature space
    n_channels = data.shape[1]
    feat = np.zeros((numWindows, n_channels * 2)) if low_pass else np.zeros((numWindows, n_channels))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+winL*sr))
        if low_pass:
            # Interleave low-pass and high-gamma features
            feat[win, 0::2] = np.mean(data_lp[start:stop,:],axis=0)  # even indices: low-pass
            feat[win, 1::2] = np.mean(data_hg[start:stop,:],axis=0)  # odd indices: high-gamma
        else:
            feat[win, :] = np.mean(data_hg[start:stop,:],axis=0)
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
        featStacked[fNum,:]=ef.flatten()
    return featStacked

def downsampleLabels(labels):
    """
    Downsamples non-numerical data by using the mode

    Parameters
    ----------
    labels : array of str
        Label time series
    sr: int
        Sampling rate of the data
    winL: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    numWindows=int(np.floor((labels.shape[0]-winL*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+winL*sr))
        vals, counts = np.unique(labels[start:stop], return_counts=True)
        newLabels[w] = vals[np.argmax(counts)]
        #newLabels[w]=scipy.stats.mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
    return newLabels

def extractMelSpecs(audio, sr):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms

    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    winL: float
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
    numWindows=int(np.floor((audio.shape[0]-winL*sr)/(frameshift*sr)))
    win = scipy.signal.windows.hann(int(np.floor(winL*sr + 1)))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(winL*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w*frameshift)*sr))
        stop_audio = int(np.floor(start_audio+winL*sr))
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


def run_extract_features(pt, initParams_list=None, SSPE=True, include_phase = True, include_pac = True, low_pass=False, saveAs = None):
    """
    Extract EEG features, process audio, align data, and save for a single participant.
    
    This is the main pipeline function that loads data from NWB files, extracts
    neural features (either SSPE or high-gamma), processes audio to mel spectrograms,
    aligns the modalities temporally, and saves all outputs.
    
    Parameters
    ----------
    pt : str
        Participant identifier (e.g., 'sub-01')
    initParams_list : list of dict
        List of initialization parameters for SSPE algorithm per channel.
        Required if SSPE=True.
    SSPE : bool, optional
        If True, use SSPE feature extraction. If False, use high-gamma extraction.
        Default is True.
    include_phase : bool, optional
        If True, include phase information in SSPE features. Default is True.
    include_pac : bool, optional
        If True, include phase-amplitude coupling in SSPE features. Default is True.
    low_pass : bool, optional
        If True, include low-pass features in HG extraction. Default is False.
    saveAs : str, optional
        Custom filename prefix for to save feature files. If None, features aren't saved.
    
    
    The function saves the following files:
    - {pt}_SSPEfeat.npy or {pt}_causalfeat.npy: Neural features
    - {pt}_procWords.npy: Downsampled word labels
    - {pt}_spec.npy: Mel spectrogram of recorded ground-truth audio
    - {pt}_feat_names.npy: Feature names with temporal context
    - {pt}_orig_audio.wav: Processed audio file
    """
    
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
        feat = extractHG(eeg, low_pass=low_pass)
    else:
        #Extract SSPE features
        feat = extractSSPE(eeg, initParams_list, include_phase = include_phase, include_pac = include_pac)
    print("extracted features")
    
    #Stack features
    feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)
    
    '''
    #Process Audio
    target_SR = 16000
    audio = scipy.signal.decimate(audio,int(audio_sr / target_SR))
    audio_sr = target_SR
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    os.makedirs(os.path.join(path_output), exist_ok=True)
    scipy.io.wavfile.write(os.path.join(path_output,f'{pt}_orig_audio.wav'),audio_sr,scaled)   

    #Extract spectrogram
    melSpec = extractMelSpecs(scaled,audio_sr)
    
    #Align to EEG features
    words = downsampleLabels(words)
    words = words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]
    melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
    #adjust length (differences might occur due to rounding in the number of windows)
    if melSpec.shape[0]!=feat.shape[0]:
        tLen = np.min([melSpec.shape[0],feat.shape[0]])
        melSpec = melSpec[:tLen,:]
        feat = feat[:tLen,:]
    
    #Create feature names by appending the temporal shift 
    feature_names = nameVector(channels[:,None], modelOrder=modelOrder)
    '''
    #Save everything
    if saveAs: 
        np.save(os.path.join(path_output,f'{pt}{saveAs}.npy'), feat)
    else:
        if SSPE:
            np.save(os.path.join(path_output,f'{pt}_SSPEfeat.npy'), feat)
        else:
            np.save(os.path.join(path_output,f'{pt}_causalfeat.npy'), feat)

    '''
    np.save(os.path.join(path_output,f'{pt}_procWords.npy'), words)
    np.save(os.path.join(path_output,f'{pt}_spec.npy'), melSpec)
    np.save(os.path.join(path_output,f'{pt}_feat_names.npy'), feature_names)
    '''



if __name__=="__main__":
    pts = ['sub-06']
    initParams_list = notebookfunctions.get_initParams('sub-06')
    run_extract_features(pts,initParams_list)
