"""
Utility functions for neural speech decoding pipeline. Methods developed and frequently used in notebooks were moved here."


This module provides helper functions for:
- Loading and preprocessing EEG data from NWB files
- Running SOMATA oscillator detection for frequency initialization
- Building frequency templates and SSPE initialization parameters
- Visualizing results and comparing methods
- Running the complete analysis pipeline
"""

import os

import numpy as np
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from extract_features import *
from reconstruction_minimal import *

from tqdm.auto import tqdm
from joblib import Parallel, delayed
import gc

from somata.oscillator_search import IterativeOscillatorModel as IterOsc
from somata.oscillator_search.helper_functions import get_knee

winL = 0.05
frameshift = 0.01
modelOrder = 4
stepSize = 5
sr=1024
pts = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05','sub-06','sub-07', 'sub-08', 'sub-09', 'sub-10']

path_bids = '/Users/lisa/Documents/DSAI_year2/Marble/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS'
path_somata = '/Users/lisa/Documents/DSAI_year2/SingleWordProductionDutch-1/somata'
path_result = '/Users/lisa/Documents/DSAI_year2/SingleWordProductionDutch-1/results'

def run_pipeline():
    """
    Demonstrates how one could run the complete neural speech decoding pipeline for all participants.
    
    This function executes the full analysis pipeline:
    1. Run SOMATA to find oscillatory compounds in the data
    2. Build frequency templates from high-gamma oscillators to complete the initial parameters for the SSPE model
    3. Extract SSPE features (amplitude, phase, and PAC combinations)
    4. Extract traditional high-gamma bandpass features
    5. Reconstruct mel spectrograms
    6. Evaluate reconstruction quality with cross-validation
    
    The pipeline extracts four feature variants:
    - SSPE with amplitude, phase, and PAC (saveAs='_pacSSPEfeat')
    - SSPE with amplitude and phase only (saveAs='_phaseSSPEfeat')
    - SSPE with amplitude only (saveAs='_ampSSPEfeat')
    - Traditional bandpass high-gamma (saveAs='_bandpassfeat')
    
    when all files are saved by running this pipeline, you can run report_figures.ipynb to generate all figures and numbers included in the report.
    """
    #run somata
    run_somata()

    #extract the features you want for each participant
    for pt in pts:
        initParams_list = get_initParams(pt)
        run_extract_features(pt, initParams_list, SSPE=True, include_phase = True, include_pac = True, saveAs= '_pacSSPEfeat')
        run_extract_features(pt, initParams_list, SSPE=True, include_phase = True, include_pac = False, saveAs= '_phaseSSPEfeat')
        run_extract_features(pt, initParams_list, SSPE=True, include_phase = False, include_pac = False, saveAs= '_ampSSPEfeat')
        run_extract_features(pt, None, SSPE=False, saveAs= '_bandpassfeat')

    #reconstruct with all feature sets
    reconstruct(pts=None,  n_comp = 9, feat_suffix='_phaseSSPEfeat.npy', unstacked =True, saveAs='phase_sppe_cropped')
    reconstruct(pts=None,  n_comp = 9, feat_suffix='_ampSSPEfeat.npy', unstacked =True, saveAs='only_AMP_sppe_cropped')
    reconstruct(pts=None,  n_comp = 9, feat_suffix='_pacSSPEfeat.npy', unstacked =True, saveAs='PAC_sspe_cropped')
    reconstruct(pts=None,  n_comp = 5, feat_suffix='_bandpassfeat.npy', unstacked =False, saveAs='CausalHG_cropped')


def get_eeg(participant, t_start=0, t_segment = 10):
    """
    Load and preprocess EEG data for a participant from NWB file.
    
    Parameters
    ----------
    participant : str
        Participant identifier (e.g., 'sub-01')
    t_start : float, optional
        Start time in seconds. Default is 0.
    t_segment : float, optional
        Length of segment to extract in seconds. Default is 10.
        
    Returns
    -------
    eeg : ndarray (samples, channels)
        Preprocessed EEG data with linear detrending and line noise removal
    """
    #Load data
    io = NWBHDF5IO(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
    nwbfile = io.read()
    #sEEG
    eeg = nwbfile.acquisition['iEEG'].data[:]
    io.close()

    start_sample = int(t_start * sr)
    end_sample = int(t_segment * sr) + start_sample
    eeg = eeg[start_sample:end_sample, :]
    
    eeg = scipy.signal.detrend(eeg, axis=0)
    
    #remove (harmonics of) line noise
    for i in range (0,5):
        sos = scipy.signal.iirfilter(4, [(i*100+49)/(sr/2),(i*100+51)/(sr/2)],btype='bandstop',output='sos')
        eeg = scipy.signal.sosfiltfilt(sos,eeg,axis=0)
    return eeg



def run_somata_single_channel(ch, eeg_data, eeg_sr):
    io1 = IterOsc(eeg_data, eeg_sr, noise_start=None, osc_range=7)
    io1.iterate(freq_res=1, plot_fit=False, verbose=False)

    ll_vec = io1.ll
    knee_idx = get_knee(ll_vec)
    best_model_for_ch = io1.get_knee_osc()

    # if the model chose knee idx 0 or 1 (so 1 or 2 oscilators), look ahead
    if knee_idx <2:
        knee_idx = 2
        best_model_for_ch = io1.fitted_osc[2]
        #print(f'changed knee_idx to {knee_idx}')

    #if somata lacks hg look if a higher iteration does 
    if not any(70 <= f <= 170 for f in best_model_for_ch.freq):
        for i, model in enumerate(io1.fitted_osc[1:], start=1):
            if 70 <= model.freq[-1] <= 170:
                best_model_for_ch = model
                knee_idx = i
                break

    # look ahead one model higher if there is an hg osc next we want to include
    if knee_idx + 1 < len(io1.fitted_osc):
        next_model = io1.fitted_osc[knee_idx+1]
        last_freq_next = next_model.freq[-1]
        if 70 <= last_freq_next <= 170:
            freq_diffs = np.abs(best_model_for_ch.freq - last_freq_next)
            if np.all(freq_diffs > 5):
                knee_idx += 1
                best_model_for_ch = next_model
                #print(f'found another hg freq at knee_idx +1 (={knee_idx})')

    #print("channel", ch)
    #print("freqs", best_model_for_ch.freq)
    
    returns = {
        'channel': ch,
        'knee_n_osc': knee_idx + 1,
        'logL': ll_vec[knee_idx],
        'freqs': best_model_for_ch.freq,
        'damping': best_model_for_ch.a,
        'sigma2': best_model_for_ch.sigma2, 
        'obs_noise_R': best_model_for_ch.R 
    }
    del io1
    del best_model_for_ch
    gc.collect()
    return returns

def run_somata():
    for participant in pts:
        eeg = get_eeg(participant)
        n_channels = eeg.shape[1]
        print(participant, n_channels)
        
        # Run with limited jobs and tqdm to keep the kernel alive
        eeg_sr = 1024
        
        results_list = Parallel(n_jobs=4)( # Only 4 jobs to save RAM
            delayed(run_somata_single_channel)(ch, eeg[:, ch], eeg_sr) 
            for ch in tqdm(range(n_channels), desc="Running SOMATA")
        )
        # Convert list of results back to a dictionary
        channel_results = {i: res for i, res in enumerate(results_list)}
        # Save
        np.save(os.path.join(path_somata, f'{participant}_somata_results.npy'), channel_results)

# Build initParams list for all channels according to rule 1.2
def get_initParams(participant):
    """
    Build initialization parameters for SSPE algorithm for all channels.
    
    This function loads SOMATA results, builds a frequency template from channels
    with high-gamma oscillators, and creates initialization parameters for all
    channels. Channels without high-gamma oscillators receive the HG template
    frequencies added to their existing frequencies.
    
    Parameters
    ----------
    participant : str
        Participant identifier (e.g., 'sub-01')
        
    Returns
    -------
    initParams_list : list of dict
        List of initialization parameter dictionaries, one per channel.
        Each dict contains: freqs, Fs, ampVec, sigmaFreqs, sigmaObs,
        windowSize, lowFreqBand
    """
    somata_results = np.load(os.path.join(path_somata,f'{participant}_somata_results.npy'),allow_pickle=True).item()
    #gather all channels that have a peak in hg range (that we thus want to use for our base osc template)    
    has_hg_osc = set()
    for idx, data in somata_results.items():
        if any(70 <= f <= 170 for f in data['freqs']):
            #has_hg_osc.add(int(data['channel']))
            has_hg_osc.add(idx)
    hg_subset_results = {idx: somata_results[idx] for idx in has_hg_osc}
    template_freqs = build_density_template(hg_subset_results)
    template_a, template_s = get_template_parameters(hg_subset_results, template_freqs) 
    template_freqs = np.array(template_freqs)
    template_a = np.array(template_a)
    template_s = np.array(template_s)
    
    # Find indices where template frequencies are in the HG range
    hg_mask = (template_freqs >= 70) & (template_freqs <= 170)
    
    hg_only_f = template_freqs[hg_mask]
    hg_only_a = template_a[hg_mask]
    hg_only_s = template_s[hg_mask]

    initParams_list = []
    for ch, _ in somata_results.items():
        initParams = {
            "freqs": somata_results[ch]['freqs'],
            "Fs": sr,
            "ampVec": somata_results[ch]['damping'],
            "sigmaFreqs": somata_results[ch]['sigma2'],
            "sigmaObs": 1,
            "windowSize": 2000,
            "lowFreqBand": None
        }
        if not any(70 <= f <= 170 for f in somata_results[ch]['freqs']):
            #print ("no hg in somata")
            initParams = {
                "freqs": list(somata_results[ch]['freqs']) + list(hg_only_f),
                "Fs": sr,
                "ampVec": list(somata_results[ch]['damping']) + list(hg_only_a),
                "sigmaFreqs": list(somata_results[ch]['sigma2']) + list(hg_only_s),
                "sigmaObs": 1,
                "windowSize": 2000,
                "lowFreqBand": None
            }
        initParams_list.append(initParams)
    return initParams_list


#average frequency template constructor by finding clusters in somata results
def plot_density_template(results_dict, template_freqs, s, dens, all_freqs):
    """
    Plot the density template visualization for frequency cluster analysis.
    
    This function creates a visualization showing the kernel density estimation
    of frequencies found by SOMATA, individual frequency lines, and the selected
    template frequencies.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of SOMATA results per channel
    template_freqs : list
        Selected template frequencies
    s : ndarray
        Frequency grid for density evaluation
    dens : ndarray
        Density values at each frequency point
    all_freqs : list
        All frequencies found by SOMATA
    """
    plt.figure(figsize=(12, 5))
    
    # plot the smooth KDE density curve
    plt.plot(s, dens, color='black', lw=2, label='Kernel Density (KDE)')
    plt.fill_between(s.flatten(), dens, alpha=0.2, color='gray')
    
    # plot the individual frequencies found by SOMATA (the blue vertical lines)
    plt.vlines(all_freqs, ymin=0, ymax=dens.max()*0.05, color='C' + str(0) , alpha=0.3, label='SOMATA Freqencies')
    
    # mark the final chosen template frequencies
    for f in template_freqs:
        # Get density value at this frequency to place the marker
        idx = np.argmin(np.abs(s - f))
        plt.plot(f, dens[idx], "ro", markersize=8)
        plt.annotate(f"{f:.0f}Hz", (f, dens[idx]), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontweight='bold',fontsize = 13, color='C3')
    
    #plt.title('Frequency Cluster Analysis & Template Selection', fontsize=14)
    #plt.xlabel('Frequency (Hz)', fontsize=12)
    #plt.ylabel('Density', fontsize=12)
    plt.xlim(70, 170)
    plt.grid(alpha=0.3)
    plt.legend(fontsize = 16)
    plt.show()


def build_density_template(results_dict, nr_osc = 3, min_hg_osc=2, verbose = False):
    """
    Build a frequency template by clustering SOMATA results using kernel density estimation.
    
    This function extracts frequencies from channels with high-gamma oscillators,
    fits a kernel density estimate, finds peaks in the density, and selects the
    strongest peaks as the template frequencies.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of SOMATA results per channel
    nr_osc : int, optional
        Number of strongest peaks to select as template. Default is 3.
    min_hg_osc : int, optional
        Minimum number of high-gamma oscillators to include in template.
        Default is 2.
    verbose : bool, optional
        If True, plot the density template visualization. Default is False.
        
    Returns
    -------
    template : list
        Sorted list of template frequencies in Hz
    """
    all_freqs = []
    for ch, data in results_dict.items():
        if any(70 <= f <= 170 for f in data['freqs']):
            all_freqs.extend(data['freqs'])

    #print(f'nr of frequencies in somata results in total: {len(all_freqs)}')
    #print(f'all frequencies found: {sorted(all_freqs)}')
    # Reshape for sklearn
    X = np.array(all_freqs).reshape(-1, 1)
    
    # Fit KDE (bandwidth=3 means it treats freqs within ~3-5Hz as 'together')
    kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(X)
    
    # Evaluate density across the spectrum (0 to 170 Hz)
    s = np.linspace(0, 170, 1000).reshape(-1, 1)
    log_dens = kde.score_samples(s)
    dens = np.exp(log_dens)
    
    # Find peaks in the density
    peaks, _ = find_peaks(dens, distance=5) # distance ensures peaks are at least 5Hz apart
    peak_freqs = s[peaks].flatten()

    #print(f'peak_freqs: {peak_freqs}')
    

    # For non HG, pick the top nr_osc strongest peaks
    # Sort peaks by their density (height)
    peaks_sorted = sorted(peak_freqs, key=lambda f: dens[np.argmin(np.abs(s-f))], reverse=True)
    template = set(peaks_sorted[:nr_osc])

    
    #if less hg peaks included then minimal:
    if len([f for f in template if 70 <= f <= 170]) <min_hg_osc:
        hg_peaks = [f for f in peak_freqs if 70 <= f <= 170]
        # Sort HG peaks by their density (height)
        hg_peaks_sorted = sorted(hg_peaks, key=lambda f: dens[np.argmin(np.abs(s-f))], reverse=True)
        template.update(hg_peaks_sorted[:min_hg_osc])

    if verbose:
        plot_density_template(results_dict, template, s, dens, all_freqs)
    
    return sorted(list(template))


def get_template_parameters(results_dict, template_freqs, window=7):
    """
    Get amplitude and sigma parameters for template frequencies from SOMATA results.
    
    For each template frequency, this function finds matching frequencies within
    a window across all channels and computes the mean amplitude and sigma values.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of SOMATA results per channel
    template_freqs : list
        Template frequencies to get parameters for
    window : int, optional
        Frequency window for matching (Hz). Default is 7, but reduced to 2 for
        frequencies below 10 Hz.
        
    Returns
    -------
    final_a : list
        Mean amplitude values for each template frequency
    final_s : list
        Mean sigma values for each template frequency
    """
    final_a = []
    final_s = []
    
    for target_f in template_freqs:
        window = 2 if target_f < 10 else 7
        matching_as = []
        matching_ss = []
        
        for ch, data in results_dict.items():
            # Extract lists from the dictionary
            ch_freqs = np.array(data['freqs'])
            ch_as = np.array(data['damping'])
            ch_ss = np.array(data['sigma2'])
            
            # Find indices where channel frequency is close to the template peak
            matches = np.where(np.abs(ch_freqs - target_f) <= window)[0]
            
            if len(matches) > 0:
                matching_as.extend(ch_as[matches])
                matching_ss.extend(ch_ss[matches])
        
        # Calculate mean for this peak. Fallback to sensible defaults if no match.
        if matching_as:
            final_a.append(np.mean(matching_as))
            final_s.append(np.mean(matching_ss))
        else:
            # Default values if a cluster has no close contributors (shouldn't happen)
            final_a.append(0.98) 
            final_s.append(0.5)
            
    return final_a, final_s


def viz_compare_results(prefix_a, prefix_b, name_a = None, name_b = None, title=None ):
    """
    Visualize comparison between two feature extraction methods.
    
    This function loads correlation results for two methods and creates a bar
    plot comparing their performance across participants.
    
    Parameters
    ----------
    prefix_a : str
        Prefix for method A result files (e.g., 'SSPE')
    prefix_b : str
        Prefix for method B result files (e.g., 'HG')
    name_a : str, optional
        Display name for method A. If None, uses prefix_a.
    name_b : str, optional
        Display name for method B. If None, uses prefix_b.
    title : str, optional
        Plot title. If None, generates default title.
        
    Notes
    -----
    Expects the following files in the results directory:
    - {prefix}linearResults.npy
    - {prefix}randomResults.npy
    - {prefix}explainedVariance.npy
    """

    #Load correlation results

    allRes_a = np.load(os.path.join(path_result, f'{prefix_a}linearResults.npy'))
    randomControl_a = np.load(os.path.join(path_result, f'{prefix_a}randomResults.npy'))
    explainedVariance_a = np.load(os.path.join(path_result,f'{prefix_a}explainedVariance.npy'))

    allRes_b = np.load(os.path.join(path_result, f'{prefix_b}linearResults.npy'))
    randomControl_b = np.load(os.path.join(path_result, f'{prefix_b}randomResults.npy'))
    explainedVariance_b = np.load(os.path.join(path_result,f'{prefix_b}explainedVariance.npy'))


    print(allRes_a.shape)
    print(allRes_b.shape)
    
    mean_a = np.mean(allRes_a, axis=(1, 2))
    std_a = np.std(allRes_a, axis=(1, 2))
    
    mean_b = np.mean(allRes_b, axis=(1, 2))
    std_b = np.std(allRes_b, axis=(1, 2))

    colors = ['C' + str(i) for i in range(10)]
    x = np.arange(len(mean_a))

    width = 0.35  # Width of the individual bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Method A bars (left side of the tick)
    rects1 = ax.bar(x - width/2, mean_a, width, yerr=std_a, 
                    label=name_a if name_a else prefix_a, alpha=0.5, color='C' + str(0)) #'steelblue')
    
    # Plot Method B bars (right side of the tick)
    rects2 = ax.bar(x + width/2, mean_b, width, yerr=std_b, 
                    label=name_b if name_b else prefix_b, alpha=0.5, color='C' + str(1)) #'indianred')

    # Add scatter points for individual observations
    for p in range(0,10):
        # Scatter for Method A
        vals_a = np.mean(allRes_a[p], axis=1)
        ax.scatter(np.repeat(p - width/2, len(vals_a)), vals_a, color='C' + str(0))
                   #color='black', s=15, alpha=0.5, zorder=3)
        
        # Scatter for Method B
        vals_b = np.mean(allRes_b[p], axis=1)
        ax.scatter(np.repeat(p + width/2, len(vals_b)), vals_b,  color='C' + str(1))
                   #color='black', s=15, alpha=0.5, zorder=3)

    # Styling
    ax.set_ylabel('Correlation', fontsize=18)
    ax.set_title(title if title else f'Performance Comparison between {prefix_a} and {prefix_b} features', fontsize=18)#, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f'sub-{i+1:02d}' for i in x], rotation=45, ha='right', fontsize=14)
    ax.set_ylim(0, 1)
    
    # Clean up spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=14, width=2)
    
    # Add a legend to distinguish methods
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(path_result,f'comparison_results.png'),dpi=600)
    plt.show()
    
    '''
    #Barplot of average results
    plt.bar(x,meanCorrs,yerr=stdCorrs,alpha=0.5,color=colors)
    for p in range(allRes.shape[0]):
        #Add mean results of each patient as scatter points
        plt.scatter(np.zeros(allRes[p,:,:].shape[0])+p,np.mean(allRes[p,:,:],axis=1),color=colors[p])

    plt.set_xticks(x)
    plt.set_xticklabels(['sub-' + "{:02d}".format(i+1) for i in x],rotation=45, ha='right',fontsize=20)
    plt.set_ylim(0,1)
    plt.set_ylabel('Correlation')
    #Title
    plt.set_title('Performance Comparison between bandpass and SSPE features',fontsize=20,fontweight="bold")
    # Make pretty
    plt.setp(plt.spines.values(), linewidth=2)
    #The ticks
    plt.xaxis.set_tick_params(width=2)
    plt.yaxis.set_tick_params(width=2)
    plt.xaxis.label.set_fontsize(20)
    plt.yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in plt.get_yticklabels()]

    #Despine
    plt.spines['right'].set_visible(False)
    plt.spines['top'].set_visible(False)


    plt.tight_layout()
    #plt.savefig(os.path.join(path_result,f'{prefix}results.png'),dpi=600)
    plt.show()
    '''

def highest_hg_mel_correlations(pt):
    """
    Find the 5 channels with highest correlation between HG features and mel spectrogram.
    
    This function computes the correlation between high-gamma features from each
    channel and the mean mel spectrogram (speech envelope) to identify the most
    speech-responsive channels.

    Parameters
    ----------
    pt : str
        Participant identifier (e.g., 'sub-01')
        
    Returns
    -------
    top_channels : ndarray (5,)
        Indices of the 5 channels with highest absolute correlation
    correlations : ndarray (n_channels,)
        Correlation coefficients for all channels
    """
    #Load data
    io = NWBHDF5IO(os.path.join(path_bids,pt,'ieeg',f'{pt}_task-wordProduction_ieeg.nwb'), 'r')
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
    """
    Get the anatomical region description for a specific channel.
    
    Parameters
    ----------
    participant_id : str
        Participant identifier (e.g., 'sub-01')
    ch_idx : int
        Channel index
        
    Returns
    -------
    region : str
        Anatomical region description for the channel
    """
    # Load channel information

    channels_df = pd.read_csv(
        os.path.join(path_bids, participant_id, 'ieeg',
                f'{participant_id}_task-wordProduction_channels.tsv'),
                delimiter='\t'
            )
    region = channels_df.iloc[ch_idx]['description']
    return region


if __name__=="__main__":

    result_path = r'./results'
    prefix = 'HG'

    allRes = np.load(os.path.join(result_path, f'{prefix}linearResults.npy'))
    randomControl = np.load(os.path.join(result_path, f'{prefix}randomResults.npy'))
    explainedVariance = np.load(os.path.join(result_path,f'{prefix}explainedVariance.npy'))


    for p in range(0,10):
        print("Sub-",p+1)
        rs = allRes[p]
        model_mean = rs.mean()
        rand_mean = randomControl[p].mean()
        rand_std = randomControl[p].std()
        z = (model_mean - rand_mean) / rand_std
        print(round(model_mean,2))
        print(round(rand_mean,2),"±",round(rand_std,2))
        print(round(explainedVariance[p].mean(),2))
        print(round(z,2))
        
    allRes = np.load(os.path.join(result_path,f'{prefix}linearResults.npy'))

    colors = ['C' + str(i) for i in range(10)]

    meanCorrs = np.mean(allRes,axis=(1,2))
    stdCorrs = np.std(allRes, axis=(1,2))

    x = range(len(meanCorrs))
    fig, ax = plt.subplots(1,2,figsize=(14,7))
    #Barplot of average results
    ax[0].bar(x,meanCorrs,yerr=stdCorrs,alpha=0.5,color=colors)
    for p in range(allRes.shape[0]):
        #Add mean results of each patient as scatter points
        ax[0].scatter(np.zeros(allRes[p,:,:].shape[0])+p,np.mean(allRes[p,:,:],axis=1),color=colors[p])

    ax[0].set_xticks(x)
    ax[0].set_xticklabels(['sub-' + "{:02d}".format(i+1) for i in x],rotation=45, ha='right',fontsize=20)
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Correlation')
    #Title
    ax[0].set_title('a',fontsize=20,fontweight="bold")
    # Make pretty
    plt.setp(ax[0].spines.values(), linewidth=2)
    #The ticks
    ax[0].xaxis.set_tick_params(width=2)
    ax[0].yaxis.set_tick_params(width=2)
    ax[0].xaxis.label.set_fontsize(20)
    ax[0].yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in ax[0].get_yticklabels()]

    #Despine
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    #Mean across folds over spectral bins
    specMean = np.mean(allRes,axis=1)
    specStd = np.std(allRes,axis=1)
    specBins = np.arange(allRes.shape[2])
    for p in range(allRes.shape[0]):
        ax[1].plot(specBins, specMean[p,:],color=colors[p])
        error = specStd[p,:]/np.sqrt(allRes.shape[1])
        #Shaded areas highlight standard error
        ax[1].fill_between(specBins,specMean[p,:]-error,specMean[p,:]+error,alpha=0.5,color=colors[p])
    ax[1].set_ylim(0,1)
    ax[1].set_xlim(0,len(specBins))
    ax[1].set_xlabel('Spectral Bin')
    ax[1].set_ylabel('Correlation')
    #Title
    ax[1].set_title('b',fontsize=20,fontweight="bold")

    #Make pretty
    plt.setp(ax[1].spines.values(), linewidth=2)
    #The ticks
    ax[1].xaxis.set_tick_params(width=2)
    ax[1].yaxis.set_tick_params(width=2)
    ax[1].xaxis.label.set_fontsize(20)
    ax[1].yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in ax[1].get_yticklabels()]
    c = [a.set_fontsize(20) for a in ax[1].get_xticklabels()]
    #Despine
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_path,f'{prefix}results.png'),dpi=600)
    plt.show()

    '''
    # Viz example spectrogram
    #Load words and spectrograms
    feat_path = r'./features'
    participant = 'sub-06'
    #Which timeframe to plot
    start_s = 5.5
    stop_s=19.5

    frameshift = 0.01
    #Load spectrograms
    rec_spec = np.load(os.path.join(result_path, f'{participant}_predicted_spec.npy'))
    spectrogram = np.load(os.path.join(feat_path, f'{participant}_spec.npy'))
    #Load prompted words
    eeg_sr= 1024
    words = np.load(os.path.join(feat_path,f'{participant}_procWords.npy'))[int(start_s*eeg_sr):int(stop_s*eeg_sr)]
    words = [words[w] for w in np.arange(1,len(words)) if words[w]!=words[w-1] and words[w]!='']
    
    cm='viridis'
    fig, ax = plt.subplots(2, sharex=True)
    #Plot spectrograms
    pSta=int(start_s*(1/frameshift));pSto=int(stop_s*(1/frameshift))
    ax[0].imshow(np.flipud(spectrogram[pSta:pSto, :].T), cmap=cm, interpolation=None,aspect='auto')
    ax[0].set_ylabel('Log Mel-Spec Bin')
    ax[1].imshow(np.flipud(rec_spec[pSta:pSto, :].T), cmap=cm, interpolation=None,aspect='auto')
    plt.setp(ax[1], xticks=np.arange(0,pSto-pSta,int(1/frameshift)), xticklabels=[str(x/int(1/frameshift)) for x in np.arange(0,pSto-pSta,int(1/frameshift))])
    plt.setp(ax[1], xticks=np.arange(int(1/frameshift),spectrogram[pSta:pSto, :].shape[0],3*int(1/frameshift)), xticklabels=words)
    ax[1].set_ylabel('Log Mel-Spec Bin')

    plt.savefig(os.path.join(result_path,'spec_example.png'),dpi=600)

    #Saving for use in Adobe Illustrator
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.savefig(os.path.join(result_path,'spec_example.pdf'),transparent=True)
    plt.show()


    # Viz waveforms
    #Load waveforms
    rate, audio = wavfile.read(os.path.join(result_path,f'{participant}_orig_synthesized.wav'))
    rate, recAudio = wavfile.read(os.path.join(result_path,f'{participant}_predicted.wav'))
    
    orig = audio[int(start_s*rate):int(stop_s*rate)]
    rec = recAudio[int(start_s*rate):int(stop_s*rate)]
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(orig)
    axarr[1].plot(rec)

    #Axis
    xts = np.arange(rate,orig.shape[0]+1,3*rate)
    axarr[1].set_xticks(xts)
    axarr[1].set_xticklabels(words)
    axarr[0].set_xlim([0,orig.shape[0]])
    axarr[0].set_ylim([-np.max(np.abs(orig)),np.max(np.abs(orig))])
    axarr[1].set_ylim([-np.max(np.abs(rec)),np.max(np.abs(rec))])
    
    #Add line indicating 3 seconds
    axarr[1].annotate("",xy=(xts[0], 27000), xycoords='data',xytext=(xts[1],27000), textcoords='data',
                arrowprops=dict(arrowstyle="-",
                                connectionstyle="arc3"),
                )
    axarr[1].annotate("3 seconds",xy=((xts[0]+xts[1])/2, 22000), horizontalalignment='center')

    #Axis labels
    axarr[0].set_ylabel('Original')
    axarr[0].set_yticks([])
    axarr[1].set_yticks([])
    axarr[1].set_ylabel('Amplitude')
    axarr[0].set_ylabel('Amplitude')
    axarr[1].text(orig.shape[0],0,'Reconstruction',horizontalalignment='left',verticalalignment='center',rotation='vertical',)
    axarr[0].text(orig.shape[0],0,'Original',horizontalalignment='left',verticalalignment='center',rotation='vertical',)

    #Make Pretty
    ax[1].xaxis.set_tick_params(width=2)
    ax[1].yaxis.set_tick_params(width=2)
    ax[1].xaxis.label.set_fontsize(20)
    ax[1].yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in ax[1].get_yticklabels()]
    c = [a.set_fontsize(20) for a in ax[1].get_xticklabels()]
    #ax.get_yticklabels().set_fontsize(28)

    #Despine
    for axes in axarr:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        #axes.spines['top'].set_visible(False)

    plt.savefig(os.path.join(result_path,'wav_example.png'),dpi=600)
    #Saving for usage in Adobe Illustrator
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.savefig(os.path.join(result_path,'wav_example.pdf'),transparent = True)
    plt.show()

'''
