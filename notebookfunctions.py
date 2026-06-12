import os
import sys

# Get the absolute path of the directory containing the notebook (notebooks/)
current_dir = os.path.abspath('')

# Add the parent directory (SingleWordProductionDutch/) to the system path
# This allows Python to find 'src'
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import scipy.signal
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import pearsonr
from pynwb import NWBHDF5IO
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy.matlib as matlib
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks

from phaseEM import *
from extract_features import *
from channel_evaluation import *
import config as cfg
from reconstruction_minimal import *

feat_path = r'./features'
result_path = r'./results'


winL = 0.05
frameshift = 0.01
modelOrder = 4
stepSize = 5
sr=1024
path_bids = '/Users/lisa/Documents/DSAI_year2/Marble/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS'
path_somata = '/Users/lisa/Documents/DSAI_year2/SingleWordProductionDutch-1/somata'
path_result = '/Users/lisa/Documents/DSAI_year2/SingleWordProductionDutch-1/results'

def get_eeg(participant, t_start=0, t_segment = 10):
    #Load data
    io = NWBHDF5IO(os.path.join(cfg.PATH_BIDS,participant,'ieeg',f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
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


# Build initParams list for all channels according to rule 1.2
def get_initParams(participant):
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
    
    #print (hg_only_f)

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
            #print ("not in somata hg")
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
    plt.savefig(os.path.join(result_path,f'comparison_results.png'),dpi=600)
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
    #plt.savefig(os.path.join(result_path,f'{prefix}results.png'),dpi=600)
    plt.show()
    '''