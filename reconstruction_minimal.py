import os

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from extract_features import *

import reconstructWave as rW
import MelFilterBank as mel


def createAudio(spectrogram, audiosr=16000, winLength=0.05, frameshift=0.01):
    """
    Create a reconstructed audio wavefrom
    
    Parameters
    ----------
    spectrogram: array
        Spectrogram of the audio
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram was calculated
    frameshift: float
        Shift (in seconds) after which next window was extracted
    Returns
    ----------
    scaled: array
        Scaled audio waveform
    """
    mfb = mel.MelFilterBank(int((audiosr*winLength)/2+1), spectrogram.shape[1], audiosr)
    nfolds = 10
    hop = int(spectrogram.shape[0]/nfolds)
    rec_audio = np.array([])
    for_reconstruction = mfb.fromLogMels(spectrogram)
    
    for w in range(0,spectrogram.shape[0],hop):
        spec = for_reconstruction[w:min(w+hop,for_reconstruction.shape[0]),:]
        
        rec = rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
        rec_audio = np.append(rec_audio,rec)
    scaled = np.int16(rec_audio/np.max(np.abs(rec_audio)) * 32767)
    return scaled

def reconstruct_old(pts, SSPEfeatures=True, hg_osc_sspe_features=False, data_dict=None, saveAs=None):

    feat_path = r'./features'
    result_path = r'./results'
    #pts = ['sub-%02d'%i for i in range(1,11)]
    if pts == None:
        pts = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05','sub-06','sub-07', 'sub-08', 'sub-09', 'sub-10']

    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000

    nfolds = 10
    kf = KFold(nfolds,shuffle=False)
    #est = LinearRegression(n_jobs=-1)
    est = LinearRegression(n_jobs=1)
    pca = PCA()
    numComps = 50
    
    #Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    allRes = np.zeros((len(pts),nfolds,23))
    explainedVariance = np.zeros((len(pts),nfolds))
    numRands = 1000
    randomControl = np.zeros((len(pts),numRands, 23))

    for pNr, pt in enumerate(pts):
        #Load the data
        if SSPEfeatures:
            data = np.load(os.path.join(feat_path,f'{pt}_SSPEfeat.npy'))
        else:
            if hg_osc_sspe_features:
                data = data_dict[pt]
            else:
                data = np.load(os.path.join(feat_path,f'{pt}_feat.npy'))
        

        #Check for NaNs or Infinities
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"!!! Warning: {pt} has {nan_count} NaNs and {inf_count} Infs in features.")
            
            # Fix: Replace NaNs with 0 and Infs with a very large/small number or 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Optional: Drop columns that are entirely NaN (all-zero oscillators)
            # This helps PCA and prevents LinearRegression from getting "confused"
            variance = np.var(data, axis=0)
            valid_cols = variance > 0
            if not np.all(valid_cols):
                print(f"Removing {np.sum(~valid_cols)} constant/zero columns from {pt}")
                data = data[:, valid_cols]
        

        spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
        labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
        featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
        
        #Check lengths, trim both to the same length
        '''
        print(data.shape[0], spectrogram.shape[0])
        min_len = min(data.shape[0], spectrogram.shape[0])
        data = data[:min_len, :]
        spectrogram = spectrogram[:min_len, :]
        '''

        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            #Z-Normalize with mean and std from the training data
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            std[std == 0] = 1.0  # Prevent division by zero

            trainData=(data[train,:]-mu)/std
            testData=(data[test,:]-mu)/std

            # Convert all NaNs to 0.0 before passing to PCA
            trainData = np.nan_to_num(trainData, nan=0.0)

            #Fit PCA to training data
            pca.fit(trainData)
            #Get percentage of explained variance by selected components
            explainedVariance[pNr,k] =  np.sum(pca.explained_variance_ratio_[:numComps])
            #Tranform data into component space
            trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)
            
            #Fit the regression model
            est.fit(trainData, spectrogram[train, :])
            #Predict the reconstructed spectrogram for the test data
            rec_spec[test, :] = est.predict(testData)

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        #Show evaluation result
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[pNr,:,:]=rs

        #Estimate random baseline
        for randRound in range(numRands):
            #Choose a random splitting point at least 10% of the dataset size away
            splitPoint = np.random.choice(np.arange(int(spectrogram.shape[0]*0.1),int(spectrogram.shape[0]*0.9)))
            #Swap the dataset on the splitting point 
            shuffled = np.concatenate((spectrogram[splitPoint:,:],spectrogram[:splitPoint,:]))
            #Calculate the correlations
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[:,specBin], shuffled[:,specBin])
                randomControl[pNr, randRound,specBin]=r


        #Save reconstructed spectrogram
        os.makedirs(os.path.join(result_path), exist_ok=True)
        np.save(os.path.join(result_path,f'{pt}_predicted_spec.npy'), rec_spec)
        
        #Synthesize waveform from spectrogram using Griffin-Lim
        print(f"Max value in rec_spec: {np.max(rec_spec)}")
        if np.max(rec_spec) > 100: # Threshold depends on your scaling
            print("Warning: Predicted spectrogram values are too high!")
            rec_spec = np.clip(rec_spec, -100, 50) # Forces values into a mathematically safe range
        reconstructedWav = createAudio(rec_spec,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
        wavfile.write(os.path.join(result_path,f'{pt}HG_predicted.wav'),int(audiosr),reconstructedWav)

        #For comparison synthesize the original spectrogram with Griffin-Lim
        origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
        wavfile.write(os.path.join(result_path,f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)

    #Save results in numpy arrays  
    if saveAs:
        np.save(os.path.join(result_path, saveAs + 'linearResults.npy'),allRes)
        np.save(os.path.join(result_path, saveAs + 'randomResults.npy'),randomControl)
        np.save(os.path.join(result_path, saveAs + 'explainedVariance.npy'),explainedVariance)
    else:
        if SSPEfeatures:
            np.save(os.path.join(result_path,'SSPElinearResults.npy'),allRes)
            np.save(os.path.join(result_path,'SSPErandomResults.npy'),randomControl)
            np.save(os.path.join(result_path,'SSPEexplainedVariance.npy'),explainedVariance)
        else:
            if not hg_osc_sspe_features:
                np.save(os.path.join(result_path,'HGlinearResults.npy'),allRes)
                np.save(os.path.join(result_path,'HGrandomResults.npy'),randomControl)
                np.save(os.path.join(result_path,'HGexplainedVariance.npy'),explainedVariance)



def reconstruct(pts, model = 'PLS', n_comp = 9, feat_suffix='_feat.npy', unstacked =False, given_data=False, data_dict=None, saveAs=None, synthesize=False):

    feat_path = r'./features'
    result_path = r'./results'
    #pts = ['sub-%02d'%i for i in range(1,11)]
    if pts == None:
        pts = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05','sub-06','sub-07', 'sub-08', 'sub-09', 'sub-10']

    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000

    nfolds = 10
    kf = KFold(nfolds,shuffle=False)



    #pca = PCA()

    if model == 'PLS': 
        est = PLSRegression(n_components=n_comp, max_iter=1000)
    elif model == 'LR':
        est = LinearRegression(n_jobs=1)
    elif model == 'Lasso':
        est = Lasso()
    else:
        print('model name not recognized, using LR')
        est = LinearRegression(n_jobs=1)




    
    #Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    allRes = np.zeros((len(pts),nfolds,23))
    explainedVariance = np.zeros((len(pts),nfolds))
    numRands = 1000
    randomControl = np.zeros((len(pts),numRands, 23))

    for pNr, pt in enumerate(pts):
        #Load the data
        if feat_suffix:
            data = np.load(os.path.join(feat_path,f'{pt}{feat_suffix}'))
            if unstacked:
                data = stackFeatures(data)
        else:
            if given_data:
                data = data_dict[pt]
    
        

        #Check for NaNs or Infinities
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"!!! Warning: {pt} has {nan_count} NaNs and {inf_count} Infs in features.")
            
            #Replace NaNs with 0 and Infs with a very large/small number or 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Optional, Drop columns that are entirely NaN (all-zero oscillators)
            variance = np.var(data, axis=0)
            valid_cols = variance > 0
            if not np.all(valid_cols):
                print(f"Removing {np.sum(~valid_cols)} constant/zero columns from {pt}")
                data = data[:, valid_cols]
        

        spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
        #labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
        #featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
        
        #Check lengths, trim both to the same length
        '''
        print(data.shape[0], spectrogram.shape[0])
        min_len = min(data.shape[0], spectrogram.shape[0])
        data = data[:min_len, :]
        spectrogram = spectrogram[:min_len, :]
        '''

        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            #Z-Normalize with mean and std from the training data
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            std[std == 0] = 1.0  # Prevent division by zero

            trainData=(data[train,:]-mu)/std
            testData=(data[test,:]-mu)/std

            # Convert all NaNs to 0.0 before passing to PCA
            trainData = np.nan_to_num(trainData, nan=0.0)

            ''''
            #Fit PCA to training data
            pca.fit(trainData)
            #Get percentage of explained variance by selected components
            explainedVariance[pNr,k] =  np.sum(pca.explained_variance_ratio_[:numComps])
            #Tranform data into component space
            trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)
            '''

            #Fit the regression model
    
            est.fit(trainData, spectrogram[train, :])
            #Predict the reconstructed spectrogram for the test data
            rec_spec[test, :] = est.predict(testData)

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        #Show evaluation result
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[pNr,:,:]=rs

        #Estimate random baseline
        for randRound in range(numRands):
            #Choose a random splitting point at least 10% of the dataset size away
            splitPoint = np.random.choice(np.arange(int(spectrogram.shape[0]*0.1),int(spectrogram.shape[0]*0.9)))
            #Swap the dataset on the splitting point 
            shuffled = np.concatenate((spectrogram[splitPoint:,:],spectrogram[:splitPoint,:]))
            
            #Calculate the correlations
            '''
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[:,specBin], shuffled[:,specBin])
                randomControl[pNr, randRound,specBin]=r
'''
            # Vectorized correlation:
            # This calculates all 23 bins in one go
            # (Subtract mean, divide by std, then dot product)
            s_ms = shuffled - shuffled.mean(axis=0)
            r_ms = spectrogram - spectrogram.mean(axis=0)
            r = np.sum(s_ms * r_ms, axis=0) / (np.sqrt(np.sum(s_ms**2, axis=0) * np.sum(r_ms**2, axis=0)))
            randomControl[pNr, randRound, :] = r


        #Save reconstructed spectrogram
        os.makedirs(os.path.join(result_path), exist_ok=True)
        np.save(os.path.join(result_path,f'{pt}_predicted_spec.npy'), rec_spec)
        
        if synthesize:
            #Synthesize waveform from spectrogram using Griffin-Lim
            print(f"Max value in rec_spec: {np.max(rec_spec)}")
            if np.max(rec_spec) > 100: # Threshold depends on your scaling
                print("Warning: Predicted spectrogram values are too high!")
                rec_spec = np.clip(rec_spec, -100, 50) # Forces values into a mathematically safe range
            reconstructedWav = createAudio(rec_spec,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
            wavfile.write(os.path.join(result_path,f'{pt}HG_predicted.wav'),int(audiosr),reconstructedWav)

            #For comparison synthesize the original spectrogram with Griffin-Lim
            origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
            wavfile.write(os.path.join(result_path,f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)

    #Save results in numpy arrays  
    if saveAs:
        if saveAs == 'dont':
            return allRes, est.coef_
        np.save(os.path.join(result_path, saveAs + 'linearResults.npy'),allRes)
        np.save(os.path.join(result_path, saveAs + 'randomResults.npy'),randomControl)
        np.save(os.path.join(result_path, saveAs + 'explainedVariance.npy'),explainedVariance)
    else:
        if SSPEfeatures:
            np.save(os.path.join(result_path,'SSPElinearResults.npy'),allRes)
            np.save(os.path.join(result_path,'SSPErandomResults.npy'),randomControl)
            np.save(os.path.join(result_path,'SSPEexplainedVariance.npy'),explainedVariance)
        else:
            if not hg_osc_sspe_features:
                np.save(os.path.join(result_path,'HGlinearResults.npy'),allRes)
                np.save(os.path.join(result_path,'HGrandomResults.npy'),randomControl)
                np.save(os.path.join(result_path,'HGexplainedVariance.npy'),explainedVariance)
    return allRes, est.coef_





if __name__=="__main__":
    reconstruct(None)