import numpy as np

'''causal phase estimates using the SP model and EM with fixed interval
% smoothing across windows of data
% primary benefit: assume a transitory burst of oscillatory activity in the
% range of your bandpass filter/ assume a peak shift in the data towards the edge
% of the band pass filter. These are problems unaddressed for instantaneous
% phase estimation right now

% Algorithm:
% after estimating reasonable initialization points we need to run EM on
% the data - at whatever rate makes it possible to run it again before
% getting the next window of data. So while it might be slow here, a C++
% implementation is likely going to be much faster meaning we have have
% small windows (up to the frequency limits of course)
% INPUT: 
% y - data/observation
% initParams - a structure containing the following parameters: 
%           .freqs - oscillator center frequencies to be tracked
%           .Fs - sampling frequency
%           .ampVec - decay rate for each oscillator (initialize to 0.99)
%           .sigmaFreqs - variance for each oscillator (.1 is good default)
%           .sigmaObs - variance for observation (1)
%           .windowSize - how large is the window used to fit the parameters
%           .lowFreqBand - where is the oscillator you wish to track, give
%           a frequency range
% OUTPUT
% phase - Estimated phase for the first oscillator in the frequency band range 
% phaseBounds - The phase credible intervals, use to judge confidence in the phase
% allX_full - Estimated state for the first oscillator in freq band range
% phaseWidth - Single value in degrees informing us of confidence in phase
% returnParams - Estimated parameter values in same format at initParams

returns [phase,phaseBounds,allX_full,phaseWidth,returnParams] 
'''
def causalPhaseEM_MKmdl_noSeg(y, initParams, flagNoFit):
    """
    Real-time causal phase estimation using oscillator-based state-space model and EM parameter updates.
    """


    freqs = np.array(initParams["freqs"])
    Fs = initParams["Fs"]
    ampVec = np.array(initParams["ampVec"])
    sigmaFreqs = np.array(initParams["sigmaFreqs"])
    sigmaObs = initParams["sigmaObs"]
    windowSize = initParams["windowSize"]
    lowFreqBand = np.array(initParams["lowFreqBand"])

    assert len(freqs) == len(ampVec), 'amplitudes and frequencies must have the same size'

    if windowSize < Fs:
        print("The window size needs to be different. Setting it equal to sampling rate.")
        windowSize = Fs

    if len(y) < 2 * windowSize:
        raise ValueError("Observation vector too short; must be at least 2x window size.")

    numSegments = int(np.floor(len(y) / windowSize))

    def ang_var2dev(v):
        return np.sqrt(-2 * np.log(v))

    data = y[:windowSize]

    if not flagNoFit:
        # Estimate oscillator parameters using EM (placeholder for your MATLAB fit_MKModel_multSines)
        omega, ampEst, allQ, R, stateVec, stateCov = fit_MKModel_multSines(
            data, freqs, Fs, ampVec, sigmaFreqs, sigmaObs
        )

        if lowFreqBand != None:
            lowFreqLoc = np.where((omega > lowFreqBand[0]) & (omega < lowFreqBand[1]))[0]
            if len(lowFreqLoc) == 0:
                lowFreqLoc = [np.argmin(np.abs(freqs - np.mean(lowFreqBand)))]

        returnParams = {
            "freqs": omega,
            "ampVec": ampEst,
            "sigmaFreqs": allQ,
            "sigmaObs": R,
        }

    else:
        omega = freqs
        ampEst = ampVec
        allQ = sigmaFreqs
        R = np.array([[sigmaObs]])
    
        returnParams = {}
        lowFreqLoc = []
        stateVec = np.zeros((len(freqs) * 2, 1))
        stateCov = np.eye(len(freqs) * 2) * 0.001
        stateCov = stateCov[:, :, np.newaxis]

    if lowFreqBand != None and len(lowFreqLoc) == 0:
        print("Low freq band incorrect or no signal; retaining initial params.")
        omega = freqs
        ampEst = ampVec
        allQ = sigmaFreqs
        R = sigmaObs
        lowFreqLoc = [np.argmin(np.abs(freqs - np.mean(lowFreqBand)))]

    phi, Q, M = genParametersSoulatMdl_sspp(omega, Fs, ampEst, allQ)

    T = len(y)
    n_freq = len(freqs)

    phase = np.zeros((T, n_freq))
    #phaseBounds = np.zeros((T, n_freq, 2)) removed when removing uncertainty estimation 
    #phaseWidth = np.zeros((T, n_freq))
    allX = np.zeros((len(freqs) * 2, len(y)))
    allP = np.zeros((len(freqs) * 2, len(freqs) * 2, len(y)))

    x = stateVec[:, -1]
    P = stateCov[:, :, -1]

    for tp in range(windowSize, len(y)):
        x_new, P_new = oneStepKFupdate_sspp(x, y[tp], phi, M, Q, R, P)
        #print(f"DEBUG: x_new shape = {x_new.shape}, allX shape = {allX.shape}, M shape = {M.shape}")

        allX[:, tp] = x_new
        P_new = (P_new + P_new.T) / 2
        allP[:, :, tp] = P_new

        #replaced this
        #real_idx = lowFreqLoc[0] * 2
        #imag_idx = real_idx + 1
        #phase[tp] = np.angle(x_new[real_idx] + 1j * x_new[imag_idx])

        #with this, so we use multiple oscilators, and not just one 
        n_freq = len(freqs)
        real_idx = np.arange(0, 2 * n_freq, 2)
        imag_idx = real_idx + 1
        phase[tp, :] = np.angle(x_new[real_idx] + 1j * x_new[imag_idx])


        # Sample credible intervals
        #mean_vec = np.array([x_new[real_idx], x_new[imag_idx]])
        #cov_mat = P_new[real_idx:imag_idx + 1, real_idx:imag_idx + 1]
        #samples = np.random.multivariate_normal(mean_vec, cov_mat, 2000)
        #sample_angles = np.angle(np.exp(1j * (np.angle(samples[:, 0] + 1j * samples[:, 1]) - phase[tp])))

        

        for f in range(n_freq):
            mu = np.array([
                x_new[real_idx[f]],
                x_new[imag_idx[f]]
            ])

            Sigma = P_new[
                real_idx[f]:imag_idx[f] + 1,
                real_idx[f]:imag_idx[f] + 1
            ]

            '''
            Nsamp = 2000
            sample_angles = np.zeros((n_freq, Nsamp))
            samples = np.random.multivariate_normal(mu, Sigma, Nsamp)

            sample_phase = np.angle(samples[:, 0] + 1j * samples[:, 1])

            sample_angles[f, :] = np.angle(
                np.exp(1j * (sample_phase - phase[tp, f]))
            )
            '''
        '''
        #lowerBnd = np.percentile(sample_angles, 2.5)
        #upperBnd = np.percentile(sample_angles, 97.5)
        #phaseBounds[tp, :] = np.sort([lowerBnd + phase[tp], upperBnd + phase[tp]])
        #phaseWidth[tp] = np.rad2deg(ang_var2dev(np.abs(np.mean(np.exp(1j * sample_angles)))))
        for f in range(n_freq):
            lower = np.percentile(sample_angles[f], 2.5)
            upper = np.percentile(sample_angles[f], 97.5)

            phaseBounds[tp, f, :] = np.sort([
                lower + phase[tp, f],
                upper + phase[tp, f]
            ])

            phaseWidth[tp, f] = np.rad2deg(
                ang_var2dev(np.abs(np.mean(np.exp(1j * sample_angles[f]))))
            )
        '''
        P = P_new
        x = x_new

    if lowFreqBand != None and lowFreqLoc[0] == 0: 
        print('lowFreqLoc = 0')
    
    # Replaced:
    #real_idx = lowFreqLoc[0] * 2
    #imag_idx = real_idx + 1
    #allX_full = allX[real_idx:imag_idx + 1, :].T

    #to this, so we return all osccilators 
    n_freq = len(freqs)
    allX_full = allX.reshape(n_freq, 2, -1).transpose(2, 0, 1)
    #return phase, phaseBounds, allX_full, phaseWidth, returnParams
    return phase, allX_full, returnParams



def genParametersSoulatMdl_sspp(freqs, Fs, ampVector, sigmaFreqs):
    n = len(freqs)
    rotMat = np.zeros((n, 2, 2))
    varMat = np.zeros((n, 2, 2))

    for i, freq in enumerate(freqs):
        rotMat[i, :, :] = createRotMat(freq, Fs) * ampVector[i]
        varMat[i, :, :] = np.diag([sigmaFreqs[i], sigmaFreqs[i]])

    phi = stackBlockMat(rotMat)
    Q = stackBlockMat(varMat)
    M = np.zeros((len(freqs) * 2, 1))
    M[::2, 0] = 1.0  # extract real component
    return phi, Q, M


def createRotMat(freq, Fs):
    """Rotation matrix for oscillator."""
    c = np.cos(2 * np.pi * freq / Fs)
    s = np.sin(2 * np.pi * freq / Fs)
    return np.array([[c, -s], [s, c]])


def stackBlockMat(allMat):
    """Construct block diagonal matrix."""
    n_blocks = allMat.shape[0]
    block_size = allMat.shape[1]
    fullMat = np.zeros((n_blocks * block_size, n_blocks * block_size))
    for i in range(n_blocks):
        start = i * block_size
        fullMat[start:start + block_size, start:start + block_size] = allMat[i, :, :]
    return fullMat


'''
def oneStepKFupdate_sspp(x, y, phi, M, Q, R, P):
    """One-step Kalman filter update for Soulat oscillator model."""
    x_one = phi @ x
    P_one = phi @ P @ phi.T + Q
    K_one = P_one @ M / (M.T @ P_one @ M + R)
    x_new = x_one + K_one.flatten() * (y - float(M.T @ x_one))
    P_new = P_one - K_one @ M.T @ P_one
    return x_new, P_new'''

def oneStepKFupdate_sspp(x, y, phi, M, Q, R, P):
    n_states = len(x)
    x = x.reshape((n_states, 1))
    M = M.reshape((n_states, 1))

    # ---- Prediction step ----
    x_pred = phi @ x
    P_pred = phi @ P @ phi.T + Q

    assert x_pred.shape == (n_states, 1), f"x_pred wrong shape: {x_pred.shape}"
    assert P_pred.shape == (n_states, n_states), f"P_pred wrong shape: {P_pred.shape}"

    # ---- Kalman gain ----
    denom = float(M.T @ P_pred @ M + R)
    K = (P_pred @ M) / denom
    assert K.shape == (n_states, 1), f"K wrong shape: {K.shape}"

    # ---- Update step ----
    innovation = float(y - (M.T @ x_pred))
    x_new = x_pred + K * innovation
    P_new = (np.eye(n_states) - K @ M.T) @ P_pred
    P_new = 0.5 * (P_new + P_new.T)

    assert x_new.shape == (n_states, 1), f"x_new wrong shape: {x_new.shape}"
    assert P_new.shape == (n_states, n_states), f"P_new wrong shape: {P_new.shape}"

    return x_new.flatten(), P_new




_eps = np.finfo(float).eps

def fit_MKModel_multSines(data, freqs, Fs, ampVec, sigmaFreqs, sigmaObs):
    """
    Fit the Soulat multi-sine oscillator model using EM (Shumway & Stoffer style).
    Returns: omega, ampEst, allQ, R, stateVec, stateCov
    """
    raise RuntimeError(
        "EM + smoothing disabled. Using fixed parameters (Option B)."
    )
    y = np.asarray(data).ravel()

    if sigmaFreqs is None or len(sigmaFreqs) == 0:
        sigmaFreqs = 0.1 * np.ones(len(freqs))
    sigmaFreqs = np.asarray(sigmaFreqs).ravel()

    if sigmaObs is None:
        sigmaObs = 10.0

    # frequency parametrization used in MATLAB code
    freqEst = np.asarray(freqs).ravel() / Fs

    # initialize phi, Q, M and observation noise
    phi, Q, M = genParametersSoulatMdl_sspp(freqs, Fs, ampVec, sigmaFreqs)
    R = float(sigmaObs)

    max_iter = 400
    iter_idx = 0
    prev_error = np.inf

    n_states = 2 * len(freqs)
    N = len(y)

    # EM loop
    while (iter_idx < max_iter) and (prev_error > 1e-3):
        # initialize KF start
        xstart = np.zeros((n_states, 1))
        Pstart = 0.001 * np.eye(n_states)

        x = xstart.copy()
        P = Pstart.copy()

        # containers for forward KF outputs
        allX = np.zeros((n_states, N))
        allP = np.zeros((n_states, n_states, N))

        # forward Kalman filter through data
        for i in range(N):
            x_new, P_new = oneStepKFupdate_sspp(x, y[i], phi, M, Q, R, P)
            # ensure column vector
            x_new = x_new.reshape((n_states,))  # flatten, but will be placed as vector
            allX[:, i] = x_new
            allP[:, :, i] = P_new
            x = x_new.reshape((n_states, 1))
            P = P_new

        # fixed interval smoother (Rauch-Tung-Striebel)
        newAllX = np.zeros_like(allX)
        newAllP = np.zeros_like(allP)
        allJ = np.zeros((n_states, n_states, N))  # smoothing gain J for each time

        # initialize with last filtered state
        x_n = allX[:, -1].reshape((n_states, 1))
        P_n = allP[:, :, -1]
        newAllX[:, -1] = x_n.ravel()
        newAllP[:, :, -1] = P_n

        # iterate backwards
        for i in range(N - 2, -1, -1):
            x_i = allX[:, i].reshape((n_states, 1))
            P_i = allP[:, :, i]

            # predicted covariance for time i->i+1
            P_pred = phi @ P_i @ phi.T + Q
            # smoothing gain
            # J = P_i * phi.T * inv(P_pred)
            # protect against singular P_pred by using solve
            J = P_i @ phi.T @ np.linalg.pinv(P_pred)  # shape n_states x n_states

            x_backone = x_i + J @ (x_n - (phi @ x_i))
            P_backone = P_i + J @ (P_n - P_pred) @ J.T

            newAllX[:, i] = x_backone.ravel()
            newAllP[:, :, i] = P_backone
            allJ[:, :, i] = J

            # update for next step backwards
            x_n = x_backone
            P_n = P_backone

        # compute cross-covariances P_t_t-1 (allP_N_N1)
        allP_N_N1 = np.zeros((n_states, n_states, N))
        # For final time N-1 (MATLAB indexing), compute as in original
        if N >= 2:
            P_tmp = phi @ allP[:, :, -2] @ phi.T + Q
            K_tmp = P_tmp @ M / (float(M.T @ P_tmp @ M) + R)
            P_N_N1 = (np.eye(n_states) - K_tmp @ M.T) @ phi @ allP[:, :, -2]
            allP_N_N1[:, :, -1] = P_N_N1

            # backward recursion to fill allP_N_N1
            # Note: MATLAB loop: for i = N-1:-1:2  (1-based); we convert to 0-based
            for i in range(N - 2, 0, -1):
                J_ip1 = allJ[:, :, i]     # J_{i+1} in 0-based
                J_i   = allJ[:, :, i - 1] # J_{i} in 0-based
                P_i = allP[:, :, i]
                P_N_N1_ip1 = allP_N_N1[:, :, i + 1]
                term = P_N_N1_ip1 - phi @ P_i
                allP_N_N1[:, :, i] = P_i @ J_i.T + J_ip1 @ term @ J_i.T
            allP_N_N1[:, :, 0] = np.eye(n_states)
        else:
            allP_N_N1[:, :, 0] = np.eye(n_states)

        # Recompute sufficient statistics A,B,C and R (observation variance)
        A = Pstart + xstart @ xstart.T
        B = np.zeros((n_states, n_states))
        C = np.zeros((n_states, n_states))
        R_num = 0.0

        for i in range(N):
            if i > 0:
                A = A + newAllP[:, :, i - 1] + np.outer(newAllX[:, i - 1], newAllX[:, i - 1])
                B = B + allP_N_N1[:, :, i] + np.outer(newAllX[:, i], newAllX[:, i - 1])
            C = C + newAllP[:, :, i] + np.outer(newAllX[:, i], newAllX[:, i])
            y_minus = y[i] - float(M.T @ newAllX[:, i].reshape((n_states, 1)))
            R_num = R_num + float(M.T @ newAllP[:, :, i] @ M) + (y_minus * y_minus)

        R = (1.0 / N) * R_num

        # Save old frequency for convergence check
        oldFreq = freqEst * Fs / (2.0 * np.pi)

        freqEst = np.zeros(len(freqs))
        ampEst = np.zeros(len(freqs))
        allQ = np.zeros(len(freqs))

        # update per-frequency parameters using 2x2 blocks
        for idx in range(len(freqs)):
            r0 = 2 * idx
            r1 = r0 + 2
            B_tmp = B[r0:r1, r0:r1]
            A_tmp = A[r0:r1, r0:r1]
            C_tmp = C[r0:r1, r0:r1]

            # lag-covariance relationship -> frequency estimate via atan of off-diagonals/trace
            numerator = (B_tmp[1, 0] - B_tmp[0, 1])
            denom = np.trace(B_tmp)
            # protect division by zero; if denom is zero, freqEst stays zero
            if np.abs(denom) < _eps:
                freqEst[idx] = 0.0
            else:
                freqEst[idx] = np.arctan2(numerator, denom)  # same as atan((B21-B12)/trace(B))

            # amplitude estimate (clamp below 1)
            amp_val = 0.0
            denomA = np.trace(A_tmp)
            if denomA < _eps:
                amp_val = 0.0
            else:
                amp_val = np.sqrt(numerator ** 2 + (np.trace(B_tmp)) ** 2) / denomA
            amp_val = min(amp_val, 1.0 - _eps)
            ampEst[idx] = amp_val

            # process noise for that oscillator
            allQ[idx] = (1.0 / (2.0 * N)) * (np.trace(C_tmp) - (ampEst[idx] ** 2) * np.trace(A_tmp))

        # convert freqEst back to Hz for phi/Q generation
        freqs_from_param = freqEst * Fs / (2.0 * np.pi)
        phi, Q, M = genParametersSoulatMdl_sspp(freqs_from_param, Fs, ampEst, allQ)

        omega = freqs_from_param.copy()
        stateVec = newAllX.copy()
        stateCov = newAllP.copy()

        # update loop control
        iter_idx += 1
        prev_error = np.sum(np.abs(omega - oldFreq))

    # ensure R is scalar float
    R = float(R)
    return omega, ampEst, allQ, R, stateVec, stateCov


# Helper: fixed-interval smoother implementation (RTS-style) - already used above
def fixedIntervalSmoother_sspp(x_t, x_next, P_t, P_next, phi, Q):
    """
    Given filtered x_t (x_t) and the smoothed next state x_next, plus covariances,
    compute smoothed state for time t (x_backone), its covariance (P_backone) and smoothing gain J.
    This implements the standard RTS equations.
    Note: inputs expected as column-vectors for x_t, x_next and 2D arrays for P_t, P_next.
    """
    # predicted covariance from time t -> t+1
    P_pred = phi @ P_t @ phi.T + Q
    # smoothing gain J = P_t * phi.T * inv(P_pred)
    J = P_t @ phi.T @ np.linalg.pinv(P_pred)
    x_backone = x_t + J @ (x_next - phi @ x_t)
    P_backone = P_t + J @ (P_next - P_pred) @ J.T
    return x_backone, P_backone, J


import numpy as np
import matplotlib.pyplot as plt

def demo_mu_rhythm_phase_estimation():
    """
    demo for testing the SSPE / causalPhaseEM_MKmdl_noSeg pipeline.
    Synthesizes a noisy mu-rhythm-like signal (~11 Hz) and runs phase estimation.
    """

    #Simulation setup
    Fs = 1000  # Hz
    t = np.arange(0, 3, 1 / Fs)
    mu_freq = 11  # Hz
    # synthetic EEG-like data: 11 Hz sine + 22 Hz harmonic + noise
    data = 0.8 * np.sin(2 * np.pi * mu_freq * t) + \
           0.2 * np.sin(2 * np.pi * 2 * mu_freq * t) + \
           0.5 * np.random.randn(len(t))



    # --- Remove slow drift with causal-like moving average ---
    kernel_size = 1000
    pad = kernel_size  # padding length (matches MATLAB trick)

    # Pad data with mean values at both ends
    padded = np.concatenate((
        np.full(pad, np.mean(data)),
        data,
        np.full(pad, np.mean(data))
    ))

    # Convolve with moving average kernel
    tmp_mean = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode="same")

    # Trim back to original data length
    start = pad
    end = pad + len(data)
    data = data - tmp_mean[start:end]

    print (f'length data {len(data)}')


    # --- Initialize model parameters ---
    initParams = {
        "freqs": [2, 11, 22],
        "Fs": Fs,
        "ampVec": [0.99, 0.99, 0.99],
        "sigmaFreqs": [0.01, 0.01, 0.01],
        "sigmaObs": 1.0,
        "windowSize": 1000,
        "lowFreqBand": [8, 14],
    }

    # --- Run causal phase estimation ---
    print("Running causal phase estimation...")
    phase, phaseBounds, allX_full, phaseWidth, returnParams = causalPhaseEM_MKmdl_noSeg(data, initParams, flagNoFit=0)

    print("Estimated parameters:")
    for k, v in returnParams.items():
        print(f"  {k}: {v}")

    # --- Plot results ---
    plt.figure(figsize=(10, 6))

    # Top plot: phase + confidence
    plt.subplot(2, 1, 1)
    plt.plot(phase, label="Estimated Phase", color="b")
    plt.fill_between(
        np.arange(len(phase)),
        phaseBounds[:, 0],
        phaseBounds[:, 1],
        color="red",
        alpha=0.3,
        label="95% Credible Interval",
    )
    plt.title("Estimated Phase for Mu-Rhythm (8–14 Hz)")
    plt.ylabel("Phase (radians)")
    plt.legend()
    plt.grid(True)

    # Bottom plot: raw data segment
    plt.subplot(2, 1, 2)
    plt.plot(data, color="gray")
    plt.title("Simulated EEG (Mu Rhythm)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return phase, phaseBounds, allX_full, phaseWidth, returnParams


# to run the demo:
if __name__ == "__main__":
    demo_mu_rhythm_phase_estimation()
