# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:02:20 2018

@author: Xiaoxuan Jia
"""

import numpy as np
from scipy import signal


def FIR_lowpass(x, fs=2500., order = 4, freq=500):
    """Apply 0 phase shift low pass filter"""
    #low pass filter designe 
    nyq = 0.5*fs
    b, a = signal.butter(order, freq/nyq, analog=False) #low pass butter filter for digital signal
    x_filtered = signal.filtfilt(b, a, x) # filtfilt 0 shift

    return x_filtered

def FIR_highpass(x, fs=2500.,  order = 4, freq=1.):
    """Same filter design as cerebus utah array."""
    #low pass filter designe 
    nyq = 0.5*fs
    b, a = signal.butter(order, freq/nyq, btype='highpass', analog=False) #low pass butter filter for digital signal
    x_filtered = signal.filtfilt(b, a, x) # filtfilt 0 shift

    return x_filtered


def noise_filter(x, fs=2500., order=4, freq_ranges=[[59.8,61.4],[94.4,95.6],[190.1,190.7]]):
    """filter 60Hz noise and harmonics
    x: signal
    freq_ranges: list of list of float numbers defining the range of noise

    """
    nyq = 0.5*fs
    sig = x
    for freq in freq_ranges:
        b, a = signal.butter(order, np.array(freq)/nyq, btype='bandstop') #low pass butter filter for digital signal
        sig_ff = signal.filtfilt(b, a, sig)
        sig = sig_ff

    return sig_ff

def filter_lfp(sig, nyq):
    """Apply filters to given lfp trace.
    LFP is sliced according to first synch on probe and last synch+post_stim on probe.
    post_stim is to allow some space for analysis after the last synch. """
    sig_lf = FIR_lowpass(sig,nyq)
    sig_lhf = FIR_highpass(sig_lf,nyq)
    #sig_lhf_n = noise_filter(sig_lhf)
    #del sig_lf, sig_lhf
    lfp_filtered=sig_lhf-np.mean(sig_lhf) #correct for shift among channels
    lfp_filtered=np.array(lfp_filtered)
    return lfp_filtered

#----------------------miscellaneous------------------------
def butter_bandpass(lowcut, highcut, nyq, order=4):
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return b, a

def apply_filter(x, B, A):
    """Apply designed filter to signal"""
    x_filtered = signal.filtfilt(B, A, x) # filtfilt 0 shift

    return x_filtered

def butter_bandpass_filter(data, sampling_freq, lowcut, highcut, order=4):
    b, a = butter_bandpass(lowcut, highcut, sampling_freq/2., order=order)
    y = signal.filtfilt(b, a, data)
    return y


def plot_bandpass_filter(self, orders, fs=2500, lowcut=0.5, highcut=500,plot=True):
    """Plot filter designe to determine the correct order of filters.
    orders: list of number or numbers
    """

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    filt = []
    for o in orders:
        b, a = butter_bandpass(lowcut, highcut, fs, order=o)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        filt.append([b,a])

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')

    filt=np.array(filt)
    return filt