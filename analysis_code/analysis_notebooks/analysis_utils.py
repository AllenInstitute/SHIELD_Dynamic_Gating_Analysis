import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal.windows import exponential
from scipy.ndimage.filters import convolve1d
from numba import njit

@njit
def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins

@njit
def makePSTH_numba(spikes, startTimes, windowDur, binSize=0.001):
    print('running')
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins

def make_neuron_time_trials_tensor(unit_ids, spike_times, stim_start_time, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a tensor with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        unit_ids: unit_id, i.e. index from units table (same form as session.units table)
        spike_times: spike times corresponding to each unit (spike_times column from units table)
        stim_start_time: the time the stimulus started for each trial
        time_before: seconds to take before each start_time in the stim_table
        trial_duration: total time in seconds to take for each trial
        bin_size: bin_size in seconds used to bin spike counts 
    OUTPUTS:
        unit_tensor: tensor storing spike counts. The value in [i,j,k] 
            is the spike count for neuron i at time bin j in the kth trial.
    '''
    neuron_number = len(unit_ids)
    trial_number = len(stim_start_time)
    unit_tensor = np.zeros((neuron_number, int(trial_duration/bin_size), trial_number))
    
    for iu,unit_id in enumerate(unit_ids):
        unit_spike_times = spike_times[unit_id]
        for tt, trial_stim_start in enumerate(stim_start_time):
            unit_tensor[iu, :, tt] = makePSTH(unit_spike_times, 
                                                [trial_stim_start-time_before], 
                                                trial_duration, 
                                                binSize=bin_size)[0]
    return unit_tensor



def make_data_array(unit_ids, spike_times, stim_start_time, time_before_flash = 0.5, trial_duration = 2, bin_size = 0.001):
    '''
    
    '''

    # Make tensor (3-D matrix [units,time,trials])
    trial_tensor = make_neuron_time_trials_tensor(unit_ids, spike_times, stim_start_time, 
                                                  time_before_flash, trial_duration, 
                                                  bin_size)
    # make xarray data array
    trial_da = xr.DataArray(trial_tensor, dims=("unit_id", "time", "trials"), 
                               coords={
                                   "unit_id": unit_ids,
                                   "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                                   "trials": stim_start_time.index.values
                                   })
    return trial_da


def region_psth(area, units, session, starttime, duration):
    
    area_units = units[units['structure_acronym'].str.contains(area)]
    
    pop_spike_times = []
    for u, _ in area_units.iterrows():
        times = session.spike_times[u]
        pop_spike_times.extend(times)

    pop_spike_times = np.sort(np.array(pop_spike_times))
    
    poppsth, time = makePSTH(pop_spike_times, [starttime], duration, 0.01)
    time = time + starttime

    return poppsth/len(area_units), time


def exponential_convolve(response_vector, tau=1, symmetrical=True):
    
    center = 0 if not symmetrical else None
    exp_filter = exponential(10*tau, center=center, tau=tau, sym=symmetrical)
    exp_filter = exp_filter/exp_filter.sum()
    filtered = convolve1d(response_vector,exp_filter[::-1])
    
    return filtered    