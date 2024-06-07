import scipy.signal
import numpy as np
import pandas as pd
from scipy.signal.windows import exponential
from scipy.ndimage.filters import convolve1d

def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins


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


def exponential_convolve(response_vector, tau=1, symmetrical=False):
    
    center = 0 if not symmetrical else None
    exp_filter = exponential(10*tau, center=center, tau=tau, sym=symmetrical)
    exp_filter = exp_filter/exp_filter.sum()
    filtered = convolve1d(response_vector,exp_filter[::-1])
    
    return filtered


def find_4Hz_bursts_for_area_just_spike_minima(area, units, session, starttime, duration):

    poppsth, time = region_psth(area, units, session, starttime, duration)
    poppsth = exponential_convolve(poppsth, symmetrical=True)
    poppsth = -poppsth

    candidate_times = [-1]
    all_peaks = []
    segments = []
    time_step = 0.25
    window_duration = 1.25
    #slide window to look for events
    for epoch in np.arange(starttime, starttime+duration, time_step):
        start_index = np.argmin(np.abs(time-epoch))
        end_index = start_index + int(window_duration*100)
        
        poppsth_segment = poppsth[start_index:end_index]
        peaks = scipy.signal.find_peaks(poppsth_segment, prominence=5, 
                                        height = -1,
                                        width=[0.1*100, 0.3*100])
        

        peak_times = peaks[0]
        if len(peak_times)>3:
            
            peak_times = np.sort(peak_times)

            eligible_starts = []
            for ind in np.arange(0, len(peak_times)-2):
                
                intervals = np.diff(peak_times[ind:ind+3])
                if min(intervals)>=0.125*100 and max(intervals)<0.35*100:
                    eligible_starts.append(ind)

            if len(eligible_starts)>0:
                
                event_time = time[start_index + peak_times[eligible_starts[0]]]
                if np.min(event_time - candidate_times)>2:
                    candidate_times.append(event_time)
                    all_peaks.append(time[start_index + peak_times])
    
    candidate_times = np.array(candidate_times)
    
    return candidate_times[candidate_times>0], all_peaks


def triggered_average(timeseries, timestamps, alignment_times, time_before, time_after):

    aligned = []
    for al in alignment_times:
        start = al - time_before
        end = al + time_after
        aligned_trial = timeseries[(timestamps>=start)&(timestamps<end)]
        aligned.append(aligned_trial)
    
    return aligned

def resample_df(df, timestamp_col, ms_per_sample):
    df_resampled = df.set_index(pd.DatetimeIndex(df[timestamp_col]*1e9))
    df_resampled = df_resampled.asfreq(str(ms_per_sample) + 'ms', method='ffill')
    df_resampled['time'] = np.arange(0, len(df_resampled))*0.01 + df[timestamp_col].iloc[0]

    return df_resampled

def crosscorrelogram(x, y):

    x1 = x - np.mean(x)
    y1 = y - np.mean(y)

    denom = np.sqrt((np.sum(x1**2)*np.sum(y1**2)))
    ccg = scipy.signal.correlate(x1, y1, 'same')
    return ccg/denom