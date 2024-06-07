import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import median_abs_deviation
from scipy.stats import kstest


def makePSTH_numba(spikes, startTimes, windowDur, binSize=0.001, convolution_kernel=0.05, avg=True):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel/2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0,windowDur+binSize,binSize)
    convkernel = np.ones(int(convolution_kernel/binSize))
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/startTimes.size
    counts = np.convolve(counts, convkernel)/(binSize*convkernel.size)
    return counts[convkernel.size-1:-convkernel.size], bins[:-convkernel.size-1]

def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    '''
    Convenience function to compute a peri-stimulus-time histogram
    (see section 7.2.2 here: https://neuronaldynamics.epfl.ch/online/Ch7.S2.html)
    INPUTS:
        spikes: spike times in seconds for one unit
        startTimes: trial start times in seconds; the first spike count 
            bin will be aligned to these times
        windowDur: trial duration in seconds
        binSize: size of spike count bins in seconds
    OUTPUTS:
        Tuple of (PSTH, bins), where:
            PSTH gives the trial-averaged spike rate for 
                each time bin aligned to the start times;
            bins are the bin edges as defined by numpy histogram
    '''
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for start in startTimes:
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins[:-1]

def make_time_trials_array(spike_times, start_times, 
                            time_before, trial_duration, 
                            bin_size=0.001):

    num_time_bins = int(trial_duration/bin_size)
    
    # Initialize array
    trial_array = np.zeros((num_time_bins, len(start_times)))
    
    # now loop through trials and make a PSTH for this unit for every trial
    for it, trial_start in enumerate(start_times):
        trial_start = trial_start - time_before
        trial_array[:, it] = makePSTH(spike_times, 
                                        [trial_start], 
                                        trial_duration, 
                                        binSize=bin_size)[0][:num_time_bins]
    
    # Make the time vector that will label the time axis
    time_vector = np.arange(num_time_bins)*bin_size - time_before

    return trial_array, time_vector


def make_neuron_time_trials_array(units, spike_times, stim_table, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a 3D array with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        units: dataframe with unit info (same form as session.units table)
        spike_times: dictionary with spike times for each unit (ie session.spike_times)
        stim_table: dataframe whose indices are trial ids and containing a
            'start_time' column indicating when each trial began
        time_before: seconds to take before each start_time in the stim_table
        trial_duration: total time in seconds to take for each trial
        bin_size: bin_size in seconds used to bin spike counts 
    OUTPUTS:
        unit_array: 3D array storing spike counts. The value in [i,j,k] 
            is the spike count for neuron i at time bin j in the kth trial.
        time_vector: vector storing the trial timestamps for the time bins
    '''
    # Get dimensions of output array
    neuron_number = len(units)
    trial_number = len(stim_table)
    num_time_bins = int(trial_duration/bin_size)
    
    # Initialize array
    unit_array = np.zeros((neuron_number, num_time_bins, trial_number))
    
    # Loop through units and trials and store spike counts for every time bin
    for u_counter, (iu, unit) in enumerate(units.iterrows()):
        
        # grab spike times for this unit
        unit_spike_times = spike_times[iu]
        
        # now loop through trials and make a PSTH for this unit for every trial
        for t_counter, (it, trial) in enumerate(stim_table.iterrows()):
            trial_start = trial.start_time - time_before
            unit_array[u_counter, :, t_counter] = makePSTH(unit_spike_times, 
                                                            [trial_start], 
                                                            trial_duration, 
                                                            binSize=bin_size)[0]
    
    # Make the time vector that will label the time axis
    time_vector = np.arange(num_time_bins)*bin_size - time_before
    
    return unit_array, time_vector


def first_spikes_after_onset(spikes, start_times, duration='', censor_period = 0.0015):
    
    start_times = start_times + censor_period
    start_times = start_times[start_times<spikes.max()]
    
    first_spike_inds = np.searchsorted(spikes, start_times)
    first_spike_times = spikes[first_spike_inds] - start_times

    return first_spike_times + censor_period


def first_spike_jitter(spikes, start_times, duration='', censor_period=0.0015):
    
    first_spike_times = first_spikes_after_onset(spikes, start_times, censor_period)
    
    return median_abs_deviation(first_spike_times)
    

def first_spike_latency(spikes, start_times, duration='', censor_period=0.0015):
    
    return np.median(first_spikes_after_onset(spikes, start_times, censor_period))
    

def trial_spike_rates(spikes, start_times, duration):
    
    spike_counts = []
    for start in start_times:
        count = len(spikes[(spikes>start) & (spikes<=start+duration)])
        spike_counts.append(count)
    
    return (np.array(spike_counts)/duration)


def baseline_spike_rate(spikes, baseline_starts, baseline_ends, binsize=1):
    baseline_counts = []
    for bs, be in zip(baseline_starts, baseline_ends):
        count = len(spikes[(spikes>bs) & (spikes<=be)])
        baseline_counts.append(count)
    
    baseline_rate = np.sum(baseline_counts)/total_baseline_time
    return baseline_rate


def mean_trial_spike_rate(spikes, start_times, duration):
    return np.mean(trial_spike_rates(spikes,start_times,duration))


def cv_trial_spike_rate(spikes, start_times, duration):
    spike_rates = trial_spike_rates(spikes,start_times,duration)
    return np.std(spike_rates)/np.mean(spike_rates)


def csalt(spikes, start_times, baseline_start_times):
    
    first_spikes = first_spikes_after_onset(spikes, start_times)
    baseline_spikes = first_spikes_after_onset(spikes, baseline_start_times)
    
    try:
        #p = wilcoxon(first_spikes, baseline_spikes[:len(first_spikes)])[1]
        p = kstest(first_spikes, baseline_spikes)[1]
    except:
        p = np.nan
    return p


def get_baseline_bins(baseline_starts, baseline_ends, binssize=0.001):
    
    all_binstarts = []
    for bs, be in zip(baseline_starts, baseline_ends):
        
        binstarts = np.arange(bs, be, binssize)
        binstarts = binstarts[binstarts+binssize<be] #make sure you don't go beyond end
        
        all_binstarts = all_binstarts + list(binstarts)
        
    return all_binstarts
    

def count_spikes_in_bin(spikes, binstart, binend):
    
    return len(spikes[(spikes>binstart)&(spikes<=binend)])


def fraction_time_responsive(spikes, start_times, time_before, duration, 
                          baseline_bin_rates, binsize=0.001):
   
    trial_array, time = make_time_trials_array(spikes, start_times, time_before, duration, binsize)
    
#     baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
#     baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
#     baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs+binsize) for bs in baseline_bins_to_use]
#     baseline_bin_rates = np.array(baseline_bin_counts)/binsize

    pvals = []
    above_baseline = []
    for timebin in trial_array:
        p = mannwhitneyu(baseline_bin_rates, timebin)
        pvals.append(p[1])
        above_baseline.append(np.mean(timebin)>np.mean(baseline_bin_rates))
    
    above_baseline = np.array(above_baseline)
    
    num_sig_bins = np.sum((np.array(pvals)<0.01)&above_baseline)
    fraction_sig_bins = num_sig_bins/len(time)
    return fraction_sig_bins    


def fraction_trials_responsive(spikes, start_times, time_before, duration, 
                          baseline_bin_rates, binsize=0.001):
   
    trial_array, time = make_time_trials_array(spikes, start_times, time_before, duration, binsize)
    
#     baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
#     baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
#     baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs+binsize) for bs in baseline_bins_to_use]
#     baseline_bin_rates = np.array(baseline_bin_counts)/binsize

    pvals = []
    above_baseline = []
    for trial in trial_array.T:
        p = mannwhitneyu(baseline_bin_rates, trial)
        pvals.append(p[1])
        above_baseline.append(np.mean(trial)>np.mean(baseline_bin_rates))
    
    above_baseline = np.array(above_baseline)
    
    num_sig_bins = np.sum((np.array(pvals)<0.01)&above_baseline)
    fraction_sig_bins = num_sig_bins/len(pvals)
    return fraction_sig_bins    


def get_baseline_bin_rates(spikes, baseline_starts, baseline_ends, binsize=0.001):
    
    baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
    baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
    baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs+binsize) for bs in baseline_bins_to_use]
    baseline_bin_rates = np.array(baseline_bin_counts)/binsize
    
    return baseline_bin_rates

def plot_raster(ax, spikes, start_times, duration=0.03):
    
    raster = []
    for start in start_times:
        r = spikes[(spikes>=start)&(spikes<=start+duration)]
        if len(r)>0:
            raster.append(r-start)
#         else:
#             raster.append([np.nan])
    
    ax.eventplot(raster)
    ax.set_xlim([0, duration])
