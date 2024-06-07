import numpy as np
import os
import argparse

from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession

save_dir = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/derived_table_info/im115_r_responses'

def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/startTimes.size
    return counts/binSize, bins


def run_image_responses(session, save_location=save_dir):
    '''
    a function that gets RF metrics for every unit
    and saves results out to csv file
    '''
    stimulus_presentations = session.stimulus_presentations
    stim_times = stimulus_presentations[stimulus_presentations['active']&
                            (stimulus_presentations['image_name']=='im115_r-1.0')]['start_time'].values

    units = session.get_units()
    quality_filter = ((units['isi_violations'] < 0.5) & 
                    (units['amplitude_cutoff']< 0.1) & 
                    (units['presence_ratio'] > 0.9) & 
                    (units['quality'] == 'good') &
                    ([True]*len(units)))

    spike_times = session.spike_times
    
    time_before = 1
    duration = 2.5
    good_units = units[quality_filter]
    for iu, unit in good_units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_change_response, bins = makePSTH(unit_spike_times, 
                                            stim_times-time_before, 
                                            duration, binSize=0.001)
        np.save(os.path.join(save_dir, str(iu)+'.npy'), unit_change_response)
    




if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    #parser.add_argument('--session_id', type=int)
    parser.add_argument('--session_path', type=str)
    args = parser.parse_args()
    
    session = DynamicGatingEcephysSession.from_nwb_path(args.session_path)

    # call the plotting function
    run_image_responses(
        session
    )