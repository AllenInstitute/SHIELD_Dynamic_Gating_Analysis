import numpy as np
import pandas as pd
import os, glob
from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession
import h5py

dynamic_gating_nwb_dir = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/nwbs'
dynamic_gating_sessions = pd.read_csv('/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/nwbs/dynamic_gating_ecephys_sessions_table_07242023.csv')

def getSpikeBins(spikeTimes,startTimes,windowDur,binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    spikes = np.zeros((len(startTimes),bins.size-1),dtype=bool)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikeTimes,start)
        endInd = np.searchsorted(spikeTimes,start+windowDur)
        spikes[i] = np.histogram(spikeTimes[startInd:endInd]-start, bins)[0]
    return spikes

windowDur = 4
binSize = 0.001
nBins = int(windowDur/binSize)

# h5Path = r'C:/Users/svc_ccg/Desktop/Analysis/vbn/vbnAllUnitSpikeTensor.hdf5'
reference_area = 'VISp'

h5Path = f'/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/dg_active_unit_tensor_{reference_area}_random_times.hdf5'
h5File = h5py.File(h5Path,'w')

burst_data_dir = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/derived_table_info/4Hz_bursts/session_analysis'
reference_area_burst_sessions = glob.glob(os.path.join(burst_data_dir, '*', f'{reference_area}_burst_times.npy'))
reference_area_burst_sessions = [os.path.basename(os.path.dirname(v)) for v in reference_area_burst_sessions]

sessionCount = 0
for session, row in dynamic_gating_sessions.iterrows():
    exp_id = row['ecephys_session_id']

    file_path = glob.glob(os.path.join(dynamic_gating_nwb_dir, str(exp_id), str(exp_id)+'.nwb'))
    if len(file_path)==0 or str(exp_id) not in reference_area_burst_sessions:
        continue
    
    session = DynamicGatingEcephysSession.from_nwb_path(file_path[0])
    stimulus_presentations = session.stimulus_presentations
    behavior_start = stimulus_presentations[stimulus_presentations['active']].iloc[0]['start_time']
    behavior_end = stimulus_presentations[stimulus_presentations['active']].iloc[-1]['end_time']
    
    sessionCount += 1
    print('session '+str(sessionCount))
    
    burst_times_file = os.path.join(burst_data_dir, str(exp_id), f'{reference_area}_burst_times.npy')
    burst_times = np.load(burst_times_file)
    
    random_times = np.random.random(len(burst_times))*(behavior_end-behavior_start)+behavior_start

    times_to_use = burst_times #either generate tensor for random times or genuine burst times

    units = session.get_units()
    good_unit_filter = ((units['isi_violations']<0.5)&
                    (units['amplitude_cutoff']<0.1)&
                    (units['presence_ratio']>0.9))
    
    goodUnits = units[good_unit_filter]

    spikeTimes = session.spike_times
    
    h5Group = h5File.create_group(str(exp_id))
    h5Group.create_dataset('unitIds',data=goodUnits.index,compression='gzip',compression_opts=4)
    spikes = h5Group.create_dataset('spikes',shape=(len(goodUnits),len(times_to_use),nBins),dtype=bool,chunks=(1,len(times_to_use),nBins),compression='gzip',compression_opts=4)
    
    i = 0
    for unitId,unitData in goodUnits.iterrows(): 
        response = getSpikeBins(spikeTimes[unitId], times_to_use-2, windowDur, binSize)
        spikes[i] = response[:, :nBins]
        i += 1

h5File.close()
    