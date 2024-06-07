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

windowDur = 0.75
binSize = 0.001
nBins = int(windowDur/binSize)

# h5Path = r'C:/Users/svc_ccg/Desktop/Analysis/vbn/vbnAllUnitSpikeTensor.hdf5'
h5Path = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/dg_active_unit_tensor.hdf5'
h5File = h5py.File(h5Path,'w')

sessionCount = 0
for session, row in dynamic_gating_sessions.iterrows():
    exp_id = row['ecephys_session_id']
    file_path = glob.glob(os.path.join(dynamic_gating_nwb_dir, str(exp_id), str(exp_id)+'.nwb'))
    if len(file_path)==0:
        continue

    session = DynamicGatingEcephysSession.from_nwb_path(file_path[0])

    sessionCount += 1
    print('session '+str(sessionCount))
    
    
    stim = session.stimulus_presentations
    # flashTimes = stim.start_time[stim.active]
    flashTimes = stim[stim['active']]['start_time'].values
    
    units = session.get_units()
    good_unit_filter = ((units['isi_violations']<0.5)&
                    (units['amplitude_cutoff']<0.1)&
                    (units['presence_ratio']>0.9))
    
    goodUnits = units[good_unit_filter]

    spikeTimes = session.spike_times
    
    h5Group = h5File.create_group(str(exp_id))
    h5Group.create_dataset('unitIds',data=goodUnits.index,compression='gzip',compression_opts=4)
    spikes = h5Group.create_dataset('spikes',shape=(len(goodUnits),len(flashTimes),nBins),dtype=bool,chunks=(1,len(flashTimes),nBins),compression='gzip',compression_opts=4)
    
    i = 0
    for unitId,unitData in goodUnits.iterrows(): 
        spikes[i] = getSpikeBins(spikeTimes[unitId],flashTimes-0.25,windowDur,binSize)
        i += 1

h5File.close()
    