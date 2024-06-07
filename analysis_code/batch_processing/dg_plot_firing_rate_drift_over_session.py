import numpy as np
import pandas as pd
import os
import argparse

from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession


import warnings
warnings.filterwarnings('ignore')
save_dir = "/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/firing_rate_drift"

def firing_rate_drift(session, save_dir=save_dir):
    
    session_id = session.metadata['ecephys_session_id']

    units = session.get_units()

    spike_times = session.spike_times

    good_units = units[(units['quality']=='good') &
                        (units['isi_violations']<0.5) &
                        (units['presence_ratio']>0.95)&
                        (units['amplitude_cutoff']<0.1)]

    stimulus_presentations = session.stimulus_presentations
    first_stim = stimulus_presentations.iloc[0]['start_time']

    all_spike_times = []
    for iu, unit in good_units.iterrows():

        uspikes = spike_times[iu]
        if len(uspikes)<2:
            continue
        all_spike_times.extend(uspikes)
    
    h, b = np.histogram(all_spike_times, bins=np.arange(int(first_stim), int(np.max(all_spike_times))))
    np.save(os.path.join(save_dir, f'{session_id}.npy'), h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_path', type=str)
    args = parser.parse_args()
    
    session = DynamicGatingEcephysSession.from_nwb_path(args.session_path)

    firing_rate_drift(
        session
    )