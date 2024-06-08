import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from vbn_utils import *
from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession

from ccf_utils import *
import analysis_utils
import plot_utils

save_dir = "/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/derived_table_info/4Hz_bursts/all_areas_using_minima_125window"
structure_tree = pd.read_csv('../ccf_structure_tree_2017.csv')


def find_4Hz_bursts(session, save_location=save_dir):

    sess_id = session.metadata['ecephys_session_id']

    channels = session.get_channels()
    units = session.get_units()
    good_unit_filter = ((units['isi_violations']<0.5)&
                        (units['amplitude_cutoff'] < 0.1)&
                        (units['presence_ratio']>0.95)&
                        (units['quality']=='good'))

    units = units[good_unit_filter]
    units = units.merge(channels, left_on='peak_channel_id', right_index=True)
    units = add_brain_division_to_units_table(units, structure_tree)

    stimulus_presentations = session.stimulus_presentations

    cortex_counts = units[units['brain_division']=='Isocortex']['structure_acronym'].value_counts()
    cortical_areas = cortex_counts[cortex_counts>20]

    start_time = 0
    duration = stimulus_presentations.iloc[-1]['end_time']

    area_4Hz_bursts = {a:[] for a in cortical_areas.index.values}
    
    for cortical_area in cortical_areas.index.values:

        bursts, _ = analysis_utils.find_4Hz_bursts_for_area_just_spike_minima(cortical_area, units, session, 
                                                        start_time, duration)
        area_4Hz_bursts[cortical_area] = bursts

    time_before = 2
    time_after = 2
    for area in area_4Hz_bursts:

        burst_times = area_4Hz_bursts[area]
        for ist, start_time in enumerate(burst_times):
            fig = plt.figure(figsize=(16, 20))
            spec = fig.add_gridspec(7, 1)

            ax_spikes = fig.add_subplot(spec[1:])

            plot_utils.plot_raster(ax_spikes, units.sort_values(by=['brain_division', 'structure_acronym', 'dorsal_ventral_ccf_coordinate'], ascending=False), 
                        session.spike_times, start_time, structure_tree, time_before=time_before, time_after= time_after)

            ax_beh = fig.add_subplot(spec[0])
            plot_utils.plot_behavior(ax_beh, session, start_time, time_before, time_after) 

            fig.savefig(os.path.join(save_dir, str(sess_id) + '_' + area + '_' + str(ist)+'.png'))
            plt.close('all')
        
        if len(burst_times)>0:
            np.save(os.path.join(save_dir, str(sess_id) + '_' + area + '_burst_times.npy'), burst_times)


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_path', type=str)
    args = parser.parse_args()
    
    session = DynamicGatingEcephysSession.from_nwb_path(args.session_path)

    # call the plotting function
    find_4Hz_bursts(
        session
    )