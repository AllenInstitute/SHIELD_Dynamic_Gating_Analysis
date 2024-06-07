import numpy as np
import pandas as pd
import os
import argparse
from vbn_utils import *
from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession
import ccf_utils
import analysis_utils

save_dir = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/derived_table_info/4Hz_bursts/session_analysis'
structure_tree = pd.read_csv('/allen/programs/mindscope/workgroups/np-behavior/ccf_structure_tree_2017.csv')
burst_data_dir = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/derived_table_info/4Hz_bursts/all_areas_using_minima_125window'

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
    units = ccf_utils.add_brain_division_to_units_table(units, structure_tree)

    stimulus_presentations = session.stimulus_presentations
    trials = session.trials
    stim_trials = stimulus_presentations.merge(trials, left_on='trials_id', right_index=True, suffixes=[None, '_trials'])

    burst_time_files = [f for f in os.listdir(burst_data_dir) if str(sess_id) in f and '.npy' in f]

    if len(burst_time_files)==0:
        return
    
    sess_save_dir = os.path.join(save_dir, str(sess_id))
    if not os.path.exists(sess_save_dir):
        os.mkdir(sess_save_dir)

    burst_time_areas = [b.split('_')[1] for b in burst_time_files]
    burst_time_areas = np.intersect1d(burst_time_areas, ['VISp', 'MOp']) #just run VISp and MOp

    behavior_start = stim_trials[stim_trials['active']].iloc[0]['start_time']
    behavior_end = stim_trials[stim_trials['active']].iloc[-1]['end_time']

    area_counts = units['structure_acronym'].value_counts()
    all_areas = area_counts[area_counts>20].index.values

    running = session.running_speed.sort_values(by='timestamps')
    running_resampled = analysis_utils.resample_df(running, 'timestamps', 10)
    
    eye_tracking = session.eye_tracking.sort_values(by='timestamps')
    eye_tracking = eye_tracking[~eye_tracking['likely_blink']]
    eye_tracking_resampled = analysis_utils.resample_df(eye_tracking, 'timestamps', 10)

    for area in burst_time_areas:
        area_file = [b for b in burst_time_files if area in b][0]
        burst_times = np.load(os.path.join(burst_data_dir, area_file))
        random_times = np.random.random(len(burst_times))*(behavior_end-behavior_start)+behavior_start

        for label, times in zip(['', '_random'], [burst_times, random_times]):
            rta = analysis_utils.triggered_average(running_resampled['speed'], running_resampled['time'], times, 5, 5)
            pta = analysis_utils.triggered_average(eye_tracking_resampled['pupil_area'], 
                            eye_tracking_resampled['time'], times,
                            5, 5)
            np.save(os.path.join(sess_save_dir, area + '_running_triggered_average' + label + '.npy'), rta)
            np.save(os.path.join(sess_save_dir, area + '_pupil_triggered_average' + label + '.npy'), pta)

        bins = np.arange(0, stimulus_presentations.iloc[-1]['end_time'], 60) - behavior_start
        h, b = np.histogram(burst_times-behavior_start, bins)
        
        np.save(os.path.join(sess_save_dir, area + '_burst_times.npy'), burst_times)

        np.save(os.path.join(sess_save_dir, area + '_burst_time_session_histogram.npy'), h)
        np.save(os.path.join(sess_save_dir, area + '_burst_time_session_histogram_bins.npy'), b)

        area_ccgs = {a:[] for a in all_areas}
        area_psths = {a:[] for a in all_areas}
        for ib, b in enumerate(burst_times):
            area_psth = analysis_utils.region_psth(area, units, session, b-2, 4)[0]
            for area2 in all_areas:
                if area2==area:
                    area2_psth = area_psth
                else:
                    area2_psth = analysis_utils.region_psth(area2, units, session, b-2, 4)[0]
                
                ccg = analysis_utils.crosscorrelogram(area_psth, area2_psth)
                area_ccgs[area2].append(ccg)
                area_psths[area2].append(area2_psth)
                    
        for area2 in area_ccgs:
            np.save(os.path.join(sess_save_dir, area + '_' + area2 + '_ccgs.npy'), area_ccgs[area2])
            np.save(os.path.join(sess_save_dir, area + '_' + area2 + '_psths.npy'), area_psths[area2])

        #Now recompute for random times
        area_ccgs = {a:[] for a in all_areas}
        area_psths = {a:[] for a in all_areas}
        for ib, b in enumerate(random_times):
            area_psth = analysis_utils.region_psth(area, units, session, b-2, 4)[0]
            for area2 in all_areas:
                if area2==area:
                    area2_psth = area_psth
                else:
                    area2_psth = analysis_utils.region_psth(area2, units, session, b-2, 4)[0]
                
                ccg = analysis_utils.crosscorrelogram(area_psth, area2_psth)
                area_ccgs[area2].append(ccg)
                area_psths[area2].append(area2_psth)

        for area2 in area_ccgs:
            np.save(os.path.join(sess_save_dir, area + '_' + area2 + '_ccgs_random.npy'), area_ccgs[area2])
            np.save(os.path.join(sess_save_dir, area + '_' + area2 + '_psths_random.npy'), area_psths[area2])


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    #parser.add_argument('--session_id', type=int)
    parser.add_argument('--session_path', type=str)
    args = parser.parse_args()
    
    #session = load_session(args.session_path)
    session = DynamicGatingEcephysSession.from_nwb_path(args.session_path)

    # call the plotting function
    find_4Hz_bursts(
        session
    )