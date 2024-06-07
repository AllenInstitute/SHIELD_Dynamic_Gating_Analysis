import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import argparse

from brain_observatory_utilities.datasets.electrophysiology.\
    receptive_field_mapping import ReceptiveFieldMapping_VBN

from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession


rf_metrics_dir = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/derived_table_info/rfs_metrics'
save_dir = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/derived_table_info/rf_mats'

def run_rfs(session, save_location=save_dir):
    '''
    a function that gets RF metrics for every unit
    and saves results out to csv file
    '''
    rf = ReceptiveFieldMapping_VBN(session)
    # read the metrics we already generated
    sess_id = session.metadata['ecephys_session_id']
    rf_metrics = pd.read_csv(os.path.join(rf_metrics_dir, str(sess_id) +'.csv'))

    rf_metrics = rf_metrics.set_index('unit_id')

    for iu, unit in rf_metrics.iterrows():
        urf = rf.get_receptive_field(iu)
        # fig, ax = plt.subplots()
        # ax.imshow(urf)
        # fig.savefig(os.path.join(save_dir, f'{iu}.png'))
        np.save(os.path.join(save_dir, str(iu)+'.npy'), urf)

    # save_path = os.path.join(save_location, str(sess_id) + '.csv')
    # rf_metrics.to_csv(save_path)


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    #parser.add_argument('--session_id', type=int)
    parser.add_argument('--session_path', type=str)
    args = parser.parse_args()
    
    session = DynamicGatingEcephysSession.from_nwb_path(args.session_path)

    # call the plotting function
    run_rfs(
        session
    )