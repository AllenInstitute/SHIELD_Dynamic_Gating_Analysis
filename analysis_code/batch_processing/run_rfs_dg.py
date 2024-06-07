import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import argparse
from brain_observatory_utilities.datasets.electrophysiology.\
    receptive_field_mapping import ReceptiveFieldMapping_VBN

from allensdk.brain_observatory.ecephys.dynamic_gating_ecephys_session import DynamicGatingEcephysSession

save_dir = '/allen/programs/mindscope/workgroups/dynamicrouting/dynamic_gating/derived_table_info/rfs_metrics'

def run_rfs(session, save_location=save_dir):
    '''
    a function that gets RF metrics for every unit
    and saves results out to csv file
    '''
    rf = ReceptiveFieldMapping_VBN(session)
    rf_metrics = rf.metrics

    sess_id = session.metadata['ecephys_session_id']
    save_path = os.path.join(save_location, str(sess_id) + '.csv')
    rf_metrics.to_csv(save_path)


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