import numpy as np
import pandas as pd
import ccf_utils
import matplotlib.colors

def plot_raster(ax, units, spike_times, alignment_time, structure_tree, time_before = 1, time_after = 1):
        
    for iu, (uid, unit) in enumerate(units.iterrows()):
        
        area = unit['structure_acronym']
        color = ccf_utils.get_area_color(area, structure_tree)
        
        spikes = spike_times[uid]
        spikes = spikes[(spikes>alignment_time-time_before)&(spikes<alignment_time+time_after)]
        
        spikes = spikes
        
        ax.eventplot(spikes, lineoffsets=iu, color=color)
    units = units.reset_index()
    for area in units['structure_acronym'].unique():
        areaunits = units[units['structure_acronym']==area]
        midpoint = np.median(areaunits.index.values)

        ax.text(alignment_time+time_after, midpoint, area, color=ccf_utils.get_area_color(area, structure_tree))


def plot_behavior(ax, session, alignment_time, time_before=1, time_after = 1):

    eye_tracking = session.eye_tracking
    eye_tracking_noblinks = eye_tracking[~eye_tracking['likely_blink']]

    running_speed = session.running_speed
    licks = session.licks
    rewards = session.rewards

    stimulus_presentations = session.stimulus_presentations

    #Get running data aligned to this reward
    trial_running = running_speed.query('timestamps >= {} and timestamps <= {} '.
                                        format(alignment_time-time_before, alignment_time+time_after))

    #Get pupil data aligned to this reward
    trial_pupil_area = eye_tracking_noblinks.query('timestamps >= {} and timestamps <= {} '.
                                        format(alignment_time-time_before, alignment_time+time_after))

    trial_licking = licks.query('timestamps >= {} and timestamps <= {} '.
                                    format(alignment_time-time_before, alignment_time+time_after))
    
    trial_rewards = rewards.query('timestamps >= {} and timestamps <= {} '.
                                    format(alignment_time-time_before, alignment_time+time_after))
    
    trial_stimuli = stimulus_presentations.query('end_time >= {} and start_time <= {}'.
                                             format(alignment_time-time_before, alignment_time+time_after))


    axr = ax
    axr.plot(trial_running['timestamps'], trial_running['speed'], 'k')
    axp = axr.twinx()
    axp.plot(trial_pupil_area['timestamps'], trial_pupil_area['pupil_area'], 'g')
    rew_handle, = axr.plot(trial_rewards['timestamps'], np.zeros(len(trial_rewards['timestamps'])), 'db', markersize=10)
    lick_handle, = axr.plot(trial_licking['timestamps'], np.zeros(len(trial_licking['timestamps'])), 'mo')

    for idx, stimulus in trial_stimuli.iterrows():
        color = '0.3'

        if stimulus['is_image_novel']:
            color = 'g'
        
        if stimulus['is_change']:
            color = 'b'
        
        if stimulus['stimulus_name'] != 'spontaneous':
            axr.axvspan(stimulus['start_time'], stimulus['end_time'], color=color, alpha=0.5)


def make_color_map_from_hex(hex_color):
    color_rgb = list(matplotlib.colors.to_rgb(hex_color))
    color_rgb.append(1)

    color_space = np.array([np.linspace(1, rgb, 256) for rgb in color_rgb]).T

    color_space[:, 3] = 1

    color_map = ListedColormap(color_space)

    return color_map


def formatFigure(fig, ax, title=None, xLabel=None, yLabel=None, xTickLabels=None, yTickLabels=None, blackBackground=False, saveName=None, no_spines=False):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    
    fig.set_facecolor('w')
    spinesToHide = ['right', 'top', 'left', 'bottom'] if no_spines else ['right', 'top']
    for spines in spinesToHide:
        ax.spines[spines].set_visible(False)

    ax.tick_params(direction='out',top=False,right=False)
    
    if title is not None:
        ax.set_title(title)
    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
        
    if blackBackground:
        ax.set_axis_bgcolor('k')
        ax.tick_params(labelcolor='w', color='w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        for side in ('left','bottom'):
            ax.spines[side].set_color('w')

        fig.set_facecolor('k')
        fig.patch.set_facecolor('k')
    if saveName is not None:
        fig.savefig(saveName, facecolor=fig.get_facecolor())