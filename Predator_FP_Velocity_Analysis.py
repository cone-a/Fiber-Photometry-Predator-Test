# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:19:32 2024

@author: aaron.cone
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import numpy as np
import os
import seaborn as sns
  
#%%

# Directory containing the files
directory = 'C:/your_directory_to_files'

# Dictionaries to store DataFrames
dfs_velo = {}


# Function to generate the dictionary key from the filename
def generate_key(filename):
    parts = filename.split('_')
    key = '_'.join(parts[2:4])  # Adjusted based on the filename structure
    return key

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.CSV'):
        filepath = os.path.join(directory, filename)
        key = generate_key(filename)
        if '_Velocity_values' in filename:
            dfs_velo[key] = pd.read_csv(filepath)
            dfs_velo[key].columns = ['frame', 'velocity']
            dfs_velo[key]['Time'] = pd.Series(np.arange(start = 0, stop=len(dfs_velo[key]), step=0.0334))


#%%  


#Peri-event histogram for continuous values.
def contvar_peh(var_ts, var_vals, ref_ts, min_max, bin_width = False):
    
    if bin_width:
        ds_ts = np.linspace(var_ts.min(), var_ts.max(), int((var_ts.max()-var_ts.min())/bin_width))
        ds_vals = np.interp(ds_ts, var_ts, var_vals)
        rate = bin_width
    
    else:
        rate = np.diff(var_ts).mean()
        ds_ts, ds_vals = (np.array(var_ts), np.array(var_vals))       
        
    left_idx = int(min_max[0]/rate)
    right_idx = int(min_max[1]/rate)
    
    all_idx = np.searchsorted(ds_ts,ref_ts, "right")   
    all_trials = np.vstack([ds_vals[idx+left_idx:idx+right_idx] for idx in all_idx])
    
    return all_trials


 #Assigns perievent per animal from timestampe

ambush_velo_events = {
    'C65_M1': [60.3538],
    'C76_M3': [61.4226],
    'C76_M1': [62.0906],
    'C76_M0': [61.957],
    'C66_F3': [60.454],
    'C69_M2': [61.289],
    'C70_F3': [62.1908],
    'C72_M2': [60.788],
    'C71_F2': [61.3224]
}



to_start = -15
to_end = 15

peri_events = {}

### 470 Signal ###
for mouse in dfs_velo:
    events = contvar_peh(var_ts = dfs_velo[mouse]['Time'], 
                          var_vals = dfs_velo[mouse]['velocity'],
                            ref_ts = ambush_velo_events[mouse], ### TImestamped ambush events
                          min_max = (to_start,to_end), 
                          bin_width = False)
    # break
    peri_events[mouse]=events ## saves array for each peri-event for all animals

#%%  

import pandas as pd
from itertools import chain


############# MAKE SURE TO CHANGE PERI_EVENT_ ######################

# creates time series for plotting, add a zero after events (e.g. num = events[0}.size])
time_peri_event = np.linspace(start = to_start, stop = to_end, num = events[0].size, retstep=0.016)


# Formatted so can process SEM and heatmap
to_line_points = chain.from_iterable(peri_events.values())#### TImestamped ambush events
# to_line_points = chain.from_iterable(peri_events_spon.values())####  ### spontanemous events based on random trial corresponding to animal velocity 10cm/s or greater

lined_up_points = np.array(list(to_line_points))

# Formatted so can process SEM and heatmap
lined_up_points = np.mean(list(peri_events.values()), axis=1)


#%%

# Calculates means for all data points
points = lined_up_points.mean(axis=0)

# Calculates standard error of mean for data points
points_sem = stats.sem(lined_up_points)

# Creates dataframe to plot 
to_plot = pd.DataFrame({'Time': time_peri_event[0], 'zdFF': points})
#%%
# Make figure
fig, ax = plt.subplots(figsize=(16, 10)) # you change dimensions of plot here Width x Length

# ax = plt.gca() # needed for line below - change y axis min and max
# ax.set_ylim([-1.5, 2.5]) #change y axis min and max

# Makes line plot
ax.plot('Time', 'zdFF', data = to_plot)
ax.fill_between(to_plot['Time'], to_plot['zdFF'] - points_sem, 
                   to_plot['zdFF'] + points_sem, alpha=0.15)
ax.set_xlabel('Time (sec)')
ax.set_ylabel('Velocity (cm/s)')
ax.margins(x=0)        
