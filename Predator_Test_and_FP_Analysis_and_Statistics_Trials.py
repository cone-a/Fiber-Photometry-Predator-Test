# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:25:17 2023

@author: aaron.cone
"""
# Load libraries

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import numpy as np
import os
import seaborn as sns
  
#%%

# Directory containing the files
directory = 'D:/Fiber Photometry PredatorTest'

# Dictionaries to store DataFrames
dfs_iso = {}
dfs_gcamp = {}

# Function to generate the dictionary key from the filename
def generate_key(filename):
    parts = filename.split('_')
    key = '_'.join(parts[2:6])  # Adjust this based on the filename structure
    return key

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.CSV'):
        filepath = os.path.join(directory, filename)
        key = generate_key(filename)
        if '415_Signal' in filename:
            dfs_iso[key] = pd.read_csv(filepath)
        elif '470_Signal' in filename:
            dfs_gcamp[key] = pd.read_csv(filepath)

# Ensure that all keys match in both dictionaries
matching_keys = set(dfs_iso.keys()).intersection(set(dfs_gcamp.keys()))
non_matching_keys = set(dfs_iso.keys()).symmetric_difference(set(dfs_gcamp.keys()))

# Make DataFrames for matching keys the same length
for key in matching_keys:
    len_iso = len(dfs_iso[key])
    len_gcamp = len(dfs_gcamp[key])
    min_length = min(len_iso, len_gcamp)
    dfs_iso[key] = dfs_iso[key].iloc[:min_length]
    dfs_gcamp[key] = dfs_gcamp[key].iloc[:min_length]

#%%  

# Rename columns
columns_to_rename = {'Region0G': 'Region'}

# Update column names in dfs_iso dictionary
for key, df in dfs_iso.items():
    dfs_iso[key] = df.rename(columns=columns_to_rename)

# Update column names in dfs_gcamp dictionary
for key, df in dfs_gcamp.items():
    dfs_gcamp[key] = df.rename(columns=columns_to_rename)

### Loop  over two dictionaries at once  
for (key, value), (key2, value2) in zip(dfs_iso.items(),dfs_gcamp.items()) :
    value["Timestamp_Cumulative"] = value['Timestamp'].diff().cumsum().fillna(0)        
    value2["Timestamp_Cumulative"] = value2['Timestamp'].diff().cumsum().fillna(0)  
    
    
    



#%% 
### Smoothes data

 ### Smoothes signal

def smooth_signal(x,window_len=10,window='flat'): 
    
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
        the smoothed signal        
    """
    
    import numpy as np

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]






import matplotlib.dates as mdates

# Choose window you would like to smooth
smooth_win = 10
i = 0

# Creates figure isobestic and gcamp signal
for (key_iso, value_iso), (key_gcamp, value_gcamp) in zip(dfs_iso.items(), dfs_gcamp.items()):
    value_iso['smooth_isobestic'] = smooth_signal(value_iso['Region'], smooth_win)
    value_gcamp['smooth_gcamp_signal'] = smooth_signal(value_gcamp['Region'], smooth_win)
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(value_iso['Timestamp_Cumulative'], value_iso['smooth_isobestic'], 'blue', linewidth=1.5)
    ax1.title.set_text('Smooth Isobestic')
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    # ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))      ### Hour Interval
    ax2 = fig.add_subplot(212)
    ax2.plot(value_gcamp['Timestamp_Cumulative'], value_gcamp['smooth_gcamp_signal'], 'purple', linewidth=1.5)
    ax2.title.set_text('Smooth Gcamp')
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    # ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3)) ### Hour Interval
    fig.suptitle(list(dfs_gcamp.keys())[i])
    i += 1


#%% 
### Finds the baseline of signal

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit

# Fits exponential curve


def func(x, a, b, c): 
    return a * np.exp(-b * x) + c

i = 0

for (key, value), (key2, value2) in zip(dfs_iso.items(),dfs_gcamp.items()):
    xvalue = np.linspace(0, len(value['smooth_isobestic']),len(value2['smooth_gcamp_signal']))
    popt_iso, pcov = curve_fit(func,xvalue,value['smooth_isobestic'], maxfev=10000)
    popt_gcamp, pcov = curve_fit(func,xvalue,value2['smooth_gcamp_signal'], maxfev=10000)
    value['iso_popt'] = func(xvalue, *popt_iso)
    value2['gcamp_popt'] = func(xvalue, *popt_gcamp)
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(value['Timestamp_Cumulative'], value['smooth_isobestic'],'red',linewidth=1.5)
    ax1.plot(value['Timestamp_Cumulative'], func(xvalue, *popt_iso),'blue',linewidth=1.5, label='iso')
    ax1.set_title('Isobestic Signal')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    ax2 = fig.add_subplot(212)
    ax2.plot(value2['Timestamp_Cumulative'], value2['smooth_gcamp_signal'],'red',linewidth=1.5)
    ax2.plot(value2['Timestamp_Cumulative'], func(xvalue, *popt_gcamp), 'blue',linewidth=1.5, label='gcamp')
    ax2.set_title('Gcamp Signal')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    fig.suptitle(list(dfs_gcamp.keys())[i])
    i += 1




#%% 

# #### Original code ####
from sklearn.linear_model import Lasso

# Starts to label each mouse/figure at position 0 
i = 0 

iso_std_values = []
gcamp_std_values = []

for (key, value), (key2, value2) in zip(dfs_iso.items(),dfs_gcamp.items()):
    reference = (value['smooth_isobestic'] - value['iso_popt']) ## removes baseline from isobestic signal
    signal = (value2['smooth_gcamp_signal'] - value2['gcamp_popt']) ## removes baseline from gcamp signal
    z_reference = np.array((reference - np.median(reference)) / np.std(reference)) ## standardize isobestic signal
    z_signal = np.array((signal - np.median(signal)) / np.std(signal)) ## standardize gcamp signal
    # iso_std = np.std(reference)
    # gcamp_std = np.std(signal)
    # iso_std_values.append(iso_std)  # append iso_std to the list
    # gcamp_std_values.append(gcamp_std)  # append gcamp_std to the list
    lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000, positive=True, 
                random_state=9999, selection='random')
    n = len(z_reference)
    lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
    z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)  ## Aligns isobestic signal to gcamp signal
    value2['zdFF'] = (z_signal - z_reference_fitted) ## subtracted, zscored, and fitted signal
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(value2['Timestamp_Cumulative'], value2['zdFF'], 'black')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('zdFF')
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format x-axis labels to show time
    fig.suptitle(list(dfs_gcamp.keys())[i])
    i += 1
    
    
#%% 


### FOR PERI-EVENT ANALYSIS ###

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




mouse_events = {
    'Vglut_flp_C70_F3': [34],
    'Vglut_flp_C65_M1': [177],
    'Vglut_flp_C66_F3': [187],
    'Vglut_flp_C72_M2': [327],
    'Vglut_flp_C71_F2': [297],
    'Vglut_flp_C69_M2': [321],
    'Vglut_flp_C76_M0': [311],
    'Vglut_flp_C76_M1': [308],
    'Vglut_flp_C76_M3': [297]
}

# mouse_events_spon = {
#     'fp_65M1': [178.859, 175.893], 
#     'fp_66F3': [196.995, 187.926], 
#     'fp_69M2': [423.648, 327.22], 
#     'fp_70F3': [160.823, 342.686],
#     'fp_71F2': [513.126, 295.922],
#     'fp_72M2': [383.668, 329.187],
#     'fp_76M0': [370.308, 310.588],
#     'fp_76M1': [561.356, 308.254], 
#     'fp_76M3': [517.101, 296.421]}


# mouse_events_spon = {
#     'fp_65M1': [178.859], 
#     'fp_66F3': [196.995], 
#     'fp_69M2': [423.648], 
#     'fp_70F3': [160.823],
#     'fp_71F2': [513.126],
#     'fp_72M2': [383.668],
#     'fp_76M0': [370.308],
#     'fp_76M1': [561.356], 
#     'fp_76M3': [517.101]}


to_start = -15
to_end = 15

peri_events = {}

### 470 Signal ###
for mouse in dfs_gcamp:
    events = contvar_peh(var_ts = dfs_gcamp[mouse]['Timestamp_Cumulative'], 
                          var_vals = dfs_gcamp[mouse]['zdFF'],
                            ref_ts = mouse_events[mouse], ### TImestamped ambush events
                          min_max = (to_start,to_end), 
                          bin_width = False)
    # break
    peri_events[mouse]=events ## saves array for each peri-event for all animals



# peri_events_spon = {}

# ### 470 Signal ###
# for mouse in dfs_gcamp:
#     events = contvar_peh(var_ts = dfs_gcamp[mouse]['Timestamp_Cumulative'], 
#                           var_vals = dfs_gcamp[mouse]['zdFF'],

#                           ref_ts = mouse_events_spon[mouse], ### spontanemous events based on random trial corresponding to animal velocity 10cm/s or greater
#                           min_max = (to_start,to_end), 
#                           bin_width = False)
#     # break
#     peri_events_spon[mouse]=events ## saves array for each peri-event for all animals

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
fig, ax = plt.subplots(figsize=(16, 10)) # you change dimensions of plot here Width x Length
# Creates heatmap
sns.heatmap(lined_up_points, xticklabels='', yticklabels='')


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
ax.set_ylabel('Z-Score')
ax.margins(x=0)        

#%%


# # Assume 'df' is your DataFrame
data_1_to_2_seconds = to_plot[(to_plot ['Time'] >= 1) & (to_plot ['Time'] <= 3)]
data_1_to_2_seconds = data_1_to_2_seconds.drop(columns = ['Time'])
data_1_to_2_seconds = np.array(data_1_to_2_seconds).T

data_minus_1_to_minus_2_seconds = to_plot[(to_plot ['Time'] >= -3) & (to_plot ['Time'] <= -1)]
data_minus_1_to_minus_2_seconds = data_minus_1_to_minus_2_seconds.drop(columns = ['Time'])
data_minus_1_to_minus_2_seconds = np.array(data_minus_1_to_minus_2_seconds).T

### Spontaneous ###
data_1_to_2_seconds_spon = to_plot[(to_plot ['Time'] >= 1) & (to_plot ['Time'] <= 3)]
data_1_to_2_seconds_spon = data_1_to_2_seconds_spon.drop(columns = ['Time'])
data_1_to_2_seconds_spon = np.array(data_1_to_2_seconds_spon).T
#%%
import numpy as np
import scipy.stats as stats

# # Split the 'lined_up_points' array into two separate arrays along axis 1
# num_columns = lined_up_points.shape[1]
# midpoint = num_columns // 2

# data_interval_minus_5_to_0 = lined_up_points[:, :midpoint]
# data_interval_0_to_5 = lined_up_points[:, midpoint:]


data_interval_minus_5_to_0 = data_1_to_2_seconds
data_interval_0_to_5 = data_minus_1_to_minus_2_seconds

# Calculate means and standard errors for -n to n seconds interval
mean_minus_5_to_0 = np.mean(data_interval_minus_5_to_0)
sem_minus_5_to_0 = np.std(data_interval_minus_5_to_0, ddof=1) / np.sqrt(len(data_interval_minus_5_to_0))

# Calculate means and standard errors for n to n seconds interval
mean_0_to_5 = np.mean(data_interval_0_to_5)
sem_0_to_5 = np.std(data_interval_0_to_5, ddof=1) / np.sqrt(len(data_interval_0_to_5))

# Calculate standard error of the difference
se_diff = np.sqrt(sem_minus_5_to_0**2 + sem_0_to_5**2)

# Choose a confidence level (e.g., 95%)
confidence_level = 0.999

# Calculate the critical value (Z) for the chosen confidence level
z = stats.norm.ppf((1 + confidence_level) / 2)

# Calculate margin of error (MOE) for -n to n seconds interval
moe_minus_5_to_0 = z * sem_minus_5_to_0

# Calculate margin of error (MOE) for n to n seconds interval
moe_0_to_5 = z * sem_0_to_5

# Calculate the confidence intervals for each interval
ci_minus_5_to_0 = (mean_minus_5_to_0 - moe_minus_5_to_0, mean_minus_5_to_0 + moe_minus_5_to_0)
ci_0_to_5 = (mean_0_to_5 - moe_0_to_5, mean_0_to_5 + moe_0_to_5)

# Calculate the confidence interval for the difference
diff_mean = mean_0_to_5 - mean_minus_5_to_0
diff_moe = z * se_diff
ci_difference = (diff_mean - diff_moe, diff_mean + diff_moe)

print("-5 to 0 seconds interval:", ci_minus_5_to_0)
print("0 to 5 seconds interval:", ci_0_to_5)
print("Difference:", ci_difference)

#%%
ambush_post = data_1_to_2_seconds
spon_post = data_1_to_2_seconds_spon

#%%

# ### FOR Predator Test - Ambush Post 5 seconds vs Spontaneous Post 5 seconds			####

# import numpy as np
# import scipy.stats as stats

# # Split the 'lined_up_points' array into two separate arrays along axis 1
# # num_columns = lined_up_points.shape[1]
# # midpoint = num_columns // 2

data_interval_minus_5_to_0 = ambush_post
data_interval_0_to_5 =  spon_post

# Calculate means and standard errors for -n to n seconds interval
mean_minus_5_to_0 = np.mean(data_interval_minus_5_to_0)
sem_minus_5_to_0 = np.std(data_interval_minus_5_to_0, ddof=1) / np.sqrt(len(data_interval_minus_5_to_0))

# Calculate means and standard errors for n to n seconds interval
mean_0_to_5 = np.mean(data_interval_0_to_5)
sem_0_to_5 = np.std(data_interval_0_to_5, ddof=1) / np.sqrt(len(data_interval_0_to_5))

# Calculate standard error of the difference
se_diff = np.sqrt(sem_minus_5_to_0**2 + sem_0_to_5**2)

# Choose a confidence level (e.g., 95%)
confidence_level = 0.999

# Calculate the critical value (Z) for the chosen confidence level
z = stats.norm.ppf((1 + confidence_level) / 2)

# Calculate margin of error (MOE) for -n to n seconds interval
moe_minus_5_to_0 = z * sem_minus_5_to_0

# Calculate margin of error (MOE) for n to n seconds interval
moe_0_to_5 = z * sem_0_to_5

# Calculate the confidence intervals for each interval
ci_minus_5_to_0 = (mean_minus_5_to_0 - moe_minus_5_to_0, mean_minus_5_to_0 + moe_minus_5_to_0)
ci_0_to_5 = (mean_0_to_5 - moe_0_to_5, mean_0_to_5 + moe_0_to_5)

# Calculate the confidence interval for the difference
diff_mean = mean_0_to_5 - mean_minus_5_to_0
diff_moe = z * se_diff
ci_difference = (diff_mean - diff_moe, diff_mean + diff_moe)

print("-5 to 0 seconds interval:", ci_minus_5_to_0)
print("0 to 5 seconds interval:", ci_0_to_5)
print("Difference:", ci_difference)

#%%


# Calculate the intersection point (x-coordinate) of the two lines
intersection_x = (ci_minus_5_to_0[0] - ci_0_to_5[0]) / (ci_0_to_5[1] - ci_minus_5_to_0[1])

# Check if the intersection point is within the range of the vertical lines
intersection_within_range = -0.5 <= intersection_x <= 0.5

if intersection_within_range:
    print("The confidence intervals intersect.")
else:
    print("The confidence intervals do not intersect.")
