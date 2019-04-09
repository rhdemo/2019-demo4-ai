#!/usr/bin/env python3
"""A library to prepare data for predictions"""

import json
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy.stats


counter = 0

def load_data(file):
    with open(file, "r") as fp:
        data = json.load(fp)

    return data

def process_crosses_data(data_crosses):
    '''Returns list of size N_samples and each element = tuple(index, label, acc+rot vel dataframe, rot angle dataframe)
    '''

    global counter
    
    data_cross_list = []
    gesture = 'draw-cross'

    for d in data_crosses: #loop over each sample
        acc_data = pd.DataFrame([x['acceleration'] for elem in data_crosses[d] for x in elem['motion']])
        acc_data.columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']

        rot_data = pd.DataFrame([x['rotationRate'] for elem in data_crosses[d] for x in elem['motion']])
        rot_data.columns = ['rotation_alpha', 'rotation_beta', 'rotation_gamma']

        timestamp_data = pd.DataFrame([x['timestamp'] for elem in data_crosses[d] for x in elem['motion']])
        timestamp_data.columns = ['timestamp']

        motion_df = pd.concat([acc_data, rot_data, timestamp_data], axis=1).sort_values('timestamp', ascending=True).drop('timestamp', axis=1)
        orientation_df = None

        data_cross_list.append((counter, gesture, motion_df, orientation_df))
        counter += 1
        
    return data_cross_list

def process_motion_rotation(example, counter):
    gesture = example.get('gesture', "")

    motion = np.array(example['motion'])
    orientation = np.array(example['orientation'])

    #process motion data
    if len(motion)>0:
        if motion.shape[1]==7: #incl. acc and rotational velocities
            motion_df = pd.DataFrame(motion, columns=['acceleration_x', 'acceleration_y', 'acceleration_z',
                                                        'rotation_alpha', 'rotation_beta', 'rotation_gamma',
                                                        'timestamp'
                                                        ])\
                            .sort_values('timestamp', ascending=True)\
                            .drop('timestamp', axis=1)
        
        elif motion.shape[1]==4: #incl. acc only
            motion_df = pd.DataFrame(motion, columns=['acceleration_x', 'acceleration_y', 'acceleration_z',
                                                        'timestamp'
                                                        ])\
                            .sort_values('timestamp', ascending=True)\
                            .drop('timestamp', axis=1)

        else: #shouldn't enter
            motion_df = pd.DataFrame()
    else: #no motion data recorded
        motion_df = pd.DataFrame()


    #process orientation data
    if len(orientation) > 0:
        orientation_df = pd.DataFrame(orientation, columns=['alpha', 'beta', 'gamma', 'timestamp'])\
                                .sort_values('timestamp', ascending=True)\
                                .drop('timestamp', axis=1)
    else:
        orientation_df = pd.DataFrame()

    return (counter, gesture, motion_df, orientation_df)

def process_dance_data(data_dance_raw):
    '''Returns list of size N_samples and each element = tuple(index, label, acc+rot vel dataframe, rot angle dataframe)
    '''

    #Ugly
    #convert appropriate data to dataframes
    global counter
    
    data = []
    #counter = 0 #counter from before
    for k in data_dance_raw.keys(): #loop over each dance type
        d = data_dance_raw[k]
        for example in d: #loop over each sample
            data.append(process_motion_rotation(example, counter))

            counter += 1

    return data

def clean_data(data):
    return [d for d in data if d[2].shape[1]==6 and d[2].shape[0]>50] # <---- clean data

def featurize(ts, bins, TAG):
    '''Take time-series and create features
    '''
    mean = np.mean(ts)
    median = np.median(ts)
    std = np.std(ts)
    length = len(ts)
    kurtosis = scipy.stats.kurtosis(ts)
    
    n,b = np.histogram(ts, bins=bins)
    n = np.array(n)/float(np.sum(n)) #normalize i.e. fraction of entries in each bin
    
    if median == 0: 
        features = {f'{TAG}_mean_over_median': 0, #dimensionless            
                    f'{TAG}_std_over_median': 0, #dimensionless            
                    f'{TAG}_length': length,
                    f'{TAG}_kurtosis': kurtosis, #already dimensionless by definition
                   }
        
    else: 
        features = {f'{TAG}_mean_over_median': mean/median, #dimensionless            
            f'{TAG}_std_over_median': std/median, #dimensionless            
            f'{TAG}_length': length,
            f'{TAG}_kurtosis': kurtosis, #already dimensionless by definition
           }
        
    for i, val in enumerate(n):
        features[f'{TAG}_binfrac_{i}'] = val
        
    
    return features

def find_bins(ts_list, method='freedman'):
    ''' Find bin edges for histograms based on different methods
    '''
    
    ts_all = np.concatenate(ts_list)
    
    plt.clf()
    if method in ['freedman', 'scott', 'knuth', 'blocks']:
        n,b = np.histogram(ts_all, bins=method)
        plt.hist(ts_all, bins=b)
    else:
        n,b,p = plt.hist(ts_all)

    return ts_all, b

def create_dataframe(data, add_label=True):
    df_list = []
    col_bins = {}

    cols = ['acceleration_x', 'acceleration_y', 'acceleration_z',
       'rotation_alpha', 'rotation_beta', 'rotation_gamma']
    method = 'plt'

    for col in cols:
        ts_all, b = find_bins([d[2][col] for d in data], method=method)
        col_bins[col] = b

    for col in cols:
        index_list, feature_list, label_list = [], [], []
        for d in data:
            features = featurize(d[2][col], bins=col_bins[col], TAG=col.upper())

            feature_list.append(features)
            index_list.append(d[0])
            label_list.append(d[1])

        feature_col_df = pd.DataFrame(feature_list)

        df_list.append(feature_col_df)
        
    df = pd.concat(df_list, axis=1)
    df['index'] = index_list
    if add_label:
        df['label'] = label_list
    
    return df




