#!/usr/bin/env python3
"""A library to prepare data for predictions"""

import json

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import scipy.stats

motion_cols = [
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'rotation_alpha', 'rotation_beta', 'rotation_gamma',
]
orientation_cols = ['alpha', 'beta', 'gamma']


def load_data(file):
    with open(file, "r") as fp:
        data = json.load(fp)

    return data


def get_raw_data(fs, path, debug=False):
    """Reads raw data from S3 - not parallelized."""
    data = {}

    gestures_paths = fs.ls(path)

    for gesture in gestures_paths:
        files_to_read = fs.ls(gesture)

        key = gesture.split('/')[-1]
        data[key] = []
        if debug:
            print(key)

        for file in files_to_read:
            with fs.open(file) as f:
                content = json.loads(f.read())
            data[key].append(content)

    return data


def process_cross_data(data, *, bins: dict = None, method: str = None):
    """Process collection of raw examples.

    :returns: bins, DataFrame
    """
    bins = bins or {}
    gesture = 'draw-cross'

    data_processed = []
    for ts, d in data.items():  # loop over each sample
        df = pd.DataFrame([], columns=motion_cols + ['timestamp'] + orientation_cols)

        acc_data = pd.DataFrame([x['acceleration'] for elem in d for x in elem['motion']])
        rot_data = pd.DataFrame([x['rotationRate'] for elem in d for x in elem['motion']])

        df[motion_cols] = pd.concat([acc_data, rot_data], axis=1)
        df['timestamp'] = pd.DataFrame(
            [x['timestamp'] for elem in d for x in elem['motion']])

        df = df \
            .sort_values('timestamp', ascending=True) \
            .drop('timestamp', axis=1)

        data_processed.append(df)

    data_clean = list(clean_data(data_processed))

    col_bins, feature_df = featurize(
        data_clean, label=gesture, col_bins=bins.get(gesture, None), method=method)

    return {gesture: col_bins}, feature_df


def process_motion_rotation(example: dict):
    """Process single raw example and returns dataframe.

    :returns: DataFrame
    """
    motion = np.array(example['motion'])
    orientation = np.array(example['orientation'])

    # process motion data
    df = pd.DataFrame([], columns=motion_cols + ['timestamp'] + orientation_cols)

    if len(motion) > 0:
        if motion.shape[1] == 7:  # incl. acc and rotational velocities
            df[motion_cols + ['timestamp']] = pd.DataFrame(motion)

    # process orientation data
    if len(orientation) > 0:
        df[orientation_cols] = pd.DataFrame(orientation[:, :-1])

    df = df \
        .sort_values('timestamp', ascending=True) \
        .drop('timestamp', axis=1)

    return df


def process_dance_data(data, *,
                       bins: dict = None,
                       method: str = None):
    """Process raw data into single dataframe.

    :returns: bins, DataFrame
    """
    df = pd.DataFrame()
    bins = bins or {}

    for gesture, d in data.items():  # loop over each dance type
        data_clean = list(clean_data(
            process_motion_rotation(example) for example in d))

        col_bins, feature_df = featurize(
            data_clean, label=gesture, col_bins=bins.get(gesture, None), method=method)

        bins[gesture] = col_bins

        df = pd.concat([df, feature_df], axis=0, ignore_index=True)

    return bins, df


def clean_data(data: list):
    """Clean collection of dataframes."""

    def condition(df): return all(df[motion_cols].any()) and len(df) > 50

    return filter(condition, map(lambda df: df[motion_cols].dropna(), data))


def featurize(data, *, label: str = None, col_bins: dict = None, method: str = None):
    """Featurize."""
    col_bins = col_bins or {}

    df = pd.DataFrame()
    for idx, col in enumerate(motion_cols):
        bins = col_bins.get(col, None)

        if bins is None:
            _, bins = find_bins(
                np.concatenate([d[col] for d in data]), method=method)
            col_bins[col] = bins

        tag = col.upper()

        feature_list = []
        for d in data:
            ts = d[col]

            mean = np.mean(ts)
            median = np.median(ts)
            std = np.std(ts)
            length = len(ts)
            kurtosis = scipy.stats.kurtosis(ts)

            n, b = np.histogram(ts, bins=bins)
            n = np.array(n) / float(np.sum(n))  # normalize i.e. fraction of entries in each bin

            if median == 0:
                features = {f'{tag}_mean_over_median': 0,  # dimensionless
                            f'{tag}_std_over_median': 0,  # dimensionless
                            f'{tag}_length': length,
                            f'{tag}_kurtosis': kurtosis,  # already dimensionless by definition
                            }

            else:
                features = {f'{tag}_mean_over_median': mean / median,  # dimensionless
                            f'{tag}_std_over_median': std / median,  # dimensionless
                            f'{tag}_length': length,
                            f'{tag}_kurtosis': kurtosis,  # already dimensionless by definition
                            }

            for i, val in enumerate(n):
                features[f'{tag}_binfrac_{i}'] = val

            feature_list.append(features)

        df = pd.concat([df, pd.DataFrame(feature_list)], axis=1)

    if label:
        df['label'] = label

    return col_bins, df.to_sparse()


def find_bins(x, method: str = None):
    """Find bin edges for histograms based on different methods."""
    if np.isscalar(x):
        x = [x]

    bins = method or mpl.rcParams['hist.bins']

    # basic input validation
    input_empty = np.size(x) == 0

    if input_empty:
        x = [np.array([])]
    else:
        x = mpl.cbook._reshape_2D(x, 'x')

    nx = len(x)  # number of datasets
    w = [None] * nx

    xmin = np.inf
    xmax = -np.inf
    for xi in x:
        if len(xi) > 0:
            xmin = min(xmin, np.nanmin(xi))
            xmax = max(xmax, np.nanmax(xi))
    bin_range = (xmin, xmax)

    # List to store all the top coordinates of the histograms
    tops = []
    mlast = None

    hist_kwargs = dict(range=bin_range)

    # Loop through datasets
    for i in range(nx):
        # this will automatically overwrite bins,
        # so that each histogram uses the same bins
        m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
        m = m.astype(float)  # causes problems later if it's an int
        tops.append(m)

    return tops, bins


def find_col_bins(data, method: str = None, export: str = None):
    """Find bins for every column in the data. Optionally exports as JSON."""
    data = list(data)

    col_bins = {}

    for idx, col in enumerate(motion_cols):
        bins = col_bins.get(col, None)

        if bins is None:
            _, bins = find_bins(
                np.concatenate([d[col] for d in data]), method=method)
            col_bins[col] = bins

    if export:
        bins_export = Path(export)
        bins_export.parent.mkdir(parents=True, exist_ok=True)

        bins_export.write_text(json.dumps({
            col: bins.tolist() for col, bins in col_bins.items()
        }))

    return col_bins
