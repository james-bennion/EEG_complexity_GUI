import numpy as np
import pandas as pd
import mne
import os
import warnings
from scipy.signal import detrend

from shiny_epoch_utils import split_epochs_by_condition

warnings.filterwarnings("ignore", category=RuntimeWarning)

#n_jobs = -1

def LZ78(X):
    """
    Compute LZ78 complexity of univariate time signal X using standard
    word-dictionary algorithm.

    Inputs:
        X -- 1D array with real-valued signal
    Outputs:
        c -- raw (unnormalised) value of LZ78 complexity (i.e., dictionary length)
    """
    if len(X.shape) > 1 and X.shape[1] > 1:
        raise ValueError("Input array must be 1D")

    v = (detrend(X) > 0).astype(int)
    s = ''.join(map(str, v))

    dictionary = {}
    w = ""

    for ch in s:
        w += ch
        if w not in dictionary:
            dictionary[w] = True
            w = ""

    c = len(dictionary)
    return c


def process_LZ78(processed_set_path, channel_labels, condition_spec=None):
    """
    Processes a preprocessed EEG '.set' file to compute LZC for each
    individual channel, split by user-defined conditions.

    Parameters:
    - processed_set_path (str): Path to the processed '.set' EEG file.
    - channel_labels (list): List of channel name strings (e.g., ['P7', 'P4', ...])
    - condition_spec (dict | None): {condition_name: [trigger_code_str, ...]}.
      If None/empty, all epochs are processed as a single condition ('all'),
      which is the resting-state case.

    Returns:
    - dict: Maps condition name -> {'lz_c': DataFrame}. Returns None if no
      usable epochs were found.
    """
    # Load processed epochs
    try:
        epochs = mne.read_epochs_eeglab(processed_set_path)
    except Exception as e:
        print(f"Error loading processed epochs: {e}")
        return None

    # Split into conditions (resting = single 'all' condition)
    cond_data = split_epochs_by_condition(epochs, condition_spec)
    if not cond_data:
        print('No valid epochs found, skipping participant')
        return None

    # Initialise results for each condition
    results = {}

    for condition, data in cond_data:
        sz = data.shape  # (n_epochs, n_channels, n_times)

        lz_values = {ch: [] for ch in channel_labels}

        for i in range(sz[0]):  # epochs
            for ch_idx, ch in enumerate(channel_labels):  # individual channels

                # Extract data for current channel and epoch
                sample_data_lz = data[i, ch_idx, :]

                # Compute LZ
                dic_vec = LZ78(sample_data_lz)

                # Normalisation
                dis_data = np.random.permutation(sample_data_lz)
                shuffled_lzc = LZ78(dis_data)
                if shuffled_lzc == 0:
                    normalised_lz = np.nan
                else:
                    normalised_lz = dic_vec / shuffled_lzc

                lz_values[ch].append(normalised_lz)

        # Create DataFrames
        df_lzc = pd.DataFrame(lz_values)
        df_lzc['epoch'] = range(sz[0])

        # Reorder columns to have 'epoch' first
        df_lzc = df_lzc[['epoch'] + channel_labels]

        results[condition] = {'lz_c': df_lzc}

        print(f'Computed LZ metrics for {condition}: {sz[0]} epochs, {sz[1]} channels')

    return results