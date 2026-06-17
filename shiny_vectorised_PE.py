import os
import numpy as np
import pandas as pd
import math
import mne
from itertools import permutations
import warnings

from shiny_epoch_utils import split_epochs_by_condition

warnings.filterwarnings("ignore", category=RuntimeWarning)

#n_jobs = -1

def compute_permutation_entropy(signal, m, tau):
    """
    Computes the normalized permutation entropy of a 1D signal.

    Parameters:
    - signal: 1D numpy array
    - m: Embedding dimension (kernel)
    - tau: Time delay (tau)

    Returns:
    - pe: Normalized permutation entropy value
    """
    # Validate enough data points
    n = len(signal)
    if n < m * tau:
        return np.nan  # Not enough data points

    # Enumerate all possible ordinal patterns
    possible_permutations = math.factorial(m)

    # Build lookup: ordinal pattern tuple into index
    perms_list = list(permutations(range(m)))
    lookup = {perm: idx for idx, perm in enumerate(perms_list)}
    n_perms = len(perms_list)

    # Build all embedded windows simultaneously: shape (n_windows, m)
    # Each row is onw window of m data points separated by tau steps
    n_windows = n - (m - 1) * tau
    window_idx = np.array([[i + j * tau  for j in range(m)]
                           for i in range(n_windows)])
    embedded = signal[window_idx] # shape (n_windows, m)

    # Argsort each window along axis=1 to get ordinal patterns
    ranked = np.argsort(embedded, axis=1)

    # Look up each ordinal pattern in dict (each row of ranked is one pattern, e.g. (1, 0, 2)
    sym_ids = np.array([lookup[tuple(row)] for row in ranked], dtype=np.int32)

    # Count occurences of each pattern and normalise to probabilities
    c = np.bincount(sym_ids, minlength=n_perms).astype(np.float64)
    c /= c.sum()

    # Compute permutation entropy
    pe = -np.nansum(c * np.log(c))
    # Normalize PE (1 = highly random, 0 = highly regular)
    pe /= np.log(possible_permutations)
    return pe


def process_permutation_entropy(processed_set_path, channel_labels, kernel, tau,
                                condition_spec=None):
    """
    Processes an EEG '.set' file to compute Permutation Entropy (PE) metrics,
    split by user-defined conditions.

    Parameters:
    - processed_set_path (str): Path to the processed '.set' EEG file.
    - channel_labels (list): Channel name strings.
    - kernel (int): The number of samples to use to transform to a symbol.
    - tau (int): The number of samples left between the ones that define a symbol.
    - condition_spec (dict | None): {condition_name: [trigger_code_str, ...]}.
      If None/empty, all epochs are processed as a single condition ('all').

    Returns:
    - dict: Contains DataFrame for PE. Returns None if no usable epochs.
    """
    print(f'Processing permutation entropy for {os.path.basename(processed_set_path)}')

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

    # Initialize list to collect results
    results = []

    for condition, data in cond_data:

        sz = data.shape  # (n_epochs, n_channels, n_times)

        for i in range(sz[0]):  # epochs
            for ch_idx, ch in enumerate(channel_labels):  # channels

                # Extract data for the current channel and epoch
                sample_data = data[i, ch_idx, :]

                # Compute PE
                try:
                    pe_value = compute_permutation_entropy(sample_data, kernel, tau)
                except Exception as e:
                    print(f"Error computing PE for epoch {i}, channel {ch}: {e}")
                    pe_value = np.nan

                # Save the result
                results.append({
                    'condition': condition,
                    'epoch': i,
                    'channel_group': ch,
                    'PE': pe_value
                })

    # Create DataFrame from results
    df_pe = pd.DataFrame(results)

    # Pivot to wide format: channels as columns
    df_pe = df_pe.pivot_table(
        index=['condition', 'epoch'],
        columns='channel_group',
        values='PE'
    ).reset_index()
    df_pe.columns.name = None

    print(f'Computed Permutation Entropy: {df_pe.shape[0]} condition-epoch '
          f'combinations, {len(channel_labels)} channels')

    return {'pe': df_pe}