import numpy as np
import pandas as pd
import mne
import os
import warnings
from mne.time_frequency import psd_array_multitaper
from fooof import FOOOF
import matplotlib.pyplot as plt
import scipy.integrate as sp

from shiny_epoch_utils import split_epochs_by_condition

warnings.filterwarnings("ignore", category=RuntimeWarning)

#n_jobs = -1


def process_psd(processed_set_path,
                normalize_total_power=True, band_power_method='fooof_peaks',
                plot_fooof=False, plot_channel=None, plots_dir=None, ppt_id=None,
                condition_spec=None):
    """
    Processes an EEG '.set' file to compute the offset, exponent, and band
    powers from power spectral analysis, split by user-defined conditions.
    Calculates power spectrum per epoch, averages across epochs, then runs
    FOOOF on the epoch-averaged spectrum (multitaper method).

    Parameters:
    - processed_set_path (str): Path to the processed '.set' EEG file.
    - normalize_total_power (bool): If True, normalizes band power by total power.
    - band_power_method (str): 'psd_integration' or 'fooof_peaks'.
    - plot_fooof (bool): If True, generates plots of the FOOOF fit.
    - plot_channel (str): Name of the channel to plot.
    - condition_spec (dict | None): {condition_name: [trigger_code_str, ...]}.
      If None/empty, all epochs are processed as a single condition ('all').

    Returns:
    - dict: Contains DataFrame of computed measures. Returns None if no usable epochs.
    """
    print(f'Processing PSD for {os.path.basename(processed_set_path)}')

    # Load processed epochs
    try:
        epochs = mne.read_epochs_eeglab(processed_set_path)
        sfreq = epochs.info['sfreq']
    except Exception as e:
        print(f"Error loading processed epochs: {e}")
        return None

    # Split into conditions (resting = single 'all' condition)
    cond_data = split_epochs_by_condition(epochs, condition_spec)
    if not cond_data:
        print('No valid epochs found, skipping participant')
        return None

    # Define frequency bands
    freq_bands = {
        'Delta': [1, 4],
        'Theta': [4, 8],
        'Alpha': [8, 13],
        'Beta':  [13, 30],
        'Gamma': [30, 48]
    }

    channel_names = epochs.ch_names
    freq_range = [1, 48]

    if plot_fooof and plot_channel is not None and plot_channel in channel_names:
        plot_ch_idx = channel_names.index(plot_channel)
        os.makedirs(plots_dir, exist_ok=True)
    else:
        plot_ch_idx = None

    results = []

    # Loop through conditions
    for condition, data in cond_data:

        n_epochs, n_channels, n_times = data.shape

        for ch_idx, channel_name in enumerate(channel_names):  # channels

            # Compute PSD per epoch for this channel
            epoch_psds = []
            for i in range(n_epochs):
                epoch_data = data[i, ch_idx, :] * 1e6  # convert to microvolts

                try:
                    psd, freqs = psd_array_multitaper(
                        epoch_data,
                        sfreq=sfreq,
                        fmin=1,
                        fmax=48,
                        normalization='full',
                        bandwidth=2.0,
                        verbose=False
                    )
                    epoch_psds.append(psd)
                except Exception as e:
                    print(f' PSD failed for epoch {i}, channel {channel_name}: {e}')
                    continue

            if len(epoch_psds) == 0:
                print(f' No valid PSDs for {channel_name} - {condition}, skipping')
                continue

            avg_psd = np.mean(epoch_psds, axis=0)
            avg_psd += 1e-12  # avoid log(0)
            n_epochs_used = len(epoch_psds)

            # Fit FOOOF on epoch-averaged spectrum
            fm = FOOOF(peak_width_limits=[2, 12], verbose=False)
            fm.fit(freqs, avg_psd, freq_range)

            # Check fit quality, reject if R2 < 0.95
            r_squared = fm.r_squared_
            fit_error = fm.error_

            if r_squared < 0.95:
                print(f' Poor FOOOF fit for {channel_name} - {condition}: R2 = {r_squared:.3f}, setting to NaN ')
                offset = np.nan
                exponent = np.nan
                band_powers = {band_name: np.nan for band_name in freq_bands}
            else:

                # Extract aperiodic parameters
                aperiodic_params = fm.aperiodic_params_
                if len(aperiodic_params) >= 2:
                    offset = aperiodic_params[0]
                    exponent = aperiodic_params[1]
                else:
                    offset = np.nan
                    exponent = np.nan

                # Compute band powers
                band_powers = {}

                if band_power_method == 'psd_integration':
                    total_power = sp.trapezoid(avg_psd, freqs)
                    for band_name, band_range in freq_bands.items():
                        band_indices = np.where(
                            (freqs >= band_range[0]) & (freqs <= band_range[1])
                        )[0]
                        if band_indices.size > 1:
                            band_psd = avg_psd[band_indices]
                            band_freqs = freqs[band_indices]
                            band_power = sp.trapezoid(band_psd, band_freqs)
                            if normalize_total_power:
                                band_power = band_power / total_power if total_power > 0 else np.nan
                        else:
                            band_power = np.nan
                        band_powers[band_name] = band_power

                elif band_power_method == 'fooof_peaks':
                    peak_params = fm.get_params('peak_params')
                    if peak_params is not None and np.array(peak_params).size > 0:
                        peak_params = np.atleast_2d(peak_params)  # ensure always 2D
                        total_peak_power = np.sum(peak_params[:, 1])
                    else:
                        peak_params = np.empty((0, 3))
                        total_peak_power = np.nan
                    for band_name, band_range in freq_bands.items():
                        band_peaks = peak_params[
                            (peak_params[:, 0] >= band_range[0]) &
                            (peak_params[:, 0] <= band_range[1])
                        ]
                        if band_peaks.size > 0:
                            band_power = np.sum(band_peaks[:, 1])
                        else:
                            band_power = np.nan
                        if normalize_total_power and total_peak_power > 0:
                            band_power = band_power / total_peak_power
                        band_powers[band_name] = band_power

                else:
                    print(f"Unknown band_power_method: {band_power_method}")
                    return None

                # Build result row
                result = {
                    'condition': condition,
                    'channel':   channel_name,
                    'n_epochs_used': n_epochs_used,
                    'Offset':    offset,
                    'Exponent':  exponent,
                }
                for band_name in freq_bands:
                    result[f'{band_name}_Power'] = band_powers.get(band_name, np.nan)
                results.append(result)

                # Plot FOOOF if enabled
                if plot_fooof and (plot_ch_idx is None or ch_idx == plot_ch_idx):
                    plt.figure(figsize=(10, 6))
                    fm.plot(plot_peaks='shade', add_legend=True)
                    plt.title(f"FOOOF Fit - Channel {channel_name} - {condition}")
                    plot_filename = f"fooof_fit_{ppt_id}_channel_{channel_name}_{condition}.png"
                    plot_path = os.path.join(plots_dir, plot_filename)
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Saved FOOOF plot to {plot_path}")

    if not results:
        print('No results computed')
        return None

    measures_df = pd.DataFrame(results)
    measures_df = measures_df.sort_values(['condition', 'channel']).reset_index(drop=True)

    print(f'Computed PSD: {measures_df.shape[0]} rows')

    return {'measures': measures_df}