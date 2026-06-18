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


def compute_band_powers(fm, avg_psd, freqs, method, normalize_total_power, freq_bands):
    """
    Compute frequency band power from FOOOF fit using 1 of 3 methods.
    Returns (band_powers_dict, column_suffix).
    Methods:
      'fooof_peaks'     -> sum of FOOOF peak heights per band        (suffix 'FOOOF')
      'psd_integration' -> trapezoid integral of raw PSD per band    (suffix 'Power')
      'aap'             -> mean(log10 obs) - mean(log10 aperiodic)    (suffix 'AAP')
    """
    band_powers = {}

    if method == 'psd_integration':
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
        return band_powers, 'Power'

    elif method == 'fooof_peaks':
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
        return band_powers, 'FOOOF'

    elif method == 'aap':
        # Aperiodic-adjusted power per band, in FOOOF's internal log space
        fooof_freqs = fm.freqs
        obs_log = fm.power_spectrum
        aper_log = fm._ap_fit
        for band_name, band_range in freq_bands.items():
            band_indices = np.where(
                (fooof_freqs >= band_range[0]) & (fooof_freqs <= band_range[1])
            )[0]
            if band_indices.size > 0:
                obs_mn = np.mean(obs_log[band_indices])
                aper_mn = np.mean(aper_log[band_indices])
                band_powers[band_name] = obs_mn - aper_mn
            else:
                band_powers[band_name] = np.nan
        return band_powers, 'AAP'

    else:
        print(f"Unknown band_power_method: {method}")
        return None, None


def process_psd(processed_set_path,
                spatial_levels=('channel',),
                band_power_methods=('fooof_peaks',),
                normalize_total_power=True, region_spec=None,
                plot_fooof=False, plot_channel=None, plots_dir=None, ppt_id=None,
                condition_spec=None):
    """
    Processes an EEG '.set' file to compute the offset, exponent, and band power, split by user-defined conditions,
    Works at one or more spatial levels, using one or more band-power metrics.

    Calculates power spectrum per epoch, averages it across epochs (and channels,
    for region/global levels), then fits FOOOF on the averaged spectrum.

    Parameters:
    - processed_set_path (str): Path to the processed '.set' EEG file.
    - spatial_levels (iterable): subset of {'channel', 'region', 'global'}.
    - band_power_methods (iterable): subset of {'fooof_peaks', 'psd_integration', 'aap'}.
    - normalize_total_power (bool): If True, normalizes band power by total power
      (applies to fooof_peaks / psd_integration only).
    - plot_fooof (bool): If True, generates plots of the FOOOF fit.
    - plot_channel (str): Name of the channel to plot.
    - condition_spec (dict | None): {condition_name: [trigger_code_str, ...]}.
      If None/empty, all epochs are processed as a single condition ('all').

    Returns:
    - dict keyed by metric name -> DataFrame of measures. Returns None if nothing computed.
    """
    print(f'Processing PSD for {os.path.basename(processed_set_path)}')

    spatial_levels = [s for s in spatial_levels if s in ('channel', 'region', 'global')]
    band_power_methods = [m for m in band_power_methods
                          if m in ('fooof_peaks', 'psd_integration', 'aap')]
    if not spatial_levels or not band_power_methods:
        print('No spatial levels or band-power metrics selected, skipping PSD')
        return None

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

    # Regions come from GUI (region_spec) rather than being hardcoded
    regions = region_spec if region_spec else {}

    channel_names = epochs.ch_names

    freq_range    = [1, 48]

    if plot_fooof and plot_channel is not None and plot_channel in channel_names:
        plot_ch_idx = channel_names.index(plot_channel)
        os.makedirs(plots_dir, exist_ok=True)
    else:
        plot_ch_idx = None

    # One results list per selected metric
    results = {m: [] for m in band_power_methods}

    # Loop through conditions
    for condition, data in cond_data:

        n_epochs, n_channels, n_times = data.shape

        # ── GLOBAL: average PSD across all epochs and all channels, then fit FOOOF ──
        if 'global' in spatial_levels:
            global_psds = []
            for i in range(n_epochs):
                for ch_idx in range(n_channels):
                    epoch_data = data[i, ch_idx, :] * 1e6  # convert to microvolts
                    try:
                        psd, freqs = psd_array_multitaper(
                            epoch_data,
                            sfreq=sfreq,
                            fmin=1,
                            fmax=48,
                            normalization='full',
                            bandwidth=2.0,
                            n_jobs=-1,
                            verbose=False
                            )
                        global_psds.append(psd)
                    except Exception as e:
                        print(f' PSD failed for epoch {i}, channel {channel_names[ch_idx]}: {e}')
                        continue

            if len(global_psds) == 0:
                print(f' No valid PSDs for global - {condition}, skipping')
            else:
                avg_psd = np.mean(global_psds, axis=0)
                avg_psd += 1e-12

                # Fit FOOOF on global epoch- and channel-averaged spectrum
                fm = FOOOF(peak_width_limits=[2, 12], verbose=False)
                fm.fit(freqs, avg_psd, freq_range)

                # Check fit quality, reject if R2 < 0.95
                r_squared = fm.r_squared_

                if r_squared < 0.95:
                    print(f' Poor FOOOF fit for global - {condition}: R2 = {r_squared:.3f}, setting to NaN ')
                    offset = np.nan
                    exponent = np.nan
                    poor_fit = True
                else:
                    poor_fit = False
                    aperiodic_params = fm.aperiodic_params_
                    if len(aperiodic_params) >= 2:
                        offset   = aperiodic_params[0]
                        exponent = aperiodic_params[1]
                    else:
                        offset   = np.nan
                        exponent = np.nan

                # Compute each selected metric from this single fit
                for method in band_power_methods:
                    if poor_fit:
                        band_powers = {b: np.nan for b in freq_bands}
                        if method == 'aap':
                            suffix = 'AAP'
                        elif method == 'fooof_peaks':
                            suffix = 'FOOOF'
                        else:
                            suffix = 'Power'
                    else:
                        band_powers, suffix = compute_band_powers(
                            fm, avg_psd, freqs, method, normalize_total_power, freq_bands)
                        if band_powers is None:
                            return None

                    result = {
                        'condition': condition,
                        'level':     'global',
                        'unit':      'Global',
                        'n_epochs_used': n_epochs,
                        'Offset':    offset,
                        'Exponent':  exponent,
                    }
                    for band_name in freq_bands:
                        result[f'{band_name}_{suffix}'] = band_powers.get(band_name, np.nan)
                    results[method].append(result)

                # Plot FOOOF if enabled
                if plot_fooof:
                    plt.figure(figsize=(10, 6))
                    fm.plot(plot_peaks='shade', add_legend=True)
                    plt.title(f"FOOOF Fit - Global - {condition}")
                    plot_filename = f"fooof_fit_{ppt_id}_global_{condition}.png"
                    plot_path = os.path.join(plots_dir, plot_filename)
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Saved FOOOF plot to {plot_path}")

        # ── REGION: average PSD across epochs and channels within each region ──
        if 'region' in spatial_levels:
            for region_name, region_channels in regions.items():

                ch_indices = [channel_names.index(c) for c in region_channels if c in channel_names]
                if len(ch_indices) == 0:
                    print(f' No channels present for {region_name} - {condition}, skipping')
                    continue

                region_psds = []
                for i in range(n_epochs):
                    for ch_idx in ch_indices:
                        epoch_data = data[i, ch_idx, :] * 1e6
                        try:
                            psd, freqs = psd_array_multitaper(
                                epoch_data,
                                sfreq=sfreq,
                                fmin=1,
                                fmax=48,
                                normalization='full',
                                bandwidth=2.0,
                                n_jobs=-1,
                                verbose=False
                            )
                            region_psds.append(psd)
                        except Exception as e:
                            print(f' PSD failed for epoch {i}, channel {channel_names[ch_idx]}: {e}')
                            continue

                if len(region_psds) == 0:
                    print(f' No valid PSDs for {region_name} - {condition}, skipping')
                    continue

                avg_psd = np.mean(region_psds, axis=0)
                avg_psd += 1e-12

                # Fit FOOOF on regional epoch-averaged spectrum
                fm = FOOOF(peak_width_limits=[2, 12], verbose=False)
                fm.fit(freqs, avg_psd, freq_range)

                # Check fit quality, reject if R2 < 0.95
                r_squared = fm.r_squared_

                if r_squared < 0.95:
                    print(f' Poor FOOOF fit for {region_name} - {condition}: R2 = {r_squared:.3f}, setting to NaN ')
                    offset = np.nan
                    exponent = np.nan
                    poor_fit = True
                else:
                    poor_fit = False
                    aperiodic_params = fm.aperiodic_params_
                    if len(aperiodic_params) >= 2:
                        offset = aperiodic_params[0]
                        exponent = aperiodic_params[1]
                    else:
                        offset = np.nan
                        exponent = np.nan

                for method in band_power_methods:
                    if poor_fit:
                        band_powers = {b: np.nan for b in freq_bands}
                        if method == 'aap':
                            suffix = 'AAP'
                        elif method == 'fooof_peaks':
                            suffix = 'FOOOF'
                        else:
                            suffix = 'Power'
                    else:
                        band_powers, suffix = compute_band_powers(
                            fm, avg_psd, freqs, method, normalize_total_power, freq_bands)
                        if band_powers is None:
                            return None

                    result = {
                        'condition': condition,
                        'level':     'region',
                        'unit':      region_name,
                        'n_epochs_used': n_epochs,
                        'Offset':    offset,
                        'Exponent':  exponent,
                    }
                    for band_name in freq_bands:
                        result[f'{band_name}_{suffix}'] = band_powers.get(band_name, np.nan)
                    results[method].append(result)

        # ── CHANNEL: average PSD across epochs for each channel, then fit FOOOF ──
        if 'channel' in spatial_levels:
            for ch_idx, channel_name in enumerate(channel_names):

                epoch_psds = []
                for i in range(n_epochs):
                    epoch_data = data[i, ch_idx, :] * 1e6
                    try:
                        psd, freqs = psd_array_multitaper(
                            epoch_data,
                            sfreq=sfreq,
                            fmin=1,
                            fmax=48,
                            normalization='full',
                            bandwidth=2.0,
                            n_jobs=-1,
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
                avg_psd += 1e-12
                n_epochs_used = len(epoch_psds)

                # Fit FOOOF on epoch-averaged spectrum
                fm = FOOOF(peak_width_limits=[2, 12], verbose=False)
                fm.fit(freqs, avg_psd, freq_range)

                # Check fit quality, reject if R2 < 0.95
                r_squared = fm.r_squared_

                if r_squared < 0.95:
                    print(f' Poor FOOOF fit for {channel_name} - {condition}: R2 = {r_squared:.3f}, setting to NaN ')
                    offset = np.nan
                    exponent = np.nan
                    poor_fit = True
                else:
                    poor_fit = False
                    aperiodic_params = fm.aperiodic_params_
                    if len(aperiodic_params) >= 2:
                        offset = aperiodic_params[0]
                        exponent = aperiodic_params[1]
                    else:
                        offset = np.nan
                        exponent = np.nan

                for method in band_power_methods:
                    if poor_fit:
                        band_powers = {b: np.nan for b in freq_bands}
                        if method == 'aap':
                            suffix = 'AAP'
                        elif method == 'fooof_peaks':
                            suffix = 'FOOOF'
                        else:
                            suffix = 'Power'
                    else:
                        band_powers, suffix = compute_band_powers(
                            fm, avg_psd, freqs, method, normalize_total_power, freq_bands)
                        if band_powers is None:
                            return None

                    result = {
                        'condition': condition,
                        'level':     'channel',
                        'unit':      channel_name,
                        'n_epochs_used': n_epochs_used,
                        'Offset':    offset,
                        'Exponent':  exponent,
                    }
                    for band_name in freq_bands:
                        result[f'{band_name}_{suffix}'] = band_powers.get(band_name, np.nan)
                    results[method].append(result)

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

    # Build one DataFrame per metric
    out = {}
    for method, rows in results.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df.sort_values(['condition', 'level', 'unit']).reset_index(drop=True)
        out[method] = df

    if not out:
        print('No results computed')
        return None

    total_rows = sum(df.shape[0] for df in out.values())
    print(f'Computed PSD: {total_rows} rows across {len(out)} metric(s)')

    return out