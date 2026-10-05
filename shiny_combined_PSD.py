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

# Multitaper frequency smoothing (Hz); also the narrowest peak the PSD can show
MT_BANDWIDTH = 2.0


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
        # Closed bands [low, high]: adjacent integrals share only a zero-width boundary
        for band_name, band_range in freq_bands.items():
            band_indices = np.where(
                (freqs >= band_range[0]) & (freqs <= band_range[1])
            )[0]
            if band_indices.size > 1:
                band_psd = avg_psd[band_indices]
                band_freqs = freqs[band_indices]
                band_powers[band_name] = sp.trapezoid(band_psd, band_freqs)
            else:
                band_powers[band_name] = np.nan
        if normalize_total_power:
            # Relative power: each band / sum of the user's bands (sums to 1)
            total_power = np.nansum(list(band_powers.values()))
            band_powers = {b: (p / total_power if total_power > 0 else np.nan)
                           for b, p in band_powers.items()}
        return band_powers, 'Power'

    elif method == 'fooof_peaks':
        peak_params = fm.get_params('peak_params')
        if peak_params is not None and np.array(peak_params).size > 0:
            peak_params = np.atleast_2d(peak_params)  # ensure always 2D
            total_peak_power = np.sum(peak_params[:, 1])
        else:
            peak_params = np.empty((0, 3))
            total_peak_power = np.nan
        # Half-open bands [low, high) so a peak on a shared edge counts once
        for band_name, band_range in freq_bands.items():
            band_peaks = peak_params[
                (peak_params[:, 0] >= band_range[0]) &
                (peak_params[:, 0] < band_range[1])
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
        # Half-open bands [low, high) so a frequency bin on a shared edge counts once
        for band_name, band_range in freq_bands.items():
            band_indices = np.where(
                (fooof_freqs >= band_range[0]) & (fooof_freqs < band_range[1])
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


def extract_aperiodic(fm, aperiodic_mode, freq_range, label, r2_threshold=0.95):
    """
    Pull aperiodic parameters from a fitted FOOOF model.
    Returns (params_dict, poor_fit). params_dict has Offset, Exponent, Knee, Knee_Freq
    (Knee / Knee_Freq are NaN in 'fixed' mode).
      fixed: aperiodic_params_ = [offset, exponent]
      knee:  aperiodic_params_ = [offset, knee, exponent]; knee freq = knee ** (1 / exponent)
    """
    params = {'Offset': np.nan, 'Exponent': np.nan, 'Knee': np.nan, 'Knee_Freq': np.nan}

    r_squared = fm.r_squared_
    if r_squared < r2_threshold:
        print(f' Poor FOOOF fit for {label}: R2 = {r_squared:.3f}, setting FOOOF-based measures to NaN')
        return params, True

    ap = fm.aperiodic_params_
    params['Offset'] = ap[0]
    params['Exponent'] = ap[-1]

    if aperiodic_mode == 'knee':
        knee = ap[1]
        params['Knee'] = knee
        if knee <= 0 or params['Exponent'] <= 0:
            print(f' Warning: knee fit for {label} has knee = {knee:.3f}, exponent = {params["Exponent"]:.3f}; '
                  f'knee frequency undefined (spectrum may have no bend), Knee_Freq set to NaN')
        else:
            knee_freq = knee ** (1 / params['Exponent'])
            params['Knee_Freq'] = knee_freq
            if not (freq_range[0] <= knee_freq <= freq_range[1]):
                print(f' Warning: knee frequency for {label} ({knee_freq:.2f} Hz) is outside the fit range '
                      f'({freq_range[0]}-{freq_range[1]} Hz); knee estimate is unreliable')

    return params, False


def process_psd(processed_set_path,
                spatial_levels=('channel',),
                band_power_methods=('fooof_peaks',),
                normalize_psd_integration=True, normalize_fooof_peaks=True, region_spec=None,
                freq_bands=None, fooof_freq_range=(1, 48), aperiodic_mode='fixed',
                peak_width_limits=(2, 6), max_n_peaks=6, min_peak_height=0.1, peak_threshold=2.0,
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
    - normalize_psd_integration (bool): If True, psd_integration band powers are divided by
      the sum of all band powers.
    - normalize_fooof_peaks (bool): If True, fooof_peaks band powers are divided by the
      summed height of all fitted peaks. (AAP is never normalised.)
    - fooof_freq_range (tuple): (low, high) Hz range for the FOOOF fit. The PSD is computed
      over this range extended to cover any bands outside it.
    - aperiodic_mode (str): 'fixed' (straight line in log-log) or 'knee' (allows a bend).
      Knee mode adds Knee and Knee_Freq (Hz) to the output; these are NaN in fixed mode.
    - peak_width_limits, max_n_peaks, min_peak_height, peak_threshold: FOOOF peak settings.
      Defaults (2-6 Hz, 6, 0.1, 2.0) follow Donoghue et al. (2020) except the lower width limit, which is
      kept at the multitaper bandwidth (2 Hz) so single peaks aren't split (see testing/06).
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
    if freq_bands is None:
        freq_bands = {
            'Delta': [1, 4],
            'Theta': [4, 8],
            'Alpha': [8, 13],
            'Beta': [13, 30],
            'Gamma': [30, 48]
        }

    # Regions come from GUI (region_spec) rather than being hardcoded
    regions = region_spec if region_spec else {}

    channel_names = epochs.ch_names

    if aperiodic_mode not in ('fixed', 'knee'):
        print(f"Unknown aperiodic_mode: {aperiodic_mode}, skipping PSD")
        return None

    # FOOOF peak settings (shared by every fit)
    peak_width_limits = list(peak_width_limits)
    if not (0 < peak_width_limits[0] < peak_width_limits[1]):
        print(f'Invalid peak_width_limits {peak_width_limits}, skipping PSD')
        return None
    if peak_width_limits[0] < MT_BANDWIDTH:
        print(f' Warning: lower peak width limit ({peak_width_limits[0]} Hz) is below the multitaper bandwidth '
              f'({MT_BANDWIDTH} Hz); single peaks may be split into several narrow ones')
    fooof_settings = dict(peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks,
                          min_peak_height=min_peak_height, peak_threshold=peak_threshold,
                          aperiodic_mode=aperiodic_mode, verbose=False)
    # Total-power normalisation is chosen per method (AAP is never normalised)
    normalize_by_method = {'psd_integration': normalize_psd_integration,
                           'fooof_peaks': normalize_fooof_peaks}

    # FOOOF fit range (user-defined)
    nyquist = sfreq / 2
    freq_range = list(fooof_freq_range)
    if not (0 < freq_range[0] < freq_range[1]):
        print(f'Invalid FOOOF fit range {freq_range}, skipping PSD')
        return None
    if freq_range[1] > nyquist:
        print(f' Warning: FOOOF fit range upper limit ({freq_range[1]} Hz) exceeds Nyquist '
              f'({nyquist} Hz); fitting up to {nyquist} Hz')
        freq_range[1] = nyquist

    # PSD range covers the fit range plus any bands outside it
    # (PSD integration can use them; FOOOF peaks / AAP only exist within freq_range)
    psd_fmin = min([freq_range[0]] + [lo for lo, hi in freq_bands.values()])
    psd_fmax = min(max([freq_range[1]] + [hi for lo, hi in freq_bands.values()]), nyquist)
    for band_name, (lo, hi) in freq_bands.items():
        if hi > nyquist:
            print(f' Warning: {band_name} band ({lo}-{hi} Hz) exceeds Nyquist ({nyquist} Hz); '
                  f'only {lo}-{nyquist} Hz can be computed')
        if lo < freq_range[0] or hi > freq_range[1]:
            fooof_methods = [m for m in band_power_methods if m in ('fooof_peaks', 'aap')]
            if fooof_methods:
                print(f' Warning: {band_name} band ({lo}-{hi} Hz) extends outside the FOOOF fit range '
                      f'({freq_range[0]}-{freq_range[1]} Hz); {", ".join(fooof_methods)} only cover '
                      f'the part inside the fit range (NaN if none)')

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
                            fmin=psd_fmin,
                            fmax=psd_fmax,
                            normalization='full',
                            bandwidth=MT_BANDWIDTH,
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
                fm = FOOOF(**fooof_settings)
                fm.fit(freqs, avg_psd, freq_range)

                # Check fit quality (reject if R2 < 0.95) and extract aperiodic parameters
                aperiodic, poor_fit = extract_aperiodic(
                    fm, aperiodic_mode, freq_range, f'global - {condition}')

                # Compute each selected metric from this single fit
                for method in band_power_methods:
                    # PSD integration doesn't use the FOOOF model, so a poor fit only blanks FOOOF-based metrics
                    if poor_fit and method != 'psd_integration':
                        band_powers = {b: np.nan for b in freq_bands}
                        suffix = 'AAP' if method == 'aap' else 'FOOOF'
                    else:
                        band_powers, suffix = compute_band_powers(
                            fm, avg_psd, freqs, method, normalize_by_method.get(method, False), freq_bands)
                        if band_powers is None:
                            return None

                    result = {
                        'condition': condition,
                        'level':     'global',
                        'unit':      'Global',
                        'n_epochs_used': n_epochs,
                        **aperiodic,
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
                                fmin=psd_fmin,
                                fmax=psd_fmax,
                                normalization='full',
                                bandwidth=MT_BANDWIDTH,
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
                fm = FOOOF(**fooof_settings)
                fm.fit(freqs, avg_psd, freq_range)

                # Check fit quality (reject if R2 < 0.95) and extract aperiodic parameters
                aperiodic, poor_fit = extract_aperiodic(
                    fm, aperiodic_mode, freq_range, f'{region_name} - {condition}')

                for method in band_power_methods:
                    # PSD integration doesn't use the FOOOF model, so a poor fit only blanks FOOOF-based metrics
                    if poor_fit and method != 'psd_integration':
                        band_powers = {b: np.nan for b in freq_bands}
                        suffix = 'AAP' if method == 'aap' else 'FOOOF'
                    else:
                        band_powers, suffix = compute_band_powers(
                            fm, avg_psd, freqs, method, normalize_by_method.get(method, False), freq_bands)
                        if band_powers is None:
                            return None

                    result = {
                        'condition': condition,
                        'level':     'region',
                        'unit':      region_name,
                        'n_epochs_used': n_epochs,
                        **aperiodic,
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
                            fmin=psd_fmin,
                            fmax=psd_fmax,
                            normalization='full',
                            bandwidth=MT_BANDWIDTH,
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
                fm = FOOOF(**fooof_settings)
                fm.fit(freqs, avg_psd, freq_range)

                # Check fit quality (reject if R2 < 0.95) and extract aperiodic parameters
                aperiodic, poor_fit = extract_aperiodic(
                    fm, aperiodic_mode, freq_range, f'{channel_name} - {condition}')

                for method in band_power_methods:
                    # PSD integration doesn't use the FOOOF model, so a poor fit only blanks FOOOF-based metrics
                    if poor_fit and method != 'psd_integration':
                        band_powers = {b: np.nan for b in freq_bands}
                        suffix = 'AAP' if method == 'aap' else 'FOOOF'
                    else:
                        band_powers, suffix = compute_band_powers(
                            fm, avg_psd, freqs, method, normalize_by_method.get(method, False), freq_bands)
                        if band_powers is None:
                            return None

                    result = {
                        'condition': condition,
                        'level':     'channel',
                        'unit':      channel_name,
                        'n_epochs_used': n_epochs_used,
                        **aperiodic,
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
