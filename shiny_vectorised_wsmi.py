import numpy as np
import pandas as pd
import mne
from mne.utils import logger, _time_mask
from scipy.signal import butter, filtfilt
import os
import warnings
import math
from itertools import permutations


def _get_weights_matrix(nsym):
    """
    Auxiliary function to create the weights matrix.
    Produces matrix with both diagonals zeroed out.
    Main diagonal = identical pattern on both channels (volume conduction with same polarity)
    Anti-diagonal = reversed pattern on both channels (volume conduction with flipped polarity)
    """
    wts = np.ones((nsym, nsym))
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    return wts


def _define_symbols(kernel):
    result_dict = dict()
    total_symbols = math.factorial(kernel)
    cursymbol = 0
    for perm in permutations(range(kernel)):
        order = ''.join(map(str, perm))
        if order not in result_dict:
            result_dict[order] = cursymbol
            cursymbol = cursymbol + 1
            result_dict[order[::-1]] = total_symbols - cursymbol
    result = []
    for v in range(total_symbols):
        for symbol, value in result_dict.items():
            if value == v:
                result += [symbol]
    return result


def _symb_python(data, kernel, tau):
    """
    Compute symbolic transform (vectorised).

    Replaces the original window-by-window loop with a single array
    indexing operation, then uses a pre-built lookup table instead of
    repeated symbols.index() calls. The count step uses np.bincount
    directly rather than apply_along_axis.

    """
    symbols = _define_symbols(kernel)
    nsym = len(symbols)
    nchannels, nsamples, ntrials = data.shape
    n_windows = nsamples - tau * (kernel - 1)

    # Pre-build lookup: ordinal pattern tuple -> symbol index
    # Avoids repeated string construction and list.index() calls
    lookup = {tuple(int(c) for c in s): i for i, s in enumerate(symbols)}

    # Build all window sample indices at once: shape (n_windows, kernel)
    window_idx = np.array([[k + j * tau for j in range(kernel)]
                           for k in range(n_windows)])

    # Extract all windows simultaneously
    # data[:, window_idx, :] shape: (nchannels, n_windows, kernel, ntrials)
    windowed = data[:, window_idx, :]

    # Argsort along kernel axis to get ordinal patterns
    # shape: (nchannels, n_windows, kernel, ntrials)
    ranked = np.argsort(windowed, axis=2)

    # Vectorise the symbol lookup by reshaping to (-1, kernel),
    # mapping each row through the lookup table, then reshaping back.
    # Transpose to (nchannels, ntrials, n_windows, kernel) first so that
    # after flattening the first three dims, each row is one (ch, tr, win).
    ranked_T = ranked.transpose(0, 3, 1, 2)          # (nch, ntr, n_win, kernel)
    flat = ranked_T.reshape(-1, kernel)
    sym_ids = np.array([lookup[tuple(row)] for row in flat], dtype=np.int32)

    # Reshape back and transpose to (nchannels, n_windows, ntrials)
    signal_sym = sym_ids.reshape(nchannels, ntrials, n_windows).transpose(0, 2, 1)

    # Count symbol occurrences per channel/trial using np.bincount
    count = np.zeros((nchannels, nsym, ntrials), dtype=np.float64)
    for tr in range(ntrials):
        for ch in range(nchannels):
            count[ch, :, tr] = np.bincount(signal_sym[ch, :, tr], minlength=nsym)
    count /= n_windows

    return signal_sym, count


def _wsmi_python(data, count, wts):
    """
    Compute wSMI (vectorised inner loops).

    Replaces the original four nested Python loops with:
      - np.bincount on a flattened index to build the joint probability
        matrix pxy in one pass (eliminates the sample loop)
      - np.outer for the marginal probability product (eliminates sc1/sc2 loops)
      - Boolean masking + vectorised log/multiply to compute MI contributions

    Outputs are numerically identical to the original (differences < 1e-15
    are floating-point rounding only).
    """
    nchannels, nsamples, ntrials = data.shape
    nsymbols = count.shape[1]
    smi  = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    wsmi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)

    for trial in range(ntrials):
        for channel1 in range(nchannels):
            for channel2 in range(channel1 + 1, nchannels):

                # ── Joint probability matrix ───────────────────────────────
                # Encode 2D symbol pair (s1, s2) as a single integer
                # s1 * nsymbols + s2, then count occurrences in one C-level pass.
                flat_idx = (data[channel1, :, trial].astype(np.int32) * nsymbols +
                            data[channel2, :, trial].astype(np.int32))
                pxy = (np.bincount(flat_idx, minlength=nsymbols ** 2)
                       .reshape(nsymbols, nsymbols)
                       .astype(np.float64))
                pxy /= nsamples

                # ── Marginal probability outer product ─────────────────────
                # marginals[i, j] = P(s1=i | ch1) * P(s2=j | ch2)
                marginals = np.outer(count[channel1, :, trial],
                                     count[channel2, :, trial])

                # ── MI contributions ───────────────────────────────────────
                # Only compute where pxy > 0 to avoid log(0)
                mask = pxy > 0
                aux = np.zeros((nsymbols, nsymbols))
                aux[mask] = pxy[mask] * np.log(pxy[mask] / marginals[mask])

                smi[channel1, channel2, trial]  = aux.sum()
                wsmi[channel1, channel2, trial] = (wts * aux).sum()

    wsmi /= np.log(nsymbols)
    smi  /= np.log(nsymbols)
    return wsmi, smi


def epochs_compute_wsmi(epochs, kernel, tau, tmin=None, tmax=None,
                        backend='python', method_params=None, n_jobs='auto'):
    if method_params is None:
        method_params = {}

    if n_jobs == 'auto':
        n_jobs = -1

    if 'bypass_csd' in method_params and method_params['bypass_csd']:
        logger.info('Bypassing CSD')
        csd_epochs = epochs
        picks = mne.pick_types(csd_epochs.info, meg=True, eeg=True)
    else:
        logger.info('Computing CSD')
        csd_epochs = mne.preprocessing.compute_current_source_density(
            epochs, lambda2=1e-5)
        picks = mne.pick_types(csd_epochs.info, csd=True)

    freq = csd_epochs.info['sfreq']

    data = csd_epochs.get_data()[:, picks, ...]
    n_epochs = len(data)

    if 'filter_freq' in method_params:
        filter_freq = method_params['filter_freq']
    else:
        filter_freq = np.double(freq) / kernel / tau
    logger.info('Filtering at %.2f Hz' % filter_freq)
    b, a = butter(6, 2.0 * filter_freq / np.double(freq), 'lowpass')
    data = np.hstack(data)

    fdata = np.transpose(np.array(
        np.split(filtfilt(b, a, data), n_epochs, axis=1)), [1, 2, 0])

    time_mask = _time_mask(epochs.times, tmin, tmax)
    fdata = fdata[:, time_mask, :]
    logger.info("Performing symbolic transformation")
    sym, count = _symb_python(fdata, kernel, tau)
    nsym = count.shape[1]
    wts = _get_weights_matrix(nsym)
    logger.info("Running wsmi with python...")
    wsmi, smi = _wsmi_python(sym, count, wts)

    return wsmi, smi


def _iter_condition_epochs(epochs, condition_spec, resting_label="all"):
    """
    Yields (condition_name, epochs_subset) for each condition with epochs.
    Yields Epochs objects rather than raw arrays because wSMI needs the
    Epochs object for CSD / filtering internally.
    """
    if not condition_spec:
        if len(epochs) == 0:
            print("No epochs found in file, skipping participant")
            return
        print(f"{resting_label} epochs: {len(epochs)}")
        yield (resting_label, epochs)
        return

    for cond_name, codes in condition_spec.items():
        code_list = [str(c).strip() for c in codes if str(c).strip()]
        if not code_list:
            print(f"Condition '{cond_name}' has no trigger codes, skipping")
            continue
        try:
            subset = epochs[code_list]
            if len(subset) == 0:
                print(f"No epochs for condition '{cond_name}' "
                      f"(codes {code_list}), skipping")
                continue
            print(f"{cond_name} epochs: {len(subset)}")
            yield (cond_name, subset)
        except KeyError:
            print(f"No epochs found for condition '{cond_name}' "
                  f"(codes {code_list}), skipping")
            continue


def process_wsmi(processed_set_path, kernel, tau,
                 tmin=None, tmax=None, backend='python', method_params=None,
                 n_jobs='auto', condition_spec=None):
    """
    Processes an EEG '.set' file to compute wSMI metrics, split by
    user-defined conditions.

    Parameters:
    - processed_set_path (str): Path to the processed '.set' EEG file.
    - kernel (int): Number of samples to use to transform to a symbol.
    - tau (int): Number of samples left between the ones that define a symbol.
    - tmin, tmax (float, optional): Time masking bounds.
    - backend (str, optional): 'python' or 'openmp'.
    - method_params (dict, optional): Additional method parameters.
    - n_jobs (int or 'auto', optional): Parallel jobs.
    - condition_spec (dict | None): {condition_name: [trigger_code_str, ...]}.
      If None/empty, all epochs are processed as a single condition ('all').

    Returns:
    - dict: Contains DataFrame for wSMI and SMI. Returns None if no usable epochs.
    """
    print(f'Processing wSMI for {os.path.basename(processed_set_path)}')

    try:
        epochs = mne.read_epochs_eeglab(processed_set_path)
    except Exception as e:
        print(f"Error loading processed epochs: {e}")
        return None

    channel_names = epochs.ch_names
    results = []

    found_any = False
    for condition, epochs_cond in _iter_condition_epochs(epochs, condition_spec):
        found_any = True

        wsmi, smi = epochs_compute_wsmi(
            epochs_cond, kernel, tau, tmin, tmax, backend, method_params, n_jobs
        )

        n_channels = len(channel_names)

        for trial in range(wsmi.shape[2]):
            for ch1 in range(n_channels):
                for ch2 in range(ch1 + 1, n_channels):
                    results.append({
                        'condition': condition,
                        'epoch': trial,
                        'channel_1': channel_names[ch1],
                        'channel_2': channel_names[ch2],
                        'wsmi': wsmi[ch1, ch2, trial],
                        'smi': smi[ch1, ch2, trial]
                    })

    if not found_any or not results:
        print('No results computed, skipping participant')
        return None

    df_wsmi = pd.DataFrame(results)
    df_wsmi = df_wsmi.sort_values(['condition', 'epoch']).reset_index(drop=True)

    print(f'Computed wSMI: {df_wsmi.shape[0]} rows')

    return {'wsmi': df_wsmi}