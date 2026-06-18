"""
shiny_epoch_utils.py
--------------
Shared helper for splitting MNE epochs into named conditions.

Used across all four metrics so condition-handling.

A "condition spec" is an ordered dict mapping a condition name to a list
of trigger-code strings:

    {'social': ['910'], 'nonsocial': ['911']}
    {'congruent': ['10', '11'], 'incongruent': ['20', '21']}   # pooled codes

Resting data (or other data without conditions) is represented by an
empty spec ({} or None). In that case, every epoch is returned under a
single condition whose name is defined by `resting_label` (default 'all').
"""

import mne

def split_epochs_by_condition(epochs, condition_spec, resting_label="all"):
    """
    Split an MNE Epochs object into a list of (condition_name, data) pairs.

    Parameters
    ----------
    epochs : mne.Epochs
        The loaded epochs object.
    condition_spec : dict | None
        Mapping {condition_name: [trigger_code_str, ...]} (see above). If empty/None,
        all epochs returned as a single condition named `resting_label`.
    resting_label : str
        Name to use for the single condition when condition_spec is empty.

    Returns
    -------
    list of (str, np.ndarray)
        One tuple per condition that had any epochs. The array
        has shape (n_epochs, n_channels, n_times). Skips conditions with
        no epochs and prints alert message for those.

        Returns empty list if no epochs were found at all.
    """
    # ── Resting / no-condition case ─────────────────────────────────────────
    if not condition_spec:
        data = epochs.get_data()
        if data.shape[0] == 0:
            print("No epochs found in file, skipping participant")
            return []
        print(f"{resting_label} epochs: {data.shape[0]}")
        return [(resting_label, data)]

    # ── Task case: one entry per named condition ────────────────────────────
    out = []
    for cond_name, codes in condition_spec.items():
        # MNE accepts a list of strings to pool multiple codes
        code_list = [str(c).strip() for c in codes if str(c).strip()]
        if not code_list:
            print(f"Condition '{cond_name}' has no trigger codes, skipping")
            continue
        try:
            selected = epochs[code_list]
            data = selected.get_data()
            if data.shape[0] == 0:
                print(f"No epochs for condition '{cond_name}' "
                      f"(codes {code_list}), skipping")
                continue
            print(f"{cond_name} epochs: {data.shape[0]}")
            out.append((cond_name, data))
        except KeyError:
            print(f"No epochs found for condition '{cond_name}' "
                  f"(codes {code_list}), skipping")
            continue

    return out


def count_epochs_by_condition(epochs, condition_spec, resting_label="all"):
    """
    Return {condition_name: n_epochs} for the tracker. Conditions with no
    matching epochs report 0 rather than being omitted, so the tracker
    columns are stable across participants.
    """
    counts = {}
    if not condition_spec:
        counts[resting_label] = int(epochs.get_data().shape[0])
        return counts

    for cond_name, codes in condition_spec.items():
        code_list = [str(c).strip() for c in codes if str(c).strip()]
        try:
            counts[cond_name] = len(epochs[code_list])
        except (KeyError, ValueError):
            counts[cond_name] = 0
    return counts
