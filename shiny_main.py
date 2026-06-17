"""
EEG Complexity Pipeline — Shiny GUI
-----------------------------------
GUI for the LZC / Permutation Entropy / wSMI / PSD complexity processing pipeline.

Supports:
  - Resting data (epoched, no conditions)
  - Task data with any number of user-defined conditions, each matching
    one or more trigger codes.

Simple way to run:
 - First, double-click the "install.bat" file and follow instructions
 - Second, double-click the "Start App.bat" file and follow instructions

To run directly:
    In the terminal, run by typing:
        shiny run --reload shiny_main.py
    or:
        python -m shiny run --reload shiny_main.py

    Then open the URL displayed in the terminal (http://127.0.0.1:8000) and wait for the app to open in your browser (e.g. Google Chrome).
"""

import os
import glob
import threading
import traceback
from datetime import datetime

import pandas as pd
from shiny import App, reactive, render, ui

# ─────────────────────────────────────────────────────────────────────────────
# Import complexity processing functions.
# GUI still loads if missing dependency, but displays warning
# ─────────────────────────────────────────────────────────────────────────────
IMPORT_ERROR = None
try:
    import mne
    from shiny_LZC import process_LZ78
    from vectorised_PE import process_permutation_entropy
    from vectorised_wsmi import process_wsmi
    from shiny_PSD_multitaper import process_psd
    from shiny_epoch_utils import count_epochs_by_condition
except Exception as e:
    IMPORT_ERROR = f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT = ""
DEFAULT_OUTPUT = ""
DEFAULT_TRACKER = ""

DEFAULT_CHANNELS = (
    "P7,P4,Cz,Pz,P3,P8,Oz,T8,PO8,C4,F4,AF8,Fz,C3,F3,AF7,T7,PO7,Fpz"
)

DEFAULT_PE_TAUS = "1,3,6,10,21,41"
DEFAULT_WSMI_TAUS = "1,3,6,10,21,41"

DEFAULT_CONDITIONS = "social: 910\nnonsocial: 911"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def parse_int_list(text):
    """'1, 3, 6' -> [1, 3, 6]. Ignores empties/whitespace."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out

def parse_str_list(text):
    """'P7, P4, Cz' -> ['P7','P4','Cz']."""
    return [p.strip() for p in str(text).split(",") if p.strip()]

def parse_condition_spec(text):
    """
    Parse the conditions text area into an ordered dict:
        'social: 910\\nnonsocial: 911'  ->  {'social': ['910'], 'nonsocial': ['911']}
        'congruent: 10, 11'             ->  {'congruent': ['10', '11']}

    Lines without a colon, or blank lines, are ignored. Raises ValueError
    with message if line misformed
    """
    spec = {}
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(
                f"Condition line missing ':' -> {raw_line!r}. "
                f"Use 'name: code1, code2'.")
        name, codes_str = line.split(":", 1)
        name = name.strip()
        codes = [c.strip() for c in codes_str.split(",") if c.strip()]
        if not name:
            raise ValueError(f"Condition line has no name -> {raw_line!r}")
        if not codes:
            raise ValueError(
                f"Condition '{name}' has no trigger codes -> {raw_line!r}")
        if name in spec:
            raise ValueError(f"Duplicate condition name '{name}'")
        spec[name] = codes
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Setup"),
        ui.input_action_button(
            "show_guide", "User Guide",
            class_="btn-outline-secondary w-100 mb-3"
        ),
        ui.input_action_button(
            "show_measures_expln", "Measures Explanation",
            class_="btn-outline-secondary w-100 mb-3"
        ),

        ui.accordion(
            ui.accordion_panel(
                "Paths",
                ui.input_text("input_path", "Input Folder (.set files)", DEFAULT_INPUT),
                ui.input_text("output_path", "Output Path (file, not folder)", DEFAULT_OUTPUT),
                ui.input_text("tracker_path", "Tracker Path (file, not folder)", DEFAULT_TRACKER),
                ui.input_text(
                    "processed_prefix",
                    "Filename prefix to remove for participant ID",
                    ""
                ),
                ui.input_text(
                    "processed_suffix",
                    "Filename suffix to remove for participant ID",
                    "_processed_data.set"
                ),
                ui.input_text(
                    "skip_pattern",
                    "Skip files containing this text (leave blank for no skipping)",
                    "no_usable"
                ),
            ),
            ui.accordion_panel(
                "Pipelines to Run",
                ui.input_checkbox("run_lz", "LZ78", True),
                ui.input_checkbox("run_pe", "Permutation Entropy", True),
                ui.input_checkbox("run_wsmi", "wSMI", True),
                ui.input_checkbox("run_psd", "PSD", False),
            ),
            ui.accordion_panel(
                "Data Type & Conditions",
                ui.input_radio_buttons(
                    "data_type", "Data type",
                    {"task": "Task (Split by Conditions)",
                     "resting": "Resting (No Conditions)"},
                    selected="task",
                ),
                ui.panel_conditional(
                    "input.data_type === 'task'",
                    ui.input_text_area(
                        "conditions",
                        "Conditions (1 per line)",
                        DEFAULT_CONDITIONS, rows=4,
                    ),
                    ui.help_text(
                        "Example — social: 910 ⏎ nonsocial: 911. "
                        "Add multiple trigger codes with commas (congruent: 10, 11)."
                    ),
                ),
                ui.panel_conditional(
                    "input.data_type === 'resting'",
                    ui.input_text(
                        "resting_label", "Condition label for all epochs", "all"),
                ),
            ),
            ui.accordion_panel(
                "Channels",
                ui.input_text_area(
                    "channels", "Channel Labels (comma-separated)",
                    DEFAULT_CHANNELS, rows=3,
                ),
            ),
            ui.accordion_panel(
                "Permutation Entropy",
                ui.input_numeric("pe_kernel", "Kernel", 3, min=1, step=1),
                ui.input_text("pe_taus", "Taus (Comma-separated)", DEFAULT_PE_TAUS),
            ),
            ui.accordion_panel(
                "wSMI",
                ui.input_numeric("wsmi_kernel", "Kernel", 3, min=1, step=1),
                ui.input_text("wsmi_taus", "Taus (Comma-separated)", DEFAULT_WSMI_TAUS),
                ui.input_checkbox("wsmi_bypass_csd", "Bypass CSD?", True),
            ),
            ui.accordion_panel(
                "PSD",
                ui.input_select(
                    "psd_band_power_method", "Band power method",
                    {"fooof_peaks": "FOOOF Peaks", "psd_integration": "PSD Integration"},
                    selected="fooof_peaks",
                ),
                ui.input_checkbox("psd_normalize_total_power", "Normalize total power?", True),
            ),
            open=["Paths", "Pipelines to run", "Data type & conditions"],
            multiple=True,
        ),

        ui.hr(),
        ui.input_action_button("scan", "Scan for files", class_="btn-secondary w-100 mb-2"),
        ui.input_action_button("run", "▶ Run pipeline", class_="btn-success w-100 mb-2"),
        ui.input_action_button("stop", "■ Stop", class_="btn-danger w-100"),
        width=380,
    ),

    ui.h2("EEG Complexity Pipeline"),
    ui.output_ui("import_warning"),

    ui.layout_columns(
        ui.value_box("Files Found", ui.output_text("n_files")),
        ui.value_box("Processed", ui.output_text("n_done")),
        ui.value_box("Status", ui.output_text("status_text")),
        col_widths=[4, 4, 4],
    ),

    ui.card(
        ui.card_header("Progress Log"),
        ui.output_text_verbatim("log_output", placeholder=True),
        height="450px",
        full_screen=True,
    ),

    ui.card(
        ui.card_header("Files to Process"),
        ui.output_data_frame("file_table"),
        height="300px",
        full_screen=True,
    ),

    title="EEG Complexity Pipeline",
    fillable=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────
def server(input, output, session):

    # Reactive state
    log_lines = reactive.value([])           # list[str]
    files = reactive.value([])               # list[str] of paths
    status = reactive.value("Idle")          # "Idle" | "Running" | "Done" | "Stopped" | "Error"
    n_processed = reactive.value(0)

    # Thread coordination (plain objects, not reactive)
    stop_flag = {"stop": False}
    worker = {"thread": None}

    # A lock-free queue-ish buffer the worker appends to; the polling
    # reactive drains it into log_lines on the main thread.
    msg_buffer = []
    buffer_lock = threading.Lock()

    def emit(msg):
        """Thread-safe: push a timestamped line into the buffer."""
        stamp = datetime.now().strftime("%H:%M:%S")
        with buffer_lock:
            msg_buffer.append(f"[{stamp}] {msg}")

    # Poll the buffer ~3x/sec and flush into the reactive log.
    @reactive.effect
    def _drain_buffer():
        reactive.invalidate_later(0.35)
        with buffer_lock:
            if not msg_buffer:
                return
            new = msg_buffer[:]
            msg_buffer.clear()
        current = log_lines.get()
        log_lines.set(current + new)

    @reactive.effect
    @reactive.event(input.show_guide)
    def _show_guide():
        m = ui.modal(
            ui.markdown("""
            ## EEG Complexity Pipeline — User Guide

            Hi! Thanks for using the EEG Complexity Pipeline App. Here is a quick guide on how to use it.

            ---

            ### Overview

            The idea of the app is to provide an easy interface for calculating some (by no means all!) of the common complexity measures from EEG data.
            It can extract:

            - **Lempel-Ziv Complexity (LZC)**
            - **Permutation Entropy (PE)**
            - **Weighted Symbolic Mutual Information (wSMI)**
            - **Power Spectral Density (PSD)** — not strictly a complexity measure, but included for convenience.
              For PSD, we calculate the aperiodic components (offset & exponent) and periodic power components.

            You can find more about these specific measures online.
            
            If you want to look under the hood at how any measures are calculated, you can find the scripts in the same folder as this app.

            **Layout**
            - **Sidebar (left):** configuration options for setting up and running the pipeline.
            - **Top of page:** general progress monitors (files found, files processed, status).
            - **Bottom of page:** detailed progress log.
            
            **Input Files**
            Prior to running the pipelines in this app, you will need to have preprocessed your data.
            There are various options for preprocessing depending on your specific requirements.
            The only thing required for this pipeline is that the input files are .set files.
            These should also automatically come with a .fdt file.
            
            Where possible, put your files in participant subfolders in the data folder, i.e. datafolder\participant1, participant2, etc.
            
            There are 2 options in the Setup section regarding input files.
            - Filename suffix - if all your preprocessed files have a suffix, e.g. "file1_processed_data.set" this will remove "_processed_data" so the final spreadsheets have neat participant IDs
            - Skip pattern - if some file names contain clues to skip them, e.g. "file1_no_usable_data.set", it will skip these files 

            **Output files**

            When the pipeline runs, it produces:
            - One spreadsheet per measure containing results across all participants.
            - A **tracker** spreadsheet (.csv) summarising key information (epoch counts, failed measures, etc.) across all participants and measures.

            ---

            ### Configuration

            #### Paths

            - **Input Folder** — paste the folder path containing your preprocessed `.set` files.
              You can use *Copy as path* in File Explorer and paste directly.
            - **Output Path** — paste your desired output filepath. This is used as a *prefix*:
              `_LZC.csv`, `_PE.csv` etc. will be appended automatically.
              E.g. `C:/data/results` → `C:/data/results_LZC.csv`
            - **Tracker CSV Path** — paste the full filepath for the tracker file, including `.csv`.
              E.g. `C:/data/complexity_tracker.csv`

            ---

            #### Pipelines to Run

            Tick the boxes for the measures you wish to compute.

            ---

            #### Data Type & Conditions

            - **Task data** — data split into conditions defined by trigger codes in the EEG.
              If you have multiple triggers for one condition, list them together on the same line.
              Enter one condition per line in the format:
            
            ```
              condition1: trigger1, trigger2
              condition2: trigger3, trigger4
            ```
              
            - **Resting state** — data with no conditions. All epochs are processed together
            under a single label (default: `all`).

            ---

            #### Channel Labels

            Enter a comma-separated list of your channel labels in order, e.g.:
            'P7, P4, Cz, Pz, etc'
            
            ---

            #### Permutation Entropy

            - **Kernel** — number of consecutive data points that form one symbol.
              E.g. `kernel=3` transforms the signal into symbols of 3 data points.
            - **Taus** — time lag between data points within a symbol (unit: samples).
              Multiple values can be entered as a comma-separated list; each is run separately.
              E.g. `kernel=3, tau=2` gives symbols of 3 data points each 2 samples apart.
            
            ---

            #### wSMI
            
            Kernel and tau parameters have the same meaning as for Permutation Entropy above.
            
            - **Bypass CSD** — CSD = Current Source Density, a spatial filter that can sharpen signal at each electrode, but may not be appropriate for low channel data. 
            
            ---
            
            #### Power Spectral Density
            
            - **Band Power Method**
              - *FOOOF Peaks* — fits peaks to the aperiodic-corrected spectrum, isolating
                oscillatory activity from the 1/f background.
              - *PSD Integration* — integrates the raw power spectrum within each frequency
                band directly, without separating oscillatory from aperiodic activity.
            - **Normalise Total Power** — divides each band power value by total power across
              the spectrum, giving a value between 0 and 1.
            """),
            title="EEG Complexity Pipeline App - User Guide",
            easy_close=True,
            footer=ui.modal_button("Close"),
            size="l",
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.show_measures_expln)
    def _show_measures_expln():
        m = ui.modal(
            ui.img(src="measures.png", style="max-width:100%;"),
            title="Guide to Measures",
            easy_close=True,
            footer=ui.modal_button("Close"),
            size="l",
        )
        ui.modal_show(m)

    # ── Scan for files ────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.scan)
    def _scan():
        path = input.input_path().strip().strip('"').strip("'")
        if not os.path.isdir(path):
            emit(f"⚠ Input folder does not exist: {path}")
            files.set([])
            return
        found = glob.glob(os.path.join(path, "**", "*.set"), recursive=True)
        files.set(sorted(found))
        emit(f"Found {len(found)} .set files in {path}")

    # ── Stop ────────────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.stop)
    def _stop():
        if worker["thread"] and worker["thread"].is_alive():
            stop_flag["stop"] = True
            emit("Stop requested — will halt after current participant.")
        else:
            emit("Nothing is running.")

    # ── Run ──────────────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.run)
    def _run():
        if IMPORT_ERROR:
            emit(f"✗ Cannot run — processing modules failed to import: {IMPORT_ERROR}")
            return
        if worker["thread"] and worker["thread"].is_alive():
            emit("Already running — ignoring.")
            return

        # Build the condition spec from the GUI.
        #   Task mode    -> parse the conditions text area
        #   Resting mode -> empty spec (functions treat as single condition)
        try:
            if input.data_type() == "resting":
                condition_spec = {}
                resting_label = input.resting_label().strip() or "all"
            else:
                condition_spec = parse_condition_spec(input.conditions())
                resting_label = "all"
                if not condition_spec:
                    emit("✗ Task mode selected but no conditions defined.")
                    return
        except ValueError as e:
            emit(f"✗ Bad condition definition: {e}")
            return

        # Snapshot all parameters NOW (can't read inputs from the worker thread)
        try:
            cfg = {
                "input_path": input.input_path().strip().strip('"').strip("'"),
                "output_path": input.output_path().strip().strip('"').strip("'"),
                "tracker_path": input.tracker_path().strip().strip('"').strip("'"),
                "processed_prefix": input.processed_prefix().strip(),
                "processed_suffix": input.processed_suffix().strip(),
                "skip_pattern": input.skip_pattern().strip(),
                "channels": parse_str_list(input.channels()),
                "condition_spec": condition_spec,
                "resting_label": resting_label,
                "run_lz": input.run_lz(),
                "run_pe": input.run_pe(),
                "run_wsmi": input.run_wsmi(),
                "run_psd": input.run_psd(),
                "pe_kernel": int(input.pe_kernel()),
                "pe_taus": parse_int_list(input.pe_taus()),
                "wsmi_kernel": int(input.wsmi_kernel()),
                "wsmi_taus": parse_int_list(input.wsmi_taus()),
                "wsmi_bypass_csd": input.wsmi_bypass_csd(),
                "psd_band_power_method": input.psd_band_power_method(),
                "psd_normalize_total_power": input.psd_normalize_total_power(),
            }
        except ValueError as e:
            emit(f"✗ Bad parameter value: {e}")
            return

        # Use already-scanned files if present, else scan now.
        file_list = files.get()
        if not file_list:
            if not os.path.isdir(cfg["input_path"]):
                emit(f"✗ Input folder does not exist: {cfg['input_path']}")
                return
            file_list = sorted(
                glob.glob(os.path.join(cfg["input_path"], "**", "*.set"), recursive=True)
            )
            files.set(file_list)

        if not file_list:
            emit("✗ No .set files found. Nothing to do.")
            return

        # Reset state
        stop_flag["stop"] = False
        log_lines.set([])
        n_processed.set(0)
        status.set("Running")

        progress = {"done": 0}

        def report_progress(done):
            progress["done"] = done

        t = threading.Thread(
            target=run_pipeline,
            args=(cfg, file_list, emit, stop_flag, report_progress),
            daemon=True,
        )
        worker["thread"] = t
        t.start()

        if cfg["condition_spec"]:
            cond_str = ", ".join(cfg["condition_spec"].keys())
            emit(f"▶ Started. {len(file_list)} files queued. Conditions: {cond_str}")
        else:
            emit(f"▶ Started. {len(file_list)} files queued. Resting "
                 f"(label '{cfg['resting_label']}').")

        # Mirror the worker's progress counter into the reactive value.
        @reactive.effect
        def _mirror_progress():
            reactive.invalidate_later(0.4)
            n_processed.set(progress["done"])
            if not t.is_alive():
                if stop_flag["stop"]:
                    status.set("Stopped")
                else:
                    status.set(status.get() if status.get() == "Error" else "Done")
                n_processed.set(progress["done"])

    # ── Outputs ───────────────────────────────────────────────────────────────
    @render.text
    def n_files():
        return str(len(files.get()))

    @render.text
    def n_done():
        return str(n_processed.get())

    @render.text
    def status_text():
        return status.get()

    @render.text
    def log_output():
        lines = log_lines.get()
        if not lines:
            return "Log is empty. Configure parameters and press ▶ Run pipeline."
        return "\n".join(lines[-400:])

    @render.data_frame
    def file_table():
        flist = files.get()
        if not flist:
            return render.DataGrid(pd.DataFrame({"file": []}))
        df = pd.DataFrame({
            "file": [os.path.basename(f) for f in flist],
            "folder": [os.path.dirname(f) for f in flist],
        })
        return render.DataGrid(df, height="260px")

    @render.ui
    def import_warning():
        if IMPORT_ERROR:
            return ui.div(
                ui.tags.b("⚠ Processing modules not importable: "),
                ui.tags.code(IMPORT_ERROR),
                ui.br(),
                "The GUI works, but pressing Run will not process anything until "
                "shiny_LZC / shiny_PE / shiny_wSMI / shiny_PSD_multitaper / shiny_epoch_utils "
                "(and mne) are importable from this folder.",
                class_="alert alert-warning",
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EEG Complexity Processing Pipeline
# Runs in a background thread.
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(cfg, set_files, emit, stop_flag, report_progress):
    all_lz, all_pe, all_wsmi, all_psd = [], [], [], []
    tracker_rows = []

    output_path = cfg["output_path"]
    tracker_path = cfg["tracker_path"]
    condition_spec = cfg["condition_spec"]
    resting_label = cfg["resting_label"]

    # Names of the conditions we expect, for stable tracker columns.
    if condition_spec:
        condition_names = list(condition_spec.keys())
    else:
        condition_names = [resting_label]

    try:
        for idx, file_path in enumerate(set_files):

            if stop_flag["stop"]:
                emit("■ Stopped before next participant.")
                break

            file_name = os.path.basename(file_path)

            # ── Skip "no usable" files ──────────────────────────────────────
            if cfg["skip_pattern"] and cfg["skip_pattern"] in file_name:
                emit(f"Skipping {file_name} — no usable data")
                row = {
                    "participant_id": file_name.replace(
                        "_no_usable_data_all_bad_epochs.set", ""),
                    "skipped": True,
                    "skip_reason": "no_usable_data",
                    "lz_status": "skipped",
                    "pe_taus_completed": None,
                    "pe_taus_failed": None,
                    "wsmi_taus_completed": None,
                    "wsmi_taus_failed": None,
                    "psd_status": "skipped",
                    "error_details": None,
                }
                for cname in condition_names:
                    row[f"n_epochs_{cname}"] = None
                tracker_rows.append(row)
                pd.DataFrame(tracker_rows).to_csv(tracker_path, index=False)
                report_progress(idx + 1)
                continue

            ppt_id = file_name
            if cfg["processed_prefix"] and ppt_id.startswith(cfg["processed_prefix"]):
                ppt_id = ppt_id[len(cfg["processed_prefix"]):]
            if cfg["processed_suffix"] and ppt_id.endswith(cfg["processed_suffix"]):
                ppt_id = ppt_id[:-len(cfg["processed_suffix"])]

            emit("=" * 50)
            emit(f"Processing: {ppt_id}")

            tracker = {
                "participant_id": ppt_id,
                "skipped": False,
                "skip_reason": None,
                "lz_status": None,
                "pe_taus_completed": [],
                "pe_taus_failed": [],
                "wsmi_taus_completed": [],
                "wsmi_taus_failed": [],
                "psd_status": None,
                "error_details": None,
            }
            # Per-condition epoch-count columns (stable across participants)
            for cname in condition_names:
                tracker[f"n_epochs_{cname}"] = None

            # ── Epoch counts ────────────────────────────────────────────────
            try:
                epochs_check = mne.read_epochs_eeglab(file_path, verbose=False)
                counts = count_epochs_by_condition(
                    epochs_check, condition_spec, resting_label)
                for cname in condition_names:
                    tracker[f"n_epochs_{cname}"] = counts.get(cname, 0)
                del epochs_check
            except Exception as e:  # noqa: BLE001
                tracker["skip_reason"] = f"failed to load: {e}"
                tracker["skipped"] = True
                tracker_rows.append(tracker)
                emit(f"  ✗ Failed to load epochs: {e}")
                pd.DataFrame(tracker_rows).to_csv(tracker_path, index=False)
                report_progress(idx + 1)
                continue

            counts_str = ", ".join(
                f"{c}={tracker[f'n_epochs_{c}']}" for c in condition_names)
            emit(f"  epochs: {counts_str}")

            # ── LZ ──────────────────────────────────────────────────────────
            if cfg["run_lz"]:
                emit("  -- LZC --")
                try:
                    lz_results = process_LZ78(
                        processed_set_path=file_path,
                        channel_labels=cfg["channels"],
                        condition_spec=condition_spec,
                    )
                    if lz_results is not None:
                        for condition, payload in lz_results.items():
                            df = payload["lz_c"].copy()
                            df["participant_id"] = ppt_id
                            df["condition"] = condition
                            df["metric"] = "LZ"
                            all_lz.append(df)
                        tracker["lz_status"] = "success"
                        emit("     LZ78 ✓")
                    else:
                        tracker["lz_status"] = "failed"
                        emit("     LZ78 returned None")
                except Exception as e:  # noqa: BLE001
                    tracker["lz_status"] = "error"
                    tracker["error_details"] = f"LZ: {traceback.format_exc()}"
                    emit(f"     LZ error: {e}")
            else:
                tracker["lz_status"] = "disabled"

            # ── PE ──────────────────────────────────────────────────────────
            if cfg["run_pe"]:
                emit("  -- Permutation Entropy --")
                for tau in cfg["pe_taus"]:
                    if stop_flag["stop"]:
                        break
                    try:
                        pe_results = process_permutation_entropy(
                            processed_set_path=file_path,
                            channel_labels=cfg["channels"],
                            kernel=cfg["pe_kernel"],
                            tau=tau,
                            condition_spec=condition_spec,
                        )
                        if pe_results is not None:
                            df = pe_results["pe"].copy()
                            df["participant_id"] = ppt_id
                            df["metric"] = "PE"
                            df["kernel"] = cfg["pe_kernel"]
                            df["tau"] = tau
                            all_pe.append(df)
                            tracker["pe_taus_completed"].append(tau)
                            emit(f"     PE tau={tau} ✓")
                        else:
                            tracker["pe_taus_failed"].append(tau)
                            emit(f"     PE tau={tau} returned None")
                    except Exception as e:  # noqa: BLE001
                        tracker["pe_taus_failed"].append(tau)
                        tracker["error_details"] = (
                            f"PE tau={tau}: {traceback.format_exc()}")
                        emit(f"     PE tau={tau} error: {e}")
            else:
                tracker["pe_taus_completed"] = "disabled"

            # ── wSMI ────────────────────────────────────────────────────────
            if cfg["run_wsmi"]:
                emit("  -- wSMI --")
                method_params = {}
                if cfg["wsmi_bypass_csd"]:
                    method_params["bypass_csd"] = True
                for tau in cfg["wsmi_taus"]:
                    if stop_flag["stop"]:
                        break
                    try:
                        wsmi_results = process_wsmi(
                            processed_set_path=file_path,
                            kernel=cfg["wsmi_kernel"],
                            tau=tau,
                            backend="python",
                            method_params=method_params,
                            condition_spec=condition_spec,
                        )
                        if wsmi_results is not None:
                            df = wsmi_results["wsmi"].copy()
                            df["participant_id"] = ppt_id
                            df["metric"] = "wSMI"
                            df["kernel"] = cfg["wsmi_kernel"]
                            df["tau"] = tau
                            all_wsmi.append(df)
                            tracker["wsmi_taus_completed"].append(tau)
                            emit(f"     wSMI tau={tau} ✓")
                        else:
                            tracker["wsmi_taus_failed"].append(tau)
                            emit(f"     wSMI tau={tau} returned None")
                    except Exception as e:  # noqa: BLE001
                        tracker["wsmi_taus_failed"].append(tau)
                        tracker["error_details"] = (
                            f"wSMI tau={tau}: {traceback.format_exc()}")
                        emit(f"     wSMI tau={tau} error: {e}")
            else:
                tracker["wsmi_taus_completed"] = "disabled"

            # ── PSD ─────────────────────────────────────────────────────────
            if cfg["run_psd"]:
                emit("  -- PSD --")
                try:
                    psd_results = process_psd(
                        processed_set_path=file_path,
                        band_power_method=cfg["psd_band_power_method"],
                        normalize_total_power=cfg["psd_normalize_total_power"],
                        plot_fooof=False,
                        condition_spec=condition_spec,
                    )
                    if psd_results is not None:
                        df = psd_results["measures"].copy()
                        df["participant_id"] = ppt_id
                        df["metric"] = "PSD"
                        all_psd.append(df)
                        tracker["psd_status"] = "success"
                        emit("     PSD ✓")
                    else:
                        tracker["psd_status"] = "failed"
                        emit("     PSD returned None")
                except Exception as e:  # noqa: BLE001
                    tracker["psd_status"] = "error"
                    tracker["error_details"] = f"PSD: {traceback.format_exc()}"
                    emit(f"     PSD error: {e}")
            else:
                tracker["psd_status"] = "disabled"

            # ── Stringify lists for CSV ──────────────────────────────────────
            for key in ("pe_taus_completed", "pe_taus_failed",
                        "wsmi_taus_completed", "wsmi_taus_failed"):
                if isinstance(tracker[key], list):
                    tracker[key] = str(tracker[key])

            tracker_rows.append(tracker)

            # Save tracker after every participant (crash recovery)
            pd.DataFrame(tracker_rows).to_csv(tracker_path, index=False)
            emit(f"  Tracker updated: {tracker_path}")
            report_progress(idx + 1)

        # ── Save final results ────────────────────────────────────────────────
        emit("=" * 50)
        emit("Saving results...")

        try:
            xlsx_path = tracker_path.replace(".csv", ".xlsx")
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                if tracker_rows:
                    pd.DataFrame(tracker_rows).to_excel(
                        writer, sheet_name="tracker", index=False)
            emit(f"  Saved tracker xlsx: {xlsx_path}")
        except Exception as e:  # noqa: BLE001
            emit(f"  ⚠ Could not write tracker xlsx: {e}")

        if all_lz:
            pd.concat(all_lz, ignore_index=True).to_csv(
                output_path + "_LZ.csv", index=False)
            emit(f"  Saved {output_path}_LZ.csv")
        if all_pe:
            pd.concat(all_pe, ignore_index=True).to_csv(
                output_path + "_PE.csv", index=False)
            emit(f"  Saved {output_path}_PE.csv")
        if all_wsmi:
            pd.concat(all_wsmi, ignore_index=True).to_csv(
                output_path + "_wSMI.csv", index=False)
            emit(f"  Saved {output_path}_wSMI.csv")
        if all_psd:
            pd.concat(all_psd, ignore_index=True).to_csv(
                output_path + "_PSD.csv", index=False)
            emit(f"  Saved {output_path}_PSD.csv")

        if stop_flag["stop"]:
            emit("■ Pipeline stopped by user.")
        else:
            emit("✓ All results saved. Complexity processing loop complete.")

    except Exception as e:  # noqa: BLE001 — catch-all so the thread never dies silently
        emit(f"✗ FATAL pipeline error: {e}")
        emit(traceback.format_exc())


app = App(app_ui, server, static_assets=os.path.join(os.path.dirname(__file__), "www"))