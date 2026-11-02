# PolStudio

Desktop GUI (PyQt6 + pyqtgraph) for offline analysis of FLASH ToF data with
[ToFPipeline](ToFPipeline/README.md): pick HDF5 files, inspect runs / trains / pulses,
tune the preprocessing on a live preview, find peaks, build or load calibrations, fit the
polarization and store everything in a reproducible session folder.

## Install and run

```bash
pip install -r requirements.txt
```

```bash
python -m polstudio
```

Optional start-up arguments: `--data <folder>` scans a data folder, `--session <folder>` reopens a
saved session, `--config <config.yaml>` uses another pipeline config (default `ToFPipeline/config.yaml`).

## Workflow

| Step | Where | What happens |
|---|---|---|
| 1. Choose files | **Data** dock → *Open data folder…* | `FlashH5Loader.scan` lists every run (from `run<N>` in the file name) with its files, train count, trainId range, wavelength and start time. Tick runs/files, optionally restrict the trains (positional start/stop/step per file) and the ADC channels, then *Load selection*. |
| 2. Inspect & split pulses | **Inspect** | Step through trains. The raw 16000-sample train trace is shown with the pulse windows shaded; below, all windows of that train are overlaid (aligned peaks = correct period) and the GMD energy per pulse. Pulse starts are `round(offset + k · period)` with a float period. Only the window (default: the full period, capped at `floor(period)` so windows never overlap) is kept; the rest of each period is discarded, which saves memory in proportion. *Apply split* re-cuts all loaded trains. |
| 3. Preprocess | **Preprocess** | Enable, order (drag) and parametrise the `PeakFinder` steps: GMD/height filters, periodic-noise and FFT filters, tShift, stacking, smoothing, normalisation. *Preview* runs the chain on N trains starting at the train selected in Inspect (raw in grey, processed in blue). *Apply to all* processes the full selection. |
| 4. Peaks | **Peaks** | `PeakFinder.process` parameters (ROI and noise region can be dragged on the plots). Traces with peak, FWHM and baseline markers, x axis in samples / ns / eV, the full results table and a position histogram. |
| 4b. Heatmap | **Heatmap** | Processed traces of one pulse (or the mean of all pulses) as a polar heatmap (angle = detector angle, radius = sample; interpolated between detectors or one wedge per detector) or a Cartesian one (sample vs detector number or angle). Sample range, interpolation, peak overlay, raster resolution, colormap and a colour bar with draggable levels, as in PolScope. |
| 5. Calibration | **Calibration** | Photon energy per run (from the wavelength channel, with overrides), energy calibration (needs ≥ 3 photon energies) and transmission coefficients via `Calibrate`. *Load / Save calibration* reads/writes JSON (default folder `results/calibrations/`). A loaded calibration is checked against the current pulse-split and shift settings. |
| 6. Polarization | **Polarization** | `Fitter.pol` with optional transmission correction, and the `Plotter` polar intensity map of one pulse. |
| 7. Save | *File → Save session…* (Ctrl+S) or *Quick save* | Writes a session folder (below). *File → Open session…* restores settings, peaks, calibration and processed traces; *Load selection* reloads the raw data. |

## Update button

*Update → <tab>* (next to *Load selection*) runs the pipeline for the current selection up to the selected
tab: Inspect = load, Preprocess / Heatmap = processed traces, Peaks / Calibration = peak results,
Polarization = peak results and the polarization fit. Only what changed is recomputed:

- unticked ADC channels are removed from the processed data and peak results without recomputing;
- newly ticked channels are loaded, processed and peak-searched on their own and merged in; products
  beyond the selected tab keep their channels and are completed by a later update on their tab (the Data
  dock shows e.g. `peaks: 13/14 channels (missing [13])`);
- results made with other files, trains, pulse split, preprocessing steps or peak settings are redone
  when the selected tab needs them, otherwise dropped;
- with `normalize` enabled the processed traces depend on all channels, so adding one reprocesses all;
- the polarization fit is redone on the Polarization tab when its inputs changed, otherwise marked outdated.

Calibrations are never recomputed by *Update*. *Load selection* still starts from scratch.

## Parallel processing

*Apply to all* and *Find peaks* run one detector per worker process (setting *worker processes* in the
Data dock; auto = min(detectors, CPU cores − 1), 1 = everything in the GUI process). Each worker reads
only its own ADC channel from the files, so decompression is parallel as well. `normalize` needs all
detectors and runs in the main process between worker phases. The pool is started on first use (a few
seconds) and reused. On a full run of the example data (14 detectors, 7 workers) preprocessing took
16 s instead of 61 s and peak finding on unstacked data 7 s instead of 15 s.

## Session folder

`results/<runs>_<YYYYMMDD-HHMMSS>/`

| File | Content |
|---|---|
| `config.yaml` | Every setting used, loadable by `GlobalConfig.load`. The `PolStudio` section holds files, train selection, step order and GUI parameters. |
| `session.json` | Provenance: input files, runs, trainId range, pulse starts, enabled steps, git commits of PolStudio and ToFPipeline, polarization result. |
| `peaks.csv` | `PeakFinder.results` plus `Angles`, `daq_run` and `Photon Energy`. |
| `calibration.json` | Energy and transmission tables with metadata (runs, binding energy, loader settings). |
| `polarization.csv` | Fit parameters and errors. |
| `processed.h5` | Preprocessed traces (`data[detector, pulse, sample]`, `trainId`, `pulseId`, `detector`, `sample`). |
| `plots/` | Matplotlib figures from the pipeline (PNG + PDF) and PNG snapshots of the GUI views. |

## Layout

```
polstudio/            GUI package
  session.py          Qt-free analysis state (load, steps, peaks, calibration, save/open)
  export.py           session folder + plot export
  main_window.py      window, menus, background tasks
  widgets/            data panel and one module per tab
scripts/
  smoke_flash_h5.py   headless test of the loader / pipeline path
  smoke_gui.py        drives the GUI through a full analysis and saves screenshots
ToFPipeline/          analysis library (submodule)
```

Checks:

```bash
python scripts/smoke_flash_h5.py ../2024_hdf5
```

```bash
python scripts/smoke_gui.py ../2024_hdf5 results/smoke --offscreen
```
