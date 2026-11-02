"""
Qt-free analysis state for PolStudio.

An ``AnalysisSession`` owns the configuration (same sections as ``ToFPipeline/config.yaml``), the
loaded data and every intermediate result. The GUI only edits settings and calls the methods here,
so an analysis can also be reproduced headlessly from a saved session folder.
"""
import copy
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from . import DEFAULT_CONFIG, ROOT, __version__
from ToFPipeline import (GlobalConfig, FlashH5Loader, PeakFinder, Calibrate, Fitter,  # noqa: E402
                         saveCalibration, loadCalibration, energyCalibFunc)

SESSION_KEY = "PolStudio"

# Preprocessing steps offered in the GUI: PeakFinder method name -> label and default kwargs.
# None means "fall back to the method's default".
STEP_SPECS = {
    "filterByIntensity": {"label": "Filter by GMD pulse energy",
                          "params": {"lowerThreshold": None, "upperThreshold": None}},
    "filterByHeight": {"label": "Filter by trace height",
                       "params": {"lowerThreshold": None, "upperThreshold": None}},
    "subtractPeriodicNoise": {"label": "Subtract periodic noise",
                              "params": {"noiseRegion": [0, 100]}},
    "fftNoiseSubtraction": {"label": "FFT band-stop filter",
                            "params": {"minFreq": 0.01, "maxFreq": 0.1, "sampleSlice": None}},
    "tShift": {"label": "Extra sample shift (tShift)", "params": {"shift": 0}},
    "stack": {"label": "Stack (average) trains / pulses",
              "params": {"stackTrains": True, "trainStackSize": None, "stackPulses": False,
                         "pulseStackStart": None, "pulseStackStop": None, "pulseStackSize": None}},
    "smooth": {"label": "Smooth (rolling mean)", "params": {"windowSize": 5}},
    "normalize": {"label": "Normalize to maximum", "params": {"ToF": None}},
}

DEFAULT_STEPS = [
    ("filterByIntensity", False), ("filterByHeight", False), ("subtractPeriodicNoise", False),
    ("fftNoiseSubtraction", False), ("tShift", False), ("stack", True), ("smooth", True), ("normalize", False),
]

DEFAULT_PEAK_PARAMS = {"threshold": 0.2, "peakNo": 1, "roi": [None, None], "distanceFactor": 2,
                       "widthFraction": 0.5, "symmetric": False, "minWidth": False, "slopeLength": False,
                       "maxSlope": False, "slopeStartHeight": None, "noiseRegion": [0, 100]}

# Peak arguments that are off via a sentinel and take a value when on: key -> (value when switched on, off value)
PEAK_TOGGLES = {"slopeLength": (5, False), "maxSlope": (4.0, False), "slopeStartHeight": (0.5, None)}

DEFAULT_POL_PARAMS = {"peakNo": 0, "beta": 2.0, "setPlin": None, "setPhi": None, "fitBeta": False,
                      "intMethod": "height", "useTransmission": True, "Ebind": 0.0}

LOADER_KEYS_FOR_CALIBRATION = ("pulseOffset", "pulsePeriod", "pulseWindow", "sampleShift", "baselineRegion")


def _gitCommit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _jsonable(obj):
    """Convert numpy scalars / arrays and tuples so they survive yaml/json dumping."""
    if isinstance(obj, dict):
        return {(int(k) if isinstance(k, np.integer) else k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def runStepChain(data, steps, peakConfig=None, pulseEnergy=None, log=print):
    """Apply the enabled preprocessing steps in order. Returns the PeakFinder."""
    pf = PeakFinder(data, config=peakConfig)
    for step in steps:
        if not step.get("enabled"):
            continue
        name, params = step["name"], dict(step.get("params", {}))
        if name == "filterByIntensity":
            if pulseEnergy is None:
                log("Skipping filterByIntensity: no GMD data")
                continue
            pf.filterByIntensity(pulseEnergy, **params)
        elif name == "stack":
            # A preview of a few trains may hold fewer trains/pulses than the stack sizes
            nTrains = len(np.unique(pf.data["trainId"].values))
            nPulses = len(np.unique(pf.data["pulseId"].values))
            if params.get("trainStackSize") and params["trainStackSize"] > nTrains:
                params["trainStackSize"] = nTrains
            if params.get("pulseStackSize") and params["pulseStackSize"] > nPulses:
                params["pulseStackSize"] = nPulses
            pf.stack(**params)
        elif name == "tShift":
            pf.tShift(params.get("shift") or 0)
        else:
            getattr(pf, name)(**params)
    return pf


class AnalysisSession:
    def __init__(self, configPath=DEFAULT_CONFIG):
        self.configPath = Path(configPath)
        GlobalConfig.load(self.configPath)
        # Shared with GlobalConfig so every pipeline class sees the GUI's edits
        self.config = GlobalConfig._config
        self.config.setdefault("FlashH5Loader", {})
        self.config.setdefault("PeakFinder", {})
        self.config.setdefault("Calibrate", {})
        self._executor = None
        self._executorWorkers = None
        self.state = self._defaultState()
        state = self.config.get(SESSION_KEY)
        if state:
            state = copy.deepcopy(state)
            # Keep defaults for parameters added after the session was saved
            for key in ("peaks", "calibration", "polarization"):
                if isinstance(state.get(key), dict):
                    state[key] = {**self.state[key], **state[key]}
            self.state.update(state)
        self.reset()

    # ------------------------------------------------------------------ state
    @staticmethod
    def _defaultState():
        return {
            "dataFolder": None,
            "files": [],
            "channels": None,
            "trainStart": None, "trainStop": None, "trainStep": None,
            "steps": [{"name": n, "enabled": e, "params": copy.deepcopy(STEP_SPECS[n]["params"])}
                      for n, e in DEFAULT_STEPS],
            "peaks": copy.deepcopy(DEFAULT_PEAK_PARAMS),
            "photonEnergy": {},          # {run: eV} overrides
            "calibration": {"bindingEnergy": 0.0, "peakNo": 1, "transmissionPeakNo": 0,
                            "setBeta": 2.0, "setPhi": 0.0, "setPlin": 0.0, "intMethod": "height",
                            "file": None},
            "polarization": copy.deepcopy(DEFAULT_POL_PARAMS),
            "workers": 0,                # worker processes per detector; 0 = auto, 1 = no pool
        }

    def reset(self):
        """Drop all data products (settings are kept)."""
        if getattr(self, "loader", None) is not None:
            self.loader.close()
        self.overview = None
        self.loader = None
        self.raw = None            # after loader.defaultPreprocessing (lazy)
        self.processed = None      # after preprocessing steps (in memory)
        self.peakFinder = None
        self.results = None
        self.energyCalib = None
        self.transmissionCalib = None
        self.calibrationMeta = {}
        self.polarization = None   # dict with params / errors
        self.calibrateObj = None

    def close(self):
        """Release files and worker processes."""
        self.reset()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def nWorkers(self):
        from .parallel import defaultWorkers
        nDet = self.raw.sizes["detector"] if self.raw is not None else 16
        requested = int(self.state.get("workers") or 0)
        return min(requested, nDet) if requested > 0 else defaultWorkers(nDet)

    def executor(self):
        """Persistent process pool (spawning workers costs a few seconds of imports, so reuse it)."""
        from .parallel import makeExecutor
        n = self.nWorkers()
        if self._executor is None or self._executorWorkers != n:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
            self._executor = makeExecutor(n, _jsonable(self.config))
            self._executorWorkers = n
        return self._executor

    @property
    def loaderConfig(self):
        return self.config["FlashH5Loader"]

    @property
    def sampleRate(self):
        return float(self.loaderConfig.get("sampleRate", 1.8e9))

    # ------------------------------------------------------------------ files
    def scan(self, folder):
        self.state["dataFolder"] = str(folder)
        self.overview = FlashH5Loader.scan(folder)
        return self.overview

    def load(self, files=None, trainStart=None, trainStop=None, trainStep=None, channels=None):
        """Open the selected files, cut pulses and apply the loader's default preprocessing (lazy)."""
        files = [str(f) for f in (files if files is not None else self.state["files"])]
        if not files:
            raise ValueError("No files selected")
        self.state.update({"files": files, "trainStart": trainStart, "trainStop": trainStop,
                           "trainStep": trainStep, "channels": channels})
        if self.loader is not None:
            self.loader.close()
        self.processed = self.peakFinder = self.results = None
        self.loader = FlashH5Loader(files, config=self.loaderConfig)
        self.loader.load(trainStart=trainStart, trainStop=trainStop, trainStep=trainStep,
                         channels=channels, photonEnergy=self.state["photonEnergy"])
        tof = self.loaderConfig.get("ToF")
        if channels is not None:
            tof = [c for c in (tof or channels) if c in channels]
        self.loader.defaultPreprocessing(ToF=tof)
        self.raw = self.loader.data
        return self

    def refreshPulseSplit(self):
        """Re-cut the pulses after pulseOffset / pulsePeriod / nPulses / pulseWindow changed."""
        s = self.state
        return self.load(s["files"], s["trainStart"], s["trainStop"], s["trainStep"], s["channels"])

    # ------------------------------------------------------------------ inspection
    def trainIds(self):
        return self.loader.trainTraces["trainId"].values if self.loader is not None else np.array([])

    def runOfTrain(self, trainId):
        tt = self.loader.trainTraces
        return int(tt["daq_run"].values[np.searchsorted(tt["trainId"].values, trainId)])

    def trainTrace(self, trainId, detectors=None):
        """Raw (unsplit, uncorrected) traces of one train: DataArray (detector, sample)."""
        tt = self.loader.trainTraces
        pos = int(np.nonzero(tt["trainId"].values == trainId)[0][0])
        tr = tt.isel(train=pos)
        if detectors is not None:
            tr = tr.sel(detector=[d for d in detectors if d in tr.detector.values])
        return tr.compute()

    def pulseStarts(self, nSamplesTrain=None):
        nSamplesTrain = nSamplesTrain or (self.loader.trainTraces.sizes["sample"] if self.loader else 16000)
        helper = self.loader or FlashH5Loader([], config=self.loaderConfig)
        helper.config = self.loaderConfig
        return helper.pulseStartIndices(nSamplesTrain)

    def gmdOfTrain(self, trainId):
        pe = self.loader.pulseEnergy if self.loader is not None else None
        if pe is None:
            return None
        return pe[pe["trainId"] == trainId]

    # ------------------------------------------------------------------ preprocessing
    def _subsetTrains(self, data, trainIds):
        mask = np.isin(data["trainId"].values, np.asarray(trainIds))
        return data.isel(pulse=np.nonzero(mask)[0])

    def runSteps(self, data, steps=None, log=print):
        """Apply the enabled preprocessing steps in order (in this process). Returns the PeakFinder."""
        steps = steps if steps is not None else self.state["steps"]
        pulseEnergy = self.loader.pulseEnergy if self.loader is not None else None
        return runStepChain(data, steps, self.config["PeakFinder"], pulseEnergy, log=log)

    def preview(self, trainIds, steps=None, log=print):
        """Run the step chain on a few trains only. Returns (rawSubset, processedSubset) in memory."""
        rawSubset = self._subsetTrains(self.raw, trainIds).compute()
        pf = self.runSteps(rawSubset.copy(), steps, log=log)
        return rawSubset, pf.data.compute()

    def applySteps(self, log=print):
        if self.raw is None:
            raise ValueError("Load data first")
        if self.nWorkers() > 1:
            from . import parallel
            log(f"Preprocessing {self.raw.sizes['detector']} detectors in {self.nWorkers()} worker processes")
            self.processed = parallel.preprocess(self, self.executor(), self.state["steps"], log=log)
        else:
            self.processed = self.runSteps(self.raw, log=log).data.compute()
        self.peakFinder = None
        self.results = None
        return self.processed

    # ------------------------------------------------------------------ peaks
    def findPeaks(self, params=None, data=None):
        params = dict(params or self.state["peaks"])
        self.state["peaks"] = params
        data = data if data is not None else (self.processed if self.processed is not None else None)
        if data is None:
            raise ValueError("Apply the preprocessing first")
        roi = params.get("roi") or [None, None]
        params["roi"] = [None if v in ("", None) else int(v) for v in roi]
        # The baseline walk needs both slopeLength and maxSlope, but process() starts it when either is set
        if (params.get("slopeLength") is False) != (params.get("maxSlope") is False):
            missing = "slopeLength" if params.get("slopeLength") is False else "maxSlope"
            params[missing] = PEAK_TOGGLES[missing][0]
            print(f"Per-peak baseline: {missing} not set, using {params[missing]}")
        self.peakFinder = PeakFinder(data, config=self.config["PeakFinder"])
        if self.nWorkers() > 1 and data.sizes["detector"] > 1:
            from . import parallel
            self.peakFinder.results = parallel.findPeaks(self, self.executor(), data, params)
        else:
            self.peakFinder.process(**params)
        self.results = self.enrichResults(self.peakFinder.results)
        self.peakFinder.results = self.results
        return self.results

    def enrichResults(self, results):
        """Add Angles, daq_run and Photon Energy columns (needed by Calibrate / Fitter / Plotter)."""
        if results is None or len(results) == 0:
            return results
        results = results.drop(columns=[c for c in ("Angles", "daq_run", "Photon Energy") if c in results])
        results = results.merge(self.loader.angles, on="detector", how="left")
        pe = self.loader.photonEnergy
        if pe is not None:
            pe = pe.copy()
            for run, energy in self.state["photonEnergy"].items():
                pe.loc[pe["daq_run"] == int(run), "Photon Energy"] = float(energy)
            # Stacked results carry the first trainId of each block, which is in this table
            results = results.merge(pe, on="trainId", how="left")
        return results

    def setPhotonEnergies(self, mapping):
        """{run: eV} overrides; updates the current results in place."""
        self.state["photonEnergy"] = {int(k): float(v) for k, v in mapping.items() if v not in (None, "")}
        if self.results is not None:
            self.results = self.enrichResults(self.results)
            if self.peakFinder is not None:
                self.peakFinder.results = self.results

    def runPhotonEnergies(self):
        """DataFrame run -> photon energy from the files (median) and the override."""
        if self.loader is None or self.loader.photonEnergy is None:
            return pd.DataFrame(columns=["run", "from file", "override"])
        pe = self.loader.photonEnergy.groupby("daq_run")["Photon Energy"].median().reset_index()
        pe.columns = ["run", "from file"]
        pe["override"] = pe["run"].map(lambda r: self.state["photonEnergy"].get(int(r), np.nan))
        return pe

    # ------------------------------------------------------------------ calibration
    def calibrationSettingsSnapshot(self):
        snapshot = {k: _jsonable(self.loaderConfig.get(k)) for k in LOADER_KEYS_FOR_CALIBRATION}
        cfg = self.loaderConfig
        # Store the window actually used (null in the config means "full period")
        snapshot["pulseWindow"] = FlashH5Loader.windowLength(float(cfg.get("pulsePeriod", 0.0)), cfg.get("pulseWindow"),
                                                             int(cfg.get("nPulses", 1)), warn=False)
        return snapshot

    def computeEnergyCalibration(self, peakNo=None, bindingEnergy=None):
        c = self.state["calibration"]
        peakNo = c["peakNo"] if peakNo is None else peakNo
        bindingEnergy = c["bindingEnergy"] if bindingEnergy is None else bindingEnergy
        self._requireResults()
        nEnergies = self.results["Photon Energy"].nunique()
        if nEnergies < 3:
            raise ValueError(f"Energy calibration needs results at >= 3 photon energies, found {nEnergies}. "
                             "Load more runs or set photon energies per run.")
        cal = Calibrate(self.results.copy(), config=self.config["Calibrate"], bindingEnergy=bindingEnergy)
        cal.energy(peakNo=peakNo)
        self.calibrateObj = cal
        self.energyCalib = cal.energyParam
        self._updateCalibrationMeta(bindingEnergy=bindingEnergy, energyPeakNo=peakNo)
        return self.energyCalib

    def computeTransmission(self, peakNo=None, setBeta=None, setPhi=None, setPlin=None, intMethod=None,
                            bindingEnergy=None):
        c = self.state["calibration"]
        self._requireResults()
        bindingEnergy = c["bindingEnergy"] if bindingEnergy is None else bindingEnergy
        cal = Calibrate(self.results.copy(), config=self.config["Calibrate"], bindingEnergy=bindingEnergy)
        cal.transmission(peakNo=c["transmissionPeakNo"] if peakNo is None else peakNo,
                         setBeta=c["setBeta"] if setBeta is None else setBeta,
                         setPhi=c["setPhi"] if setPhi is None else setPhi,
                         setPlin=c["setPlin"] if setPlin is None else setPlin,
                         intMethod=c["intMethod"] if intMethod is None else intMethod)
        self.calibrateObj = cal
        self.transmissionCalib = cal.transmissionParam
        self._updateCalibrationMeta(bindingEnergy=bindingEnergy)
        return self.transmissionCalib

    def _updateCalibrationMeta(self, **kwargs):
        runs = sorted({int(r) for r in self.results["daq_run"].dropna().unique()}) if "daq_run" in self.results else []
        self.calibrationMeta.update({
            "runs": runs,
            "files": [Path(f).name for f in self.state["files"]],
            "sampleRate": self.sampleRate,
            "loader": self.calibrationSettingsSnapshot(),
            "peakFinder": _jsonable(self.state["peaks"]),
            **_jsonable(kwargs),
        })

    def saveCalibration(self, path):
        if self.energyCalib is None and self.transmissionCalib is None:
            raise ValueError("No calibration to save")
        saveCalibration(path, energyParam=self.energyCalib, transmissionParam=self.transmissionCalib,
                        meta=self.calibrationMeta)
        self.state["calibration"]["file"] = str(path)
        return path

    def loadCalibration(self, path):
        """Load a calibration JSON. Returns a list of warnings about mismatching loader settings."""
        content = loadCalibration(path)
        self.energyCalib = content["energy"]
        self.transmissionCalib = content["transmission"]
        self.calibrationMeta = content.get("meta", {}) or {}
        self.state["calibration"]["file"] = str(path)
        warnings = []
        stored = self.calibrationMeta.get("loader", {})
        # JSON turns dict keys into strings: compare both sides after the same round trip
        current = json.loads(json.dumps(self.calibrationSettingsSnapshot()))
        for key, value in stored.items():
            if key in current and current[key] != value:
                warnings.append(f"{key}: calibration {value} vs current {current[key]}")
        return warnings

    def energyOf(self, detector, samples):
        """Map sample positions to energy (eV) with the energy calibration, or None."""
        if self.energyCalib is None or len(self.energyCalib) == 0:
            return None
        row = self.energyCalib[self.energyCalib["detector"] == detector]
        if row.empty or row[["p0", "p1", "p2", "p3", "p4"]].isna().any(axis=None):
            return None
        p = row.iloc[0][["p0", "p1", "p2", "p3", "p4"]].astype(float).values
        with np.errstate(invalid="ignore", divide="ignore"):
            return energyCalibFunc(np.asarray(samples, dtype=float), *p)

    # ------------------------------------------------------------------ polarization
    def fitPolarization(self, params=None, ax=None):
        params = dict(params or self.state["polarization"])
        self.state["polarization"] = dict(params)
        self._requireResults()
        useTransmission = params.pop("useTransmission", True)
        transParam = self.transmissionCalib if (useTransmission and self.transmissionCalib is not None
                                                and len(self.transmissionCalib)) else None
        fitter = Fitter(self.results, config=self.config.get("Fitter"), Ebind=params.pop("Ebind", 0.0))
        fitParams, errors = fitter.pol(transParam=transParam, plot=ax is not None, ax=ax, savename=False,
                                       show=False, **params)
        names = ["Plin", "phi", "scale"] + (["beta2"] if params.get("fitBeta") else [])
        self.polarization = {
            "params": {n: float(v) for n, v in zip(names, fitParams)},
            "errors": {n: float(e) for n, e in zip(names, errors)},
            "settings": _jsonable({**params, "useTransmission": useTransmission and transParam is not None}),
        }
        return self.polarization

    def _requireResults(self):
        if self.results is None or len(self.results) == 0:
            raise ValueError("Run the peak finder first")

    # ------------------------------------------------------------------ persistence
    def configForSaving(self):
        cfg = copy.deepcopy(self.config)
        cfg[SESSION_KEY] = copy.deepcopy(self.state)
        return _jsonable(cfg)

    def saveConfig(self, path):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.configForSaving(), f, sort_keys=False)

    def manifest(self):
        selection = None
        if self.loader is not None:
            tids = self.trainIds()
            selection = {"nTrains": int(len(tids)),
                         "trainId min": int(tids.min()) if len(tids) else None,
                         "trainId max": int(tids.max()) if len(tids) else None,
                         "runs": sorted({int(r) for r in self.loader.trainTraces["daq_run"].values}),
                         "pulseStarts": _jsonable(self.loader.pulseStarts.to_dict())}
        return {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "polstudio": {"version": __version__, "commit": _gitCommit(ROOT)},
            "tofpipeline": {"commit": _gitCommit(ROOT / "ToFPipeline")},
            "baseConfig": str(self.configPath),
            "files": self.state["files"],
            "selection": selection,
            "steps": [s["name"] for s in self.state["steps"] if s.get("enabled")],
            "polarization": self.polarization,
        }

    def save(self, folder, processed=True):
        """Write config, tables and calibration to a session folder (plots are added by export.py)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.saveConfig(folder / "config.yaml")
        if self.results is not None:
            self.results.to_csv(folder / "peaks.csv", index=False)
        if self.energyCalib is not None or self.transmissionCalib is not None:
            saveCalibration(folder / "calibration.json", energyParam=self.energyCalib,
                            transmissionParam=self.transmissionCalib, meta=self.calibrationMeta)
        if self.polarization is not None:
            rows = [{"parameter": k, "value": v, "error": self.polarization["errors"].get(k)}
                    for k, v in self.polarization["params"].items()]
            pd.DataFrame(rows).to_csv(folder / "polarization.csv", index=False)
        if processed and self.processed is not None:
            self.saveProcessed(folder / "processed.h5")
        with open(folder / "session.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(self.manifest()), f, indent=2, default=str)
        return folder

    def saveProcessed(self, path):
        import h5py
        data = self.processed
        index = data["pulse"].to_index()
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.asarray(data.values, dtype=np.float32), compression="gzip")
            f["data"].attrs["dims"] = list(data.dims)
            f.create_dataset("detector", data=data["detector"].values)
            f.create_dataset("trainId", data=index.get_level_values("trainId").values)
            f.create_dataset("pulseId", data=index.get_level_values("pulseId").values)
            f.create_dataset("sample", data=data["sample"].values)

    @staticmethod
    def loadProcessed(path):
        import h5py
        with h5py.File(path, "r") as f:
            idx = pd.MultiIndex.from_arrays([f["trainId"][:], f["pulseId"][:]], names=["trainId", "pulseId"])
            da_ = xr.DataArray(f["data"][:], dims=["detector", "pulse", "sample"],
                               coords={"detector": f["detector"][:], "sample": f["sample"][:]})
        return da_.assign_coords(xr.Coordinates.from_pandas_multiindex(idx, "pulse"))

    @classmethod
    def open(cls, folder):
        """Restore a saved session: settings, tables and calibration. Data is reloaded on demand."""
        folder = Path(folder)
        session = cls(folder / "config.yaml")
        if (folder / "peaks.csv").exists():
            session.results = pd.read_csv(folder / "peaks.csv")
        if (folder / "calibration.json").exists():
            session.loadCalibration(folder / "calibration.json")
        if (folder / "processed.h5").exists():
            session.processed = cls.loadProcessed(folder / "processed.h5")
        if (folder / "session.json").exists():
            with open(folder / "session.json", encoding="utf-8") as f:
                session.polarization = json.load(f).get("polarization")
        return session
