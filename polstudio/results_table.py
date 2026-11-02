"""
Polarization results table: one CSV row per fit, so runs can be analysed one after another into one file.

Layout next to the table (e.g. ``results/polarization_results.csv``)::

    polarization_results.csv
    configs/config_<hash>.yaml            all settings used (without the run selection)
    calibrations/calibration_<hash>.json  calibration used

Config and calibration files are named after a hash of their content and written only when that content is
new, so many rows share the same files. The run-specific selection (files, trains, photon energy overrides)
is stored in the row itself.
"""
import copy
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ToFPipeline import FlashH5Loader, saveCalibration

from .session import SESSION_KEY, _gitCommit, _jsonable
from . import ROOT

# Session state that describes the run selection or the GUI, not the analysis settings
RUN_SPECIFIC_STATE = ("dataFolder", "files", "trainStart", "trainStop", "trainStep", "photonEnergy", "workers",
                      "resultsTable")

LEADING_COLUMNS = ["timestamp", "runs", "Plin", "Plin error", "phi", "phi error", "scale", "beta2", "beta2 error",
                   "photon energy", "peakNo", "intMethod", "beta", "fitBeta", "setPlin", "setPhi",
                   "useTransmission", "Ebind", "detectors", "nDetectors", "nPeaks", "nTraces", "files",
                   "trainStart", "trainStop", "trainStep", "trainId min", "trainId max", "nTrains",
                   "config", "calibration", "calibration source", "polstudio", "tofpipeline"]


def _hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def settingsConfig(session):
    """Full config with every setting used, minus the run selection (that goes into the row)."""
    cfg = session.configForSaving()
    state = cfg.get(SESSION_KEY, {})
    for key in RUN_SPECIFIC_STATE:
        state.pop(key, None)
    state.get("calibration", {}).pop("file", None)
    return cfg


class ResultsTable:
    def __init__(self, path):
        self.path = Path(path)
        self.folder = self.path.parent

    def read(self):
        if not self.path.exists():
            return pd.DataFrame(columns=LEADING_COLUMNS)
        return pd.read_csv(self.path)

    def _storeConfig(self, session):
        text = yaml.safe_dump(settingsConfig(session), sort_keys=True)
        target = self.folder / "configs" / f"config_{_hash(text)}.yaml"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return target.relative_to(self.folder).as_posix()

    def _storeCalibration(self, session):
        energy, trans = session.energyCalib, session.transmissionCalib
        if (energy is None or not len(energy)) and (trans is None or not len(trans)):
            return ""
        content = json.dumps(_jsonable({
            "energy": None if energy is None else json.loads(energy.to_json(orient="records")),
            "transmission": None if trans is None else json.loads(trans.to_json(orient="records")),
            "meta": session.calibrationMeta}), sort_keys=True, default=str)
        target = self.folder / "calibrations" / f"calibration_{_hash(content)}.json"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            saveCalibration(target, energyParam=energy, transmissionParam=trans, meta=session.calibrationMeta)
        return target.relative_to(self.folder).as_posix()

    def row(self, session):
        pol = session.polarization
        if pol is None:
            raise ValueError("No polarization fit to save – fit first")
        s, results = session.state, session.results
        params, errors, settings = pol["params"], pol["errors"], pol["settings"]
        peakNo = settings.get("peakNo", s["polarization"].get("peakNo", 0))
        used = results[results["peakNo"] == peakNo] if results is not None else pd.DataFrame()
        runs = sorted({FlashH5Loader.runNumber(f) for f in s["files"]} - {None})
        tids = session.trainIds() if session.loader is not None else np.array([])
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "runs": ";".join(str(r) for r in runs),
            "Plin": params.get("Plin"), "Plin error": errors.get("Plin"),
            "phi": params.get("phi"), "phi error": errors.get("phi"),
            "scale": params.get("scale"),
            "beta2": params.get("beta2", settings.get("beta")), "beta2 error": errors.get("beta2"),
            "photon energy": float(used["Photon Energy"].median()) if "Photon Energy" in used and len(used) else None,
            "detectors": ";".join(str(d) for d in sorted(session.resultsDetectors)),
            "nDetectors": len(session.resultsDetectors),
            "nPeaks": len(used),
            "nTraces": session.processed.sizes["pulse"] if session.processed is not None else None,
            "files": ";".join(Path(f).name for f in s["files"]),
            "trainStart": s.get("trainStart"), "trainStop": s.get("trainStop"), "trainStep": s.get("trainStep"),
            "trainId min": int(tids.min()) if len(tids) else None,
            "trainId max": int(tids.max()) if len(tids) else None,
            "nTrains": len(tids) if len(tids) else None,
            "config": self._storeConfig(session),
            "calibration": self._storeCalibration(session),
            "calibration source": s["calibration"].get("file") or "",
            "polstudio": _gitCommit(ROOT), "tofpipeline": _gitCommit(ROOT / "ToFPipeline"),
        }
        for key in ("peakNo", "intMethod", "beta", "fitBeta", "setPlin", "setPhi", "useTransmission", "Ebind"):
            row[key] = settings.get(key, s["polarization"].get(key))
        return row

    def append(self, session):
        """Append the session's current polarization fit. Returns the new row."""
        row = self.row(session)
        self.folder.mkdir(parents=True, exist_ok=True)
        table = pd.concat([self.read(), pd.DataFrame([row])], ignore_index=True)
        columns = LEADING_COLUMNS + [c for c in table.columns if c not in LEADING_COLUMNS]
        tmp = self.path.with_suffix(".tmp")
        table[columns].to_csv(tmp, index=False)
        tmp.replace(self.path)   # never leave a half-written table behind
        return row
