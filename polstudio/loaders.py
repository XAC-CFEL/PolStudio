"""
Data formats PolStudio can open: FLASH DAQ HDF5 files (``*.h5``, FlashH5Loader) and P04 NeXus files
(``*.nxs``, NXSLoader). Both loaders offer the interface the session and the GUI use: ``scan``, ``runNumber``,
``load(trainStart, trainStop, trainStep, channels, key, photonEnergy)``, ``defaultPreprocessing(ToF)``, ``close``
and the attributes ``data``, ``trainTraces``, ``pulseStarts``, ``pulseEnergy``, ``photonEnergy``, ``angles``.
"""
import re
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import xarray as xr

from ToFPipeline import FlashH5Loader, GlobalConfig, NXSLoader


class NxsFileLoader(NXSLoader):
    """
    P04 NeXus files (``*_<run>.nxs``): one file per run with one ToF histogram per channel and scan point.

    Keeps NXSLoader's layout and settings (config section ``NXSLoader``): detector = histogram channel − 1,
    trainId = run number, pulseId = scan point, so the positional train slice selects scan points. There is no
    pulse split and no GMD; the photon energy is the monochromator energy stored in the file. ``trainTraces``
    holds the raw histograms summed over the selected scan points (one "train" per run), ``pointTraces`` the raw
    histograms of every scan point (``data`` before ``defaultPreprocessing``).
    """
    CONFIG_KEY = "NXSLoader"
    RUN_PATTERN = re.compile(r"_(\d+)\.nxs$")
    HISTOGRAM = "scan/instrument/histogram_ch{ch:02d}"
    N_DETECTORS = 15   # NXSLoader._loadSingleFile reads histogram_ch01 … ch15
    PHOTON_ENERGY_PATH = "scan/instrument/mono/energy"

    def __init__(self, files, config=None):
        if isinstance(files, (str, Path)):
            files = [files]
        self.files = sorted((Path(f) for f in files), key=lambda f: self.runNumber(f) or 0)
        super().__init__(self.files[0].parent if self.files else ".", config=config)
        self.trainTraces = None
        self.pointTraces = None
        self.pulseStarts = None    # no pulse split: every scan point is a pulse
        self.pulseEnergy = None    # no GMD
        self.photonEnergy = None
        self.sampleRate = None     # from the histogram bin size

        # As in FlashH5Loader: keep every channel's angle, so channels outside ToF can be loaded too
        fullTheta = np.array(self.config.get("angles", np.linspace(0, 360, 16, endpoint=False)))
        detectors = self.config.get("ToF", list(range(len(fullTheta))))
        if len(fullTheta) != len(detectors):
            detectors = list(range(len(fullTheta)))
        self.angles = pd.DataFrame(detectors, columns=["detector"])
        self.angles["Angles"] = fullTheta

    @classmethod
    def runNumber(cls, filePath):
        match = cls.RUN_PATTERN.search(Path(filePath).name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _photonEnergy(f, path):
        if path not in f:
            return np.nan
        values = np.asarray(f[path][()], dtype=float).ravel()
        return float(np.nanmedian(values)) if len(values) else np.nan

    @classmethod
    def scan(cls, paths, config=None):
        """
        Cheap overview of .nxs files: only shapes and scalars are read.

        Returns a DataFrame with one row per file: file, run, fileNo, created (scan start), channels, nPoints,
        nSamples, photonEnergy (eV).
        """
        cfg = GlobalConfig.get_for_class(cls.CONFIG_KEY)
        cfg.update(config or {})
        energyPath = cfg.get("photonEnergyPath", cls.PHOTON_ENERGY_PATH)

        if isinstance(paths, (str, Path)) and Path(paths).is_dir():
            paths = sorted(Path(paths).glob("*.nxs"))
        elif isinstance(paths, (str, Path)):
            paths = [paths]

        rows = []
        for path in paths:
            path = Path(path)
            row = {"file": str(path), "run": cls.runNumber(path), "fileNo": None}
            try:
                with h5py.File(path, "r") as f:
                    start = f["scan/start_time"][()] if "scan/start_time" in f else ""
                    row["created"] = start.decode() if isinstance(start, bytes) else str(start)
                    channels = [det for det in range(cls.N_DETECTORS)
                                if f"{cls.HISTOGRAM.format(ch=det + 1)}/data" in f]
                    row["channels"] = channels
                    if channels:
                        row["nPoints"], row["nSamples"] = f[f"{cls.HISTOGRAM.format(ch=channels[0] + 1)}/data"].shape
                    row["photonEnergy"] = cls._photonEnergy(f, energyPath)
            except OSError as e:
                row["error"] = str(e)
            rows.append(row)
        return pd.DataFrame(rows)

    def close(self):
        pass   # the files are read completely in load()

    def load(self, trainStart=None, trainStop=None, trainStep=None, channels=None, key="all", photonEnergy=None):
        """
        Read the histograms of the selected files into memory (the files are small).

        Parameters
        ----------
        trainStart, trainStop, trainStep : int, optional
            Positional slice of the scan points of each file.
        channels : list of int, optional
            Detectors to keep (histogram channel − 1). Defaults to all.
        key : {"all", "Photon Energy", None}
            Whether to read the photon energy.
        photonEnergy : dict {run: eV}, optional
            Override the monochromator energy. Falls back to ``photonEnergyOverride`` in the config.

        Returns
        -------
        self
        """
        trainStart = trainStart if trainStart is not None else self.config.get("trainStart", None)
        trainStop = trainStop if trainStop is not None else self.config.get("trainStop", None)
        trainStep = trainStep if trainStep is not None else self.config.get("trainStep", None)
        photonEnergy = photonEnergy if photonEnergy is not None else (self.config.get("photonEnergyOverride", None) or {})
        energyPath = self.config.get("photonEnergyPath", self.PHOTON_ENERGY_PATH)
        roi = self.config.get("roi", None)
        pointSlice = slice(trainStart, trainStop, trainStep)

        blocks, sums, runs, pulseIds, energies = [], [], [], [], []
        detectors = binSize = None
        for path in self.files:
            run = self.runNumber(path)
            if run is None:
                raise ValueError(f"No run number in {path.name} (expected *_<run>.nxs)")
            if run in runs:
                raise ValueError(f"Two files for run {run}")
            fileData = self._loadSingleFile(path, roi=roi)
            data = fileData["data"]   # (detector, point, sample)
            if detectors is None:
                detectors = list(range(data.shape[0])) if channels is None else \
                    [int(d) for d in channels if 0 <= int(d) < data.shape[0]]
                if not detectors:
                    raise ValueError(f"None of the channels {list(channels)} is in {path.name}")
            points = np.arange(fileData["nPulses"])[pointSlice]
            if not len(points):
                print(f"Warning: no scan points selected in {path.name}")
                continue
            data = data[detectors][:, points]
            blocks.append(data)
            sums.append(data.sum(axis=1))
            runs.append(run)
            pulseIds.append(points)
            with h5py.File(path, "r") as f:
                energies.append(self._photonEnergy(f, energyPath))
                sizePath = f"{self.HISTOGRAM.format(ch=detectors[0] + 1)}/x_pixel_size"
                if binSize is None and sizePath in f:
                    binSize = float(np.ravel(f[sizePath][()])[0])   # ns

        if not blocks:
            raise ValueError("No data could be loaded")
        nSamples = blocks[0].shape[-1]
        if any(b.shape[-1] != nSamples for b in blocks):
            raise ValueError("The files have different numbers of samples")
        sampleStart = int(roi[0] or 0) if roi is not None else 0
        samples = np.arange(sampleStart, sampleStart + nSamples, dtype=np.int64)
        runs = np.asarray(runs, dtype=np.int64)
        self.sampleRate = 1e9 / binSize if binSize else None

        self.trainTraces = xr.DataArray(
            da.from_array(np.stack(sums, axis=1), chunks=(1, -1, -1)), dims=["detector", "train", "sample"],
            coords={"detector": np.asarray(detectors, dtype=np.int64), "trainId": ("train", runs),
                    "daq_run": ("train", runs), "sample": samples},
            name="trainTraces")

        counts = [len(p) for p in pulseIds]
        pulseIndex = pd.MultiIndex.from_arrays([np.repeat(runs, counts), np.concatenate(pulseIds).astype(np.int64)],
                                               names=["trainId", "pulseId"])
        self.data = xr.DataArray(
            da.from_array(np.concatenate(blocks, axis=1), chunks=(1, -1, -1)), dims=["detector", "pulse", "sample"],
            coords={"detector": np.asarray(detectors, dtype=np.int64), "sample": samples},
            name="adc")
        self.data = self.data.assign_coords(xr.Coordinates.from_pandas_multiindex(pulseIndex, "pulse"))
        self.data = self.data.assign_coords(daq_run=("pulse", np.repeat(runs, counts)))
        self.pointTraces = self.data   # kept unprocessed for the Inspect tab

        self.photonEnergy = None
        if key in ("all", "Photon Energy"):
            pe = pd.DataFrame({"trainId": runs, "daq_run": runs, "Photon Energy": energies})
            for run, energy in (photonEnergy or {}).items():
                pe.loc[pe["daq_run"] == int(run), "Photon Energy"] = float(energy)
            self.photonEnergy = pe
        return self

    def defaultPreprocessing(self, ToF=None, baselineRegion=None):
        """NXSLoader's baseline subtraction and sampleShift for the loaded detectors (points are sliced in load)."""
        ToF = ToF if ToF is not None else self.config.get("ToF", list(self.data.detector.values))
        ToF = [d for d in ToF if d in self.data.detector.values]
        return super().defaultPreprocessing(ToF=ToF, baselineRegion=baselineRegion)


LOADERS = {"h5": FlashH5Loader, "nxs": NxsFileLoader}
SUFFIXES = {".h5": "h5", ".nxs": "nxs"}


def fileFormat(path):
    return SUFFIXES.get(Path(path).suffix.lower(), "h5")


def filesFormat(files):
    """Format of a file selection; mixing formats is not supported."""
    formats = {fileFormat(f) for f in files}
    if len(formats) > 1:
        raise ValueError("Select either .h5 or .nxs files, not both")
    return formats.pop() if formats else "h5"


def folderFormat(folder):
    """Format of the files in a folder (HDF5 if it holds both)."""
    folder = Path(folder)
    if not any(folder.glob("*.h5")) and any(folder.glob("*.nxs")):
        return "nxs"
    return "h5"


def loaderClass(fmt):
    return LOADERS[fmt]


def configKey(fmt):
    cls = LOADERS[fmt]
    return cls.CONFIG_KEY or cls.__name__


def runNumber(path):
    return LOADERS[fileFormat(path)].runNumber(path)
