"""
Per-detector process pool.

Every worker handles one detector: it opens the HDF5 files itself and reads only its own ADC channel
(so decompression runs in parallel too), applies the loader's default preprocessing and the step chain,
or runs PeakFinder.process on its detector's slice. Processes rather than threads are used because the
peak finder is a Python loop that holds the GIL.

Steps that need all detectors at once (normalize) split the chain into phases: the workers run the
steps up to it, the main process normalizes, and the workers continue on in-memory slices.
"""
import contextlib
import io
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import xarray as xr

GLOBAL_STEPS = {"normalize"}


def defaultWorkers(nDetectors):
    return max(1, min(int(nDetectors), (os.cpu_count() or 2) - 1))


def makeExecutor(nWorkers, config):
    return ProcessPoolExecutor(max_workers=nWorkers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_initWorker, initargs=(config,))


# ---------------------------------------------------------------------- worker side
def _initWorker(config):
    import dask
    # One process per detector already: keep dask single-threaded inside it to avoid oversubscription
    dask.config.set(scheduler="synchronous")
    import polstudio  # noqa: F401  (puts ToFPipeline on sys.path)
    from ToFPipeline import GlobalConfig
    GlobalConfig._config = config


def _captured(fn, job):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = fn(job)
    return job["detector"], result, buffer.getvalue()


def _loadAndPreprocess(job):
    from ToFPipeline import FlashH5Loader
    from .session import runStepChain
    det = job["detector"]
    loader = FlashH5Loader(job["files"], config=job["loaderConfig"])
    try:
        loader.load(trainStart=job["trainStart"], trainStop=job["trainStop"], trainStep=job["trainStep"],
                    channels=[det], key="GMD" if job["needsGMD"] else None, photonEnergy={})
        loader.defaultPreprocessing(ToF=[det])
        data = loader.data.compute()
        pf = runStepChain(data, job["steps"], job["peakConfig"], loader.pulseEnergy)
        return pf.data.compute()
    finally:
        loader.close()


def _stepsOnData(job):
    from .session import runStepChain
    pf = runStepChain(job["data"], job["steps"], job["peakConfig"], job["pulseEnergy"])
    return pf.data.compute()


def _findPeaks(job):
    from ToFPipeline import PeakFinder
    pf = PeakFinder(job["data"], config=job["peakConfig"])
    pf.process(**job["params"])
    return pf.results


def workerLoadAndPreprocess(job):
    return _captured(_loadAndPreprocess, job)


def workerStepsOnData(job):
    return _captured(_stepsOnData, job)


def workerFindPeaks(job):
    return _captured(_findPeaks, job)


# ---------------------------------------------------------------------- main side
def _runJobs(executor, fn, jobs, what, log):
    out = {}
    futures = [executor.submit(fn, job) for job in jobs]
    for i, future in enumerate(as_completed(futures), 1):
        det, result, text = future.result()
        for line in text.strip().splitlines():
            log(f"[ToF {det}] {line}")
        log(f"{what}: ToF {det} done ({i}/{len(jobs)})")
        out[det] = result
    return [out[job["detector"]] for job in jobs]


def splitPhases(steps):
    """[(globalStep or None, [per-detector steps]), ...] for the enabled steps."""
    phases = [(None, [])]
    for step in steps:
        if not step.get("enabled"):
            continue
        if step["name"] in GLOBAL_STEPS:
            phases.append((step, []))
        else:
            phases[-1][1].append(step)
    return phases


def preprocess(session, executor, steps, log=print, detectors=None):
    """Parallel equivalent of runStepChain(session.raw.sel(detector=detectors), steps).data.compute()."""
    from ToFPipeline import PeakFinder
    s = session.state
    detectors = [int(d) for d in (detectors if detectors is not None else session.raw["detector"].values)]
    phases = splitPhases(steps)
    peakConfig = dict(session.config["PeakFinder"])
    needsGMD = any(st["name"] == "filterByIntensity" for _, seg in phases for st in seg)
    jobs = [{"detector": det, "files": s["files"], "loaderConfig": dict(session.loaderConfig),
             "trainStart": s["trainStart"], "trainStop": s["trainStop"], "trainStep": s["trainStep"],
             "steps": phases[0][1], "peakConfig": peakConfig, "needsGMD": needsGMD} for det in detectors]
    parts = _runJobs(executor, workerLoadAndPreprocess, jobs, "preprocessing", log)
    data = xr.concat(parts, dim="detector")

    for globalStep, segment in phases[1:]:
        pf = PeakFinder(data, config=peakConfig)
        getattr(pf, globalStep["name"])(**dict(globalStep.get("params", {})))
        data = pf.data.compute()
        if segment:
            jobs = [{"detector": det, "data": data.sel(detector=[det]), "steps": segment, "peakConfig": peakConfig,
                     "pulseEnergy": session.loader.pulseEnergy if session.loader is not None else None}
                    for det in detectors]
            data = xr.concat(_runJobs(executor, workerStepsOnData, jobs, "preprocessing", log), dim="detector")
    return data


def findPeaks(session, executor, data, params, log=print):
    """Parallel PeakFinder.process; returns the combined results DataFrame."""
    peakConfig = dict(session.config["PeakFinder"])
    jobs = [{"detector": int(det), "data": data.sel(detector=[det]), "params": params, "peakConfig": peakConfig}
            for det in data["detector"].values]
    parts = [r for r in _runJobs(executor, workerFindPeaks, jobs, "peak finding", log)
             if isinstance(r, pd.DataFrame)]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
