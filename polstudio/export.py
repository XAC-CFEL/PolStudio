"""Session folder export: tables via AnalysisSession.save, plus matplotlib and pyqtgraph figures."""
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import RESULTS_DIR


def defaultSessionFolder(session):
    runs = []
    if session.loader is not None:
        runs = sorted({int(r) for r in session.loader.trainTraces["daq_run"].values})
    elif session.state.get("files"):
        from .loaders import runNumber
        runs = sorted({runNumber(f) for f in session.state["files"]} - {None})
    label = "-".join(str(r) for r in runs[:4]) + ("+" if len(runs) > 4 else "") if runs else "session"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RESULTS_DIR / f"{label}_{stamp}"


def _saveFigure(fig, base, formats=("png", "pdf")):
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", dpi=300 if fmt == "png" else None, bbox_inches="tight")


def exportPlots(session, folder, views=None, polarizationFigure=None, log=print):
    """
    Write plots/ into a session folder.

    views : dict name -> pyqtgraph GraphicsScene-holding widget (exported as PNG, what the user sees)
    polarizationFigure : matplotlib Figure of the Polarization tab
    """
    plots = Path(folder) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written = []

    pf = session.peakFinder
    if pf is not None and session.results is not None and len(session.results):
        try:
            first = session.results.iloc[0]
            pf.plot(trainIndex=int(first["trainId"]), pulseIndex=int(first["pulseId"]), savename=False,
                    energyCalib=session.energyCalib if session.energyCalib is not None and len(session.energyCalib)
                    else None)
            _saveFigure(pf.fig, plots / "traces_peaks")
            plt.close(pf.fig)
            written.append("traces_peaks")
        except Exception as e:
            log(f"traces plot skipped: {e}")

    cal = session.calibrateObj
    if cal is not None:
        if isinstance(cal.energyParam, pd.DataFrame) and len(cal.energyParam):
            try:
                cal.plotEnergy(peakNo=session.state["calibration"]["peakNo"], savename=False, show=False)
                _saveFigure(cal.fig, plots / "energy_calibration")
                plt.close(cal.fig)
                written.append("energy_calibration")
            except Exception as e:
                log(f"energy calibration plot skipped: {e}")
        if isinstance(cal.transmissionParam, pd.DataFrame) and len(cal.transmissionParam):
            try:
                cal.plotTransmission(savename=False)
                _saveFigure(cal.fig, plots / "transmission")
                plt.close(cal.fig)
                written.append("transmission")
            except Exception as e:
                log(f"transmission plot skipped: {e}")

    if polarizationFigure is not None and session.polarization is not None:
        _saveFigure(polarizationFigure, plots / "polarization")
        written.append("polarization")

    if views:
        import pyqtgraph.exporters as exporters
        for name, widget in views.items():
            try:
                scene = widget.scene() if hasattr(widget, "scene") else widget.view.scene()
                exporter = exporters.ImageExporter(scene)
                exporter.parameters()["width"] = max(int(scene.sceneRect().width() * 2), 800)
                exporter.export(str(plots / f"view_{name}.png"))
                written.append(f"view_{name}")
            except Exception as e:
                log(f"view {name} not exported: {e}")
    return written


def exportSession(session, folder=None, views=None, polarizationFigure=None, processed=True, log=print):
    folder = Path(folder) if folder else defaultSessionFolder(session)
    session.save(folder, processed=processed)
    written = exportPlots(session, folder, views=views, polarizationFigure=polarizationFigure, log=log)
    log(f"Session saved to {folder} ({len(written)} plots)")
    return folder
