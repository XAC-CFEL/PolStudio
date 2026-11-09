"""
Drive the PolStudio window through a full analysis without user input and save screenshots.

    python scripts/smoke_gui.py ../2024_hdf5 <screenshot folder> [--offscreen]
"""
import os
import sys
import time
from pathlib import Path

if "--offscreen" in sys.argv:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.argv.remove("--offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pyqtgraph as pg  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from polstudio.main_window import MainWindow  # noqa: E402
from polstudio.session import AnalysisSession  # noqa: E402


def wait(window, app, timeout=600):
    t0 = time.time()
    app.processEvents()
    while window._tasks:
        app.processEvents()
        time.sleep(0.05)
        if time.time() - t0 > timeout:
            raise TimeoutError("task did not finish")
    for _ in range(5):
        app.processEvents()


def shot(window, app, outDir, name, tab=None):
    if tab is not None:
        window.tabs.setCurrentWidget(tab)
    for _ in range(10):
        app.processEvents()
    window.grab().save(str(outDir / f"{name}.png"))


def main(dataDir, outDir):
    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])
    pg.setConfigOptions(background="w", foreground="k")
    window = MainWindow(AnalysisSession())
    window.error = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))  # fail loudly
    window.show()

    window.openFolder(dataDir)
    wait(window, app)
    files = window.dataPanel.selectedFiles()
    assert window.dataPanel.tree.topLevelItemCount() >= 1, "no runs listed"
    # tick the first two runs' files
    from PyQt6.QtCore import Qt
    for i in range(min(2, window.dataPanel.tree.topLevelItemCount())):
        window.dataPanel.tree.topLevelItem(i).setCheckState(0, Qt.CheckState.Checked)
    window.dataPanel.trainEdits["trainStop"].setText("60")
    selection = window.dataPanel.selection()
    print("selection:", [Path(f).name for f in selection["files"]], selection["trainStop"])
    window.loadSelection(selection)
    wait(window, app)
    shot(window, app, outDir, "1_inspect", window.inspectTab)

    window.inspectTab.periodSpin.setValue(1794.6)
    window.inspectTab.navigator.setPosition(5)
    shot(window, app, outDir, "1b_inspect_train5", window.inspectTab)

    stack = next(s for s in window.session.state["steps"] if s["name"] == "stack")
    stack["params"]["trainStackSize"] = 10
    window.preprocessTab.loadSteps()
    window.preprocessTab.nTrainsSpin.setValue(10)
    window.preprocessTab._requestPreview()
    wait(window, app)
    shot(window, app, outDir, "2_preview", window.preprocessTab)

    window.applySteps()
    wait(window, app)
    window.peaksTab.findRequested.emit(window.peaksTab.params())
    wait(window, app)
    assert window.session.results is not None and len(window.session.results), "no peaks"
    shot(window, app, outDir, "3_peaks", window.peaksTab)
    window.peaksTab.axisCombo.setCurrentIndex(1)
    shot(window, app, outDir, "3b_peaks_ns", window.peaksTab)

    window.computeTransmission()
    wait(window, app)
    shot(window, app, outDir, "4_calibration", window.calibrationTab)

    window.fitPolarization(window.polarizationTab.params())
    window.fitRuns(window.polarizationTab.params())
    wait(window, app)
    assert window.session.runPolarization is not None and len(window.session.runPolarization), "no per-run fits"
    window.heatmapTab.transmissionCheck.setChecked(True)
    shot(window, app, outDir, "5_polarization", window.polarizationTab)
    assert window.heatmapTab.parent() is window.polarizationTab.heatmapHost, "heatmap not shown on Polarization"

    sessionDir = outDir / "session"
    window._save(str(sessionDir))
    print("session files:", sorted(p.name for p in sessionDir.iterdir()))
    print("plots:", sorted(p.name for p in (sessionDir / "plots").iterdir()))

    window.openSession(str(sessionDir))
    wait(window, app)
    assert window.session.results is not None and len(window.session.results)
    shot(window, app, outDir, "6_reopened", window.peaksTab)
    print("GUI SMOKE TEST PASSED")
    window.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
