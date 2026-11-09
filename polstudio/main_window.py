from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtGui import QAction, QKeySequence, QTextCursor
from PyQt6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QMainWindow, QMessageBox, QPlainTextEdit,
                             QProgressBar, QTabWidget)

from . import RESULTS_DIR, __version__
from .export import defaultSessionFolder, exportSession
from .session import LEVEL_LOADED, LEVEL_POLARIZATION, LEVEL_PROCESSED, LEVEL_RESULTS, AnalysisSession
from .widgets.calibration_tab import CalibrationTab
from .widgets.data_panel import DataPanel
from .widgets.inspect_tab import InspectTab
from .widgets.peaks_tab import PeaksTab
from .widgets.heatmap_tab import HeatmapTab
from .widgets.polarization_tab import PolarizationTab
from .widgets.preprocess_tab import PreprocessTab
from .workers import Task


class MainWindow(QMainWindow):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.pool = QThreadPool.globalInstance()
        self._tasks = set()
        self.setWindowTitle(f"PolStudio {__version__}")
        self.resize(1500, 950)

        self.dataPanel = DataPanel(session)
        dock = QDockWidget("Data", self)
        dock.setObjectName("dataDock")
        dock.setWidget(self.dataPanel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.dataDock = dock

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        logDock = QDockWidget("Log", self)
        logDock.setObjectName("logDock")
        logDock.setWidget(self.log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, logDock)
        self.logDock = logDock

        self.tabs = QTabWidget()
        self.inspectTab = InspectTab(session)
        self.preprocessTab = PreprocessTab(session)
        self.peaksTab = PeaksTab(session)
        self.calibrationTab = CalibrationTab(session)
        self.heatmapTab = HeatmapTab(session)
        self.polarizationTab = PolarizationTab(session)
        for tab, name in ((self.inspectTab, "Inspect"), (self.preprocessTab, "Preprocess"),
                          (self.peaksTab, "Peaks"), (self.heatmapTab, "Heatmap"),
                          (self.calibrationTab, "Calibration"), (self.polarizationTab, "Polarization")):
            self.tabs.addTab(tab, name)
        self.setCentralWidget(self.tabs)
        self.resizeDocks([self.dataDock], [430], Qt.Orientation.Horizontal)
        self.resizeDocks([self.logDock], [130], Qt.Orientation.Vertical)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(200)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

        self._buildMenus()
        self._connect()

    # ------------------------------------------------------------------ setup
    def _buildMenus(self):
        fileMenu = self.menuBar().addMenu("&File")
        self._addAction(fileMenu, "Open data folder…", self.dataPanel.chooseFolder, QKeySequence.StandardKey.Open)
        fileMenu.addSeparator()
        self._addAction(fileMenu, "Open session…", self.openSessionDialog)
        self._addAction(fileMenu, "Save session…", self.saveSessionDialog, QKeySequence.StandardKey.Save)
        self._addAction(fileMenu, "Quick save session (results/…)", self.quickSaveSession)
        fileMenu.addSeparator()
        self._addAction(fileMenu, "Load config.yaml…", self.loadConfigDialog)
        self._addAction(fileMenu, "Save config.yaml…", self.saveConfigDialog)
        fileMenu.addSeparator()
        self._addAction(fileMenu, "Quit", self.close, QKeySequence.StandardKey.Quit)
        calMenu = self.menuBar().addMenu("&Calibration")
        self._addAction(calMenu, "Load calibration…", self.loadCalibrationDialog)
        self._addAction(calMenu, "Save calibration…", self.saveCalibrationDialog)
        viewMenu = self.menuBar().addMenu("&View")
        viewMenu.addAction(self.dataDock.toggleViewAction())
        viewMenu.addAction(self.logDock.toggleViewAction())

    def _addAction(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut is not None:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    def _connect(self):
        self.dataPanel.folderRequested.connect(self.openFolder)
        self.dataPanel.loadRequested.connect(self.loadSelection)
        self.dataPanel.updateRequested.connect(self.updateSelection)
        self.tabs.currentChanged.connect(lambda _: self.dataPanel.setUpdateTarget(self.tabs.tabText(self.tabs.currentIndex())))
        self.dataPanel.setUpdateTarget(self.tabs.tabText(self.tabs.currentIndex()))
        self.inspectTab.recutRequested.connect(self.recut)
        self.inspectTab.trainChanged.connect(self.preprocessTab.setStartTrain)
        self.preprocessTab.previewRequested.connect(self.preview)
        self.preprocessTab.applyRequested.connect(self.applySteps)
        self.peaksTab.findRequested.connect(self.findPeaks)
        self.calibrationTab.computeEnergyRequested.connect(self.computeEnergy)
        self.calibrationTab.computeTransmissionRequested.connect(self.computeTransmission)
        self.calibrationTab.loadRequested.connect(self.loadCalibrationDialog)
        self.calibrationTab.saveRequested.connect(self.saveCalibrationDialog)
        self.calibrationTab.photonEnergiesChanged.connect(self.onResultsChanged)
        self.polarizationTab.fitRequested.connect(self.fitPolarization)
        self.polarizationTab.chooseTableRequested.connect(self.chooseResultsTable)
        self.polarizationTab.appendRequested.connect(lambda: self.appendToResultsTable(auto=False))

    # ------------------------------------------------------------------ helpers
    def appendLog(self, text):
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def error(self, message):
        print(f"ERROR: {message}")
        QMessageBox.warning(self, "PolStudio", message)

    def setBusy(self, busy, message=""):
        self.tabs.setEnabled(not busy)
        self.dataPanel.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.statusBar().showMessage(message if busy else "Ready")
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def runTask(self, fn, onDone, message):
        """Run fn in the thread pool with the window locked; onDone(result) runs in the GUI thread."""
        if self._tasks:
            self.statusBar().showMessage("Busy – please wait", 3000)
            return
        task = Task(fn)
        self._tasks.add(task)

        def finished(result):
            self._tasks.discard(task)
            self.setBusy(False)
            try:
                onDone(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.error(f"{type(e).__name__}: {e}")

        def failed(message):
            self._tasks.discard(task)
            self.setBusy(False)
            self.error(message)

        task.signals.finished.connect(finished)
        task.signals.error.connect(failed)
        self.setBusy(True, message)
        print(f"> {message}")
        self.pool.start(task)

    # ------------------------------------------------------------------ data
    def openFolder(self, folder):
        self.runTask(lambda: self.session.scan(folder), lambda overview: self.onScanned(overview, folder),
                     f"Scanning {folder}")

    def onScanned(self, overview, folder):
        self.dataPanel.setOverview(overview, folder)
        self.inspectTab.updateFormat()

    def loadSelection(self, selection):
        self.inspectTab.applySettingsToConfig()
        self.runTask(lambda: self.session.load(**selection), lambda _: self.onDataLoaded(),
                     f"Loading {len(selection['files'])} file(s)")

    def _tabLevel(self):
        """How far 'Update' runs the pipeline for the selected tab."""
        tab = self.tabs.currentWidget()
        if tab is self.polarizationTab:
            return LEVEL_POLARIZATION
        if tab in (self.peaksTab, self.calibrationTab):
            return LEVEL_RESULTS
        if tab in (self.preprocessTab, self.heatmapTab):
            return LEVEL_PROCESSED
        return LEVEL_LOADED

    def updateSelection(self, selection):
        """Run the pipeline for the selection up to the selected tab, recomputing only what changed."""
        if not selection["files"]:
            return self.error("Select files first")
        # Settings as currently shown in the tabs
        self.inspectTab.applySettingsToConfig()
        self.preprocessTab.steps()
        self.peaksTab.params()
        self.polarizationTab.params()
        level = self._tabLevel()
        tab = self.tabs.tabText(self.tabs.currentIndex())
        self.runTask(lambda: self.session.update(level=level, **selection), self.onUpdated,
                     f"Updating up to {tab}")

    def onUpdated(self, report):
        if report["reloaded"]:
            self.onDataLoaded()
        else:
            self.onProcessedChanged()
            self.onResultsChanged()
            self._refreshInfo()
        self.preprocessTab.restrictPreview(self.session.raw["detector"].values)
        if report["fitPolarization"]:
            self.fitPolarization(self.polarizationTab.params())
        elif self.session.polarization is None:
            self.polarizationTab.markOutdated()
        self.statusBar().showMessage("Update: " + "; ".join(report["actions"]), 10000)

    def _refreshInfo(self):
        s = self.session
        if s.raw is None:
            return
        raw = s.raw
        if s.splitsPulses:
            text = (f"Loaded {len(s.trainIds())} trains, {raw.sizes['pulse']} pulses, {raw.sizes['detector']} "
                    f"detectors, {raw.sizes['sample']} samples/pulse.\nPulse starts: {s.loader.pulseStarts.tolist()}")
        else:
            text = (f"Loaded {len(s.trainIds())} runs, {raw.sizes['pulse']} scan points, {raw.sizes['detector']} "
                    f"detectors, {raw.sizes['sample']} samples/point (.nxs).")
        status = s.productStatus()
        self.dataPanel.setLoadedInfo(text + (f"\n{status}" if status else ""))

    def recut(self):
        if self.session.loader is None:
            self.inspectTab.applySettingsToConfig()
            return
        self.inspectTab.applySettingsToConfig()
        self.runTask(self.session.refreshPulseSplit, lambda _: self.onDataLoaded(),
                     "Reloading with the new pulse split / baseline")

    def onDataLoaded(self):
        s = self.session
        self._refreshInfo()
        self.inspectTab.onDataLoaded()
        self.calibrationTab.onDataLoaded()
        self.onProcessedChanged()
        self.onResultsChanged()
        print(f"Loaded runs {sorted(set(s.loader.trainTraces['daq_run'].values.tolist()))}")

    # ------------------------------------------------------------------ preprocessing / peaks
    def preview(self, trainIds):
        if self.session.raw is None:
            return self.error("Load data first")
        if not trainIds:
            return self.error("No trains selected for the preview")
        self.runTask(lambda: self.session.preview(trainIds), lambda res: self.preprocessTab.showPreview(*res),
                     f"Preview on {len(trainIds)} trains")

    def applySteps(self):
        if self.session.raw is None:
            return self.error("Load data first")
        self.preprocessTab.steps()
        self.runTask(self.session.applySteps, lambda _: self.onProcessedChanged(), "Preprocessing all trains")

    def onProcessedChanged(self):
        p = self.session.processed
        if p is not None:
            self.preprocessTab.setProcessedInfo(f"Processed data: {dict(p.sizes)}")
        self.peaksTab.onProcessedChanged()
        self.heatmapTab.onProcessedChanged()
        self._refreshInfo()
        self.polarizationTab.onProcessedChanged()

    def findPeaks(self, params):
        if self.session.processed is None:
            return self.error("Apply the preprocessing first (Preprocess tab)")
        self.runTask(lambda: self.session.findPeaks(params), lambda _: self.onResultsChanged(), "Finding peaks")

    def onResultsChanged(self):
        self.peaksTab.onResultsChanged()
        self.calibrationTab.onResultsChanged()
        self.heatmapTab.redraw()
        self._refreshInfo()

    # ------------------------------------------------------------------ calibration / polarization
    def computeEnergy(self):
        self.calibrationTab.params()
        self.runTask(self.session.computeEnergyCalibration, lambda _: self.onCalibrationChanged(),
                     "Energy calibration")

    def computeTransmission(self):
        self.calibrationTab.params()
        self.runTask(self.session.computeTransmission, lambda _: self.onCalibrationChanged(),
                     "Transmission calibration")

    def onCalibrationChanged(self, warnings=None):
        self.calibrationTab.onCalibrationChanged(warnings)
        self.peaksTab.redraw()

    def _calibrationDir(self):
        folder = RESULTS_DIR / "calibrations"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def loadCalibrationDialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load calibration", str(self._calibrationDir()),
                                              "Calibration (*.json)")
        if not path:
            return
        try:
            warnings = self.session.loadCalibration(path)
        except Exception as e:
            return self.error(f"Could not load {path}: {e}")
        print(f"Loaded calibration {path}")
        for w in warnings:
            print(f"Warning: {w}")
        self.onCalibrationChanged(warnings)

    def saveCalibrationDialog(self):
        if self.session.energyCalib is None and self.session.transmissionCalib is None:
            return self.error("No calibration to save – compute or load one first")
        default = self._calibrationDir() / (defaultSessionFolder(self.session).name + ".json")
        path, _ = QFileDialog.getSaveFileName(self, "Save calibration", str(default), "Calibration (*.json)")
        if not path:
            return
        self.session.saveCalibration(path)
        print(f"Saved calibration to {path}")
        self.onCalibrationChanged()

    def fitPolarization(self, params):
        # Runs in the GUI thread: the fit is fast and draws into the embedded matplotlib figure
        try:
            ax = self.polarizationTab.resetPolAxis()
            result = self.session.fitPolarization(params, ax=ax)
        except Exception as e:
            self.polarizationTab.canvas.draw_idle()
            return self.error(f"Polarization fit failed: {type(e).__name__}: {e}")
        print(f"Polarization: {result['params']} ± {result['errors']}")
        self.polarizationTab.showFit(result)
        if self.session.state["resultsTable"].get("auto"):
            self.appendToResultsTable(auto=True)

    def chooseResultsTable(self):
        current = self.session.state["resultsTable"].get("file") or str(RESULTS_DIR / "polarization_results.csv")
        Path(current).parent.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Polarization results table (new or existing)", current,
                                              "CSV table (*.csv)", options=QFileDialog.Option.DontConfirmOverwrite)
        if path:
            self.session.state["resultsTable"]["file"] = path
            self.session.lastTableInputs = None
            self.polarizationTab.showTable()
            print(f"Results table: {path}")

    def appendToResultsTable(self, auto=False):
        """Append the current fit; auto=True skips a fit that is already in the table."""
        from .results_table import ResultsTable
        s = self.session
        if s.polarization is None:
            return None if auto else self.error("No polarization fit to append – fit first")
        if not s.state["resultsTable"].get("file"):
            if auto:
                print("Results table: no file chosen, fit not appended")
                return None
            self.chooseResultsTable()
            if not s.state["resultsTable"].get("file"):
                return None
        if auto and s.lastTableInputs == s.polarizationInputs:
            return None
        try:
            row = ResultsTable(s.state["resultsTable"]["file"]).append(s)
        except Exception as e:
            return self.error(f"Could not append to the results table: {type(e).__name__}: {e}")
        s.lastTableInputs = s.polarizationInputs
        message = f"appended run {row['runs']}: P_lin = {row['Plin']:.3f}, φ = {row['phi']:.1f}°"
        print(f"Results table: {message}")
        self.polarizationTab.showTable(message)
        return row

    # ------------------------------------------------------------------ sessions / config
    def _views(self):
        return {"inspect": self.inspectTab.view, "preprocess": self.preprocessTab.grid.view,
                "peaks": self.peaksTab.grid.view, "peak_histogram": self.peaksTab.histPlot,
                "energy_calibration": self.calibrationTab.energyPlot,
                "transmission": self.calibrationTab.transPlot,
                "heatmap": self.heatmapTab.currentView()}

    def _save(self, folder):
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            folder = exportSession(self.session, folder, views=self._views(),
                                   polarizationFigure=self.polarizationTab.figure)
        except Exception as e:
            return self.error(f"Saving failed: {type(e).__name__}: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Saved session to {folder}", 10000)

    def quickSaveSession(self):
        self._save(None)

    def saveSessionDialog(self):
        default = defaultSessionFolder(self.session)
        default.parent.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Save session as folder", str(default), "Session folder (*)")
        if path:
            self._save(path)

    def openSessionDialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Open session folder", str(RESULTS_DIR))
        if not folder:
            return
        if not (Path(folder) / "config.yaml").exists():
            return self.error("Not a session folder (config.yaml missing)")
        self.openSession(folder)

    def openSession(self, folder):
        try:
            session = AnalysisSession.open(folder)
        except Exception as e:
            return self.error(f"Could not open session: {type(e).__name__}: {e}")
        self.setSession(session)
        print(f"Opened session {folder}. Click 'Load selection' to reload the raw data.")

    def loadConfigDialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", str(RESULTS_DIR), "YAML (*.yaml *.yml)")
        if path:
            self.setSession(AnalysisSession(path))
            print(f"Loaded config {path}")

    def saveConfigDialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", str(RESULTS_DIR / "config.yaml"),
                                              "YAML (*.yaml)")
        if path:
            self.preprocessTab.steps()
            self.session.saveConfig(path)
            print(f"Saved config to {path}")

    def setSession(self, session):
        if self.session is not None and self.session is not session:
            self.session.close()
        self.session = session
        for widget in (self.dataPanel, self.inspectTab, self.preprocessTab, self.peaksTab, self.heatmapTab,
                       self.calibrationTab, self.polarizationTab):
            widget.session = session
        self.inspectTab.loadSplitSettings(session.loaderConfig)
        self.inspectTab._updateStartsLabel()
        self.inspectTab.updateFormat()
        self.dataPanel.workersSpin.setValue(int(session.state.get("workers") or 0))
        self.dataPanel.separateRunsCheck.setChecked(bool(session.state.get("separateRuns")))
        for key, edit in self.dataPanel.trainEdits.items():
            value = session.state.get(key)
            edit.setText("" if value is None else str(value))
        self.preprocessTab.loadSteps()
        self.peaksTab.loadParams()
        self.calibrationTab.reloadSettings()
        self.polarizationTab.reloadSettings()
        self.onProcessedChanged()
        self.onResultsChanged()
        self.onCalibrationChanged()
        if session.polarization:
            p = session.polarization
            self.polarizationTab.resultLabel.setText(
                "\n".join(f"{k} = {v:.4g} ± {p['errors'].get(k, np.nan):.2g}" for k, v in p["params"].items()))
        folder = session.state.get("dataFolder")
        if folder and Path(folder).exists():
            self.openFolder(folder)

    def closeEvent(self, event):
        if self.session is not None:
            self.session.close()
        super().closeEvent(event)
