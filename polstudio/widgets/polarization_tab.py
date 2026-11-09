"""
Polarization fit (Fitter.pol, embedded matplotlib canvas), the shared polar heatmap (the Heatmap tab's panel,
moved here while this tab is shown), a per-run overview (every run fitted on its own) and the results table.
"""
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .heatmap_tab import PanelHost
from .pandas_model import PandasModel, makeTableView
from .params import ParamForm
from .plot_grid import TOL

TABLE_VIEW_COLUMNS = ["timestamp", "runs", "Plin", "Plin error", "phi", "phi error", "photon energy", "nDetectors",
                      "useTransmission", "config", "calibration"]

POL_TOOLTIPS = {
    "peakNo": "Peak used for the fit",
    "beta": "β₂ (fixed unless fitBeta)",
    "setPlin": "Fix the linear polarization (empty = fit)",
    "setPhi": "Fix the polarization angle in degrees (empty = fit)",
    "fitBeta": "Also fit β₂",
    "intMethod": "Results column used as intensity",
    "useTransmission": "Correct intensities with the transmission calibration",
    "Ebind": "Binding energy used to match the transmission calibration",
}

OVERVIEW_X = {"run": "run", "photon energy": "photon energy (eV)"}


class PolarizationTab(QWidget):
    fitRequested = pyqtSignal(dict)
    fitRunsRequested = pyqtSignal(dict)
    chooseTableRequested = pyqtSignal()
    appendRequested = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        side = QWidget()
        sideLayout = QVBoxLayout(side)
        box = QGroupBox("Fitter.pol")
        boxLayout = QVBoxLayout(box)
        self.form = ParamForm(dict(self.session.state["polarization"]), tooltips=POL_TOOLTIPS)
        boxLayout.addWidget(self.form)
        self.fitButton = QPushButton("Fit polarization")
        self.fitButton.setToolTip("Fit all peak results together")
        self.fitButton.clicked.connect(lambda: self.fitRequested.emit(self.params()))
        boxLayout.addWidget(self.fitButton)
        self.resultLabel = QLabel("")
        self.resultLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.resultLabel.setWordWrap(True)
        boxLayout.addWidget(self.resultLabel)
        sideLayout.addWidget(box)

        runs = QGroupBox("Per-run overview")
        runsLayout = QFormLayout(runs)
        self.fitRunsButton = QPushButton("Fit every run")
        self.fitRunsButton.setToolTip("Fit the peak results of every run on its own with the settings above "
                                      "(use 'Process selected runs separately' so the runs are not stacked together)")
        self.fitRunsButton.clicked.connect(lambda: self.fitRunsRequested.emit(self.params()))
        runsLayout.addRow(self.fitRunsButton)
        self.overviewXCombo = QComboBox()
        self.overviewXCombo.addItems(list(OVERVIEW_X))
        self.overviewXCombo.currentIndexChanged.connect(self._drawOverview)
        runsLayout.addRow("x axis", self.overviewXCombo)
        self.overviewLabel = QLabel("")
        self.overviewLabel.setWordWrap(True)
        runsLayout.addRow(self.overviewLabel)
        sideLayout.addWidget(runs)

        table = QGroupBox("Results table")
        tableLayout = QVBoxLayout(table)
        self.tableLabel = QLabel("")
        self.tableLabel.setWordWrap(True)
        tableLayout.addWidget(self.tableLabel)
        row = QHBoxLayout()
        chooseButton = QPushButton("Choose file…")
        chooseButton.clicked.connect(self.chooseTableRequested)
        self.appendButton = QPushButton("Append current fit")
        self.appendButton.clicked.connect(self.appendRequested)
        row.addWidget(chooseButton)
        row.addWidget(self.appendButton)
        tableLayout.addLayout(row)
        self.autoCheck = QCheckBox("append every fit")
        self.autoCheck.setToolTip("Append a row after every polarization fit, e.g. after selecting another run and "
                                  "pressing 'Update → Polarization'. Config and calibration are stored once and "
                                  "referenced from the rows.")
        self.autoCheck.toggled.connect(lambda on: self.session.state["resultsTable"].__setitem__("auto", on))
        tableLayout.addWidget(self.autoCheck)
        sideLayout.addWidget(table)
        sideLayout.addStretch(1)
        splitter.addWidget(side)

        right = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)
        plots = QWidget()
        plotsLayout = QVBoxLayout(plots)
        plotsLayout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        plotsLayout.addWidget(NavigationToolbar2QT(self.canvas, self))
        plotsLayout.addWidget(self.canvas, stretch=1)
        self.polAx = self.figure.add_subplot(1, 1, 1, projection="polar")
        top.addWidget(plots)
        # The Heatmap tab's panel is moved in here while this tab is shown (same view and settings)
        self.heatmapHost = PanelHost()
        top.addWidget(self.heatmapHost)
        top.setSizes([450, 650])
        right.addWidget(top)

        bottom = QTabWidget()
        overview = QSplitter(Qt.Orientation.Horizontal)
        self.overviewView = pg.GraphicsLayoutWidget()
        self.plinPlot = self.overviewView.addPlot(row=0, col=0)
        self.plinPlot.setLabel("left", "P_lin")
        self.phiPlot = self.overviewView.addPlot(row=1, col=0)
        self.phiPlot.setLabel("left", "φ (°)")
        self.phiPlot.setXLink(self.plinPlot)
        for plot in (self.plinPlot, self.phiPlot):
            plot.showGrid(x=True, y=True, alpha=0.3)
        overview.addWidget(self.overviewView)
        self.overviewModel = PandasModel()
        overview.addWidget(makeTableView(self.overviewModel))
        overview.setSizes([550, 550])
        bottom.addTab(overview, "Per-run overview")
        self.tableModel = PandasModel()
        bottom.addTab(makeTableView(self.tableModel), "Results table")
        right.addWidget(bottom)
        right.setSizes([550, 330])
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1100])
        self.reloadTableSettings()
        self.showRunOverview(self.session.runPolarization)

    def params(self):
        params = self.form.params()
        self.session.state["polarization"] = dict(params)
        return params

    def reloadSettings(self):
        self.form.setParams(dict(self.session.state["polarization"]))
        self.reloadTableSettings()
        self.showRunOverview(self.session.runPolarization)

    def reloadTableSettings(self):
        settings = self.session.state["resultsTable"]
        self.autoCheck.blockSignals(True)
        self.autoCheck.setChecked(bool(settings.get("auto")))
        self.autoCheck.blockSignals(False)
        self.showTable()

    def showTable(self, message=None):
        """Show the table file and its rows (newest first)."""
        from ..results_table import ResultsTable
        path = self.session.state["resultsTable"].get("file")
        if not path:
            self.tableLabel.setText("No table file chosen")
            self.tableModel.setDataFrame(None)
            return
        table = ResultsTable(path).read()
        columns = [c for c in TABLE_VIEW_COLUMNS if c in table.columns]
        self.tableModel.setDataFrame(table[columns].iloc[::-1].reset_index(drop=True))
        text = f"{Path(path).name}: {len(table)} rows"
        self.tableLabel.setText(text + (f"\n{message}" if message else ""))
        self.tableLabel.setToolTip(str(path))

    # ------------------------------------------------------------------ per-run overview
    def showRunOverview(self, overview):
        self.overviewModel.setDataFrame(overview)
        self.refreshOverviewStatus()
        self._drawOverview()

    def refreshOverviewStatus(self):
        overview = self.session.runPolarization
        if overview is None:
            self.overviewLabel.setText("")
            return
        failed = int(overview["error"].notna().sum()) if "error" in overview else 0
        text = f"{len(overview)} runs fitted" + (f", {failed} failed (see the table)" if failed else "")
        if not self.session.runPolarizationIsCurrent():
            text = "Outdated – the peak results or settings changed; fit again\n" + text
        self.overviewLabel.setText(text)

    def _drawOverview(self):
        for plot in (self.plinPlot, self.phiPlot):
            plot.clear()
        overview = self.session.runPolarization
        if overview is None or not len(overview) or "Plin" not in overview:
            return
        xKey = self.overviewXCombo.currentText()
        xKey = xKey if xKey in overview else "run"
        self.phiPlot.setLabel("bottom", OVERVIEW_X[xKey])
        x = overview[xKey].to_numpy(dtype=float)
        for plot, key, color in ((self.plinPlot, "Plin", TOL["blue"]), (self.phiPlot, "phi", TOL["red"])):
            y = overview[key].to_numpy(dtype=float)
            err = overview.get(f"{key} error")
            err = np.nan_to_num(err.to_numpy(dtype=float)) if err is not None else np.zeros_like(y)
            ok = np.isfinite(x) & np.isfinite(y)
            plot.addItem(pg.ErrorBarItem(x=x[ok], y=y[ok], height=2 * err[ok], beam=0, pen=pg.mkPen(color)))
            plot.plot(x[ok], y[ok], pen=None, symbol="o", symbolSize=6, symbolBrush=color, symbolPen=None)

    # ------------------------------------------------------------------ fit
    def onProcessedChanged(self):
        self.refreshOverviewStatus()

    def markOutdated(self):
        if self.resultLabel.text() and not self.resultLabel.text().startswith("Outdated"):
            self.resultLabel.setText("Outdated – the channel selection or inputs changed; fit again\n\n"
                                     + self.resultLabel.text())
        self.refreshOverviewStatus()

    def resetPolAxis(self):
        self.figure.delaxes(self.polAx)
        self.polAx = self.figure.add_subplot(1, 1, 1, projection="polar")
        return self.polAx

    def showFit(self, polarization):
        """Called after Fitter.pol drew into self.polAx."""
        p, e = polarization["params"], polarization["errors"]
        lines = [f"P_lin = {p['Plin']:.3f} ± {e['Plin']:.3f}", f"φ = {p['phi']:.1f} ± {e['phi']:.1f}°",
                 f"scale = {p['scale']:.4g}"]
        if "beta2" in p:
            lines.append(f"β₂ = {p['beta2']:.3f} ± {e['beta2']:.3f}")
        if not polarization["settings"].get("useTransmission"):
            lines.append("(no transmission correction)")
        else:
            calRuns = set(self.session.calibrationMeta.get("runs") or [])
            results = self.session.results
            fitRuns = set(results["daq_run"].dropna().astype(int)) if "daq_run" in results else set()
            if calRuns & fitRuns:
                lines.append(f"Note: transmission was derived from runs {sorted(calRuns & fitRuns)}, "
                             "which are also fitted here – the result reproduces the reference distribution.")
        self.resultLabel.setText("\n".join(lines))
        # Fitter.pol labels the fit with a LaTeX table meant for usetex; show plain text instead
        lineArtists = self.polAx.get_lines()
        if lineArtists:
            lineArtists[-1].set_label(", ".join(lines[:2]))
        self.polAx.legend(loc="upper left", bbox_to_anchor=(-0.1, 1.15), fontsize=8)
        self.polAx.set_title("Polarization fit")
        self.canvas.draw_idle()
