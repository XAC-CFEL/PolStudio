"""Photon energies per run, energy / transmission calibration: compute, load and save."""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QTabWidget,
                             QVBoxLayout, QWidget)

from .pandas_model import PandasModel, makeTableView
from .params import ParamForm
from .plot_grid import CYCLE, TOL

CAL_TOOLTIPS = {
    "bindingEnergy": "Subtracted from the photon energy (Calibrate(bindingEnergy=...))",
    "peakNo": "Peak used for the energy calibration",
    "transmissionPeakNo": "Peak used for the transmission coefficients",
    "setBeta": "β₂ of the reference distribution", "setPhi": "φ (deg) of the reference distribution",
    "setPlin": "Linear polarization of the reference distribution",
    "intMethod": "Results column used as intensity (height, fwhm area, ...)",
}


class CalibrationTab(QWidget):
    computeEnergyRequested = pyqtSignal()
    computeTransmissionRequested = pyqtSignal()
    loadRequested = pyqtSignal()
    saveRequested = pyqtSignal()
    photonEnergiesChanged = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        side = QWidget()
        sideLayout = QVBoxLayout(side)
        energies = QGroupBox("Photon energy per run (eV)")
        energiesLayout = QVBoxLayout(energies)
        energiesLayout.addWidget(QLabel("Values from the files come from the wavelength channel (.h5) or the "
                                        "monochromator energy (.nxs); type an override to correct them."))
        self.energyModel = PandasModel(editableColumns=("override",))
        self.energyTable = makeTableView(self.energyModel)
        self.energyTable.setMaximumHeight(180)
        energiesLayout.addWidget(self.energyTable)
        applyEnergies = QPushButton("Apply photon energies to results")
        applyEnergies.clicked.connect(self._applyEnergies)
        energiesLayout.addWidget(applyEnergies)
        sideLayout.addWidget(energies)

        settings = QGroupBox("Calibration settings")
        settingsLayout = QVBoxLayout(settings)
        self.form = ParamForm(self._formParams(), tooltips=CAL_TOOLTIPS)
        self.form.changed.connect(self.params)
        settingsLayout.addWidget(self.form)
        row = QHBoxLayout()
        self.energyButton = QPushButton("Compute energy calibration")
        self.energyButton.clicked.connect(self.computeEnergyRequested)
        self.transButton = QPushButton("Compute transmission")
        self.transButton.clicked.connect(self.computeTransmissionRequested)
        row.addWidget(self.energyButton)
        row.addWidget(self.transButton)
        settingsLayout.addLayout(row)
        row = QHBoxLayout()
        loadButton = QPushButton("Load calibration…")
        loadButton.clicked.connect(self.loadRequested)
        saveButton = QPushButton("Save calibration…")
        saveButton.clicked.connect(self.saveRequested)
        row.addWidget(loadButton)
        row.addWidget(saveButton)
        settingsLayout.addLayout(row)
        self.infoLabel = QLabel("No calibration")
        self.infoLabel.setWordWrap(True)
        self.infoLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        settingsLayout.addWidget(self.infoLabel)
        sideLayout.addWidget(settings)
        sideLayout.addStretch(1)
        splitter.addWidget(side)

        right = QSplitter(Qt.Orientation.Vertical)
        plotBox = QWidget()
        plotLayout = QVBoxLayout(plotBox)
        row = QHBoxLayout()
        row.addWidget(QLabel("detector"))
        self.detectorCombo = QComboBox()
        self.detectorCombo.currentIndexChanged.connect(self.redraw)
        row.addWidget(self.detectorCombo)
        row.addStretch(1)
        plotLayout.addLayout(row)
        self.energyPlot = pg.PlotWidget(title="Energy calibration: photon energy − E_bind vs peak position")
        self.energyPlot.setLabel("bottom", "pos (sample)")
        self.energyPlot.setLabel("left", "kinetic energy (eV)")
        self.energyPlot.showGrid(x=True, y=True, alpha=0.3)
        plotLayout.addWidget(self.energyPlot, stretch=1)
        self.transPlot = pg.PlotWidget(title="Transmission coefficient per detector")
        self.transPlot.setLabel("bottom", "detector")
        self.transPlot.addLegend()
        plotLayout.addWidget(self.transPlot, stretch=1)
        right.addWidget(plotBox)
        tables = QTabWidget()
        self.energyParamModel = PandasModel()
        tables.addTab(makeTableView(self.energyParamModel), "Energy parameters")
        self.transModel = PandasModel()
        tables.addTab(makeTableView(self.transModel), "Transmission")
        right.addWidget(tables)
        right.setSizes([600, 250])
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 900])

    # ------------------------------------------------------------------
    def _formParams(self):
        c = self.session.state["calibration"]
        return {k: c[k] for k in ("bindingEnergy", "peakNo", "transmissionPeakNo", "setBeta", "setPhi",
                                  "setPlin", "intMethod")}

    def params(self):
        self.session.state["calibration"].update(self.form.params())
        return self.session.state["calibration"]

    def reloadSettings(self):
        self.form.setParams(self._formParams())

    def _applyEnergies(self):
        df = self.energyModel.dataFrame()
        if df is None or df.empty:
            return
        mapping = {int(r): v for r, v in zip(df["run"], df["override"]) if pd.notna(v)}
        self.session.setPhotonEnergies(mapping)
        self.photonEnergiesChanged.emit()

    def onDataLoaded(self):
        self.energyModel.setDataFrame(self.session.runPhotonEnergies())

    def onResultsChanged(self):
        results = self.session.results
        self.detectorCombo.blockSignals(True)
        current = self.detectorCombo.currentData()
        self.detectorCombo.clear()
        if results is not None and len(results):
            for d in sorted(results["detector"].unique()):
                self.detectorCombo.addItem(f"ToF {d}", int(d))
        idx = self.detectorCombo.findData(current)
        self.detectorCombo.setCurrentIndex(idx if idx >= 0 else 0)
        self.detectorCombo.blockSignals(False)
        self.redraw()

    def onCalibrationChanged(self, warnings=None):
        s = self.session
        self.energyParamModel.setDataFrame(s.energyCalib)
        self.transModel.setDataFrame(s.transmissionCalib)
        lines = []
        if s.state["calibration"].get("file"):
            lines.append(f"File: {s.state['calibration']['file']}")
        meta = s.calibrationMeta or {}
        if meta.get("runs"):
            lines.append(f"Runs: {meta['runs']}")
        if "bindingEnergy" in meta:
            lines.append(f"Binding energy: {meta['bindingEnergy']} eV")
        if s.energyCalib is not None:
            ok = s.energyCalib[["p0", "p1", "p2", "p3", "p4"]].notna().all(axis=1).sum()
            lines.append(f"Energy fits: {ok}/{len(s.energyCalib)} detectors")
        if s.transmissionCalib is not None:
            lines.append(f"Transmission: {len(s.transmissionCalib)} rows")
        if warnings:
            lines.append("<b>Warning – the calibration was made with different loader settings:</b>")
            lines.extend(warnings)
        self.infoLabel.setText("<br>".join(lines) if lines else "No calibration")
        self.redraw()

    def redraw(self):
        s = self.session
        self.energyPlot.clear()
        self.transPlot.clear()
        det = self.detectorCombo.currentData()
        c = s.state["calibration"]
        results = s.results
        if results is not None and len(results) and det is not None and "Photon Energy" in results:
            sel = results[(results["detector"] == det) & (results["peakNo"] == c["peakNo"])]
            avg = sel.groupby("Photon Energy")["pos"].mean().reset_index()
            self.energyPlot.plot(avg["pos"].values, avg["Photon Energy"].values - (c["bindingEnergy"] or 0),
                                 pen=None, symbol="o", symbolBrush=TOL["blue"], symbolPen=None)
        if det is not None and s.energyCalib is not None:
            row = s.energyCalib[s.energyCalib["detector"] == det]
            if len(row) and np.isfinite(row.iloc[0].get("pos_min", np.nan)):
                lo, hi = row.iloc[0]["pos_min"], row.iloc[0]["pos_max"]
                span = hi - lo
                xs = np.linspace(lo - 0.2 * span, hi + 0.2 * span, 300)
                self.energyPlot.plot(xs, s.energyOf(det, xs), pen=pg.mkPen(TOL["red"], width=2))
        trans = s.transmissionCalib
        if trans is not None and len(trans):
            for k, (ee, group) in enumerate(trans.groupby("Electron Energy")):
                group = group.sort_values("detector")
                self.transPlot.plot(group["detector"].values, group["Transmission Coefficient"].values,
                                    pen=pg.mkPen(CYCLE[k % len(CYCLE)]), symbol="o",
                                    symbolBrush=CYCLE[k % len(CYCLE)], name=f"{ee:g} eV")
