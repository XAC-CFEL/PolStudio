"""Polarization fit (Fitter.pol) and polar intensity map (Plotter) on embedded matplotlib canvases."""
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QSplitter, QVBoxLayout,
                             QWidget, QFormLayout)

from ToFPipeline import Plotter

from .params import ParamForm

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


class PolarizationTab(QWidget):
    fitRequested = pyqtSignal(dict)

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
        self.fitButton.clicked.connect(lambda: self.fitRequested.emit(self.params()))
        boxLayout.addWidget(self.fitButton)
        self.resultLabel = QLabel("")
        self.resultLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.resultLabel.setWordWrap(True)
        boxLayout.addWidget(self.resultLabel)
        sideLayout.addWidget(box)

        heat = QGroupBox("Polar intensity map")
        heatLayout = QFormLayout(heat)
        self.pulseSpin = QSpinBox()
        heatLayout.addRow("pulse index", self.pulseSpin)
        drawButton = QPushButton("Draw map")
        drawButton.clicked.connect(self.drawHeatmap)
        heatLayout.addRow(drawButton)
        sideLayout.addWidget(heat)
        sideLayout.addStretch(1)
        splitter.addWidget(side)

        plots = QWidget()
        plotsLayout = QVBoxLayout(plots)
        self.figure = Figure(figsize=(10, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        plotsLayout.addWidget(NavigationToolbar2QT(self.canvas, self))
        plotsLayout.addWidget(self.canvas, stretch=1)
        self.polAx = self.figure.add_subplot(1, 2, 1, projection="polar")
        self.heatAx = self.figure.add_subplot(1, 2, 2, projection="polar")
        splitter.addWidget(plots)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1000])

    def params(self):
        params = self.form.params()
        self.session.state["polarization"] = dict(params)
        return params

    def reloadSettings(self):
        self.form.setParams(dict(self.session.state["polarization"]))

    def onProcessedChanged(self):
        data = self.session.processed
        self.pulseSpin.setMaximum(max((data.sizes["pulse"] if data is not None else 1) - 1, 0))

    def markOutdated(self):
        if self.resultLabel.text() and not self.resultLabel.text().startswith("Outdated"):
            self.resultLabel.setText("Outdated – the channel selection or inputs changed; fit again\n\n"
                                     + self.resultLabel.text())

    def resetPolAxis(self):
        self.figure.delaxes(self.polAx)
        self.polAx = self.figure.add_subplot(1, 2, 1, projection="polar")
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

    def drawHeatmap(self):
        data = self.session.processed
        if data is None or self.session.loader is None:
            return
        for ax in list(self.figure.axes):  # map and its colorbar
            if ax is not self.polAx:
                self.figure.delaxes(ax)
        self.heatAx = self.figure.add_subplot(1, 2, 2, projection="polar")
        tid, pid = data["pulse"].to_index()[min(self.pulseSpin.value(), data.sizes["pulse"] - 1)]
        angles = self.session.loader.angles
        cfg = {"ToF": [int(d) for d in angles["detector"]], "angles": [float(a) for a in angles["Angles"]]}
        trans = self.session.transmissionCalib
        Plotter(data, results=self.session.results, config=cfg).plot(
            trainId=tid, pulseId=pid, ax=self.heatAx, savename=False, show=False,
            transParam=trans if trans is not None and len(trans) else None,
            title=f"trainId {tid} / pulse {pid}")
        self.canvas.draw_idle()
