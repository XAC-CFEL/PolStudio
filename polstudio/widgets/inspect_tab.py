"""Inspect raw train traces, tune the pulse split and baseline region and look at the per-pulse GMD energy.
For .nxs files (no pulse split) a "train" is a run and its pulses are the scan points."""
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)

from ToFPipeline import FlashH5Loader

from .navigator import DetectorSelector, TraceNavigator
from .plot_grid import CYCLE, TOL


class InspectTab(QWidget):
    recutRequested = pyqtSignal()
    trainChanged = pyqtSignal(int)   # trainId

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._trace = None
        self._pulses = None     # .nxs: raw scan points of the current run
        self._regions = []

        layout = QHBoxLayout(self)
        side = QVBoxLayout()
        layout.addLayout(side, stretch=0)

        nav = QGroupBox("Train")
        self.navBox = nav
        navLayout = QFormLayout(nav)
        self.navigator = TraceNavigator(levels=("run", "trainId"), slider=True)
        self.navigator.changed.connect(self._trainIndexChanged)
        navLayout.addRow(self.navigator)
        self.detectorSelector = DetectorSelector("sum of all")
        self.detectorSelector.changed.connect(self.redraw)
        navLayout.addRow("detector", self.detectorSelector)
        side.addWidget(nav)

        split = QGroupBox("Pulse split")
        self.splitBox = split
        splitLayout = QFormLayout(split)
        self.offsetSpin = QDoubleSpinBox(decimals=2, maximum=1e6, singleStep=1.0)
        self.periodSpin = QDoubleSpinBox(decimals=3, maximum=1e6, singleStep=0.1)
        self.nPulsesSpin = QSpinBox(minimum=1, maximum=1000)
        # 0 = full period; the maximum follows floor(period) so windows cannot overlap
        self.windowSpin = QSpinBox(minimum=0, maximum=1_000_000)
        self.windowSpin.setSpecialValueText("full period")
        self.windowSpin.setToolTip("Samples kept from each pulse start; the rest of the period is discarded")
        for label, spin in (("first pulse offset", self.offsetSpin), ("period (samples)", self.periodSpin),
                            ("number of pulses", self.nPulsesSpin), ("window (samples)", self.windowSpin)):
            splitLayout.addRow(label, spin)
            spin.valueChanged.connect(self._splitChanged)
        self.startsLabel = QLabel("")
        self.startsLabel.setWordWrap(True)
        self.startsLabel.setTextFormat(Qt.TextFormat.RichText)
        splitLayout.addRow(self.startsLabel)
        self.recutButton = QPushButton("Apply split (re-cut pulses)")
        self.recutButton.setToolTip("Re-cuts all loaded trains with these settings. "
                                    "Preprocessing and peaks have to be redone afterwards.")
        self.recutButton.clicked.connect(self.recutRequested)
        splitLayout.addRow(self.recutButton)
        side.addWidget(split)

        baseline = QGroupBox("Baseline (subtracted when loading)")
        baselineLayout = QFormLayout(baseline)
        self.baselineStartSpin = QSpinBox(minimum=0, maximum=1_000_000)
        # 0 = up to the end of the trace (an empty region is never useful)
        self.baselineStopSpin = QSpinBox(minimum=0, maximum=1_000_000)
        self.baselineStopSpin.setSpecialValueText("end")
        for label, spin in (("start (sample)", self.baselineStartSpin), ("stop (sample)", self.baselineStopSpin)):
            spin.setToolTip("The mean of samples [start, stop) of each pulse window (.h5) or histogram (.nxs) is "
                            "subtracted from every trace. Shown as the grey band in the overlay, drag it there.")
            baselineLayout.addRow(label, spin)
            spin.valueChanged.connect(self._baselineChanged)
        self.baselineButton = QPushButton("Apply baseline (reload)")
        self.baselineButton.setToolTip("Reloads the selection with this baseline region. "
                                       "Preprocessing and peaks have to be redone afterwards.")
        self.baselineButton.clicked.connect(self.recutRequested)
        baselineLayout.addRow(self.baselineButton)
        side.addWidget(baseline)
        self.loadSplitSettings(self.session.loaderConfig)
        side.addStretch(1)

        self.view = pg.GraphicsLayoutWidget()
        layout.addWidget(self.view, stretch=1)
        self.trainPlot = self.view.addPlot(row=0, col=0, title="Raw train trace (ADC counts)")
        self.trainPlot.setLabel("bottom", "sample")
        self.trainPlot.showGrid(x=True, y=True, alpha=0.3)
        self.overlayPlot = self.view.addPlot(row=1, col=0, title="Pulse windows overlaid (baseline subtracted)")
        self.overlayPlot.setLabel("bottom", "sample within pulse window")
        self.overlayPlot.showGrid(x=True, y=True, alpha=0.3)
        self.overlayPlot.addLegend(offset=(-10, 10))
        self.gmdPlot = self.view.addPlot(row=2, col=0, title="GMD pulse energy")
        self.gmdPlot.setLabel("bottom", "pulseId")
        self.view.ci.layout.setRowStretchFactor(0, 3)
        self.view.ci.layout.setRowStretchFactor(1, 3)
        self.view.ci.layout.setRowStretchFactor(2, 1)
        self._updateStartsLabel()
        self.updateFormat()

    # ------------------------------------------------------------------
    def updateFormat(self):
        """Show the pulse split for FLASH HDF5 data only; another format brings its own loader settings."""
        fmt = self.session.state["dataFormat"]
        if fmt != getattr(self, "_format", fmt):
            self.loadSplitSettings(self.session.loaderConfig)
            self._updateStartsLabel()
        self._format = fmt
        split = self.session.splitsPulses
        self.splitBox.setVisible(split)
        self.navBox.setTitle("Train" if split else "Run")
        self.trainPlot.setTitle("Raw train trace (ADC counts)" if split
                                else "Raw histogram (counts, sum of the scan points)")
        self.overlayPlot.setTitle("Pulse windows overlaid (baseline subtracted)" if split
                                  else "Scan points overlaid (baseline subtracted)")
        self.overlayPlot.setLabel("bottom", "sample within pulse window" if split else "sample")
        self.gmdPlot.setTitle("GMD pulse energy" if split else "GMD pulse energy (not in .nxs files)")

    def loadSplitSettings(self, cfg):
        """Pulse split and baseline region from the loader config."""
        spins = (self.offsetSpin, self.periodSpin, self.nPulsesSpin, self.windowSpin,
                 self.baselineStartSpin, self.baselineStopSpin)
        for spin in spins:
            spin.blockSignals(True)
        self.offsetSpin.setValue(float(cfg.get("pulseOffset", 0.0)))
        self.periodSpin.setValue(float(cfg.get("pulsePeriod", 1000.0)))
        self.nPulsesSpin.setValue(int(cfg.get("nPulses", 1)))
        self._updateWindowMaximum()
        self.windowSpin.setValue(int(cfg.get("pulseWindow") or 0))
        start, stop = (list(cfg.get("baselineRegion") or [0, 10]) + [None])[:2]
        self.baselineStartSpin.setValue(int(start or 0))
        self.baselineStopSpin.setValue(int(stop or 0))
        for spin in spins:
            spin.blockSignals(False)

    def _nSamplesTrain(self):
        return self._trace.sizes["sample"] if self._trace is not None else 16000

    def _updateWindowMaximum(self):
        self.windowSpin.setMaximum(FlashH5Loader.windowLength(self.periodSpin.value(), None,
                                                              self.nPulsesSpin.value(), self._nSamplesTrain()))

    def splitSettings(self):
        return {"pulseOffset": self.offsetSpin.value(), "pulsePeriod": self.periodSpin.value(),
                "nPulses": self.nPulsesSpin.value(), "pulseWindow": self.windowSpin.value() or None}

    def baselineRegion(self):
        return [self.baselineStartSpin.value(), self.baselineStopSpin.value() or None]

    def applySettingsToConfig(self):
        if self.session.splitsPulses:
            self.session.loaderConfig.update(self.splitSettings())
        self.session.loaderConfig["baselineRegion"] = self.baselineRegion()

    def onDataLoaded(self):
        self.updateFormat()
        tt = self.session.loader.trainTraces
        self.navigator.setTraces(pd.DataFrame({"run": tt["daq_run"].values, "trainId": tt["trainId"].values}))
        self._updateWindowMaximum()
        self.detectorSelector.setDetectors(tt["detector"].values)
        self._trainIndexChanged()

    def currentTrainId(self):
        current = self.navigator.current()
        return None if current is None else int(current["trainId"])

    def _trainIndexChanged(self):
        tid = self.currentTrainId()
        if tid is None:
            return
        split = self.session.splitsPulses
        self._trace = self.session.trainTrace(tid)
        self._pulses = None if split else self.session.pulsesOfTrain(tid)
        self.redraw()
        self.trainChanged.emit(tid)

    def _splitChanged(self, *args):
        self._updateWindowMaximum()
        self._updateStartsLabel()
        self.redraw()

    def _baselineChanged(self, *args):
        start, stop = self.baselineRegion()
        if stop is not None and stop <= start:
            self.baselineStopSpin.blockSignals(True)
            self.baselineStopSpin.setValue(start + 1)
            self.baselineStopSpin.blockSignals(False)
        self.redraw()

    def _baselineDragged(self, x0, nSamples, region):
        lo, hi = (int(round(v - x0)) for v in region.getRegion())
        lo = min(max(lo, 0), nSamples - 1)
        for spin in (self.baselineStartSpin, self.baselineStopSpin):
            spin.blockSignals(True)
        self.baselineStartSpin.setValue(lo)
        self.baselineStopSpin.setValue(0 if hi >= nSamples else max(hi, lo + 1))
        for spin in (self.baselineStartSpin, self.baselineStopSpin):
            spin.blockSignals(False)
        # Rebuild the band after the drag signal has returned
        QTimer.singleShot(0, self.redraw)

    def _subtractBaseline(self, traces):
        """Subtract the mean of the baseline region along the last axis, as the loaders do."""
        start, stop = self.baselineRegion()
        traces = np.asarray(traces, dtype=float)
        with np.errstate(invalid="ignore"):
            return traces - np.nanmean(traces[..., start:stop], axis=-1, keepdims=True)

    def _drawBaselineBand(self, x0, nSamples):
        start, stop = self.baselineRegion()
        stop = nSamples if stop is None else min(stop, nSamples)
        band = pg.LinearRegionItem(values=(x0 + start, x0 + stop), brush=pg.mkBrush(128, 128, 128, 60))
        band.setToolTip("Baseline region: drag to move, drag an edge to resize")
        band.sigRegionChangeFinished.connect(lambda r: self._baselineDragged(x0, nSamples, r))
        self.overlayPlot.addItem(band)

    def _starts(self):
        s = self.splitSettings()
        nSamples = self._nSamplesTrain()
        window = FlashH5Loader.windowLength(s["pulsePeriod"], s["pulseWindow"], s["nPulses"], nSamples, warn=False)
        starts = np.round(s["pulseOffset"] + np.arange(s["nPulses"]) * s["pulsePeriod"]).astype(int)
        valid = (starts >= 0) & (starts + window <= nSamples)
        return starts, valid, window

    def _updateStartsLabel(self):
        starts, valid, window = self._starts()
        text = ", ".join(str(s) if v else f"<s>{s}</s>" for s, v in zip(starts, valid))
        text = f"starts: {text}" + ("" if valid.all() else "<br>struck-out windows do not fit")
        nKept = int(valid.sum()) * window
        text += (f"<br>kept {window} samples per pulse, {nKept} of {self._nSamplesTrain()} per train "
                 f"({100 * nKept / self._nSamplesTrain():.0f} %)")
        self.startsLabel.setText(text)

    def _regionDragged(self, start, region):
        lo, hi = region.getRegion()
        for spin in (self.offsetSpin, self.windowSpin):
            spin.blockSignals(True)
        # Shift by the dragged distance so the fractional offset is kept (no drift when only the right edge moves)
        self.offsetSpin.setValue(max(0.0, self.offsetSpin.value() + round(lo - start)))
        window = int(round(hi - lo))
        # windowSpin's maximum is floor(period): longer windows are capped; 0 shows as "full period"
        self.windowSpin.setValue(window if window < self.windowSpin.maximum() else 0)
        for spin in (self.offsetSpin, self.windowSpin):
            spin.blockSignals(False)
        # Rebuild the regions after the drag signal has returned
        QTimer.singleShot(0, self._splitChanged)

    def redraw(self):
        self.trainPlot.clear()
        self.overlayPlot.clear()
        self.gmdPlot.clear()
        if self._trace is None:
            return
        det = self.detectorSelector.current()
        trace = self._trace
        x = trace["sample"].values
        if det is None:
            y = trace.values.astype(float)
            y = (y - np.median(y[:, :200], axis=1, keepdims=True)).sum(axis=0)
        else:
            y = trace.sel(detector=det).values.astype(float)
        self.trainPlot.plot(x, y, pen=pg.mkPen(TOL["blue"], width=1))
        if self._pulses is not None:
            return self._drawPoints(det)

        starts, valid, window = self._starts()
        for k, (s, v) in enumerate(zip(starts, valid)):
            color = pg.mkColor(CYCLE[k % len(CYCLE)]) if v else pg.mkColor("#999999")
            color.setAlpha(40)
            # Drag a window to move all of them (offset), drag its edges to change the window length
            region = pg.LinearRegionItem(values=(s, s + window), movable=bool(v), brush=color)
            region.setToolTip("Drag to move all pulse windows, drag an edge to change the window length")
            region.sigRegionChangeFinished.connect(lambda r, start=s: self._regionDragged(start, r))
            self.trainPlot.addItem(region)
            label = pg.TextItem(str(k), color=CYCLE[k % len(CYCLE)] if v else "#999999", anchor=(0, 0))
            label.setPos(s, float(np.nanmax(y)))
            self.trainPlot.addItem(label)

        for k, (s, v) in enumerate(zip(starts, valid)):
            if not v:
                continue
            seg = self._subtractBaseline(y[s:s + window])
            self.overlayPlot.plot(np.arange(len(seg)), seg, pen=pg.mkPen(CYCLE[k % len(CYCLE)], width=1),
                                  name=f"pulse {k}")
        if valid.any():
            self._drawBaselineBand(0, window)

        gmd = self.session.gmdOfTrain(self.currentTrainId())
        if gmd is not None and len(gmd):
            bars = pg.BarGraphItem(x=gmd["pulseId"].values, height=np.nan_to_num(gmd["Pulse Energy"].values),
                                   width=0.7, brush=TOL["green"])
            self.gmdPlot.addItem(bars)

    def _drawPoints(self, det):
        pulses = self._pulses
        if det is None:
            pulses = pulses.sum(dim="detector")
        elif det in pulses["detector"].values:
            pulses = pulses.sel(detector=det)
        else:
            return
        x = pulses["sample"].values
        values = self._subtractBaseline(pulses.values)
        for k, pid in enumerate(pulses["pulseId"].values):
            self.overlayPlot.plot(x, values[k], pen=pg.mkPen(CYCLE[k % len(CYCLE)], width=1), name=f"point {pid}")
        self._drawBaselineBand(int(x[0]), len(x))
