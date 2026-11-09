"""
Heatmap of the processed traces of one pulse (or the average of all pulses), adapted from PolScope's
AngularHeatmapCanvas / CartesianHeightCanvas:

- Polar: angle = detector angle (0° on +x, counter-clockwise), radius = sample. Either interpolated
  between detectors over the full circle, or one wedge per detector.
- Cartesian: sample (x) vs detector number or detector angle (y), optionally interpolated between rows.

Options as in PolScope: sample range, interpolate, show peaks (marker + width bar);
plus colormap, a colour bar with draggable levels, the choice of run / train / pulse and scaling by the
transmission calibration. The same panel is shown on the Polarization tab (it moves to the visible tab via
PanelHost), so both tabs always show the same map with the same settings.
"""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QSpinBox,
                             QStackedWidget, QVBoxLayout, QWidget)

from .navigator import TraceNavigator, traceTable
from .plot_grid import TOL

COLORMAPS = ["viridis", "inferno", "plasma", "magma", "cividis", "turbo"]
PEAK_COLOR = "#EE6677"     # Tol red: readable on all colormaps above
N_THETA = 720
RASTER_SIZE = 1024   # polar raster in pixels; the plot is redrawn on demand only, so render it at full quality


def angularGrid(traces, anglesRad, nTheta=N_THETA):
    """Periodic linear interpolation of traces (n_det, n_samples) vs angle -> (nTheta, n_samples)."""
    thetaGrid = np.linspace(0, 2 * np.pi, nTheta, endpoint=False)
    order = np.argsort(anglesRad)
    sortedAngles, sortedTraces = anglesRad[order], traces[order]
    xp = np.concatenate([sortedAngles - 2 * np.pi, sortedAngles, sortedAngles + 2 * np.pi])
    fp = np.vstack([sortedTraces, sortedTraces, sortedTraces])
    idx = np.clip(np.searchsorted(xp, thetaGrid) - 1, 0, len(xp) - 2)
    t = (thetaGrid - xp[idx]) / (xp[idx + 1] - xp[idx])
    return fp[idx] * (1.0 - t)[:, None] + fp[idx + 1] * t[:, None]


def nearestGrid(traces, anglesRad, nTheta=N_THETA):
    """Each angle takes the trace of the nearest detector (circular distance)."""
    thetaGrid = np.linspace(0, 2 * np.pi, nTheta, endpoint=False)
    diff = np.abs(thetaGrid[:, None] - anglesRad[None, :])
    diff = np.minimum(diff, 2 * np.pi - diff)
    return traces[np.argmin(diff, axis=1)]


def _levels(values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    return lo, hi if hi > lo else lo + 1e-9


class _HeatmapBase(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = self.addPlot(row=0, col=0)
        self.plot.hideButtons()
        self.colorBar = pg.ColorBarItem(values=(0, 1), colorMap=pg.colormap.get("viridis"), interactive=True,
                                        width=15)
        self.addItem(self.colorBar, row=0, col=1)
        self.colorBar.sigLevelsChanged.connect(self._levelsFromBar)
        self._levelItems = []    # items whose levels follow the colour bar
        self.peakLines = pg.PlotDataItem([], [], pen=pg.mkPen(PEAK_COLOR, width=2), connect="pairs")
        self.peakScatter = pg.ScatterPlotItem(size=7, pen=pg.mkPen("w", width=0.5), brush=pg.mkBrush(PEAK_COLOR))
        for item, z in ((self.peakLines, 10), (self.peakScatter, 11)):
            item.setZValue(z)
            self.plot.addItem(item)

    def _levelsFromBar(self, bar):
        levels = bar.levels()
        for item in self._levelItems:
            item.setLevels(levels)

    def _setColor(self, cmap, levels):
        self.colorBar.setColorMap(cmap)
        self.colorBar.blockSignals(True)
        self.colorBar.setLevels(levels)
        self.colorBar.blockSignals(False)

    def setPeaks(self, xs, ys, segX, segY):
        self.peakScatter.setData(x=np.asarray(xs, float), y=np.asarray(ys, float))
        self.peakLines.setData(np.asarray(segX, float), np.asarray(segY, float))


class PolarHeatmap(_HeatmapBase):
    """Angle around the circle, sample along the radius (displayed range stretched over the whole disk)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot.getViewBox().setAspectLocked(True)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.setTitle("Polar heatmap")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self._wedges = []
        self._grid = []
        self._labels = []
        self._rasterKey = None
        self._rasterMaps = None
        self._buildGrid()

    def _buildGrid(self):
        pen = pg.mkPen("#bbbbbb", width=0.5)
        for _ in range(3):
            ring = pg.PlotDataItem([], [], pen=pen)
            ring.setZValue(5)
            self.plot.addItem(ring)
            self._grid.append(ring)
            label = pg.TextItem("", anchor=(0, 1), color="#555555")
            label.setZValue(6)
            self.plot.addItem(label)
            self._labels.append(label)
        for deg in range(0, 360, 45):
            ray = pg.PlotDataItem([], [], pen=pen)
            ray.setZValue(5)
            self.plot.addItem(ray)
            self._grid.append(ray)
            label = pg.TextItem(f"{deg}°", anchor=(0.5, 0.5), color="#555555")
            self.plot.addItem(label)
            self._labels.append(label)

    def _rasterTables(self, nTheta, nR, size, rMax):
        """Bilinear (theta, r) lookup for every pixel of a size x size raster; cached per configuration."""
        key = (nTheta, nR, size, rMax)
        if key == self._rasterKey:
            return self._rasterMaps
        half = 1.25 * rMax
        coords = np.linspace(-half, half, size, dtype=np.float32)
        # ImageItem is column-major: image[ix, iy] -> build the raster with x on the first axis
        xx, yy = np.meshgrid(coords, coords, indexing="ij")
        r = np.hypot(xx, yy)
        theta = np.arctan2(yy, xx) % (2 * np.pi)
        thIdx = theta / (2 * np.pi) * nTheta
        rIdx = r / rMax * (nR - 1)
        th0 = np.clip(thIdx.astype(np.int32), 0, nTheta - 1)
        r0 = np.clip(rIdx.astype(np.int32), 0, nR - 1)
        th1 = (th0 + 1) % nTheta          # wrap around 360°
        r1 = np.clip(r0 + 1, 0, nR - 1)
        wt = np.clip(thIdx - th0, 0, 1)
        wr = np.clip(rIdx - r0, 0, 1)
        valid = np.flatnonzero(~(r > rMax))
        flat = lambda a: a.ravel()[valid]  # noqa: E731
        self._rasterKey = key
        self._rasterMaps = (flat(th0 * nR + r0), flat(th1 * nR + r0), flat(th0 * nR + r1), flat(th1 * nR + r1),
                            flat((1 - wt) * (1 - wr)), flat(wt * (1 - wr)), flat((1 - wt) * wr), flat(wt * wr),
                            valid)
        return self._rasterMaps

    def update(self, traces, samples, anglesDeg, peaks, interpolate, cmap, levels):
        """traces (n_det, n_samples); anglesDeg per row; peaks: list of (row, pos, widthL, widthR)."""
        for item in self._wedges:
            self.plot.removeItem(item)
        self._wedges = []
        self._levelItems = []
        self._setColor(cmap, levels)
        anglesRad = np.deg2rad(np.asarray(anglesDeg, float)) % (2 * np.pi)
        sMin, sMax = float(samples[0]), float(samples[-1])
        rMax = max(sMax, 1.0)
        span = max(sMax - sMin, 1e-9)
        rmap = lambda s: (np.asarray(s, float) - sMin) * (rMax / span)  # noqa: E731

        if interpolate:
            grid = angularGrid(traces, anglesRad)
            nTheta, nR = grid.shape
            if nR < 2:
                grid, nR = np.hstack([grid, grid]), 2
            size = RASTER_SIZE
            ia, ib, ic, id_, w00, w10, w01, w11, valid = self._rasterTables(nTheta, nR, size, rMax)
            g = np.ascontiguousarray(grid, dtype=np.float32).ravel()
            img = np.full(size * size, np.nan, dtype=np.float32)
            img[valid] = g[ia] * w00 + g[ib] * w10 + g[ic] * w01 + g[id_] * w11
            half = 1.25 * rMax
            self.image.setImage(img.reshape(size, size), autoLevels=False, levels=levels)
            self.image.setColorMap(cmap)
            self.image.setRect(QRectF(-half, -half, 2 * half, 2 * half))
            self.image.setVisible(True)
            self._levelItems.append(self.image)
        else:
            self.image.setVisible(False)
            order = np.argsort(anglesRad)
            sortedAngles = anglesRad[order]
            gaps = np.diff(sortedAngles, append=sortedAngles[0] + 2 * np.pi) if len(order) > 1 else np.array([2 * np.pi])
            halfLeft = np.empty_like(gaps)
            halfRight = np.empty_like(gaps)
            halfLeft[order] = np.roll(gaps, 1) / 2
            halfRight[order] = gaps / 2
            dr = float(samples[1] - samples[0]) if len(samples) > 1 else 1.0
            rEdges = np.clip(rmap(np.concatenate([[samples[0] - dr / 2], samples + dr / 2])), 0, None)
            for k, angle in enumerate(anglesRad):
                th, rr = np.meshgrid([angle - halfLeft[k], angle + halfRight[k]], rEdges, indexing="ij")
                wedge = pg.PColorMeshItem(rr * np.cos(th), rr * np.sin(th), traces[k][None, :],
                                          colorMap=cmap, levels=levels)
                self.plot.addItem(wedge)
                self._wedges.append(wedge)
                self._levelItems.append(wedge)

        # grid: rings at start / middle / end of the displayed sample range, rays every 45°
        circle = np.linspace(0, 2 * np.pi, 300)
        for i, s in enumerate((sMin, (sMin + sMax) / 2, sMax)):
            r = float(rmap(s))
            self._grid[i].setData(r * np.cos(circle), r * np.sin(circle))
            self._labels[i].setText(f"{s:.0f}")
            self._labels[i].setPos(r * np.cos(np.pi / 8), r * np.sin(np.pi / 8))
        for j, deg in enumerate(range(0, 360, 45)):
            rad = np.deg2rad(deg)
            self._grid[3 + j].setData([0, rMax * np.cos(rad)], [0, rMax * np.sin(rad)])
            self._labels[3 + j].setPos(1.12 * rMax * np.cos(rad), 1.12 * rMax * np.sin(rad))
        half = 1.25 * rMax
        self.plot.setXRange(-half, half, padding=0)
        self.plot.setYRange(-half, half, padding=0)

        xs, ys, segX, segY = [], [], [], []
        for row, pos, wl, wr in peaks:
            if not (sMin <= pos <= sMax):
                continue
            c, s_ = np.cos(anglesRad[row]), np.sin(anglesRad[row])
            p, a, b = (float(np.clip(rmap(v), 0, rMax)) for v in (pos, pos + wl, pos + wr))
            xs.append(p * c)
            ys.append(p * s_)
            segX += [a * c, b * c]
            segY += [a * s_, b * s_]
        self.setPeaks(xs, ys, segX, segY)


class CartesianHeatmap(_HeatmapBase):
    """Sample on x, detector number (or detector angle) on y."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot.setTitle("Cartesian heatmap")
        self.plot.setLabel("bottom", "sample")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)

    def update(self, traces, samples, detIds, anglesDeg, peaks, useAngle, interpolate, cmap, levels):
        self._setColor(cmap, levels)
        self.image.setColorMap(cmap)
        x0, width = float(samples[0]), float(max(samples[-1] - samples[0], 1))
        axis = self.plot.getAxis("left")
        if useAngle:
            anglesRad = np.deg2rad(np.asarray(anglesDeg, float)) % (2 * np.pi)
            grid = angularGrid(traces, anglesRad) if interpolate else nearestGrid(traces, anglesRad)
            self.image.setImage(grid.T, autoLevels=False, levels=levels)
            self.image.setRect(QRectF(x0, 0.0, width, 360.0))
            self.plot.setLabel("left", "angle (°)")
            axis.setTicks(None)
            self.plot.setYRange(0, 360, padding=0.02)
            rowY = np.degrees(anglesRad)
        else:
            order = np.argsort(detIds)
            grid = traces[order]
            n = len(order)
            if interpolate and n > 1:
                # linear interpolation between neighbouring detector rows (row centres at integer y)
                k = 16
                y = np.clip((np.arange(n * k) + 0.5) / k - 0.5, 0, n - 1)
                lo = np.minimum(y.astype(int), n - 2)
                w = (y - lo)[:, None]
                grid = grid[lo] * (1 - w) + grid[lo + 1] * w
            self.image.setImage(grid.T, autoLevels=False, levels=levels)
            self.image.setRect(QRectF(x0, -0.5, width, n))
            self.plot.setLabel("left", "detector")
            axis.setTicks([[(i, str(detIds[j])) for i, j in enumerate(order)]])
            self.plot.setYRange(-0.5, n - 0.5, padding=0.02)
            rowY = np.empty(n)
            rowY[order] = np.arange(n)
        self.plot.setXRange(x0, x0 + width, padding=0.02)
        self._levelItems = [self.image]

        xs, ys, segX, segY = [], [], [], []
        for row, pos, wl, wr in peaks:
            if not (samples[0] <= pos <= samples[-1]):
                continue
            xs.append(pos)
            ys.append(rowY[row])
            segX += [pos + wl, pos + wr]
            segY += [rowY[row], rowY[row]]
        self.setPeaks(xs, ys, segX, segY)


class PanelHost(QWidget):
    """Empty container a shared panel can be moved into (a widget has one parent at a time)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def adopt(self, panel):
        if panel.parent() is not self:
            self._layout.addWidget(panel)
        panel.show()


class HeatmapTab(QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QHBoxLayout(self)

        self.stack = QStackedWidget()
        self.polar = PolarHeatmap()
        self.cartesian = CartesianHeatmap()
        self.stack.addWidget(self.polar)
        self.stack.addWidget(self.cartesian)
        layout.addWidget(self.stack, stretch=1)

        side = QVBoxLayout()
        data = QGroupBox("Data")
        dataLayout = QFormLayout(data)
        self.navigator = TraceNavigator()
        self.navigator.changed.connect(self.redraw)
        dataLayout.addRow(self.navigator)
        self.averageCheck = QCheckBox("average over all pulses")
        dataLayout.addRow(self.averageCheck)
        self.transmissionCheck = QCheckBox("scale by transmission")
        self.transmissionCheck.setToolTip(
            "Multiply every detector's trace by its transmission coefficient (Calibration tab). With several "
            "calibration sets the one closest to the trace's electron energy (photon energy − Ebind) is used.")
        dataLayout.addRow(self.transmissionCheck)
        side.addWidget(data)

        settings = QGroupBox("Heatmap settings")
        form = QFormLayout(settings)
        self.cartesianCheck = QCheckBox("Cartesian view")
        form.addRow(self.cartesianCheck)
        self.angleAxisCheck = QCheckBox("angle on y axis (Cartesian)")
        self.angleAxisCheck.setToolTip("Cartesian view: detector angle (0–360°) instead of detector number")
        form.addRow(self.angleAxisCheck)
        self.sMinSpin = QSpinBox(maximum=1_000_000)
        self.sMaxSpin = QSpinBox(maximum=1_000_000)
        form.addRow("sample min", self.sMinSpin)
        form.addRow("sample max", self.sMaxSpin)
        self.interpolateCheck = QCheckBox("interpolate")
        self.interpolateCheck.setChecked(True)
        self.interpolateCheck.setToolTip("Polar: interpolate between detector angles (off = one wedge per "
                                         "detector). Cartesian: interpolate between rows.")
        form.addRow(self.interpolateCheck)
        self.peaksCheck = QCheckBox("show peaks")
        self.peaksCheck.setChecked(True)
        self.peaksCheck.setToolTip("Peak position and width from the Peaks tab (single pulse only)")
        form.addRow(self.peaksCheck)
        self.cmapCombo = QComboBox()
        self.cmapCombo.addItems(COLORMAPS)
        form.addRow("colormap", self.cmapCombo)
        self.autoLevelsCheck = QCheckBox("auto levels")
        self.autoLevelsCheck.setChecked(True)
        self.autoLevelsCheck.setToolTip("Off: keep the colour bar levels (drag the bar ends to change them)")
        form.addRow(self.autoLevelsCheck)
        side.addWidget(settings)
        self.infoLabel = QLabel("Apply the preprocessing to see a heatmap")
        self.infoLabel.setWordWrap(True)
        self.infoLabel.setTextFormat(Qt.TextFormat.RichText)
        side.addWidget(self.infoLabel)
        side.addStretch(1)
        sideWidget = QWidget()
        sideWidget.setLayout(side)
        sideWidget.setMaximumWidth(280)
        layout.addWidget(sideWidget)

        for spin in (self.sMinSpin, self.sMaxSpin):
            spin.valueChanged.connect(self.redraw)
        for check in (self.averageCheck, self.transmissionCheck, self.cartesianCheck, self.angleAxisCheck,
                      self.interpolateCheck, self.peaksCheck, self.autoLevelsCheck):
            check.toggled.connect(self.redraw)
        self.cmapCombo.currentIndexChanged.connect(self.redraw)
        self._nSamples = None

    # ------------------------------------------------------------------
    def currentView(self):
        return self.stack.currentWidget()

    def _angles(self, detectors):
        if self.session.loader is not None:
            mapping = dict(zip(self.session.loader.angles["detector"], self.session.loader.angles["Angles"]))
        else:
            cfg = self.session.loaderConfig
            angles, tof = list(cfg.get("angles", [])), cfg.get("ToF", [])
            mapping = dict(zip(tof, angles)) if len(angles) == len(tof) else dict(enumerate(angles))
        return np.array([float(mapping.get(int(d), np.nan)) for d in detectors])

    def onProcessedChanged(self):
        data = self.session.processed
        if data is None:
            self.redraw()
            return
        n = data.sizes["sample"]
        self.navigator.setTraces(traceTable(data))
        for spin in (self.sMinSpin, self.sMaxSpin):
            spin.blockSignals(True)
        self.sMinSpin.setMaximum(n - 1)
        self.sMaxSpin.setMaximum(n - 1)
        if self._nSamples != n:   # new window length: show the whole window
            self._nSamples = n
            self.sMinSpin.setValue(0)
            self.sMaxSpin.setValue(n - 1)
        for spin in (self.sMinSpin, self.sMaxSpin):
            spin.blockSignals(False)
        self.redraw()

    def redraw(self):
        data = self.session.processed
        self.stack.setCurrentWidget(self.cartesian if self.cartesianCheck.isChecked() else self.polar)
        self.angleAxisCheck.setEnabled(self.cartesianCheck.isChecked())
        self.navigator.setEnabled(not self.averageCheck.isChecked())
        hasTransmission = self.session.transmissionCalib is not None and len(self.session.transmissionCalib) > 0
        self.transmissionCheck.setEnabled(hasTransmission)
        if data is None or data.sizes["pulse"] == 0 or self.navigator.position() is None:
            self.infoLabel.setText("Apply the preprocessing to see a heatmap")
            return
        lo, hi = sorted((self.sMinSpin.value(), self.sMaxSpin.value()))
        if hi <= lo:
            hi = min(lo + 1, data.sizes["sample"] - 1)
        window = data.isel(sample=slice(lo, hi + 1))
        detectors = [int(d) for d in window["detector"].values]
        angles = self._angles(detectors)
        keep = np.isfinite(angles)
        notes = []
        if self.averageCheck.isChecked():
            traces = window.mean("pulse").values
            notes.append(f"mean of {data.sizes['pulse']} pulses")
            peaks, tid = [], None
        else:
            i = min(self.navigator.position(), data.sizes["pulse"] - 1)
            tid, pid = data["pulse"].to_index()[i]
            traces = window.isel(pulse=i).values
            peaks = self._peaks(tid, pid, detectors) if self.peaksCheck.isChecked() else []
        traces = np.nan_to_num(np.asarray(traces, dtype=float))
        if hasTransmission and self.transmissionCheck.isChecked():
            factors, electronEnergy = self.session.transmissionFactors(
                detectors, self.session.photonEnergyOfTrain(tid))
            unscaled = [d for d, f in zip(detectors, factors) if not np.isfinite(f)]
            traces = traces * np.where(np.isfinite(factors), factors, 1.0)[:, None]
            notes.append("scaled by transmission" + (f" (E_e = {electronEnergy:.1f} eV)"
                                                     if electronEnergy is not None else ""))
            if unscaled:
                notes.append(f"no coefficient for ToF {unscaled}: unscaled")
        samples = window["sample"].values.astype(float)
        cmap = pg.colormap.get(self.cmapCombo.currentText())
        view = self.currentView()
        levels = _levels(traces) if self.autoLevelsCheck.isChecked() else tuple(view.colorBar.levels())

        if self.cartesianCheck.isChecked():
            useAngle = self.angleAxisCheck.isChecked()
            rows = np.flatnonzero(keep) if useAngle else np.arange(len(detectors))
            remap = {int(r): k for k, r in enumerate(rows)}
            self.cartesian.update(traces[rows], samples, [detectors[r] for r in rows], angles[rows],
                                  [(remap[r], p, a, b) for r, p, a, b in peaks if r in remap],
                                  useAngle, self.interpolateCheck.isChecked(), cmap, levels)
        else:
            rows = np.flatnonzero(keep)
            remap = {int(r): k for k, r in enumerate(rows)}
            self.polar.update(traces[rows], samples, angles[rows],
                              [(remap[r], p, a, b) for r, p, a, b in peaks if r in remap],
                              self.interpolateCheck.isChecked(), cmap, levels)
        missing = [d for d, ok in zip(detectors, keep) if not ok]
        if missing:
            notes.append(f"no angle for ToF {missing}: not shown")
        self.infoLabel.setText("<br>".join([f"{len(detectors)} detectors, samples {lo}–{hi}"] + notes))

    def _peaks(self, tid, pid, detectors):
        results = self.session.results
        if results is None or not len(results):
            return []
        sel = results[(results["trainId"] == tid) & (results["pulseId"] == pid)]
        rowOf = {d: k for k, d in enumerate(detectors)}
        return [(rowOf[int(r["detector"])], float(r["pos"]), float(r["width left"]), float(r["width right"]))
                for _, r in sel.iterrows() if int(r["detector"]) in rowOf]
