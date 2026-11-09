"""Grid of pyqtgraph plots, one per detector, with optional time / energy axis labels."""
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout, QWidget

# Paul Tol bright palette (same as the pipeline's matplotlib plots)
TOL = {"blue": "#4477AA", "cyan": "#66CCEE", "green": "#228833", "yellow": "#CCBB44",
       "red": "#EE6677", "purple": "#AA3377", "grey": "#BBBBBB", "black": "#000000"}
CYCLE = [TOL[c] for c in ("blue", "red", "green", "yellow", "cyan", "purple", "grey", "black")]


class MappedAxis(pg.AxisItem):
    """Bottom axis whose tick labels are ``mapper(sample)`` (e.g. ns or eV) instead of the sample index."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = None
        self.fmt = "{:.1f}"

    def setMapper(self, mapper, label=None, fmt="{:.1f}"):
        self.mapper, self.fmt = mapper, fmt
        if label is not None:
            self.setLabel(label)
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if self.mapper is None:
            return super().tickStrings(values, scale, spacing)
        mapped = self.mapper(np.asarray(values, dtype=float))
        if mapped is None:
            return super().tickStrings(values, scale, spacing)
        return ["" if not np.isfinite(v) else self.fmt.format(v) for v in mapped]


class PlotGrid(QWidget):
    def __init__(self, columns=4, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.view = pg.GraphicsLayoutWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.plots = {}
        self.axes = {}

    def setDetectors(self, detectors, linkY=True):
        detectors = [int(d) for d in detectors]
        if list(self.plots) == detectors:
            return
        # Unlink before removing: a linked ViewBox left pointing at a removed one crashes on the next paint
        for plot in self.plots.values():
            plot.setXLink(None)
            plot.setYLink(None)
        self.view.clear()
        self.plots, self.axes = {}, {}
        first = None
        for i, det in enumerate(detectors):
            axis = MappedAxis(orientation="bottom")
            plot = self.view.addPlot(row=i // self.columns, col=i % self.columns, title=f"ToF {det}",
                                     axisItems={"bottom": axis})
            plot.showGrid(x=True, y=True, alpha=0.3)
            plot.setMenuEnabled(True)
            if first is None:
                first = plot
            else:
                plot.setXLink(first)
                if linkY:
                    plot.setYLink(first)
            self.plots[det] = plot
            self.axes[det] = axis

    def clear(self):
        for plot in self.plots.values():
            plot.clear()

    def setAxisMapper(self, mapperForDetector, label=None, fmt="{:.1f}"):
        """mapperForDetector(det) -> callable or None."""
        for det, axis in self.axes.items():
            axis.setMapper(mapperForDetector(det) if mapperForDetector else None, label, fmt)
            if mapperForDetector is None:
                axis.setLabel("")
