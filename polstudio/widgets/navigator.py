"""Step through traces by run, trainId and pulseId separately, and through detectors."""
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QSlider, QSpinBox, QToolButton, QWidget

LEVELS = ("run", "trainId", "pulseId")
NO_RUN = -1   # stands for "run unknown" inside the table


def traceTable(data):
    """Run, trainId and pulseId of every trace of data (along pulse); run is NO_RUN when the data has no daq_run."""
    index = data["pulse"].to_index()
    runs = data["daq_run"].values if "daq_run" in data.coords else np.full(len(index), NO_RUN)
    return pd.DataFrame({"run": runs, "trainId": index.get_level_values("trainId"),
                         "pulseId": index.get_level_values("pulseId")})


class TraceNavigator(QWidget):
    """
    One spin box per level (run, trainId, pulseId). Each steps through the values that exist for the levels above
    it, so runs, trains and pulses are flipped through separately; a lower level keeps its value when the level
    above changes and the value exists there. Levels that never take more than one value are hidden; the summary
    line always names the current trace. ``position()`` is the index of that trace in the table.
    """
    changed = pyqtSignal()

    def __init__(self, levels=LEVELS, slider=False, parent=None):
        super().__init__(parent)
        self.levels = tuple(levels)
        self._table = pd.DataFrame(columns=list(self.levels))
        self._values = {}
        self._options = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        self.spins, self._labels, self._rows = {}, {}, {}
        for row, level in enumerate(self.levels):
            name = QLabel(level)
            spin = QSpinBox()
            spin.setToolTip(f"Step through the {level} values (index within the levels above)")
            spin.valueChanged.connect(lambda _, level=level: self._stepped(level))
            value = QLabel("–")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name, row, 0)
            grid.addWidget(spin, row, 1)
            grid.addWidget(value, row, 2)
            self.spins[level], self._labels[level], self._rows[level] = spin, value, (name, spin, value)
        row = len(self.levels)
        self.slider = None
        if slider and "trainId" in self.levels:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.valueChanged.connect(self.spins["trainId"].setValue)
            self.spins["trainId"].valueChanged.connect(self.slider.setValue)
            grid.addWidget(self.slider, row, 0, 1, 3)
            row += 1
        self.summary = QLabel("–")
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(self.summary, row, 0, 1, 3)
        grid.setColumnStretch(2, 1)

    # ------------------------------------------------------------------
    def setTraces(self, table):
        """table: DataFrame with one column per level (row i = trace i). Keeps the current trace when possible."""
        self._table = table[list(self.levels)].reset_index(drop=True)
        for k, level in enumerate(self.levels):
            above = list(self.levels[:k])
            counts = (self._table.groupby(above)[level].nunique() if above
                      else pd.Series([self._table[level].nunique()]))
            varies = bool(len(counts)) and int(counts.max()) > 1
            for widget in self._rows[level]:
                widget.setVisible(varies)
            if self.slider is not None and level == "trainId":
                self.slider.setVisible(varies)
        # New traces: keep the current values where they exist, otherwise start at the first one
        self._refresh(0, keepIndex=False)

    def position(self):
        """Index of the current trace, or None without traces."""
        if not len(self._table):
            return None
        mask = np.ones(len(self._table), dtype=bool)
        for level in self.levels:
            mask &= self._table[level].values == self._values.get(level)
        hits = np.flatnonzero(mask)
        return int(hits[0]) if len(hits) else 0

    def setPosition(self, pos):
        if not len(self._table):
            return
        row = self._table.iloc[min(max(int(pos), 0), len(self._table) - 1)]
        self._values = {level: row[level] for level in self.levels}
        self._refresh(0)
        self.changed.emit()

    def current(self):
        """{level: value} of the current trace (run None when unknown), or None without traces."""
        if not len(self._table):
            return None
        values = {level: self._values[level].item() if hasattr(self._values[level], "item") else self._values[level]
                  for level in self.levels}
        if values.get("run") == NO_RUN:
            values["run"] = None
        return values

    # ------------------------------------------------------------------
    def _stepped(self, level):
        options = self._options.get(level)
        if options is None or not len(options):
            return
        self._values[level] = options[min(self.spins[level].value(), len(options) - 1)]
        self._refresh(self.levels.index(level) + 1)
        self.changed.emit()

    def _refresh(self, start, keepIndex=True):
        """Recompute the options of the levels from ``start`` on, keeping their values where they still exist
        (otherwise their index with keepIndex, else the first value)."""
        mask = np.ones(len(self._table), dtype=bool)
        for k, level in enumerate(self.levels):
            column = self._table[level].values
            if k >= start:
                options = np.unique(column[mask])
                spin = self.spins[level]
                old = self._values.get(level)
                hits = np.flatnonzero(options == old) if old is not None else []
                fallback = min(spin.value(), max(len(options) - 1, 0)) if keepIndex else 0
                index = int(hits[0]) if len(hits) else fallback
                self._options[level] = options
                spin.blockSignals(True)
                spin.setMaximum(max(len(options) - 1, 0))
                spin.setValue(index)
                spin.setEnabled(len(options) > 1)
                spin.blockSignals(False)
                if self.slider is not None and level == "trainId":
                    self.slider.blockSignals(True)
                    self.slider.setMaximum(max(len(options) - 1, 0))
                    self.slider.setValue(index)
                    self.slider.blockSignals(False)
                self._values[level] = options[index] if len(options) else None
                self._labels[level].setText(f"{self._values[level]}  ({index + 1}/{len(options)})"
                                            if len(options) else "–")
            mask &= column == self._values.get(level)
        self._updateSummary()

    def _updateSummary(self):
        values = self.current()
        if values is None:
            self.summary.setText("–")
            return
        parts = [str(values[level]) for level in ("trainId", "pulseId") if level in values]
        text = " / ".join(parts)
        if values.get("run") is not None:
            text += f"  (run {values['run']})"
        self.summary.setText(text)


class DetectorSelector(QWidget):
    """'all detectors' or one detector, with buttons to step through them."""
    changed = pyqtSignal()

    def __init__(self, allLabel="all detectors", parent=None):
        super().__init__(parent)
        self.allLabel = allLabel
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.addItem(allLabel, None)
        self.combo.currentIndexChanged.connect(self.changed)
        previous, following = QToolButton(), QToolButton()
        previous.setArrowType(Qt.ArrowType.LeftArrow)
        following.setArrowType(Qt.ArrowType.RightArrow)
        previous.setToolTip("Previous detector")
        following.setToolTip("Next detector")
        previous.clicked.connect(lambda: self._step(-1))
        following.clicked.connect(lambda: self._step(1))
        layout.addWidget(previous)
        layout.addWidget(self.combo, stretch=1)
        layout.addWidget(following)

    def setDetectors(self, detectors):
        current = self.combo.currentData()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(self.allLabel, None)
        for det in detectors:
            self.combo.addItem(f"ToF {int(det)}", int(det))
        index = self.combo.findData(current)
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        self.combo.blockSignals(False)

    def current(self):
        """Selected detector, or None for all."""
        return self.combo.currentData()

    def select(self, detectors):
        """The detectors to show out of ``detectors``."""
        det = self.current()
        return [int(d) for d in detectors] if det is None or det not in set(int(d) for d in detectors) else [det]

    def _step(self, step):
        n = self.combo.count()
        if n > 1:
            # cycle through the single detectors; from "all" go to the first / last one
            index = self.combo.currentIndex()
            index = (1 if step > 0 else n - 1) if index == 0 else (index - 1 + step) % (n - 1) + 1
            self.combo.setCurrentIndex(index)
