"""Ordered, toggleable preprocessing steps with a fast preview on a few trains."""
import copy

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from ..session import STEP_SPECS, traceLabel
from .params import ParamForm
from .plot_grid import TOL, PlotGrid

STEP_HELP = {
    "filterByIntensity": "Keep pulses whose GMD pulse energy lies between the thresholds.",
    "filterByHeight": "Set traces whose maximum lies outside the thresholds to NaN.",
    "subtractPeriodicNoise": "Tile the noiseRegion [start, stop) over the trace and subtract it.",
    "fftNoiseSubtraction": "Zero FFT components between minFreq and maxFreq (cycles/sample).",
    "tShift": "Additional shift of all detectors (int) or per detector ({det: n}).",
    "stack": "stackTrains: average blocks of trainStackSize consecutive trains per pulseId "
             "(empty = all trains); labelled with the block's first trainId. "
             "stackPulses: within each train, average groups of pulseStackSize consecutive pulseIds "
             "(empty = all pulses) between pulseStackStart and pulseStackStop; labelled with the group's "
             "first pulseId. Incomplete blocks/groups at the end are dropped.",
    "smooth": "Centred rolling mean of windowSize samples.",
    "normalize": "Divide by the global maximum, or by the maximum of detector ToF.",
}


class PreprocessTab(QWidget):
    previewRequested = pyqtSignal(list)    # trainIds
    applyRequested = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._currentRow = -1
        self._startTrainId = None

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        side = QWidget()
        sideLayout = QVBoxLayout(side)
        sideLayout.addWidget(QLabel("Steps (tick to enable, drag to reorder, duplicate to run a step again):"))
        self.stepList = QListWidget()
        self.stepList.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.stepList.currentRowChanged.connect(self._selectStep)
        self.stepList.model().rowsMoved.connect(self._syncStepsFromList)
        self.stepList.itemChanged.connect(self._syncStepsFromList)
        sideLayout.addWidget(self.stepList, stretch=1)
        stepButtons = QHBoxLayout()
        self.duplicateButton = QPushButton("Duplicate")
        self.duplicateButton.setToolTip("Add a copy of the selected step below it, e.g. to smooth twice")
        self.duplicateButton.clicked.connect(self._duplicateStep)
        self.removeButton = QPushButton("Remove")
        self.removeButton.setToolTip("Remove the selected copy (one step of each kind always stays in the list)")
        self.removeButton.clicked.connect(self._removeStep)
        stepButtons.addWidget(self.duplicateButton)
        stepButtons.addWidget(self.removeButton)
        sideLayout.addLayout(stepButtons)

        self.paramBox = QGroupBox("Parameters")
        paramLayout = QVBoxLayout(self.paramBox)
        self.helpLabel = QLabel("")
        self.helpLabel.setWordWrap(True)
        paramLayout.addWidget(self.helpLabel)
        self.form = ParamForm()
        self.form.changed.connect(self._storeParams)
        paramLayout.addWidget(self.form)
        sideLayout.addWidget(self.paramBox)

        preview = QGroupBox("Preview")
        previewLayout = QFormLayout(preview)
        self.nTrainsSpin = QSpinBox(minimum=1, maximum=100000)
        self.nTrainsSpin.setValue(20)
        self.nTrainsSpin.setToolTip("Number of trains, starting at the train selected in Inspect")
        previewLayout.addRow("trains", self.nTrainsSpin)
        self.startLabel = QLabel("–")
        previewLayout.addRow("first trainId", self.startLabel)
        self.pulseCombo = QComboBox()
        self.pulseCombo.currentIndexChanged.connect(self.redraw)
        previewLayout.addRow("show pulse", self.pulseCombo)
        self.showRaw = QCheckBox("overlay raw pulse (same trainId / pulseId, rescaled)")
        self.showRaw.setChecked(True)
        self.showRaw.toggled.connect(self.redraw)
        previewLayout.addRow(self.showRaw)
        self.baselineLabel = QLabel("")
        self.baselineLabel.setWordWrap(True)
        self.baselineLabel.setToolTip("Subtracted by the loader before these steps; change it in the Inspect tab")
        previewLayout.addRow(self.baselineLabel)
        buttons = QHBoxLayout()
        self.previewButton = QPushButton("Preview")
        self.previewButton.clicked.connect(self._requestPreview)
        self.applyButton = QPushButton("Apply to all loaded trains")
        self.applyButton.clicked.connect(self.applyRequested)
        buttons.addWidget(self.previewButton)
        buttons.addWidget(self.applyButton)
        previewLayout.addRow(buttons)
        self.statusLabel = QLabel("")
        self.statusLabel.setWordWrap(True)
        self.statusLabel.setTextFormat(Qt.TextFormat.RichText)
        previewLayout.addRow(self.statusLabel)
        sideLayout.addWidget(preview)
        splitter.addWidget(side)

        self.grid = PlotGrid()
        splitter.addWidget(self.grid)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 900])

        self._raw = None
        self._processed = None
        self.loadSteps()

    def showEvent(self, event):
        start, stop = (list(self.session.loaderConfig.get("baselineRegion") or [0, 10]) + [None])[:2]
        self.baselineLabel.setText(f"Baseline: samples [{start or 0}, {'end' if stop is None else stop}) are "
                                   "subtracted when loading (set in Inspect)")
        super().showEvent(event)

    # ------------------------------------------------------------------ steps
    @staticmethod
    def _makeItem(step):
        item = QListWidgetItem(STEP_SPECS[step["name"]]["label"])
        item.setData(Qt.ItemDataRole.UserRole, copy.deepcopy(step))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDropEnabled)
        item.setCheckState(Qt.CheckState.Checked if step.get("enabled") else Qt.CheckState.Unchecked)
        return item

    def loadSteps(self):
        self.stepList.blockSignals(True)
        self.stepList.clear()
        for step in self.session.state["steps"]:
            self.stepList.addItem(self._makeItem(step))
        self._relabel()
        self.stepList.blockSignals(False)
        self.stepList.setCurrentRow(0)

    def _stepNames(self):
        return [self.stepList.item(i).data(Qt.ItemDataRole.UserRole)["name"] for i in range(self.stepList.count())]

    def _relabel(self):
        """Number the copies of steps that occur more than once, in list order."""
        names = self._stepNames()
        seen = {}
        blocked = self.stepList.blockSignals(True)
        for i, name in enumerate(names):
            seen[name] = seen.get(name, 0) + 1
            suffix = f" ({seen[name]})" if names.count(name) > 1 else ""
            self.stepList.item(i).setText(STEP_SPECS[name]["label"] + suffix)
        self.stepList.blockSignals(blocked)

    def _hasCopies(self, row):
        names = self._stepNames()
        return 0 <= row < len(names) and names.count(names[row]) > 1

    def _duplicateStep(self):
        row = self.stepList.currentRow()
        if row < 0:
            return
        self._storeParams()
        step = copy.deepcopy(self.stepList.item(row).data(Qt.ItemDataRole.UserRole))
        step["enabled"] = True
        self.stepList.insertItem(row + 1, self._makeItem(step))
        self._syncStepsFromList()
        self.stepList.setCurrentRow(row + 1)

    def _removeStep(self):
        row = self.stepList.currentRow()
        if not self._hasCopies(row):
            return
        self._currentRow = -1   # the form still shows the removed step: do not store it anywhere
        self.stepList.takeItem(row)
        self._syncStepsFromList()
        self._selectStep(self.stepList.currentRow())

    def _selectStep(self, row):
        self._currentRow = row
        self.removeButton.setEnabled(self._hasCopies(row))
        if row < 0:
            return
        step = self.stepList.item(row).data(Qt.ItemDataRole.UserRole)
        self.paramBox.setTitle(f"Parameters – {step['name']}")
        self.helpLabel.setText(STEP_HELP.get(step["name"], ""))
        params = dict(STEP_SPECS[step["name"]]["params"])
        params.update(step.get("params", {}))
        self.form.setParams(params)

    def _storeParams(self):
        if self._currentRow < 0:
            return
        item = self.stepList.item(self._currentRow)
        step = item.data(Qt.ItemDataRole.UserRole)
        step["params"] = self.form.params()
        item.setData(Qt.ItemDataRole.UserRole, step)
        self._syncStepsFromList()

    def _syncStepsFromList(self, *args):
        steps = []
        for i in range(self.stepList.count()):
            item = self.stepList.item(i)
            step = item.data(Qt.ItemDataRole.UserRole)
            step["enabled"] = item.checkState() == Qt.CheckState.Checked
            steps.append(step)
        self.session.state["steps"] = steps
        self._relabel()
        self.removeButton.setEnabled(self._hasCopies(self.stepList.currentRow()))

    def steps(self):
        self._storeParams()
        return self.session.state["steps"]

    # ------------------------------------------------------------------ preview
    def setStartTrain(self, trainId):
        self._startTrainId = trainId
        self.startLabel.setText(str(trainId))

    def previewTrainIds(self):
        tids = self.session.trainIds()
        if not len(tids):
            return []
        start = int(np.searchsorted(tids, self._startTrainId)) if self._startTrainId is not None else 0
        return [int(t) for t in tids[start:start + self.nTrainsSpin.value()]]

    def _requestPreview(self):
        self.steps()
        self.previewRequested.emit(self.previewTrainIds())

    def showPreview(self, raw, processed):
        self._raw, self._processed = raw, processed
        current = self.pulseCombo.currentData()
        self.pulseCombo.blockSignals(True)
        self.pulseCombo.clear()
        index = processed["pulse"].to_index()
        for i in range(len(index)):
            self.pulseCombo.addItem(traceLabel(processed, i, "trainId {tid} / pulse {pid}"), i)
        idx = self.pulseCombo.findData(current)
        self.pulseCombo.setCurrentIndex(idx if idx >= 0 else 0)
        self.pulseCombo.blockSignals(False)
        text = f"raw {dict(raw.sizes)} → processed {dict(processed.sizes)}"
        if processed.sizes["pulse"] and raw.sizes["pulse"] > processed.sizes["pulse"]:
            text += f"<br>each processed trace averages ≈ {raw.sizes['pulse'] / processed.sizes['pulse']:.3g} raw pulses"
        self.statusLabel.setText(text)
        self.redraw()

    def redraw(self):
        if self._processed is None:
            return
        processed, raw = self._processed, self._raw
        self.grid.setDetectors(processed["detector"].values)
        self.grid.clear()
        i = self.pulseCombo.currentData() or 0
        tid, pid = processed["pulse"].to_index()[i]
        x = processed["sample"].values
        for det, plot in self.grid.plots.items():
            y = processed.sel(detector=det).isel(pulse=i).values
            if self.showRaw.isChecked():
                # The stacked trace carries the first trainId / pulseId of its block: show that raw pulse
                rawIdx = raw["pulse"].to_index()
                rawTids = rawIdx.get_level_values("trainId")
                rawPids = rawIdx.get_level_values("pulseId")
                matches = np.nonzero((rawTids == tid) & (rawPids == pid))[0]
                if not len(matches):
                    matches = np.nonzero(rawPids == pid)[0]
                if len(matches):
                    r = raw.sel(detector=det).isel(pulse=matches[0]).values
                    scale = np.nanmax(np.abs(y)) / np.nanmax(np.abs(r)) if np.nanmax(np.abs(r)) > 0 else 1
                    plot.plot(raw["sample"].values, r * scale, pen=pg.mkPen(TOL["grey"], width=1))
            plot.plot(x, y, pen=pg.mkPen(TOL["blue"], width=1))

    def restrictPreview(self, detectors):
        """Drop deselected channels from the current preview (new channels appear with the next preview)."""
        if self._processed is None:
            return
        keep = [d for d in self._processed["detector"].values if d in set(detectors)]
        self._processed = self._processed.sel(detector=keep)
        self._raw = self._raw.sel(detector=[d for d in self._raw["detector"].values if d in set(detectors)])
        self.redraw()

    def setProcessedInfo(self, text):
        self.statusLabel.setText(text)
