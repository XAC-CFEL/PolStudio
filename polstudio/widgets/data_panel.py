"""Left dock: choose a data folder, select runs/files, train range and channels."""
import os

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .params import fromText, toText


class DataPanel(QWidget):
    folderRequested = pyqtSignal(str)
    loadRequested = pyqtSignal(dict)
    updateRequested = pyqtSignal(dict)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self.folderButton = QPushButton("Open data folder…")
        self.folderButton.clicked.connect(self.chooseFolder)
        row.addWidget(self.folderButton)
        layout.addLayout(row)
        self.folderLabel = QLabel("No folder")
        self.folderLabel.setWordWrap(True)
        layout.addWidget(self.folderLabel)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Run / file", "Trains", "trainId range", "λ (nm)", "Created"])
        self.tree.setRootIsDecorated(True)
        self.tree.itemChanged.connect(self._itemChanged)
        layout.addWidget(self.tree, stretch=1)

        trains = QGroupBox("Trains (positional slice per file)")
        grid = QGridLayout(trains)
        self.trainEdits = {}
        for col, key in enumerate(("trainStart", "trainStop", "trainStep")):
            grid.addWidget(QLabel(key[5:].lower()), 0, col)
            edit = QLineEdit(toText(self.session.state.get(key)))
            edit.setPlaceholderText("all")
            grid.addWidget(edit, 1, col)
            self.trainEdits[key] = edit
        layout.addWidget(trains)

        channels = QGroupBox("ADC channels (angle)")
        self.channelGrid = QGridLayout(channels)
        self.channelBoxes = {}
        layout.addWidget(channels)
        self._buildChannels(range(16))

        workers = QFormLayout()
        self.workersSpin = QSpinBox(minimum=0, maximum=os.cpu_count() or 1)
        self.workersSpin.setSpecialValueText("auto")
        self.workersSpin.setToolTip("Processes for 'Apply to all' and 'Find peaks', one detector each. "
                                    "auto = min(detectors, CPU cores − 1); 1 = no extra processes.")
        self.workersSpin.setValue(int(self.session.state.get("workers") or 0))
        self.workersSpin.valueChanged.connect(lambda v: self.session.state.__setitem__("workers", v))
        workers.addRow("worker processes", self.workersSpin)
        layout.addLayout(workers)

        buttons = QHBoxLayout()
        self.loadButton = QPushButton("Load selection")
        self.loadButton.setEnabled(False)
        self.loadButton.setToolTip("Load the selection from scratch (drops processed data and peaks)")
        self.loadButton.clicked.connect(self._emitLoad)
        buttons.addWidget(self.loadButton)
        self.updateButton = QPushButton("Update")
        self.updateButton.setEnabled(False)
        self.updateButton.setToolTip(
            "Run the pipeline for the current selection up to the selected tab, recomputing only what changed:\n"
            "deselected channels are removed, newly ticked channels are processed on their own and merged in;\n"
            "results made with other files, trains, pulse split, steps or peak settings are redone.")
        self.updateButton.clicked.connect(lambda: self.updateRequested.emit(self.selection()))
        buttons.addWidget(self.updateButton)
        layout.addLayout(buttons)
        self.infoLabel = QLabel("")
        self.infoLabel.setWordWrap(True)
        layout.addWidget(self.infoLabel)

    # ------------------------------------------------------------------
    def _angles(self):
        cfg = self.session.loaderConfig
        angles = list(cfg.get("angles", []))
        tof = cfg.get("ToF", list(range(len(angles))))
        if len(angles) == len(tof):
            return dict(zip(tof, angles))
        return dict(enumerate(angles))

    def _buildChannels(self, channels):
        for box in self.channelBoxes.values():
            box.setParent(None)
        self.channelBoxes = {}
        angles = self._angles()
        selected = self.session.state.get("channels") or self.session.loaderConfig.get("ToF") or list(channels)
        for i, ch in enumerate(channels):
            angle = angles.get(ch)
            box = QCheckBox(f"CH{ch:02d}" + (f" {angle:g}°" if angle is not None else ""))
            box.setChecked(ch in selected)
            self.channelGrid.addWidget(box, i // 4, i % 4)
            self.channelBoxes[ch] = box

    def chooseFolder(self):
        start = self.session.state.get("dataFolder") or ""
        folder = QFileDialog.getExistingDirectory(self, "Select folder with FLASH HDF5 files", start)
        if folder:
            self.folderRequested.emit(folder)

    def setOverview(self, overview, folder):
        """Fill the run/file tree from FlashH5Loader.scan()."""
        self.folderLabel.setText(str(folder))
        self.tree.blockSignals(True)
        self.tree.clear()
        selectedFiles = set(self.session.state.get("files") or [])
        if overview is None or overview.empty:
            self.folderLabel.setText(f"{folder}\nNo .h5 files found")
        else:
            for run, group in overview.groupby("run", dropna=False):
                runItem = QTreeWidgetItem(self.tree)
                runItem.setText(0, f"Run {run}")
                runItem.setText(1, str(int(group["nTrains"].sum())) if "nTrains" in group else "")
                runItem.setText(2, f"{int(group['trainId min'].min())} – {int(group['trainId max'].max())}"
                                if "trainId min" in group else "")
                wl = group["wavelength"].dropna() if "wavelength" in group else []
                runItem.setText(3, f"{wl.iloc[0]:.3f}" if len(wl) else "")
                runItem.setText(4, str(group["created"].min()) if "created" in group else "")
                runItem.setFlags(runItem.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
                for _, row in group.iterrows():
                    item = QTreeWidgetItem(runItem)
                    name = row["file"].replace("\\", "/").split("/")[-1]
                    item.setText(0, name)
                    item.setToolTip(0, row["file"])
                    item.setData(0, Qt.ItemDataRole.UserRole, row["file"])
                    if "error" in row and isinstance(row["error"], str):
                        item.setText(1, "error")
                        item.setToolTip(1, row["error"])
                        continue
                    item.setText(1, str(int(row["nTrains"])) if not np.isnan(row.get("nTrains", np.nan)) else "")
                    item.setText(2, f"{int(row['trainId min'])} – {int(row['trainId max'])}"
                                 if not np.isnan(row.get("trainId min", np.nan)) else "")
                    item.setText(3, f"{row['wavelength']:.3f}" if not np.isnan(row.get("wavelength", np.nan)) else "")
                    item.setText(4, str(row.get("created", "")))
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.CheckState.Checked if row["file"] in selectedFiles
                                       else Qt.CheckState.Unchecked)
            channels = sorted({ch for chs in overview.get("channels", []) if isinstance(chs, list) for ch in chs})
            if channels:
                self._buildChannels(channels)
        self.tree.expandAll()
        for col in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(col)
        self.tree.blockSignals(False)
        self._itemChanged()

    def _itemChanged(self, *args):
        self.loadButton.setEnabled(bool(self.selectedFiles()))
        self.updateButton.setEnabled(bool(self.selectedFiles()))

    def setUpdateTarget(self, tabName):
        self.updateButton.setText(f"Update → {tabName}")

    def selectedFiles(self):
        files = []
        for i in range(self.tree.topLevelItemCount()):
            run = self.tree.topLevelItem(i)
            for j in range(run.childCount()):
                item = run.child(j)
                if item.checkState(0) == Qt.CheckState.Checked:
                    files.append(item.data(0, Qt.ItemDataRole.UserRole))
        return files

    def selection(self):
        sel = {key: fromText(edit.text()) for key, edit in self.trainEdits.items()}
        sel["files"] = self.selectedFiles()
        sel["channels"] = [ch for ch, box in self.channelBoxes.items() if box.isChecked()]
        return sel

    def _emitLoad(self):
        self.loadRequested.emit(self.selection())

    def setLoadedInfo(self, text):
        self.infoLabel.setText(text)
