"""
Pipeline Board dialog for SystS — Windows 98 style.

Shows all 11 pipeline stages with ✅/⏳/❌/⬜ status icons, elapsed time,
and buttons to run or stop the full pipeline.
"""
from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QWidget,
    QFrame,
)

from gui.theme import COLORS, _RAISED, _SUNKEN


# Stage definitions: (stage_num, display_name)
STAGES = [
    (1, "Search"),
    (2, "Score"),
    (3, "Extract"),
    (4, "Validate"),
    (5, "Audit"),
    (6, "Synthesize"),
    (7, "Fulltext"),
    (8, "Interrater"),
    (9, "Robustness"),
    (10, "Verify"),
    (11, "Compare"),
]

STATUS_ICONS = {
    "pending":  "⬜",
    "running":  "⏳",
    "done":     "✅",
    "failed":   "❌",
}

STATUS_LABELS = {
    "pending":  "Pending",
    "running":  "Running…",
    "done":     "Done",
    "failed":   "Failed",
}


class PipelineBoardDialog(QDialog):
    """Non-modal floating dialog showing real-time pipeline stage status."""

    run_all_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("Pipeline Status Board")
        self.setMinimumSize(520, 380)
        self.setModal(False)

        # Internal state: stage_num (1-based) → {status, start_time, elapsed}
        self._stage_data: dict[int, dict] = {}
        for num, name in STAGES:
            self._stage_data[num] = {
                "name": name,
                "status": "pending",
                "start_time": None,
                "elapsed": None,
            }

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_running_stages)
        self._timer.start(500)  # update every 500 ms

        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title bar area
        title = QLabel("SystS Pipeline — Stage Status")
        title.setStyleSheet(
            f"background-color: {COLORS['primary']}; color: {COLORS['selection_fg']}; "
            f"font-weight: bold; padding: 4px 8px;"
        )
        layout.addWidget(title)

        # Stage table
        self.table = QTableWidget(len(STAGES), 4)
        self.table.setHorizontalHeaderLabels(["Stage", "Name", "Status", "Elapsed"])
        self.table.setStyleSheet(
            f"QTableWidget {{ background-color: #ffffff; {_SUNKEN} "
            f"selection-background-color: {COLORS['selection_bg']}; "
            f"selection-color: {COLORS['selection_fg']}; }}"
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)

        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self._populate_table()
        layout.addWidget(self.table)

        # Button row
        btn_frame = QFrame()
        btn_frame.setStyleSheet(f"QFrame {{ {_RAISED} background-color: {COLORS['bg']}; }}")
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(6, 4, 6, 4)

        self._run_btn = QPushButton("▶  Run All (S1→S11)")
        self._run_btn.setToolTip("Run the complete pipeline from Stage 1 through Stage 11")
        self._run_btn.clicked.connect(self._on_run_clicked)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setToolTip("Request cancellation of the current stage")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset all stages to Pending")
        reset_btn.clicked.connect(self._reset_all)

        btn_layout.addWidget(self._run_btn)
        btn_layout.addWidget(self._stop_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(reset_btn)

        layout.addWidget(btn_frame)

    def _populate_table(self) -> None:
        """Fill table rows from internal stage data."""
        for row, (num, _name) in enumerate(STAGES):
            d = self._stage_data[num]
            status = d["status"]
            icon = STATUS_ICONS.get(status, "⬜")
            status_label = STATUS_LABELS.get(status, status.title())

            # Stage number
            num_item = QTableWidgetItem(f"Stage {num}")
            num_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, num_item)

            # Stage name with icon prefix
            name_item = QTableWidgetItem(f"{icon} {d['name']}")
            self.table.setItem(row, 1, name_item)

            # Status
            status_item = QTableWidgetItem(status_label)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if status == "done":
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            elif status == "failed":
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == "running":
                status_item.setForeground(Qt.GlobalColor.darkBlue)
            self.table.setItem(row, 2, status_item)

            # Elapsed time
            elapsed = d["elapsed"]
            if elapsed is not None:
                elapsed_str = f"{elapsed:.1f}s"
            elif status == "running" and d["start_time"] is not None:
                elapsed_str = f"{time.time() - d['start_time']:.1f}s"
            else:
                elapsed_str = "—"
            elapsed_item = QTableWidgetItem(elapsed_str)
            elapsed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, elapsed_item)

    # ── Public API ────────────────────────────────────────────────────

    def update_stage(self, stage_num: int, status: str, elapsed_sec: Optional[float] = None) -> None:
        """Update a single stage row.

        Parameters
        ----------
        stage_num:
            1-based stage number (1–11).
        status:
            One of "pending", "running", "done", "failed".
        elapsed_sec:
            Elapsed seconds; computed automatically for running stages if omitted.
        """
        if stage_num not in self._stage_data:
            return
        d = self._stage_data[stage_num]
        d["status"] = status
        if status == "running" and d["start_time"] is None:
            d["start_time"] = time.time()
        if elapsed_sec is not None:
            d["elapsed"] = elapsed_sec
        elif status in ("done", "failed") and d["start_time"] is not None:
            d["elapsed"] = time.time() - d["start_time"]

        # Update run/stop button state
        any_running = any(v["status"] == "running" for v in self._stage_data.values())
        self._run_btn.setEnabled(not any_running)
        self._stop_btn.setEnabled(any_running)

        self._refresh_row(stage_num)

    def _refresh_row(self, stage_num: int) -> None:
        """Refresh a single table row without rebuilding everything."""
        row = stage_num - 1
        if row < 0 or row >= self.table.rowCount():
            return
        d = self._stage_data[stage_num]
        status = d["status"]
        icon = STATUS_ICONS.get(status, "⬜")
        status_label = STATUS_LABELS.get(status, status.title())

        name_item = self.table.item(row, 1)
        if name_item:
            name_item.setText(f"{icon} {d['name']}")

        status_item = self.table.item(row, 2)
        if status_item:
            status_item.setText(status_label)
            if status == "done":
                status_item.setForeground(Qt.GlobalColor.darkGreen)
            elif status == "failed":
                status_item.setForeground(Qt.GlobalColor.red)
            elif status == "running":
                status_item.setForeground(Qt.GlobalColor.darkBlue)
            else:
                status_item.setForeground(Qt.GlobalColor.black)

        elapsed = d["elapsed"]
        if elapsed is not None:
            elapsed_str = f"{elapsed:.1f}s"
        elif status == "running" and d["start_time"] is not None:
            elapsed_str = f"{time.time() - d['start_time']:.1f}s"
        else:
            elapsed_str = "—"
        elapsed_item = self.table.item(row, 3)
        if elapsed_item:
            elapsed_item.setText(elapsed_str)

    def _tick_running_stages(self) -> None:
        """Called by QTimer to refresh elapsed time for any running stages."""
        for num, d in self._stage_data.items():
            if d["status"] == "running":
                self._refresh_row(num)

    # ── Button slots ──────────────────────────────────────────────────

    def _on_run_clicked(self) -> None:
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self.run_all_requested.emit()

    def _on_stop_clicked(self) -> None:
        self.stop_requested.emit()

    def _reset_all(self) -> None:
        for num in self._stage_data:
            self._stage_data[num]["status"] = "pending"
            self._stage_data[num]["start_time"] = None
            self._stage_data[num]["elapsed"] = None
        self._populate_table()
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
