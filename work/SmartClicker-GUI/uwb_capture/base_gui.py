"""Base Tk GUI for UWB measurement capture."""

from __future__ import annotations

import math
import queue
import re
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .common import (
    ALERT_PROMPT_COOLDOWN_SECONDS,
    DEFAULT_DB_NAME,
    INSTABILITY_WINDOW_SIZE,
    MAX_TABLE_ROWS,
    ParsedRecord,
    export_timestamp,
    now_local_iso,
    safe_filename,
    safe_float,
    safe_int,
)
from .bluetooth_io import (
    DEFAULT_DEVICE_NAME,
    DEFAULT_LOG_TX_UUID,
    DEFAULT_PACKET_RX_UUID,
    DEFAULT_PACKET_TX_UUID,
    BluetoothDeviceInfo,
    BluetoothScanner,
    BluetoothWorker,
)
from .protocol import (
    CommandStatus,
    ImecPacket,
    MessageType,
    ProtocolError,
    build_ml_start_collection_packet,
    build_ml_start_fast_ranging_packet,
    command_result_summary,
    command_sample_count_from_packet,
    command_status_from_packet,
    decode_packet,
    format_device_id,
    next_sequence,
    packet_summary,
    parse_device_id,
    records_from_packet,
)
from .store import MeasurementStore

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = APP_DIR / "Measurements" / "GUI_Captures"
ML_COLLECTION_MODE_FULL = "full_diagnostics"
ML_COLLECTION_MODE_FAST = "fast_ranging"
ML_COLLECTION_MODE_LIVE = "live_tracking"
ML_COLLECTION_MODE_LABELS = {
    ML_COLLECTION_MODE_FULL: "Full diagnostics / CIR",
    ML_COLLECTION_MODE_FAST: "Fast ranging only",
    ML_COLLECTION_MODE_LIVE: "Live tracking",
}
ML_COLLECTION_MODE_DETAILS = {
    ML_COLLECTION_MODE_FULL: (
        "Full diagnostics / CIR sends 0x8000 and keeps post-burst diagnostic/CIR handling."
    ),
    ML_COLLECTION_MODE_FAST: (
        "Fast ranging only sends 0x8001, stores range rows only, and expects no post-burst CIR."
    ),
    ML_COLLECTION_MODE_LIVE: (
        "Live tracking sends 0x8003, keeps a watchdog heartbeat alive, and stores range rows only."
    ),
}
FULL_DIAGNOSTICS_COMMAND_TIMEOUT_MS = 75_000
FAST_RANGING_COMMAND_TIMEOUT_MS = 45_000

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_BT_DEBUG_RE = re.compile(r"<(?:dbg|err|wrn|inf)> bt_|bt_\w+:", re.IGNORECASE)
_BT_DROPPED_BANNER_RE = re.compile(r"^-+\s*\d*\s*$")
_BT_DROPPED_COUNT_RE = re.compile(r"^\d{1,6}$")
_MESSAGES_DROPPED_RE = re.compile(r"messages\s+dropped", re.IGNORECASE)


def _clean_log_line(line: str) -> str:
    return _ANSI_RE.sub("", line).strip()


def _is_bt_debug_noise(line: str) -> bool:
    if not line:
        return True
    if _BT_DEBUG_RE.search(line):
        return True
    if _MESSAGES_DROPPED_RE.search(line):
        return True
    if _BT_DROPPED_BANNER_RE.match(line):
        return True
    if _BT_DROPPED_COUNT_RE.match(line):
        return True
    return False


def _command_status_name(status: int | None) -> str:
    if status is None:
        return "UNKNOWN"
    try:
        return CommandStatus(status).name
    except ValueError:
        return f"STATUS_{status}"


def _decode_cir_bytes(cir_hex: str | None) -> list[float]:
    """Decode CIR data into magnitude values.

    The current full CIR window is 1152 bytes: 192 complex samples, each 6 bytes
    (signed 24-bit little-endian real + signed 24-bit little-endian imaginary).
    Returns the magnitude sqrt(re^2 + im^2) per point.
    """
    if not cir_hex:
        return []
    try:
        raw = bytes.fromhex(cir_hex)
    except ValueError:
        return []
    mags: list[float] = []
    for i in range(0, len(raw) - 5, 6):
        re = int.from_bytes(raw[i:i + 3], "little", signed=True)
        im = int.from_bytes(raw[i + 3:i + 6], "little", signed=True)
        mags.append(math.sqrt(re * re + im * im))
    return mags or [float(b) for b in raw]


def _cir_first_path_local_index(record: ParsedRecord, point_count: int) -> int | None:
    fp_index = record.cir_first_path_index
    if fp_index is None:
        return None
    start_index = record.cir_start_index
    local_index = fp_index - start_index if start_index is not None else fp_index
    if 0 <= local_index < point_count:
        return local_index
    return None


def _cir_record_matches_diag(record: ParsedRecord, anchor_id: str, event_seq: int | None) -> bool:
    if record.anchor_id != anchor_id:
        return False
    if event_seq is None:
        return True
    return record.event_seq == event_seq


class UwbCaptureApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("UWB Measurement Capture")
        self.geometry("1200x760")
        self.minsize(980, 620)
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.bluetooth_worker: BluetoothWorker | None = None
        self.connected_device: str | None = None
        self.bluetooth_manual_disconnect_pending = False
        self.device_infos: dict[str, BluetoothDeviceInfo] = {}
        self.protocol_sequence = 0
        self.protocol_session_seed = int(time.time()) & 0xFFFFFFFF
        self.ml_command_in_flight = False
        self.ml_pending_session: int | None = None
        self.ml_pending_seq: int | None = None
        self.ml_pending_mode: str | None = None
        self.ml_timeout_after_id: str | None = None
        self.ml_expected_sample_notifications: int | None = None
        self.ml_received_sample_notifications = 0
        self.capture_active = False
        self.current_session_id: str | None = None
        self.store: MeasurementStore | None = None
        self.table_rows = 0
        self.last_alert_prompt: dict[str, float] = {}
        self.last_instability_prompt: dict[str, float] = {}
        self.anchor_distance_windows: dict[str, list[float]] = {}
        self.last_sample_row_by_anchor: dict[str, int] = {}
        self.collection_sample_rows: list[int] = []
        self.collection_sample_rows_by_anchor: dict[str, list[int]] = {}
        self.collection_table_items: list[str] = []
        self.collection_table_items_by_anchor: dict[str, list[str]] = {}
        self.collection_cir_records_by_anchor: dict[str, list[ParsedRecord]] = {}
        self.collection_event_seq: int | None = None
        self.pending_diagnostics: list[ParsedRecord] = []
        self.cir_reassembly_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.cir_history: list[ParsedRecord] = []
        self._build_variables()
        self._build_style()
        self._build_layout()
        self.after(80, self.process_events)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_variables(self) -> None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.output_dir_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.db_path_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR / DEFAULT_DB_NAME))
        self.device_var = tk.StringVar()
        self.device_filter_var = tk.StringVar(value=DEFAULT_DEVICE_NAME)
        self.notify_uuid_var = tk.StringVar(value=DEFAULT_PACKET_TX_UUID)
        self.write_uuid_var = tk.StringVar(value=DEFAULT_PACKET_RX_UUID)
        self.log_uuid_var = tk.StringVar(value=DEFAULT_LOG_TX_UUID)
        self.auto_connect_var = tk.BooleanVar(value=False)
        self.host_id_var = tk.StringVar(value="1")
        self.clicker_id_var = tk.StringVar(value="0")
        self.ml_session_id_var = tk.StringVar(value=str(self.protocol_session_seed))
        self.ml_sample_count_var = tk.StringVar(value="8")
        self.ml_discovery_slot_count_var = tk.StringVar(value="8")
        self.ml_collection_mode_var = tk.StringVar(value=ML_COLLECTION_MODE_FULL)
        self.ml_mode_detail_var = tk.StringVar(value=ML_COLLECTION_MODE_DETAILS[ML_COLLECTION_MODE_FULL])
        self.constellation_var = tk.StringVar()
        self.los_var = tk.BooleanVar(value=True)
        self.nlos_var = tk.BooleanVar(value=False)
        self.ground_truth_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="0.50")
        self.instability_threshold_var = tk.StringVar(value="0.35")
        self.status_var = tk.StringVar(value="Disconnected")
        self.session_status_var = tk.StringVar(value="No active session")
        self.sample_count_var = tk.StringVar(value="0")
        self.alert_count_var = tk.StringVar(value="0")
        self.raw_count_var = tk.StringVar(value="0")

    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9, "bold"))

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = self._build_scrollable_frame(root, width=480)
        self._scrollable_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_connection_panel(left)
        self._build_session_panel(left)
        self._build_storage_panel(left)
        self._build_stats_panel(left)
        self._build_live_panel(right)
        self._build_log_panel(right)

    def _build_scrollable_frame(self, parent: Any, *, width: int) -> ttk.Frame:
        container = ttk.Frame(parent)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(container, width=width, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window(0, 0, window=inner, anchor="nw")
        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(inner_id, width=event.width),
        )
        self._bind_scroll_wheel(canvas)
        self._scrollable_container = container
        return inner

    def _bind_scroll_wheel(self, canvas: Any) -> None:
        def scroll(event: Any) -> None:
            if hasattr(event, "delta") and event.delta:
                canvas.yview_scroll(int(-event.delta / 120), "units")
            elif hasattr(event, "num"):
                canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

        canvas.bind_all("<MouseWheel>", scroll)
        canvas.bind_all("<Button-4>", scroll)
        canvas.bind_all("<Button-5>", scroll)

    def _build_connection_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Bluetooth Connection", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Device").grid(row=0, column=0, sticky="w", pady=2)
        self.device_combo = ttk.Combobox(frame, textvariable=self.device_var, state="normal")
        self.device_combo.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(frame, text="Scan", command=self.refresh_devices).grid(row=0, column=2, padx=(6, 0), pady=2)

        ttk.Label(frame, text="Name filter").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.device_filter_var).grid(
            row=1,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=2,
        )
        ttk.Label(frame, text="Packet TX").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.notify_uuid_var).grid(
            row=2,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=2,
        )
        ttk.Label(frame, text="Packet RX").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.write_uuid_var).grid(
            row=3,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=2,
        )
        ttk.Label(frame, text="Log TX").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.log_uuid_var).grid(
            row=4,
            column=1,
            columnspan=2,
            sticky="ew",
            pady=2,
        )
        ttk.Checkbutton(frame, text="Auto-connect", variable=self.auto_connect_var).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
            pady=2,
        )

        button_row = ttk.Frame(frame)
        button_row.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)
        self.connect_button = ttk.Button(button_row, text="Connect", command=self.connect_selected_device)
        self.connect_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(button_row, text="Disconnect", command=self.disconnect_transport).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(5, 0),
        )

        ttk.Label(frame, textvariable=self.status_var, style="Status.TLabel").grid(
            row=7,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

    def _build_session_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Measurement Session", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        rows = [
            ("Constellation", self.constellation_var),
            ("Laser ground truth m", self.ground_truth_var),
            ("LOS alert threshold m", self.threshold_var),
            ("Stability alert std m", self.instability_threshold_var),
        ]
        for row_index, (label, variable) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row_index, column=0, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=variable).grid(row=row_index, column=1, columnspan=2, sticky="ew", pady=2)

        condition_row = ttk.Frame(frame)
        condition_row.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(condition_row, text="LOS", variable=self.los_var).pack(side=tk.LEFT)
        ttk.Checkbutton(condition_row, text="NLOS", variable=self.nlos_var).pack(side=tk.LEFT, padx=(14, 0))

        self._build_serial_control(frame, 5)

        action_row = ttk.Frame(frame)
        action_row.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        action_row.columnconfigure(0, weight=1)
        action_row.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(action_row, text="Start Recording", style="Accent.TButton", command=self.start_capture)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.stop_button = ttk.Button(action_row, text="Stop Recording", command=self.stop_capture)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        ttk.Label(
            action_row,
            text="Recording saves incoming ML samples to the database. Start it once, then trigger captures below.",
            wraplength=320,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Label(frame, textvariable=self.session_status_var, style="Status.TLabel").grid(
            row=7,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(8, 0),
        )

    def _build_serial_control(self, parent: ttk.Frame, row: int) -> None:
        control = ttk.LabelFrame(parent, text="ML Collection", padding=6)
        control.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        control.columnconfigure(1, weight=1)
        control.columnconfigure(3, weight=1)

        ttk.Label(control, text="Host ID").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(control, textvariable=self.host_id_var, width=18).grid(
            row=0,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=(6, 0),
            pady=2,
        )
        ttk.Label(control, text="Clicker ID").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(control, textvariable=self.clicker_id_var, width=18).grid(
            row=1,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=(6, 0),
            pady=2,
        )

        ttk.Label(control, text="Session ID").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(control, textvariable=self.ml_session_id_var, width=12).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(6, 12),
            pady=(8, 0),
        )
        ttk.Label(control, text="Samples / anchor").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(control, textvariable=self.ml_sample_count_var, width=8).grid(
            row=2,
            column=3,
            sticky="ew",
            padx=(6, 0),
            pady=(8, 0),
        )

        ttk.Label(control, text="Discovery slots").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(control, textvariable=self.ml_discovery_slot_count_var, width=8).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(6, 0),
            pady=2,
        )

        ttk.Label(control, text="Collection mode").grid(row=4, column=0, sticky="w", pady=(8, 0))
        mode_frame = ttk.Frame(control)
        mode_frame.grid(row=4, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=(8, 0))
        for column in range(2):
            mode_frame.columnconfigure(column, weight=1)
        ttk.Radiobutton(
            mode_frame,
            text=ML_COLLECTION_MODE_LABELS[ML_COLLECTION_MODE_FULL],
            value=ML_COLLECTION_MODE_FULL,
            variable=self.ml_collection_mode_var,
            command=self._update_ml_mode_detail,
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Radiobutton(
            mode_frame,
            text=ML_COLLECTION_MODE_LABELS[ML_COLLECTION_MODE_FAST],
            value=ML_COLLECTION_MODE_FAST,
            variable=self.ml_collection_mode_var,
            command=self._update_ml_mode_detail,
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(
            control,
            textvariable=self.ml_mode_detail_var,
            wraplength=320,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))

        self.ml_collect_button = ttk.Button(
            control,
            text="Trigger ML Collection",
            style="Accent.TButton",
            command=self.send_ml_start_collection,
        )
        self.ml_collect_button.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        ttk.Label(
            control,
            text="Sample count and discovery slots are sent with either command. Fast mode lets firmware reuse a recent four-anchor ranging cache or run discovery automatically.",
            wraplength=320,
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.ml_status_var = tk.StringVar(value="No collection command sent")
        ttk.Label(control, textvariable=self.ml_status_var, style="Status.TLabel").grid(
            row=8,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(6, 0),
        )

    def _build_storage_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Storage", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Output folder").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.output_dir_var).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(frame, text="Browse", command=self.choose_output_dir).grid(row=0, column=2, padx=(6, 0), pady=2)

        ttk.Label(frame, text="SQLite DB").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=self.db_path_var).grid(row=1, column=1, columnspan=2, sticky="ew", pady=2)

        ttk.Button(frame, text="Export Current Session to Excel", command=self.export_current_session).grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(8, 0),
        )

    def _build_stats_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Current Session Counts", padding=8)
        frame.pack(fill=tk.X, pady=(0, 8))
        rows = [
            ("Samples", self.sample_count_var),
            ("Alerts", self.alert_count_var),
            ("Raw lines", self.raw_count_var),
        ]
        for index, (label, variable) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=index, column=0, sticky="w", pady=2)
            ttk.Label(frame, textvariable=variable, style="Status.TLabel").grid(row=index, column=1, sticky="e", pady=2)
        frame.columnconfigure(1, weight=1)

    def _build_live_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Current Live Table", style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Clear", command=self.clear_live_view).grid(row=0, column=1, sticky="e")

        columns = ("time", "anchor", "sample", "distance", "error", "rx", "fp", "cir", "status", "source")
        self.sample_table = ttk.Treeview(frame, columns=columns, show="headings", height=14)
        headings = {
            "time": "Time",
            "anchor": "Anchor",
            "sample": "Sample",
            "distance": "Distance m",
            "error": "Error m",
            "rx": "RX dBm",
            "fp": "FP dBm",
            "cir": "CIR power",
            "status": "Status",
            "source": "Source",
        }
        widths = {
            "time": 80,
            "anchor": 90,
            "sample": 80,
            "distance": 100,
            "error": 90,
            "rx": 90,
            "fp": 90,
            "cir": 95,
            "status": 120,
            "source": 110,
        }
        for column in columns:
            self.sample_table.heading(column, text=headings[column])
            self.sample_table.column(column, width=widths[column], anchor="center")

        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.sample_table.yview)
        self.sample_table.configure(yscrollcommand=yscroll.set)
        self.sample_table.grid(row=1, column=0, sticky="nsew")
        yscroll.grid(row=1, column=1, sticky="ns")
        self.sample_table.tag_configure("alert", background="#f8d7da")
        self.sample_table.tag_configure("failure", background="#f5c2c7")
        self.sample_table.tag_configure("summary", background="#d1e7dd")

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky="nsew")

        raw_frame = ttk.Frame(notebook)
        raw_frame.rowconfigure(0, weight=1)
        raw_frame.columnconfigure(0, weight=1)
        self.raw_text = tk.Text(raw_frame, height=10, wrap="none")
        raw_scroll = ttk.Scrollbar(raw_frame, orient=tk.VERTICAL, command=self.raw_text.yview)
        self.raw_text.configure(yscrollcommand=raw_scroll.set)
        self.raw_text.grid(row=0, column=0, sticky="nsew")
        raw_scroll.grid(row=0, column=1, sticky="ns")
        notebook.add(raw_frame, text="Raw data view")

        alert_frame = ttk.Frame(notebook)
        alert_frame.rowconfigure(0, weight=1)
        alert_frame.columnconfigure(0, weight=1)
        self.alert_text = tk.Text(alert_frame, height=10, wrap="word")
        alert_scroll = ttk.Scrollbar(alert_frame, orient=tk.VERTICAL, command=self.alert_text.yview)
        self.alert_text.configure(yscrollcommand=alert_scroll.set)
        self.alert_text.grid(row=0, column=0, sticky="nsew")
        alert_scroll.grid(row=0, column=1, sticky="ns")
        notebook.add(alert_frame, text="Alerts")

        self.cir_notebook_tab = self._build_cir_panel(notebook)

    def _build_cir_panel(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)

        header = ttk.Frame(frame)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Channel Impulse Response samples", style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.cir_status_var = tk.StringVar(value="No CIR samples yet")
        ttk.Label(header, textvariable=self.cir_status_var).grid(row=1, column=0, sticky="w")

        list_frame = ttk.Frame(frame)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.cir_listbox = tk.Listbox(list_frame, height=12, exportselection=False)
        cir_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.cir_listbox.yview)
        self.cir_listbox.configure(yscrollcommand=cir_scroll.set)
        self.cir_listbox.grid(row=0, column=0, sticky="nsew")
        cir_scroll.grid(row=0, column=1, sticky="ns")
        self.cir_listbox.bind("<<ListboxSelect>>", self._on_cir_selected)

        self.cir_canvas = tk.Canvas(frame, background="#ffffff", highlightthickness=1, highlightbackground="#c8c8c8")
        self.cir_canvas.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.cir_canvas.bind("<Configure>", lambda _event: self._draw_cir_plot())

        notebook.add(frame, text="CIR Visualizer")
        return frame

    def _on_cir_selected(self, _event: Any = None) -> None:
        self._draw_cir_plot()

    def _draw_cir_plot(self) -> None:
        canvas = getattr(self, "cir_canvas", None)
        if canvas is None:
            return
        canvas.delete("all")
        if not hasattr(self, "cir_listbox"):
            return
        selection = self.cir_listbox.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index >= len(self.cir_history):
            return
        record = self.cir_history[index]
        values = _decode_cir_bytes(record.cir_raw)
        if not values:
            canvas.create_text(
                canvas.winfo_width() // 2 or 100,
                canvas.winfo_height() // 2 or 50,
                text="No CIR data in this sample",
                fill="#666666",
            )
            return

        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 150)
        left, top, right, bottom = 40, 24, width - 16, height - 32
        plot_w = max(right - left, 1)
        plot_h = max(bottom - top, 1)
        max_val = max(values)
        min_val = 0
        span = max(max_val - min_val, 1)
        n = len(values)

        # Draw axes
        canvas.create_line(left, bottom, right, bottom, fill="#cccccc")
        canvas.create_line(left, top, left, bottom, fill="#cccccc")
        canvas.create_text(left - 6, top - 6, text=f"{max_val:.0f}", anchor="e", fill="#666666")
        canvas.create_text(left - 6, bottom + 6, text="0", anchor="e", fill="#666666")

        # Draw magnitude as a filled line graph
        points = []
        for i, value in enumerate(values):
            x = left + (plot_w * i / max(n - 1, 1))
            y = bottom - (value - min_val) / span * plot_h
            points.extend([x, y])
        if len(points) >= 4:
            canvas.create_line(points, fill="#1f77b4", width=1, smooth=False)

        # Draw first path marker. The firmware reports an absolute accumulator
        # index; the plot window starts at cir_start_index when available.
        fp_index = record.cir_first_path_index
        fp_local_index = _cir_first_path_local_index(record, n)
        if fp_index is not None and fp_local_index is not None:
            fp_x = left + (plot_w * fp_local_index / max(n - 1, 1))
            canvas.create_line(fp_x, top, fp_x, bottom, fill="#d62728", width=2, dash=(4, 2))
            canvas.create_text(fp_x, top - 8, text=f"FP={fp_index}", anchor="s", fill="#d62728")

        # X-axis labels (start, middle, end)
        start_idx = getattr(record, "cir_start_index", None) or 0
        for frac, label in [(0, str(start_idx)), (0.5, str(start_idx + n // 2)), (1, str(start_idx + n - 1))]:
            x = left + plot_w * frac
            canvas.create_text(x, bottom + 12, text=label, anchor="n", fill="#666666")

        # Info text
        anchor = record.anchor_id or "?"
        sample_idx = record.sample_index if record.sample_index is not None else "?"
        distance = f"{record.distance_m:.3f} m" if record.distance_m is not None else "n/a"
        status = record.status or "?"
        n_text = f"{n} samples"
        fp_text = f"  FP={fp_index}" if fp_index is not None else ""
        canvas.create_text(
            right,
            top - 8,
            text=f"anchor {anchor}  sample {sample_idx}  {distance}  {status}  {n_text}{fp_text}",
            anchor="ne",
            fill="#333333",
        )

    def _add_cir_sample(self, record: ParsedRecord) -> None:
        if record.kind not in ("sample", "failure"):
            return
        self.cir_history.append(record)
        if len(self.cir_history) > 200:
            del self.cir_history[: len(self.cir_history) - 200]
        label = self._cir_label(record, len(self.cir_history) - 1)
        self.cir_listbox.insert(tk.END, label)
        self.cir_listbox.selection_clear(0, tk.END)
        self.cir_listbox.selection_set(tk.END)
        self.cir_listbox.see(tk.END)
        self.cir_status_var.set(f"{len(self.cir_history)} CIR sample(s) stored")
        if record.cir_raw:
            self._draw_cir_plot()

    def _cir_label(self, record: ParsedRecord, index: int) -> str:
        anchor = record.anchor_id or "?"
        sample_idx = record.sample_index if record.sample_index is not None else "?"
        distance = f"{record.distance_m:.3f}m" if record.distance_m is not None else "n/a"
        return f"#{index} anchor {anchor[-8:]} s{sample_idx} {distance} {record.status or ''}"

    def _reset_trigger_collection_state(self) -> None:
        self.collection_sample_rows.clear()
        self.collection_sample_rows_by_anchor.clear()
        self.collection_table_items.clear()
        self.collection_table_items_by_anchor.clear()
        self.collection_cir_records_by_anchor.clear()
        self.collection_event_seq = None
        self.pending_diagnostics.clear()
        self.cir_reassembly_groups.clear()

    def _ml_collection_mode(self, override: str | None = None) -> str:
        mode = override if override is not None else self.ml_collection_mode_var.get()
        if mode not in ML_COLLECTION_MODE_LABELS:
            raise ProtocolError("Collection mode must be Full diagnostics / CIR or Fast ranging only.")
        return mode

    def _update_ml_mode_detail(self) -> None:
        try:
            mode = self._ml_collection_mode()
        except ProtocolError:
            mode = ML_COLLECTION_MODE_FULL
            self.ml_collection_mode_var.set(mode)
        self.ml_mode_detail_var.set(ML_COLLECTION_MODE_DETAILS[mode])

    @staticmethod
    def _ml_command_name(mode: str) -> str:
        if mode == ML_COLLECTION_MODE_LIVE:
            return "ML_START_LIVE_TRACKING"
        if mode == ML_COLLECTION_MODE_FAST:
            return "ML_START_FAST_RANGING"
        return "ML_START_COLLECTION"

    @staticmethod
    def _ml_command_timeout_ms(mode: str) -> int:
        if mode == ML_COLLECTION_MODE_FAST:
            return FAST_RANGING_COMMAND_TIMEOUT_MS
        return FULL_DIAGNOSTICS_COMMAND_TIMEOUT_MS

    @staticmethod
    def _ml_expected_sample_notifications(
        sample_count: int | None,
        discovery_slot_count: int | None,
    ) -> int | None:
        if sample_count is None or discovery_slot_count is None:
            return None
        return sample_count * discovery_slot_count

    def _ml_progress_text(self) -> str:
        received = self.ml_received_sample_notifications
        expected = self.ml_expected_sample_notifications
        if expected is None:
            return f"{received} sample row(s) received"
        return f"{received}/{expected} sample row(s) received"

    def _update_ml_progress_status(self) -> None:
        if not self.ml_command_in_flight or not hasattr(self, "ml_status_var"):
            return
        mode = self.ml_pending_mode or ML_COLLECTION_MODE_FULL
        self.ml_status_var.set(f"{ML_COLLECTION_MODE_LABELS[mode]}: {self._ml_progress_text()}")

    def _note_ml_sample_notification(self, record: ParsedRecord) -> None:
        if not self.ml_command_in_flight:
            return
        if self.ml_pending_mode != ML_COLLECTION_MODE_LIVE and record.scheduled_sample_count is not None:
            self.ml_expected_sample_notifications = record.scheduled_sample_count
        self.ml_received_sample_notifications += 1
        self._update_ml_progress_status()

    def _start_ml_command_timeout(self, mode: str) -> None:
        self._cancel_ml_command_timeout()
        self.ml_timeout_after_id = self.after(
            self._ml_command_timeout_ms(mode),
            self._handle_ml_command_timeout,
        )

    def _cancel_ml_command_timeout(self) -> None:
        after_id = self.ml_timeout_after_id
        self.ml_timeout_after_id = None
        if after_id is None:
            return
        try:
            self.after_cancel(after_id)
        except Exception:
            pass

    def _clear_ml_command_tracking(self) -> None:
        self.ml_command_in_flight = False
        self.ml_pending_session = None
        self.ml_pending_seq = None
        self.ml_pending_mode = None
        self.ml_expected_sample_notifications = None
        self.ml_received_sample_notifications = 0

    def _handle_ml_command_timeout(self) -> None:
        self.ml_timeout_after_id = None
        if not self.ml_command_in_flight:
            return
        mode = self.ml_pending_mode or ML_COLLECTION_MODE_FULL
        progress = self._ml_progress_text()
        timeout_s = self._ml_command_timeout_ms(mode) / 1000
        self._flush_diagnostics_to_all_samples()
        self._clear_ml_command_tracking()
        self._set_ml_collect_state(False)
        self._reset_trigger_collection_state()
        label = ML_COLLECTION_MODE_LABELS[mode]
        self.log_raw(
            f"# {label} timed out after {timeout_s:.0f}s without final command result "
            f"({progress})."
        )
        if hasattr(self, "ml_status_var"):
            self.ml_status_var.set(f"{label} timed out ({progress})")

    def _abort_ml_command(self, reason: str) -> None:
        if not self.ml_command_in_flight and self.ml_timeout_after_id is None:
            return
        mode = self.ml_pending_mode or ML_COLLECTION_MODE_FULL
        label = ML_COLLECTION_MODE_LABELS[mode]
        self._cancel_ml_command_timeout()
        self._flush_diagnostics_to_all_samples()
        self._clear_ml_command_tracking()
        self._set_ml_collect_state(False)
        self._reset_trigger_collection_state()
        self.log_raw(f"# {label} canceled: {reason}")
        if hasattr(self, "ml_status_var"):
            self.ml_status_var.set(f"{label} canceled: {reason}")

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_dir_var.get(), parent=self)
        if not path:
            return
        self.output_dir_var.set(str(path))
        self.db_path_var.set(str(Path(path) / DEFAULT_DB_NAME))

    def refresh_devices(self) -> None:
        self.status_var.set("Scanning for Bluetooth devices...")
        BluetoothScanner(self.events, name_filter=self.device_filter_var.get()).start()

    def selected_device(self) -> str:
        return self.device_var.get().split(" - ", 1)[0].strip()

    def connect_selected_device(self) -> None:
        if self.bluetooth_worker is not None:
            return
        device = self.selected_device()
        if not device:
            messagebox.showwarning("No device selected", "Select or enter a BLE device before connecting.", parent=self)
            return
        self.bluetooth_worker = BluetoothWorker(
            device=device,
            notify_uuid=self.notify_uuid_var.get(),
            write_uuid=self.write_uuid_var.get(),
            events=self.events,
            log_uuid=self.log_uuid_var.get(),
        )
        self.bluetooth_worker.start()
        self.status_var.set(f"Connecting to {device}...")

    def _reconnect_bluetooth_immediately(self, reason: str) -> None:
        if self.bluetooth_manual_disconnect_pending or self.bluetooth_worker is not None:
            return
        device = self.selected_device()
        if not device:
            self.log_raw(f"# Bluetooth {reason}; no selected device to reconnect.")
            return
        self.log_raw(f"# Bluetooth {reason}; reconnecting immediately to {device}.")
        self.connect_selected_device()

    def disconnect_transport(self) -> None:
        worker = self.bluetooth_worker
        self.bluetooth_manual_disconnect_pending = worker is not None
        self._abort_ml_command("Bluetooth disconnected")
        if worker is not None:
            worker.stop()
            self.bluetooth_worker = None
        self.connected_device = None
        self.status_var.set("Disconnected")
        if worker is None:
            self.bluetooth_manual_disconnect_pending = False

    def start_capture(self) -> None:
        if self.capture_active:
            return
        metadata = self.session_metadata()
        if not metadata["condition_los"] and not metadata["condition_nlos"]:
            proceed = messagebox.askyesno(
                "No LOS/NLOS label",
                "Neither LOS nor NLOS is checked. Continue with an unlabeled session?",
                parent=self,
            )
            if not proceed:
                return
        output_dir = Path(self.output_dir_var.get()).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(self.db_path_var.get()).expanduser()
        if not db_path.name:
            db_path = output_dir / DEFAULT_DB_NAME
            self.db_path_var.set(str(db_path))
        if self.store is not None:
            self.store.close()
        self.store = MeasurementStore(db_path)
        self.current_session_id = self.store.create_session(metadata)
        self.capture_active = True
        self.sample_count_var.set("0")
        self.alert_count_var.set("0")
        self.raw_count_var.set("0")
        self.last_alert_prompt.clear()
        self.last_instability_prompt.clear()
        self.anchor_distance_windows.clear()
        self.last_sample_row_by_anchor.clear()
        self._reset_trigger_collection_state()
        self.session_status_var.set(f"Capturing: {self.current_session_id[:8]}")
        self.log_raw(f"# Started session {self.current_session_id}")
        self.log_raw("# Waiting for ML click reports. Send an ML collection command to begin streaming samples.")

    def stop_capture(self) -> None:
        if not self.capture_active:
            return
        self.capture_active = False
        if self.store is not None and self.current_session_id is not None:
            self.store.finish_session(self.current_session_id)
            counts = self.store.counts(self.current_session_id)
            self.session_status_var.set(f"Stopped: {counts['samples']} samples, {counts['alerts']} alerts")
            self.log_raw(f"# Stopped session {self.current_session_id}")

    def send_ml_start_collection(self, *, collection_mode: str | None = None) -> bool:
        if self.bluetooth_worker is None:
            self.log_raw("# Did not send ML collection; Bluetooth is not connected.")
            return False
        if self.ml_command_in_flight:
            self.log_raw("# ML collection already in flight; wait for the command result before retrying.")
            return False
        try:
            mode = self._ml_collection_mode(collection_mode)
            sample_count = self._ml_sample_count()
            discovery_slots = self._ml_discovery_slot_count()
            session_id = self._ml_session_id()
            source_id = parse_device_id(self.host_id_var.get(), default=0)
            if source_id == 0:
                raise ProtocolError("Host ID must be a stable non-zero device ID.")
            destination_id = parse_device_id(self.clicker_id_var.get(), default=0)
            self.protocol_sequence = next_sequence(self.protocol_sequence)
            builder = (
                build_ml_start_fast_ranging_packet
                if mode == ML_COLLECTION_MODE_FAST
                else build_ml_start_collection_packet
            )
            packet = builder(
                source_id=source_id,
                destination_id=destination_id,
                session_id=session_id,
                sequence=self.protocol_sequence,
                sample_count=sample_count,
                discovery_slot_count=discovery_slots,
            )
        except ProtocolError as exc:
            messagebox.showerror("Invalid ML collection command", str(exc), parent=self)
            return False
        sent = self.bluetooth_worker.send_packet(packet)
        if not sent:
            self.log_raw("# Did not send ML collection; Bluetooth is not ready.")
            return False
        self._reset_trigger_collection_state()
        self.ml_command_in_flight = True
        self.ml_pending_session = session_id
        self.ml_pending_seq = self.protocol_sequence
        self.ml_pending_mode = mode
        self.ml_received_sample_notifications = 0
        self.ml_expected_sample_notifications = self._ml_expected_sample_notifications(
            sample_count,
            discovery_slots,
        )
        self._start_ml_command_timeout(mode)
        self._set_ml_collect_state(True)
        self._update_ml_progress_status()
        command_name = self._ml_command_name(mode)
        self.log_raw(
            f"# Queued {command_name} to {self.clicker_id_var.get().strip() or 'broadcast'} "
            f"(session={session_id} seq={self.protocol_sequence})"
        )
        return True

    def _ml_sample_count(self) -> int | None:
        text = self.ml_sample_count_var.get().strip()
        if not text:
            return None
        value = safe_int(text)
        if value is None or not (1 <= value <= 100):
            raise ProtocolError("Samples per anchor must be from 1 to 100 (blank uses firmware default 8).")
        return value

    def _ml_discovery_slot_count(self) -> int | None:
        text = self.ml_discovery_slot_count_var.get().strip()
        if not text:
            return None
        value = safe_int(text)
        if value is None or not (1 <= value <= 8):
            raise ProtocolError("Discovery slots must be from 1 to 8 (blank uses firmware default 8).")
        return value

    def _ml_session_id(self) -> int:
        value = safe_int(self.ml_session_id_var.get())
        if value is None or value < 0 or value > 0xFFFFFFFF:
            raise ProtocolError("Session ID must be a 32-bit unsigned integer.")
        return value

    def _set_ml_collect_state(self, in_flight: bool) -> None:
        if hasattr(self, "ml_collect_button"):
            self.ml_collect_button.configure(state="disabled" if in_flight else "normal")
        if hasattr(self, "ml_status_var"):
            mode = self.ml_pending_mode or ML_COLLECTION_MODE_FULL
            self.ml_status_var.set(
                f"{ML_COLLECTION_MODE_LABELS[mode]} in flight..." if in_flight else "Ready to collect"
            )

    def handle_protocol_packet(self, data: bytes) -> None:
        try:
            packet = decode_packet(data)
        except ProtocolError as exc:
            self.log_alert(f"Protocol packet rejected: {exc}")
            return

        summary = packet_summary(packet)
        self.log_raw(f"# RX {summary}")
        try:
            self.log_protocol_message(packet)
            records = records_from_packet(packet)
        except ProtocolError as exc:
            self.log_alert(f"Could not parse {summary}: {exc}")
            records = []

        if packet.msg_type == MessageType.COMMAND_RESULT:
            self._handle_command_result(packet)

        if self.capture_active and self.store is not None and self.current_session_id is not None:
            self.store.insert_raw_line(self.current_session_id, f"{summary} {data.hex(' ')}", bool(records))
        if records:
            self.handle_records(records)

    def _handle_command_result(self, packet: ImecPacket) -> None:
        matches = (
            self.ml_pending_session == packet.session_id
            and self.ml_pending_seq == packet.sequence
        )
        if not matches:
            return
        mode = self.ml_pending_mode or ML_COLLECTION_MODE_FULL
        status = command_status_from_packet(packet)
        sample_count = command_sample_count_from_packet(packet)
        self._cancel_ml_command_timeout()
        progress = self._ml_progress_text()
        self._clear_ml_command_tracking()
        self._set_ml_collect_state(False)
        self._flush_diagnostics_to_all_samples()
        self._reset_trigger_collection_state()
        samples_text = f" ({sample_count} sample notifications)" if sample_count is not None else ""
        if status == CommandStatus.COMMAND_BUSY:
            if mode == ML_COLLECTION_MODE_LIVE:
                self.log_raw("# Live tracking rejected as BUSY; stop the prior live run or wait for timeout.")
            elif mode == ML_COLLECTION_MODE_FAST:
                self.log_raw("# Fast ranging rejected as BUSY; wait for the prior request to finish, then retry.")
            else:
                self.log_raw("# ML collection rejected as BUSY; wait for the prior collection to finish, then retry.")
            if hasattr(self, "ml_status_var"):
                self.ml_status_var.set("Busy - retry after prior collection finishes")
        elif status == CommandStatus.COMMAND_TIMEOUT:
            if mode == ML_COLLECTION_MODE_LIVE:
                self.log_raw(f"# Live tracking ended by watchdog timeout{samples_text}.")
            elif mode == ML_COLLECTION_MODE_FAST:
                self.log_raw(
                    f"# Fast ranging timed out with no anchor replies{samples_text}. "
                    "Firmware may have attempted fresh discovery; check anchor placement/power and retry."
                )
            else:
                self.log_raw(
                    f"# ML collection timed out with no anchor replies{samples_text}. "
                    "Check anchor placement/power and retry."
                )
            if hasattr(self, "ml_status_var"):
                if mode == ML_COLLECTION_MODE_LIVE:
                    self.ml_status_var.set(f"Live tracking watchdog timeout{samples_text}")
                else:
                    self.ml_status_var.set("Timeout - no anchors replied")
        elif status is None or status == CommandStatus.COMMAND_OK:
            if mode == ML_COLLECTION_MODE_LIVE:
                self.log_raw(f"# Live tracking command result: OK{samples_text}")
            elif mode == ML_COLLECTION_MODE_FAST:
                self.log_raw(
                    f"# Fast ranging command result: OK{samples_text}; no post-burst CIR expected."
                )
            else:
                self.log_raw(f"# ML collection command result: OK{samples_text}")
            if hasattr(self, "ml_status_var"):
                if mode == ML_COLLECTION_MODE_LIVE:
                    self.ml_status_var.set(f"Live tracking complete{samples_text}")
                elif mode == ML_COLLECTION_MODE_FAST:
                    suffix = samples_text or f" ({progress})"
                    self.ml_status_var.set(f"Fast ranging complete{suffix}")
                else:
                    self.ml_status_var.set(f"Collection complete{samples_text}")
        else:
            name = _command_status_name(status)
            if mode == ML_COLLECTION_MODE_LIVE:
                self.log_raw(f"# Live tracking command result: {name}{samples_text}")
            elif mode == ML_COLLECTION_MODE_FAST:
                self.log_raw(f"# Fast ranging command result: {name}{samples_text}")
            else:
                self.log_raw(f"# ML collection command result: {name}{samples_text}")
            if hasattr(self, "ml_status_var"):
                self.ml_status_var.set(f"Result: {name}{samples_text}")

    def log_protocol_message(self, packet: ImecPacket) -> None:
        if packet.msg_type == MessageType.COMMAND_RESULT:
            self.log_raw(f"# Command result: {command_result_summary(packet)}")
        elif packet.msg_type == MessageType.MSG_ERROR:
            self.log_alert(f"Protocol error from {format_device_id(packet.source_id)}")

    def session_metadata(self) -> dict[str, Any]:
        ground_truth = safe_float(self.ground_truth_var.get())
        threshold = safe_float(self.threshold_var.get())
        return {
            "port": self.selected_device(),
            "baud": None,
            "tag_id": "",
            "building_label": "",
            "constellation_label": self.constellation_var.get().strip(),
            "condition_los": self.los_var.get(),
            "condition_nlos": self.nlos_var.get(),
            "ground_truth_m": ground_truth,
            "outlier_threshold_m": threshold,
            "notes": "BLE protocol capture",
            "send_start_stop": False,
            "start_command": "",
            "stop_command": "",
        }

    def process_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "devices":
                    infos = list(payload)
                    self.device_infos = {info.address: info for info in infos}
                    values = [info.label for info in infos]
                    self.device_combo.configure(values=values)
                    if values and not self.device_var.get().strip():
                        self.device_var.set(values[0])
                    self.status_var.set(f"Found {len(values)} Bluetooth device(s)")
                    if self.auto_connect_var.get() and values and self.bluetooth_worker is None:
                        self.connect_selected_device()
                elif event == "connected":
                    self.bluetooth_manual_disconnect_pending = False
                    self.connected_device = str(payload)
                    self.status_var.set(f"Connected to {payload}")
                    self.log_raw(f"# Connected to {payload}")
                elif event == "disconnected":
                    manual_disconnect = self.bluetooth_manual_disconnect_pending
                    self._abort_ml_command("Bluetooth disconnected")
                    self.bluetooth_worker = None
                    self.connected_device = None
                    self.status_var.set("Disconnected")
                    self.log_raw("# Disconnected")
                    if manual_disconnect:
                        self.bluetooth_manual_disconnect_pending = False
                    else:
                        self._reconnect_bluetooth_immediately("disconnected")
                elif event == "error":
                    manual_disconnect = self.bluetooth_manual_disconnect_pending
                    was_connected = self.connected_device is not None
                    self._abort_ml_command("Bluetooth error")
                    self.status_var.set(str(payload))
                    self.log_alert(str(payload))
                    self.bluetooth_worker = None
                    if manual_disconnect:
                        self.bluetooth_manual_disconnect_pending = False
                    elif was_connected:
                        self._reconnect_bluetooth_immediately("error")
                elif event == "command_sent":
                    self.log_raw(f"# TX {payload}")
                elif event == "packet":
                    self.handle_protocol_packet(bytes(payload))
                elif event == "line":
                    self.handle_serial_line(str(payload))
        except queue.Empty:
            pass
        self.after(80, self.process_events)

    def handle_serial_line(self, line: str) -> None:
        cleaned = _clean_log_line(line)
        if _is_bt_debug_noise(cleaned):
            return
        self.log_raw(cleaned)
        if self.capture_active and self.store is not None and self.current_session_id is not None:
            self.store.insert_raw_line(self.current_session_id, cleaned, False)


    def handle_records(self, records: list[ParsedRecord]) -> None:
        for record in records:
            if record.kind == "diagnostic_fragment":
                self._buffer_diagnostic_fragment(record)
                continue
            if record.kind in ("sample", "failure"):
                self._note_ml_sample_notification(record)
            if self.store is not None and self.current_session_id is not None:
                if record.kind == "summary":
                    self.store.insert_summary(self.current_session_id, record)
                    alert_error = self.check_los_alert(record)
                    self.add_record_to_table(record, alert_error)
                else:
                    row_id = self.store.insert_sample(self.current_session_id, record)
                    if record.event_seq is not None:
                        self.collection_event_seq = record.event_seq
                    self.collection_sample_rows.append(row_id)
                    if record.anchor_id:
                        self.last_sample_row_by_anchor[record.anchor_id] = row_id
                        self.collection_sample_rows_by_anchor.setdefault(record.anchor_id, []).append(row_id)
                    alert_error = self.check_los_alert(record)
                    self.check_instability_alert(record)
                    self.add_record_to_table(record, alert_error)
                    self._add_cir_sample(record)
                    if record.anchor_id:
                        self.collection_cir_records_by_anchor.setdefault(record.anchor_id, []).append(record)
                    if self.collection_sample_rows:
                        item = self.sample_table.get_children()[-1]
                        self.collection_table_items.append(item)
                        if record.anchor_id:
                            self.collection_table_items_by_anchor.setdefault(record.anchor_id, []).append(item)
            else:
                self.add_record_to_table(record)
                self._add_cir_sample(record)
        self.update_counts()

    def _buffer_diagnostic_fragment(self, record: ParsedRecord) -> None:
        self.pending_diagnostics.append(record)
        if record.tlv_json:
            self._reassemble_cir_from_fragment(record)

    def _reassemble_cir_from_fragment(self, record: ParsedRecord) -> None:
        try:
            tlvs = json.loads(record.tlv_json or "{}")
        except (ValueError, TypeError):
            return
        chunk_hex_list = tlvs.get("UWB_CIR_FULL_CHUNK", [])
        offset_hex_list = tlvs.get("UWB_CIR_BYTE_OFFSET", [])
        total_bytes = self._first_tlv_uint(tlvs, "UWB_CIR_TOTAL_BYTES")
        if not chunk_hex_list:
            return

        event_seq = record.event_seq if record.event_seq is not None else self.collection_event_seq
        group_key = (
            event_seq,
            record.anchor_id or "",
            record.burst_id,
            record.cir_start_index,
            record.cir_first_path_index,
            total_bytes,
        )
        group = self.cir_reassembly_groups.setdefault(
            group_key,
            {
                "anchor_id": record.anchor_id,
                "event_seq": event_seq,
                "burst_id": record.burst_id,
                "cir_start_index": record.cir_start_index,
                "cir_first_path_index": record.cir_first_path_index,
                "total": total_bytes,
                "bytes": bytearray(),
            },
        )
        if total_bytes is not None:
            group["total"] = total_bytes

        reassembly = group["bytes"]
        for chunk_index, chunk_hex in enumerate(chunk_hex_list):
            try:
                chunk = bytes.fromhex(chunk_hex)
            except ValueError:
                continue
            offset = 0
            if offset_hex_list:
                offset_hex = offset_hex_list[min(chunk_index, len(offset_hex_list) - 1)]
                try:
                    offset = int.from_bytes(bytes.fromhex(offset_hex), "little")
                except ValueError:
                    offset = 0
            end = offset + len(chunk)
            if end > len(reassembly):
                reassembly.extend(bytearray(end - len(reassembly)))
            reassembly[offset:end] = chunk

    def _flush_diagnostics_to_all_samples(self) -> None:
        if not self.pending_diagnostics or not self.collection_sample_rows:
            self._clear_pending_diagnostics()
            return

        diagnostics_by_anchor: dict[str, list[ParsedRecord]] = {}
        for record in self.pending_diagnostics:
            if record.anchor_id:
                diagnostics_by_anchor.setdefault(record.anchor_id, []).append(record)

        for anchor_id, diagnostics in diagnostics_by_anchor.items():
            row_ids = self.collection_sample_rows_by_anchor.get(anchor_id, [])
            if not row_ids:
                continue
            reassembled_cir = self._best_reassembled_cir_for_anchor(anchor_id, diagnostics)
            merged_diag = self._merge_fragments(diagnostics, reassembled_cir)

            for row_id in row_ids:
                if self.store is not None:
                    self.store.merge_diagnostic_into_sample(row_id, merged_diag)
            if reassembled_cir and self.store is not None:
                cir_hex = reassembled_cir.hex()
                for row_id in row_ids:
                    self.store.update_sample_cir_full(row_id, cir_hex)

            for item_id in self.collection_table_items_by_anchor.get(anchor_id, []):
                self._update_table_row_with_diag(item_id, merged_diag)

            self._apply_diag_to_cir_history(anchor_id, merged_diag)

        self._clear_pending_diagnostics()

    def _clear_pending_diagnostics(self) -> None:
        self.pending_diagnostics.clear()
        self.cir_reassembly_groups.clear()

    @staticmethod
    def _first_tlv_uint(tlvs: dict[str, list[str]], name: str) -> int | None:
        values = tlvs.get(name) or []
        if not values:
            return None
        try:
            return int.from_bytes(bytes.fromhex(values[0]), "little")
        except ValueError:
            return None

    def _best_reassembled_cir_for_anchor(
        self, anchor_id: str, diagnostics: list[ParsedRecord]
    ) -> bytes | None:
        event_seqs = {record.event_seq for record in diagnostics if record.event_seq is not None}
        burst_ids = {record.burst_id for record in diagnostics if record.burst_id is not None}
        candidates: list[dict[str, Any]] = []
        for group in self.cir_reassembly_groups.values():
            if group.get("anchor_id") != anchor_id:
                continue
            if event_seqs and group.get("event_seq") not in event_seqs:
                continue
            if burst_ids and group.get("burst_id") not in burst_ids:
                continue
            if group.get("bytes"):
                candidates.append(group)
        if not candidates:
            return None

        def score(group: dict[str, Any]) -> tuple[int, int]:
            data_len = len(group["bytes"])
            total = int(group.get("total") or 0)
            complete = int(bool(total and data_len >= total))
            return complete, data_len

        group = max(candidates, key=score)
        data = bytes(group["bytes"])
        total = int(group.get("total") or 0)
        return data[:total] if total and len(data) > total else data

    def _merge_fragments(
        self, diagnostics: list[ParsedRecord], reassembled_cir: bytes | None
    ) -> ParsedRecord:
        merged = self._merge_diag_record(diagnostics[0], reassembled_cir)
        fields = (
            "event_seq", "rx_power_dbm", "phy_config_id", "burst_id", "exchange_stride_us",
            "burst_duration_ms", "diag_status_flags", "diag_bytes_captured",
            "diag_bytes_transmitted", "report_fragment_count", "uwb_clock_offset_raw",
            "uwb_carrier_integrator", "clicker_diag_bytes", "cir_raw", "tlv_json",
            "cir_first_path_index", "cir_start_index", "diag_source",
        )
        for extra in diagnostics[1:]:
            for field in fields:
                if getattr(merged, field, None) is None and getattr(extra, field, None) is not None:
                    setattr(merged, field, getattr(extra, field))
        if reassembled_cir:
            merged.cir_raw = reassembled_cir.hex()
        return merged

    def _apply_diag_to_cir_history(self, anchor_id: str, diag: ParsedRecord) -> None:
        changed = False
        event_seq = diag.event_seq if diag.event_seq is not None else self.collection_event_seq
        for record in self.collection_cir_records_by_anchor.get(anchor_id, []):
            if not _cir_record_matches_diag(record, anchor_id, event_seq):
                continue
            if diag.cir_raw:
                record.cir_raw = diag.cir_raw
            if diag.cir_first_path_index is not None:
                record.cir_first_path_index = diag.cir_first_path_index
            if diag.cir_start_index is not None:
                record.cir_start_index = diag.cir_start_index
            changed = True
        if changed:
            self._draw_cir_plot()

    def _update_table_row_with_diag(self, item_id: str, diag: ParsedRecord) -> None:
        try:
            values = list(self.sample_table.item(item_id, "values"))
        except Exception:
            return
        if len(values) < 10:
            return
        if diag.rx_power_dbm is not None and not values[5]:
            values[5] = f"{float(diag.rx_power_dbm):.2f}"
        if diag.cir_raw and not values[7]:
            values[7] = f"{len(bytes.fromhex(diag.cir_raw))}B"
        self.sample_table.item(item_id, values=values)

    @staticmethod
    def _merge_diag_record(diag: ParsedRecord, reassembled_cir: bytes | None) -> ParsedRecord:
        return ParsedRecord(
            kind="diagnostic_merge",
            anchor_id=diag.anchor_id,
            event_seq=diag.event_seq,
            rx_power_dbm=diag.rx_power_dbm,
            phy_config_id=diag.phy_config_id,
            burst_id=diag.burst_id,
            exchange_stride_us=diag.exchange_stride_us,
            burst_duration_ms=diag.burst_duration_ms,
            diag_status_flags=diag.diag_status_flags,
            diag_bytes_captured=diag.diag_bytes_captured,
            diag_bytes_transmitted=diag.diag_bytes_transmitted,
            report_fragment_count=diag.report_fragment_count,
            uwb_clock_offset_raw=diag.uwb_clock_offset_raw,
            uwb_carrier_integrator=diag.uwb_carrier_integrator,
            clicker_diag_bytes=diag.clicker_diag_bytes,
            cir_raw=reassembled_cir.hex() if reassembled_cir else diag.cir_raw,
            tlv_json=diag.tlv_json,
            cir_first_path_index=diag.cir_first_path_index,
            cir_start_index=diag.cir_start_index,
            diag_source=diag.diag_source,
        )

    def check_los_alert(self, record: ParsedRecord) -> float | None:
        if not self.los_var.get() or self.nlos_var.get():
            return None
        ground_truth = safe_float(self.ground_truth_var.get())
        threshold = safe_float(self.threshold_var.get()) or 0.0
        distance = record.distance_m if record.distance_m is not None else record.mean_distance_m
        if ground_truth is None or ground_truth <= 0 or distance is None:
            return None
        absolute_error = abs(float(distance) - float(ground_truth))
        if absolute_error > threshold and self.store is not None and self.current_session_id is not None:
            anchor = record.anchor_id or "unknown"
            message = (
                f"LOS measurement from anchor {anchor} is {absolute_error:.3f} m away "
                f"from ground truth ({float(distance):.3f} m measured vs {ground_truth:.3f} m true)."
            )
            self.store.insert_alert(self.current_session_id, anchor, float(distance), ground_truth, absolute_error, message)
            self.log_alert(message)
            current_time = time.monotonic()
            last_prompt = self.last_alert_prompt.get(anchor, 0.0)
            if current_time - last_prompt > ALERT_PROMPT_COOLDOWN_SECONDS:
                self.last_alert_prompt[anchor] = current_time
                messagebox.showwarning("LOS measurement warning", message, parent=self)
        return absolute_error

    def check_instability_alert(self, record: ParsedRecord) -> float | None:
        if record.kind == "summary" or record.distance_m is None:
            return None
        threshold = safe_float(self.instability_threshold_var.get())
        if threshold is None or threshold <= 0:
            return None
        anchor = record.anchor_id or "unknown"
        values = self.anchor_distance_windows.setdefault(anchor, [])
        values.append(float(record.distance_m))
        if len(values) > INSTABILITY_WINDOW_SIZE:
            del values[0 : len(values) - INSTABILITY_WINDOW_SIZE]
        if len(values) < min(5, INSTABILITY_WINDOW_SIZE):
            return None
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        if std > threshold and self.store is not None and self.current_session_id is not None:
            now = time.monotonic()
            last = self.last_instability_prompt.get(anchor, 0.0)
            if now - last > ALERT_PROMPT_COOLDOWN_SECONDS:
                self.last_instability_prompt[anchor] = now
                message = f"Anchor {anchor} distance is unstable: std {std:.3f} m over {len(values)} samples."
                self.store.insert_alert(self.current_session_id, anchor, record.distance_m, None, std, message)
                self.log_alert(message)
        return std

    def update_counts(self) -> None:
        if self.store is None or self.current_session_id is None:
            return
        counts = self.store.counts(self.current_session_id)
        self.sample_count_var.set(str(counts["samples"]))
        self.alert_count_var.set(str(counts["alerts"]))
        self.raw_count_var.set(str(counts["raw_lines"]))

    def add_record_to_table(self, record: ParsedRecord, alert_error: float | None = None) -> None:
        display_time = datetime.now().strftime("%H:%M:%S")
        distance = record.distance_m if record.distance_m is not None else record.mean_distance_m
        row = (
            display_time,
            record.anchor_id or "",
            "" if record.sample_index is None else str(record.sample_index),
            "" if distance is None else f"{float(distance):.3f}",
            "" if alert_error is None else f"{float(alert_error):.3f}",
            "" if record.rx_power_dbm is None else f"{float(record.rx_power_dbm):.2f}",
            "" if record.fp_power_dbm is None else f"{float(record.fp_power_dbm):.2f}",
            "" if record.cir_power is None else f"{float(record.cir_power):.2f}",
            record.status or "",
            record.source or "",
        )
        tags: tuple[str, ...] = ()
        threshold = safe_float(self.threshold_var.get())
        if alert_error is not None and threshold is not None and alert_error > threshold:
            tags = ("alert",)
        if record.kind == "failure":
            tags = ("failure",)
        elif record.kind == "summary":
            tags = ("summary",)
        self.sample_table.insert("", tk.END, values=row, tags=tags)
        self.table_rows += 1
        if self.table_rows > MAX_TABLE_ROWS:
            first = self.sample_table.get_children()[0]
            self.sample_table.delete(first)
            self.table_rows -= 1
        self.sample_table.yview_moveto(1.0)

    def clear_live_view(self) -> None:
        for item in self.sample_table.get_children():
            self.sample_table.delete(item)
        self.table_rows = 0
        self.raw_text.delete("1.0", tk.END)
        self.alert_text.delete("1.0", tk.END)
        self.cir_history.clear()
        if hasattr(self, "cir_listbox"):
            self.cir_listbox.delete(0, tk.END)
        if hasattr(self, "cir_status_var"):
            self.cir_status_var.set("No CIR samples yet")
        if hasattr(self, "cir_canvas"):
            self.cir_canvas.delete("all")

    def log_raw(self, message: str) -> None:
        self.raw_text.insert(tk.END, f"{message}\n")
        self.raw_text.see(tk.END)

    def log_alert(self, message: str) -> None:
        self.alert_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} {message}\n")
        self.alert_text.see(tk.END)

    def export_current_session(self) -> None:
        if self.store is None or self.current_session_id is None:
            messagebox.showwarning("No session", "Start a capture session before exporting.", parent=self)
            return
        if self.capture_active:
            proceed = messagebox.askyesno(
                "Session still running",
                "The session is still capturing. Export a snapshot of the data collected so far?",
                parent=self,
            )
            if not proceed:
                return
        session = self.store.session_row(self.current_session_id)
        condition_bits = []
        if session["condition_los"]:
            condition_bits.append("LOS")
        if session["condition_nlos"]:
            condition_bits.append("NLOS")
        condition = "-".join(condition_bits) or "unlabeled"
        constellation = safe_filename(session["constellation_label"] or "constellation")
        default_name = f"{constellation}_{condition}_{export_timestamp()}.xlsx"
        output_path = filedialog.asksaveasfilename(
            initialdir=self.output_dir_var.get(),
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx")],
            parent=self,
        )
        if not output_path:
            return
        try:
            self.store.export_session_to_excel(self.current_session_id, Path(output_path))
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)
            return
        messagebox.showinfo("Export complete", f"Wrote {output_path}", parent=self)

    def on_close(self) -> None:
        if self.capture_active:
            proceed = messagebox.askyesno("Capture active", "Stop logging and close the app?", parent=self)
            if not proceed:
                return
            self.stop_capture()
        self._cancel_ml_command_timeout()
        self.disconnect_transport()
        if self.store is not None:
            self.store.close()
        self.destroy()


def main() -> None:
    app = UwbCaptureApp()
    app.mainloop()


if __name__ == "__main__":
    main()
