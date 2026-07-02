"""Serial port discovery and background reading."""

from __future__ import annotations

import queue
import threading
from typing import Any

try:
    import serial
    from serial import SerialException
    from serial.tools import list_ports
except Exception:  # pragma: no cover - lets the GUI open even without pyserial.
    serial = None
    list_ports = None

    class SerialException(Exception):
        pass

class SerialWorker(threading.Thread):
    def __init__(self, port: str, baud: int, events: queue.Queue[tuple[str, Any]]) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.events = events
        self.stop_event = threading.Event()
        self._serial = None
        self._lock = threading.Lock()

    def run(self) -> None:
        if serial is None:
            self.events.put(("error", "pyserial is not installed. Install it with: pip install pyserial"))
            return
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.2)
            with self._lock:
                self._serial = ser
            self.events.put(("connected", self.port))
            while not self.stop_event.is_set():
                try:
                    data = ser.readline()
                except SerialException as exc:
                    self.events.put(("error", str(exc)))
                    break
                if not data:
                    continue
                line = data.decode("utf-8", errors="replace").strip()
                if line:
                    self.events.put(("line", line))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            with self._lock:
                try:
                    if self._serial is not None and self._serial.is_open:
                        self._serial.close()
                except SerialException:
                    pass
                self._serial = None
            self.events.put(("disconnected", self.port))

    def stop(self) -> None:
        self.stop_event.set()
        with self._lock:
            try:
                if self._serial is not None and self._serial.is_open:
                    self._serial.close()
            except SerialException:
                pass

    def send_command(self, command: str) -> bool:
        text = command.strip()
        if not text:
            return False
        with self._lock:
            if self._serial is None or not self._serial.is_open:
                return False
            self._serial.write((text + "\n").encode("utf-8"))
            self._serial.flush()
            return True

