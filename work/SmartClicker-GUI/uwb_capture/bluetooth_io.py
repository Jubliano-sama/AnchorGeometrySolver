"""Bluetooth Low Energy discovery and packet transport for the ML clicker."""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any

from .protocol import ImecPacketStream, cobs_encode

try:
    from bleak import BleakClient, BleakScanner
except Exception:  # pragma: no cover - keeps tests and non-BLE systems usable.
    BleakClient = None
    BleakScanner = None

# IMEC ML Clicker GATT service and characteristics (see ML BLE GUI Integration.md).
DEFAULT_DEVICE_NAME = "IMEC ML Clicker"
DEFAULT_SERVICE_UUID = "494d4543-0001-4757-8000-000000000001"
DEFAULT_PACKET_TX_UUID = "494d4543-0001-4757-8000-000000000002"
DEFAULT_PACKET_RX_UUID = "494d4543-0001-4757-8000-000000000003"
DEFAULT_LOG_TX_UUID = "494d4543-0001-4757-8000-000000000004"


class BluetoothError(Exception):
    pass


@dataclass(frozen=True)
class BluetoothDeviceInfo:
    address: str
    name: str

    @property
    def label(self) -> str:
        return f"{self.address} - {self.name}" if self.name else self.address


class BluetoothScanner(threading.Thread):
    def __init__(
        self,
        events: queue.Queue[tuple[str, Any]],
        timeout_s: float = 5.0,
        name_filter: str | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.events = events
        self.timeout_s = timeout_s
        self.name_filter = (name_filter or "").strip().lower()

    def run(self) -> None:
        if BleakScanner is None:
            self.events.put(("error", "bleak is not installed. Install it with: pip install bleak"))
            return
        try:
            devices = asyncio.run(BleakScanner.discover(timeout=self.timeout_s))
            infos = [
                BluetoothDeviceInfo(
                    address=str(getattr(device, "address", "")),
                    name=str(getattr(device, "name", "") or ""),
                )
                for device in devices
                if getattr(device, "address", "")
            ]
            if self.name_filter:
                infos = [info for info in infos if self.name_filter in info.name.lower()]
            self.events.put(("devices", infos))
        except Exception as exc:
            self.events.put(("error", f"Bluetooth scan failed: {exc}"))


class BluetoothWorker(threading.Thread):
    def __init__(
        self,
        *,
        device: str,
        notify_uuid: str,
        write_uuid: str,
        events: queue.Queue[tuple[str, Any]],
        log_uuid: str = "",
    ) -> None:
        super().__init__(daemon=True)
        self.device = device.strip()
        self.notify_uuid = notify_uuid.strip()
        self.write_uuid = write_uuid.strip()
        self.log_uuid = log_uuid.strip()
        self.events = events
        self.stop_event = threading.Event()
        self._send_queue: queue.Queue[bytes] = queue.Queue()
        self._packet_stream = ImecPacketStream()

    def run(self) -> None:
        if BleakClient is None or BleakScanner is None:
            self.events.put(("error", "bleak is not installed. Install it with: pip install bleak"))
            return
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self.events.put(("error", str(exc)))

    async def _run(self) -> None:
        if not self.device:
            raise BluetoothError("Enter a Bluetooth device name or address before connecting.")
        if not self.notify_uuid or not self.write_uuid:
            raise BluetoothError("Enter BLE notify and write characteristic UUIDs before connecting.")

        address = await self._resolve_device_address(self.device)
        async with BleakClient(address) as client:
            self.events.put(("connected", address))
            await client.start_notify(self.notify_uuid, self._handle_notification)
            if self.log_uuid:
                try:
                    await client.start_notify(self.log_uuid, self._handle_log_notification)
                except Exception as exc:
                    self.events.put(("error", f"Could not subscribe to log characteristic: {exc}"))
            try:
                while not self.stop_event.is_set():
                    await self._flush_send_queue(client)
                    await asyncio.sleep(0.05)
            finally:
                try:
                    await client.stop_notify(self.notify_uuid)
                except Exception:
                    pass
                if self.log_uuid:
                    try:
                        await client.stop_notify(self.log_uuid)
                    except Exception:
                        pass
        self.events.put(("disconnected", address))

    async def _resolve_device_address(self, value: str) -> str:
        value = value.split(" - ", 1)[0].strip()
        if not value:
            raise BluetoothError("Bluetooth device is empty.")
        compact = value.replace(":", "").replace("-", "")
        if ":" in value or "-" in value or (len(compact) == 12 and all(char in "0123456789abcdefABCDEF" for char in compact)):
            return value

        devices = await BleakScanner.discover(timeout=5.0)
        for device in devices:
            name = str(getattr(device, "name", "") or "")
            address = str(getattr(device, "address", "") or "")
            if value.lower() in {name.lower(), address.lower()}:
                return address
        raise BluetoothError(f"Bluetooth device not found: {value}")

    async def _flush_send_queue(self, client: Any) -> None:
        while True:
            try:
                packet = self._send_queue.get_nowait()
            except queue.Empty:
                return
            frame = cobs_encode(packet) + b"\x00"
            await client.write_gatt_char(self.write_uuid, frame, response=True)
            self.events.put(("command_sent", packet.hex(" ")))

    def _handle_notification(self, _sender: Any, data: bytearray) -> None:
        packets = self._packet_stream.feed(bytes(data))
        for packet in packets:
            self.events.put(("packet", packet))

    def _handle_log_notification(self, _sender: Any, data: bytearray) -> None:
        try:
            text = bytes(data).decode("utf-8").strip()
        except UnicodeDecodeError:
            text = ""
        if text:
            for line in text.splitlines():
                self.events.put(("line", line.strip()))

    def send_packet(self, packet: bytes) -> bool:
        if self.stop_event.is_set():
            return False
        self._send_queue.put(packet)
        return True

    def stop(self) -> None:
        self.stop_event.set()
