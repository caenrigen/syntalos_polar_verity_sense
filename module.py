import asyncio
from dataclasses import asdict, dataclass, field
import json
import sys
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleakheart import PolarMeasurementData

from PyQt6 import uic
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QDialog, QLayout

import syntalos_mlink as syl

UI_FILE_PATH = Path(__file__).resolve().with_name("settings.ui")

NS_PER_SEC = 1_000_000_000
POLAR_ERR_ALREADY_IN_STATE = 6
VALID_PPG_SAMPLE_RATES = [28, 44, 55, 135, 176]
ASYNC_LOOP_ADVANCE_S = 0.010
ASYNC_LOOP_WRAPUP_S = 0.100


@dataclass
class Settings:
    device_address: str = ""
    device_name: str = ""
    # NB the device sends ~35-50 samples per frame, 64 is a reasonable batch size
    batch_size: int = 64
    sampling_rate: int = 176  # possible values: 28, 44, 55, 135, 176
    # These are the only valid settings for PPG on Verity Sense
    resolution: int = 22
    channels: int = 4


@dataclass
class State:
    settings: Settings | None = None
    running: bool = False
    settings_dialog: QDialog | None = None
    scan_thread: QThread | None = None
    scan_worker: QObject | None = None
    scan_selected_address: str = ""

    loop: asyncio.AbstractEventLoop | None = None
    client: BleakClient | None = None
    pmd: PolarMeasurementData | None = None
    ppg_queue: asyncio.Queue | None = None
    offset_start_us: int | None = None
    offset_ns: int | None = None
    ppg_timestamps_us: list[int] = field(default_factory=list)
    ppg_rows: list[list[int]] = field(default_factory=list)


def clear_state() -> None:
    # STATE.settings must stay persistent across runs.

    # STATE.loop must stay persistent.
    # `bleak` associates a persistent `BlueZManager()` with `asyncio.get_running_loop()`.
    # See bleak.backends.bluezdbus.manager.get_global_bluez_manager() for details.
    # The issue is that for some reason the `loop` is not garbage collected and leaves opened DBus
    # connections lingering around. By default, Linux has a 256 connections limit which prevents us
    # from running >~110 "launch sync" runs in Syntalos
    # (x2 for two polar devices, ~220 bus connections + system connections = 256 limit).

    STATE.running = False
    STATE.client = None
    STATE.ppg_queue = None
    STATE.pmd = None
    STATE.offset_ns = None
    STATE.offset_start_us = None
    STATE.ppg_timestamps_us.clear()
    STATE.ppg_rows.clear()


STATE = State()
App: QApplication | None = None
MLink: syl.SyntalosLink | None = None
out: syl.OutputPort | None = None


def serialise_settings(settings: Settings) -> bytes:
    return json.dumps(asdict(settings)).encode()


def deserialise_settings(settings: bytes) -> Settings:
    return Settings(**json.loads(settings.decode()))  # pyright: ignore[reportAny]


def close_settings_dialog() -> None:
    dialog = STATE.settings_dialog
    if dialog is not None:
        _ = dialog.close()


def fit_dialog_to_contents(dialog: QDialog) -> None:
    layout = dialog.layout()
    if layout is not None:
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
    dialog.adjustSize()


async def scan_for_device(device_address: str):
    device_address = device_address.strip()
    if device_address:
        addr = device_address.lower()
        device: BLEDevice | None = await BleakScanner.find_device_by_filter(
            lambda dev, adv: bool(dev.address and dev.address.lower() == addr)
        )
        if device is None:
            raise RuntimeError(f"Polar device not found at address: {device_address}")
        return device

    devices = await scan_for_polar_devices()
    if not devices:
        raise RuntimeError("Polar device not found")
    return devices[0]


async def scan_for_polar_devices() -> list[BLEDevice]:
    devices = await BleakScanner.discover()
    return [dev for dev in devices if dev.name and "polar " in dev.name.lower()]


async def ensure_not_streaming(pmd: PolarMeasurementData, measurement: str):
    err_code, err_msg = await pmd.stop_streaming(measurement)
    if err_code not in (0, POLAR_ERR_ALREADY_IN_STATE):
        raise RuntimeError(f"Failed to stop {measurement} before restart: {err_code} {err_msg}")
    if err_code == 0:
        print(f"Stopped existing {measurement} stream/state before start")


async def start_sdk_mode():
    assert STATE.pmd is not None
    await ensure_not_streaming(STATE.pmd, "PPG")
    await ensure_not_streaming(STATE.pmd, "SDK")
    err_code, err_msg, _ = await STATE.pmd.start_streaming("SDK")
    if err_code != 0:
        raise RuntimeError(f"Failed to start SDK mode: {err_code} {err_msg}")


async def start_ppg_streaming():
    assert STATE.pmd is not None
    assert STATE.settings is not None
    err_code, err_msg, _ = await STATE.pmd.start_streaming(
        "PPG",
        sample_rate=STATE.settings.sampling_rate,
        resolution=STATE.settings.resolution,
        channels=STATE.settings.channels,
    )
    if err_code != 0:
        raise RuntimeError(f"Failed to start PPG stream: {err_code} {err_msg}")


async def stop_streaming():
    pmd = STATE.pmd
    assert pmd is not None
    try:
        await pmd.stop_streaming("PPG")
    except Exception as exc:
        print(f"PPG stop failed: {exc.__class__.__name__}({exc})")
    try:
        await pmd.stop_streaming("SDK")
    except Exception as exc:
        print(f"SDK stop failed: {exc.__class__.__name__}({exc})")


def submit_batch(timestamps_us: list[int], rows: list[list[int]]) -> None:
    assert out is not None
    block = syl.IntSignalBlock()
    block.timestamps = np.array(timestamps_us, dtype=np.uint64)
    block.data = np.array(rows, dtype=np.int32)
    out.submit(block)
    timestamps_us.clear()
    rows.clear()


def process_ppg_frame(frame, timestamps_us: list[int], rows: list[list[int]]) -> None:
    assert STATE.settings is not None
    assert STATE.offset_start_us is not None
    dtype, frame_timestamp_ns, payload = frame
    assert frame_timestamp_ns > 0, f"Negative {frame_timestamp_ns = }"
    if dtype != "PPG":
        print(f"Expected PPG frame, got {dtype}")
        return
    if not payload:
        print("Empty PPG frame")
        return

    n_samples = len(payload)
    if STATE.offset_ns is None:
        offset_1st_sample = (
            (n_samples - 1) * NS_PER_SEC + STATE.settings.sampling_rate // 2
        ) // STATE.settings.sampling_rate
        # We add the offset to the timestamp of each data point.
        # frame_timestamp_ns: first frame is our "zero"
        # offset_1st_sample: the timestamp of the batch corresponds to the last data point, offset to first sample
        # syl.time_since_start_usec(): transparently report the delay of the first datapoint
        print(f"{STATE.offset_start_us = }")
        STATE.offset_ns = -(frame_timestamp_ns - offset_1st_sample) + STATE.offset_start_us * 1000

    for back_idx, sample in zip(range(n_samples - 1, -1, -1), payload):
        ts_ns = frame_timestamp_ns - (
            (back_idx * NS_PER_SEC + STATE.settings.sampling_rate // 2)
            // STATE.settings.sampling_rate
        )
        ts_us = (ts_ns + STATE.offset_ns) // 1_000
        assert ts_us > 0, f"Negative {ts_us = }!"
        timestamps_us.append(ts_us)
        rows.append([int(sample[0]), int(sample[1]), int(sample[2]), int(sample[3])])

    if len(timestamps_us) >= STATE.settings.batch_size:
        submit_batch(timestamps_us, rows)


def cleanup() -> None:
    loop = STATE.loop
    if loop is None:
        print("No event loop to cleanup, skipping cleanup()")
        return

    client = STATE.client
    if client is None:
        print("No client to disconnect, skipping client.disconnect()")
        return

    try:
        loop.run_until_complete(stop_streaming())
        # Advance the async loop a final bit for all pending tasks to wrap up.
        # This pervents a series of errors being printed when quiting Syntalos.
        loop.run_until_complete(asyncio.sleep(ASYNC_LOOP_WRAPUP_S))
    except Exception as exc:
        print(f"Stop streaming failed: {exc.__class__.__name__}({exc})")

    try:
        loop.run_until_complete(client.disconnect())
        # Advance the async loop a final bit for all pending tasks to wrap up.
        # This pervents a series of errors being printed when quiting Syntalos.
        loop.run_until_complete(asyncio.sleep(ASYNC_LOOP_WRAPUP_S))
    except EOFError:
        # Happens sometimes but does not seem to be problematic.
        # So far I have seen it only inside a VM.
        print("EOFError at client.disconnect()")
    except Exception as exc:
        print(f"Disconnect failed: {exc.__class__.__name__}({exc})")

    print("Cleanup complete")


# # ####################################################################################
# # Syntalos interface
# # ####################################################################################


def register_ports(mlink: syl.SyntalosLink) -> None:
    global out

    out = mlink.register_output_port("packets", "Data Packets", syl.DataType.IntSignalBlock)
    assert out is not None


def prepare():
    clear_state()
    close_settings_dialog()
    if STATE.settings is None:
        print("Settings not set, aborting prepare()")
        return False

    assert out is not None
    out.set_metadata_value("signal_names", ["PPG0", "PPG1", "PPG2", "AMBIENT"])
    out.set_metadata_value("time_unit", "microseconds")
    out.set_metadata_value("data_unit", ["raw", "raw", "raw", "raw"])

    if not STATE.settings.device_address:
        raise ValueError("Device address not set, edit the settings and try again")

    if STATE.loop is None:
        STATE.loop = asyncio.new_event_loop()

    client = BleakClient(STATE.settings.device_address)
    STATE.loop.run_until_complete(client.connect())

    STATE.client = client
    STATE.ppg_queue = asyncio.Queue()
    STATE.pmd = PolarMeasurementData(client, ppg_queue=STATE.ppg_queue)

    STATE.loop.run_until_complete(start_sdk_mode())
    return True


def start() -> None:
    assert STATE.loop is not None
    # ts_us_before = syl.time_since_start_usec() # returns ~300 us at this point
    STATE.loop.run_until_complete(start_ppg_streaming())
    STATE.offset_start_us = int(syl.time_since_start_usec())
    STATE.running = True


def event_loop_tick() -> None:
    if App is not None:
        App.processEvents()

    loop = STATE.loop
    if loop is None:
        return

    loop.run_until_complete(asyncio.sleep(ASYNC_LOOP_ADVANCE_S))

    if not STATE.running:
        return

    ppg_queue = STATE.ppg_queue
    client = STATE.client
    assert ppg_queue is not None
    assert client is not None

    while True:
        try:
            frame = ppg_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        process_ppg_frame(frame, STATE.ppg_timestamps_us, STATE.ppg_rows)

    if not client.is_connected:
        # Polar devices sometimes disconnect unexpectedly.
        # Distance to the Bluetooth receiver seemed to be one of the causes.
        raise RuntimeError("Device disconnected")


def stop() -> None:
    STATE.running = False
    cleanup()


def load_settings(settings: bytes, _base_dir: Path) -> bool:
    if not settings:
        if STATE.settings is None:
            STATE.settings = Settings()
        return True

    try:
        STATE.settings = deserialise_settings(settings)
        return True
    except Exception:
        STATE.settings = Settings()
        raise


def save_settings(_base_dir: Path) -> bytes:
    if STATE.settings is None:
        STATE.settings = Settings()
    return serialise_settings(STATE.settings)


# # ####################################################################################
# # Settings UI
# # ####################################################################################


DEVICE_NAME_ROLE = int(Qt.ItemDataRole.UserRole) + 1


class DeviceScanWorker(QObject):
    scan_done = pyqtSignal(object, str)  # list[tuple[name, address]], error_text
    finished = pyqtSignal()

    @pyqtSlot()
    def run(self):
        entries: list[tuple[str, str]] = []
        error_text = ""
        try:
            devices = asyncio.run(scan_for_polar_devices())
            for dev in devices:
                if not dev.address:
                    continue
                name = (dev.name or "").strip() or "Polar device"
                entries.append((name, dev.address))
        except Exception as exc:
            error_text = "Scan failed, check logs"
            print(f"Polar scan failed: {exc.__class__.__name__}({exc})")
        self.scan_done.emit(entries, error_text)
        self.finished.emit()


def set_scanning_ui(scanning: bool) -> None:
    dialog = STATE.settings_dialog
    if dialog is None:
        return

    dialog.deviceComboBox.setEnabled(not scanning)
    dialog.refreshDevicesButton.setEnabled(not scanning)
    if scanning:
        dialog.deviceComboBox.clear()
        dialog.deviceComboBox.setPlaceholderText("Scanning...")
        dialog.refreshDevicesButton.setToolTip("Scanning for Polar devices...")
    else:
        dialog.refreshDevicesButton.setToolTip("Scan again")


def persist_settings() -> None:
    dialog = STATE.settings_dialog
    if dialog is None:
        return

    assert STATE.settings is not None

    address = dialog.deviceComboBox.currentData()
    name = dialog.deviceComboBox.currentData(DEVICE_NAME_ROLE)
    if isinstance(address, str) and address.strip():
        STATE.scan_selected_address = address.strip()
        STATE.settings.device_address = address.strip()
        STATE.settings.device_name = name.strip() if isinstance(name, str) else ""

    sampling_rate = dialog.samplingRateComboBox.currentData()
    if sampling_rate is not None:
        STATE.settings.sampling_rate = int(sampling_rate)
    STATE.settings.batch_size = dialog.batchSizeSpinBox.value()


def cleanup_settings_dialog(_result: int) -> None:
    dialog = STATE.settings_dialog
    if dialog is None:
        return

    persist_settings()
    STATE.settings_dialog = None
    dialog.deleteLater()


def cleanup_scan_thread() -> None:
    STATE.scan_thread = None
    STATE.scan_worker = None
    set_scanning_ui(False)


def start_scan() -> None:
    dialog = STATE.settings_dialog
    if dialog is None:
        return

    thread = STATE.scan_thread
    if isinstance(thread, QThread) and thread.isRunning():
        set_scanning_ui(True)
        return

    current_address = dialog.deviceComboBox.currentData()
    if isinstance(current_address, str) and current_address.strip():
        STATE.scan_selected_address = current_address.strip()
    elif STATE.settings is not None:
        STATE.scan_selected_address = STATE.settings.device_address
    else:
        STATE.scan_selected_address = ""

    set_scanning_ui(True)
    worker = DeviceScanWorker()
    thread = QThread()
    worker.moveToThread(thread)

    worker.scan_done.connect(finish_scan)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(cleanup_scan_thread)
    thread.started.connect(worker.run)

    STATE.scan_thread = thread
    STATE.scan_worker = worker
    thread.start()


def finish_scan(entries: list[tuple[str, str]], error_text: str) -> None:
    dialog = STATE.settings_dialog
    if dialog is None:
        return

    dialog.deviceComboBox.clear()
    if error_text:
        dialog.deviceComboBox.setPlaceholderText(error_text)
        return
    if not entries:
        dialog.deviceComboBox.setPlaceholderText("No Polar devices found")
        return

    for name, address in entries:
        dialog.deviceComboBox.addItem(f"{name} ({address})", address)
        idx = dialog.deviceComboBox.count() - 1
        dialog.deviceComboBox.setItemData(idx, name, DEVICE_NAME_ROLE)

    selected_index = -1
    if STATE.scan_selected_address:
        selected_index = dialog.deviceComboBox.findData(STATE.scan_selected_address)
    if selected_index < 0 and STATE.settings is not None and STATE.settings.device_address:
        selected_index = dialog.deviceComboBox.findData(STATE.settings.device_address)
    if selected_index < 0:
        selected_index = 0
    dialog.deviceComboBox.setCurrentIndex(selected_index)


def show_settings() -> None:
    # Showing the settings UI while running prevents the module event loop from advancing.
    # Keep it simple: no settings UI while running.
    if STATE.running or (MLink is not None and MLink.is_running):
        print("Cannot show settings while running")
        return

    if STATE.settings is None:
        STATE.settings = Settings()

    dialog = STATE.settings_dialog
    if dialog is not None:
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return

    dialog = uic.loadUi(UI_FILE_PATH)
    STATE.settings_dialog = dialog
    fit_dialog_to_contents(dialog)
    assert STATE.settings is not None

    dialog.deviceComboBox.setPlaceholderText("No Polar devices found")
    dialog.batchSizeSpinBox.setValue(STATE.settings.batch_size)

    dialog.samplingRateComboBox.clear()
    for rate in VALID_PPG_SAMPLE_RATES:
        dialog.samplingRateComboBox.addItem(f"{rate} Hz", rate)
    current_index = dialog.samplingRateComboBox.findData(STATE.settings.sampling_rate)
    if current_index < 0:
        dialog.samplingRateComboBox.addItem(
            f"{STATE.settings.sampling_rate} Hz", STATE.settings.sampling_rate
        )
        current_index = dialog.samplingRateComboBox.findData(STATE.settings.sampling_rate)
    dialog.samplingRateComboBox.setCurrentIndex(current_index)
    STATE.scan_selected_address = STATE.settings.device_address

    dialog.refreshDevicesButton.clicked.connect(start_scan)
    dialog.deviceComboBox.currentIndexChanged.connect(persist_settings)
    dialog.samplingRateComboBox.currentIndexChanged.connect(persist_settings)
    dialog.batchSizeSpinBox.valueChanged.connect(persist_settings)
    dialog.finished.connect(cleanup_settings_dialog)
    if STATE.settings.device_address.strip():
        saved_name = STATE.settings.device_name.strip() or "Saved Polar device"
        saved_address = STATE.settings.device_address.strip()
        dialog.deviceComboBox.clear()
        dialog.deviceComboBox.addItem(f"{saved_name} ({saved_address}) [saved]", saved_address)
        dialog.deviceComboBox.setItemData(0, saved_name, DEVICE_NAME_ROLE)
        dialog.deviceComboBox.setCurrentIndex(0)
    else:
        start_scan()

    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def main() -> int:
    global App, MLink

    App = QApplication(sys.argv)
    App.setQuitOnLastWindowClosed(False)
    MLink = syl.init_link(rename_process=True)
    register_ports(MLink)
    MLink.on_prepare = prepare
    MLink.on_start = start
    MLink.on_stop = stop
    MLink.on_show_settings = show_settings
    MLink.on_save_settings = save_settings
    MLink.on_load_settings = load_settings
    MLink.await_data_forever(event_loop_tick)
    if STATE.running:
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
