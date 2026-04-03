import syntalos_mlink as syl

import asyncio
from dataclasses import dataclass, asdict
import json
import traceback

import numpy as np
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice
from bleakheart import PolarMeasurementData as PolarMeasurementDataOrig
from PyQt6 import uic
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QDialog, QLayout

import time


class PolarMeasurementData(PolarMeasurementDataOrig):
    async def _pmd_ctrl_request(self, request: bytearray):
        """Sends a control request to the PMD control point. Awaits the
        response with a timeout of 10 seconds."""
        # time to start notifications if not done yet
        if not self._notifications_started:
            await self._start_notifications()
        # make sure no other request can be made to ctrl point until
        # it  has responded.
        async with self._ctrl_lock:
            t0 = time.perf_counter_ns()
            self._ctrl_recv.clear()
            t_clear = (time.perf_counter_ns() - t0) // 1_000_000

            t0 = time.perf_counter_ns()
            await self.client.write_gatt_char(self.PMDCTRLPOINT, request)
            t_write_gatt = (time.perf_counter_ns() - t0) // 1_000_000

            t0 = time.perf_counter_ns()
            await asyncio.wait_for(self._ctrl_recv.wait(), timeout=10)
            t_ctrl_receive = (time.perf_counter_ns() - t0) // 1_000_000

            # grab response before releasing lock
            response = self._ctrl_response
            syl.println(f"{t_clear = }, {t_write_gatt = }, {t_ctrl_receive = }, {request.hex() = }")
        return response


def handle_fatal_exc(exc: Exception, syntalos_raise: bool, clean: bool, prefix: str = ""):
    msg = f"{prefix}{': ' if prefix else ''}{exc.__class__.__name__}({exc})"
    syl.println(f"{msg}\n{traceback.format_exc()}")
    if clean:
        cleanup()
    if syntalos_raise:
        syl.raise_error(msg)


# Path to the UI file (same directory as this script)
UI_FILE_PATH = "settings.ui"

NS_PER_SEC = 1_000_000_000
POLAR_ERR_ALREADY_IN_STATE = 6
VALID_PPG_SAMPLE_RATES = [28, 44, 55, 135, 176]


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
    stop_requested: bool = False
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


def clear_state():
    # Settings should stay persistent across runs
    STATE.stop_requested = False
    STATE.running = False
    STATE.loop = None
    STATE.client = None
    STATE.ppg_queue = None
    STATE.pmd = None
    STATE.offset_ns = None
    STATE.offset_start_us = None


STATE: State = State()


def serialise_settings(settings: Settings) -> bytes:
    return json.dumps(asdict(settings)).encode()


def deserialise_settings(settings: bytes) -> Settings:
    return Settings(**json.loads(settings.decode()))  # pyright: ignore[reportAny]


def save_current_settings() -> None:
    assert STATE.settings is not None
    syl.save_settings(serialise_settings(STATE.settings))


def close_settings_dialog() -> None:
    dialog = STATE.settings_dialog
    if dialog is not None:
        dialog.close()


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
        syl.println(f"Stopped existing {measurement} stream/state before start")


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
        syl.println(f"PPG stop failed: {exc.__class__.__name__}({exc})")
    try:
        await pmd.stop_streaming("SDK")
    except Exception as exc:
        syl.println(f"SDK stop failed: {exc.__class__.__name__}({exc})")


def submit_batch(timestamps_us: list[int], rows: list[list[int]]):
    block = syl.IntSignalBlock()
    block.timestamps = np.array(timestamps_us, dtype=np.uint64)
    block.data = np.array(rows, dtype=np.int32)
    out.submit(block)
    timestamps_us.clear()
    rows.clear()


def process_ppg_frame(frame, timestamps_us: list[int], rows: list[list[int]]):
    assert STATE.settings is not None
    assert STATE.offset_start_us is not None
    dtype, frame_timestamp_ns, payload = frame
    assert frame_timestamp_ns > 0, f"Negative {frame_timestamp_ns = }"
    if dtype != "PPG":
        syl.println(f"Expected PPG frame, got {dtype}")
        return
    if not payload:
        syl.println("Empty PPG frame")
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
        syl.println(f"{STATE.offset_start_us = }")
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


def cleanup():
    loop = STATE.loop
    if loop is None:
        syl.println("No event loop to cleanup, skipping cleanup()")
        return

    client = STATE.client
    if client is None:
        syl.println("No client to disconnect, skipping client.disconnect()")
        return

    # TODO: check the device is still connected

    try:
        loop.run_until_complete(stop_streaming())
    except Exception as exc:
        syl.println(f"Stop streaming failed: {exc.__class__.__name__}({exc})")

    try:
        loop.run_until_complete(client.disconnect())
    except EOFError:
        # Happens sometimes but does not seem to be problematic.
        # So far I have seen it only inside a VM.
        syl.println("EOFError at client.disconnect()")
    except Exception as exc:
        syl.println(f"Disconnect failed: {exc.__class__.__name__}({exc})")

    loop.close()

    # TODO: figure out how to make cleanup finish at the end of stop()
    syl.println("Cleanup complete")


# # ####################################################################################
# # Syntalos interface
# # ####################################################################################


out = syl.get_output_port("packets")
out.set_metadata_value("signal_names", ["PPG0", "PPG1", "PPG2", "AMBIENT"])
out.set_metadata_value("time_unit", "microseconds")
out.set_metadata_value("data_unit", ["raw", "raw", "raw", "raw"])


def prepare():
    clear_state()
    save_current_settings()
    close_settings_dialog()
    if STATE.settings is None:
        syl.println("Settings not set, aborting prepare()")
        return False

    if not STATE.settings.device_address:
        raise ValueError("Device address not set, edit the settings and try again")

    try:
        loop = asyncio.new_event_loop()
        STATE.loop = loop

        client = BleakClient(STATE.settings.device_address)
        loop.run_until_complete(client.connect())
        STATE.client = client
        STATE.ppg_queue = asyncio.Queue()
        STATE.pmd = PolarMeasurementData(client, ppg_queue=STATE.ppg_queue)

        loop.run_until_complete(start_sdk_mode())
        return True
    except Exception as exc:
        handle_fatal_exc(exc, syntalos_raise=True, clean=True, prefix="Prepare failed")
        return False


def start():
    assert STATE.loop is not None
    try:
        # ts_us_before = syl.time_since_start_usec() # returns ~300 us at this point
        STATE.loop.run_until_complete(start_ppg_streaming())
        # STATE.offset_start_us = int(syl.time_since_start_usec())
    except Exception as exc:
        handle_fatal_exc(exc, syntalos_raise=True, clean=True, prefix="Start failed")


def run() -> None:
    STATE.running = True
    assert STATE.loop is not None
    assert STATE.ppg_queue is not None
    assert STATE.client is not None

    timestamps_us: list[int] = []
    rows: list[list[int]] = []
    try:
        while not STATE.stop_requested and syl.is_running():
            # Give the async loop a chance to advance
            STATE.loop.run_until_complete(asyncio.sleep(0.010))
            while True:
                try:
                    frame = STATE.ppg_queue.get_nowait()
                    process_ppg_frame(frame, timestamps_us, rows)
                except asyncio.QueueEmpty:
                    break
            if not STATE.client.is_connected:
                # Polar devices sometimes disconnect unexpectedly.
                # Distance to the receiver seemed to be one of the causes.
                raise RuntimeError("Device disconnected")
            syl.wait(1)  # ms
        cleanup()
    except Exception as exc:
        handle_fatal_exc(exc, syntalos_raise=True, clean=True, prefix="Run failed")

    STATE.running = False


def stop():
    STATE.stop_requested = True
    # In case other modules trigger a premature stop(), we need to call cleanup() here
    if not STATE.running:
        cleanup()


def set_settings(settings: bytes):
    if settings:
        try:
            STATE.settings = deserialise_settings(settings)
        except Exception as exc:
            msg = f"Failed to parse settings: {exc.__class__.__name__}({exc})"
            syl.println(msg)
            syl.raise_error(msg)
            STATE.settings = Settings()
    elif STATE.settings is None:
        STATE.settings = Settings()


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
            syl.println(f"Polar scan failed: {exc.__class__.__name__}({exc})")
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
    save_current_settings()


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


def show_settings(settings: bytes):
    # Showing the settings UI while running prevents the run() loop from advancing.
    # Keep it simple: no settings UI while running.
    if STATE.running or syl.is_running():
        syl.println("Cannot show settings while running")
        return

    if not settings:
        if STATE.settings is None:
            STATE.settings = Settings()
    else:
        STATE.settings = deserialise_settings(settings)

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
    dialog.exec()


syl.call_on_show_settings(show_settings)
