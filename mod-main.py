import syntalos_mlink as syl

import asyncio
from dataclasses import dataclass, asdict
import json

import numpy as np
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice
from bleakheart import PolarMeasurementData
from PyQt6 import uic
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QDialog

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

    loop: asyncio.AbstractEventLoop | None = None
    client: BleakClient | None = None
    pmd: PolarMeasurementData | None = None
    ppg_queue: asyncio.Queue | None = None
    t0_ns: int | None = None


STATE = State()


def serialise_settings(settings: Settings) -> bytes:
    return json.dumps(asdict(settings)).encode()


def deserialise_settings(settings: bytes) -> Settings:
    return Settings(**json.loads(settings.decode()))  # pyright: ignore[reportAny]


def set_settings(settings: bytes):
    if settings:
        STATE.settings = deserialise_settings(settings)
    elif STATE.settings is None:
        STATE.settings = Settings()


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
    pmd = STATE.pmd
    assert pmd is not None
    await ensure_not_streaming(pmd, "PPG")
    await ensure_not_streaming(pmd, "SDK")
    err_code, err_msg, _ = await pmd.start_streaming("SDK")
    if err_code != 0:
        raise RuntimeError(f"Failed to start SDK mode: {err_code} {err_msg}")


async def start_ppg_streaming():
    pmd = STATE.pmd
    assert pmd is not None
    err_code, err_msg, _ = await pmd.start_streaming(
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
        syl.println(f"PPG stop failed: {exc}")
    try:
        await pmd.stop_streaming("SDK")
    except Exception as exc:
        syl.println(f"SDK stop failed: {exc}")


def submit_batch(timestamps_us: list[int], rows: list[list[int]]):
    block = syl.IntSignalBlock()
    block.timestamps = np.array(timestamps_us, dtype=np.uint64)
    block.data = np.array(rows, dtype=np.int64)
    out.submit(block)
    timestamps_us.clear()
    rows.clear()


def process_ppg_frame(frame, timestamps_us: list[int], rows: list[list[int]]):
    dtype, frame_timestamp_ns, payload = frame
    if dtype != "PPG":
        syl.println(f"Expected PPG frame, got {dtype}")
        return
    if not payload:
        syl.println("Empty PPG frame")
        return

    n_samples = len(payload)
    if STATE.t0_ns is None:
        first_offset_ns = (
            (n_samples - 1) * NS_PER_SEC + STATE.settings.sampling_rate // 2
        ) // STATE.settings.sampling_rate
        STATE.t0_ns = frame_timestamp_ns - first_offset_ns

    for back_idx, sample in zip(range(n_samples - 1, -1, -1), payload):
        ts_ns = frame_timestamp_ns - (
            (back_idx * NS_PER_SEC + STATE.settings.sampling_rate // 2)
            // STATE.settings.sampling_rate
        )
        timestamps_us.append((ts_ns - STATE.t0_ns) // 1_000)
        rows.append([int(sample[0]), int(sample[1]), int(sample[2]), int(sample[3])])

    if len(timestamps_us) >= STATE.settings.batch_size:
        submit_batch(timestamps_us, rows)


def cleanup():
    loop = STATE.loop
    assert loop is not None
    client = STATE.client
    assert client is not None

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

    STATE.loop = None
    STATE.client = None
    STATE.ppg_queue = None
    STATE.pmd = None

    STATE.stop_requested = False
    STATE.t0_ns = None
    # TODO: figure out how to make cleanup finish at the end of stop()
    syl.println("Cleanup complete")


# ## ###############################################################################################
# Syntalos interface
# ## ###############################################################################################


out = syl.get_output_port("packets")
out.set_metadata_value("signal_names", ["PPG0", "PPG1", "PPG2", "AMBIENT"])
out.set_metadata_value("time_unit", "microseconds")
out.set_metadata_value("data_unit", ["raw", "raw", "raw", "raw"])


def prepare():
    loop = asyncio.new_event_loop()
    STATE.loop = loop

    client = BleakClient(STATE.settings.device_address)
    loop.run_until_complete(client.connect())
    # syl.println(f"Connected to {client.address}")
    STATE.client = client
    STATE.ppg_queue = asyncio.Queue()
    STATE.pmd = PolarMeasurementData(client, ppg_queue=STATE.ppg_queue)
    STATE.t0_ns = None
    try:
        loop.run_until_complete(start_sdk_mode())
    except Exception as exc:
        syl.println(f"Start SDK mode failed: {exc.__class__.__name__}({exc})")
        cleanup()
        raise
    return True


def start():
    loop = STATE.loop
    assert loop is not None
    loop.run_until_complete(start_ppg_streaming())


def run():
    loop = STATE.loop
    ppg_queue = STATE.ppg_queue
    assert loop is not None
    assert ppg_queue is not None

    timestamps_us: list[int] = []
    rows: list[list[int]] = []
    try:
        while not STATE.stop_requested and syl.is_running():
            # Give the async loop a chance to advance
            loop.run_until_complete(asyncio.sleep(0.040))
            while True:
                try:
                    frame = ppg_queue.get_nowait()
                    process_ppg_frame(frame, timestamps_us, rows)
                except asyncio.QueueEmpty:
                    break
            syl.wait(20)  # ms
    finally:
        cleanup()


def stop():
    STATE.stop_requested = True


# ## ###############################################################################################
# ## Settings UI
# ## ###############################################################################################


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


def show_settings(settings: bytes):
    if not settings:
        if STATE.settings is None:
            STATE.settings = Settings()
    else:
        STATE.settings = deserialise_settings(settings)

    dialog: QDialog = uic.loadUi(UI_FILE_PATH)

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

    scan_state = {
        "thread": None,
        "worker": None,
        "selected_address": STATE.settings.device_address,
    }

    def set_scanning_ui(scanning: bool):
        try:
            dialog.deviceComboBox.setEnabled(not scanning)
            dialog.refreshDevicesButton.setEnabled(not scanning)
            if scanning:
                dialog.deviceComboBox.clear()
                dialog.deviceComboBox.setPlaceholderText("")
                dialog.refreshDevicesButton.setToolTip("Scanning devices...")
            else:
                dialog.refreshDevicesButton.setToolTip("Refresh device list.")
        except RuntimeError:
            return

    def finish_scan(entries: list[tuple[str, str]], error_text: str):
        try:
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
            selected_address = scan_state["selected_address"]
            if isinstance(selected_address, str) and selected_address:
                selected_index = dialog.deviceComboBox.findData(selected_address)
            if selected_index < 0 and STATE.settings.device_address:
                selected_index = dialog.deviceComboBox.findData(STATE.settings.device_address)
            if selected_index < 0:
                selected_index = 0
            dialog.deviceComboBox.setCurrentIndex(selected_index)
        except RuntimeError:
            return

    def cleanup_scan_thread():
        scan_state["thread"] = None
        scan_state["worker"] = None
        set_scanning_ui(False)

    def start_scan():
        thread = scan_state["thread"]
        if isinstance(thread, QThread) and thread.isRunning():
            return

        current_address = dialog.deviceComboBox.currentData()
        if isinstance(current_address, str) and current_address:
            scan_state["selected_address"] = current_address
        else:
            scan_state["selected_address"] = STATE.settings.device_address

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

        scan_state["thread"] = thread
        scan_state["worker"] = worker
        thread.start()

    dialog.refreshDevicesButton.clicked.connect(start_scan)
    if STATE.settings.device_address.strip():
        saved_name = STATE.settings.device_name.strip() or "Saved Polar device"
        saved_address = STATE.settings.device_address.strip()
        dialog.deviceComboBox.clear()
        dialog.deviceComboBox.addItem(f"{saved_name} ({saved_address}) [saved]", saved_address)
        dialog.deviceComboBox.setItemData(0, saved_name, DEVICE_NAME_ROLE)
        dialog.deviceComboBox.setCurrentIndex(0)
    else:
        start_scan()

    if dialog.exec() == QDialog.DialogCode.Accepted:
        address = dialog.deviceComboBox.currentData()
        name = dialog.deviceComboBox.currentData(DEVICE_NAME_ROLE)
        if isinstance(address, str) and address.strip():
            STATE.settings.device_address = address.strip()
            STATE.settings.device_name = name.strip() if isinstance(name, str) else ""
        sampling_rate = dialog.samplingRateComboBox.currentData()
        if sampling_rate is not None:
            STATE.settings.sampling_rate = int(sampling_rate)
        STATE.settings.batch_size = dialog.batchSizeSpinBox.value()
        syl.save_settings(serialise_settings(STATE.settings))


syl.call_on_show_settings(show_settings)
