import syntalos_mlink as syl

import asyncio
from dataclasses import dataclass, asdict
import json

import numpy as np
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice
from bleakheart import PolarMeasurementData

# Path to the UI file (same directory as this script)
UI_FILE_PATH = "settings.ui"

PPG_SAMPLE_RATE_HZ = 176
PPG_RESOLUTION_BITS = 22
PPG_CHANNELS = 4
PPG_BATCH_SIZE = 64
NS_PER_SEC = 1_000_000_000
POLAR_ERR_ALREADY_IN_STATE = 6


@dataclass
class Settings:
    tbd: str = "tbd"


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


async def scan_for_device():
    device: BLEDevice | None = await BleakScanner.find_device_by_filter(
        lambda dev, adv: bool(dev.name and "polar" in dev.name.lower())
    )
    if device is None:
        raise RuntimeError("Polar device not found")
    return device


async def ensure_not_streaming(pmd: PolarMeasurementData, measurement: str):
    err_code, err_msg = await pmd.stop_streaming(measurement)
    if err_code not in (0, POLAR_ERR_ALREADY_IN_STATE):
        raise RuntimeError(f"Failed to stop {measurement} before restart: {err_code} {err_msg}")
    if err_code == 0:
        syl.println(f"Stopped existing {measurement} stream/state before start")


async def start_sdk_mode():
    pmd = STATE.pmd
    assert pmd is not None
    await ensure_not_streaming(pmd, "SDK")
    err_code, err_msg, _ = await pmd.start_streaming("SDK")
    if err_code != 0:
        raise RuntimeError(f"Failed to start SDK mode: {err_code} {err_msg}")


async def start_ppg_streaming():
    pmd = STATE.pmd
    assert pmd is not None
    await ensure_not_streaming(pmd, "PPG")
    err_code, err_msg, _ = await pmd.start_streaming(
        "PPG",
        SAMPLE_RATE=PPG_SAMPLE_RATE_HZ,
        RESOLUTION=PPG_RESOLUTION_BITS,
        CHANNELS=PPG_CHANNELS,
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
            (n_samples - 1) * NS_PER_SEC + PPG_SAMPLE_RATE_HZ // 2
        ) // PPG_SAMPLE_RATE_HZ
        STATE.t0_ns = int(frame_timestamp_ns) - first_offset_ns

    t0_ns = STATE.t0_ns
    assert t0_ns is not None

    for back_idx, sample in zip(range(n_samples - 1, -1, -1), payload):
        ts_ns = int(frame_timestamp_ns) - (
            (back_idx * NS_PER_SEC + PPG_SAMPLE_RATE_HZ // 2) // PPG_SAMPLE_RATE_HZ
        )
        timestamps_us.append((ts_ns - t0_ns) // 1_000)
        rows.append([int(sample[0]), int(sample[1]), int(sample[2]), int(sample[3])])

    if len(timestamps_us) >= PPG_BATCH_SIZE:
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
        # Happens sometimes but does not seem to be problematic
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

    client = BleakClient(loop.run_until_complete(scan_for_device()))
    loop.run_until_complete(client.connect())
    syl.println(f"Connected to {client.address}")
    STATE.client = client
    STATE.ppg_queue = asyncio.Queue()
    STATE.pmd = PolarMeasurementData(client, ppg_queue=STATE.ppg_queue)
    STATE.t0_ns = None
    loop.run_until_complete(start_sdk_mode())
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
            loop.run_until_complete(asyncio.sleep(0.020))
            while True:
                try:
                    frame = ppg_queue.get_nowait()
                    process_ppg_frame(frame, timestamps_us, rows)
                except asyncio.QueueEmpty:
                    break
            syl.wait(5)  # ms
    finally:
        cleanup()


def stop():
    STATE.stop_requested = True
