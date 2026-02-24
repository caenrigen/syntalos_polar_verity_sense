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
PPG_BATCH_SIZE = 128  # decoded samples per Syntalos block
PPG_QUEUE_TIMEOUT_SEC = 0.005
PPG_MAX_FRAMES_PER_SLICE = 16
NS_PER_SEC = 1_000_000_000


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
    ppg_stream_started: bool = False
    sdk_mode_started: bool = False
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


async def start_ppg_streaming():
    pmd = STATE.pmd
    queue = STATE.ppg_queue
    assert pmd is not None
    assert queue is not None

    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    err_code, err_msg, _ = await pmd.start_streaming("SDK")
    if err_code != 0:
        raise RuntimeError(f"Failed to start SDK mode: {err_code} {err_msg}")
    STATE.sdk_mode_started = True

    err_code, err_msg, _ = await pmd.start_streaming(
        "PPG",
        SAMPLE_RATE=PPG_SAMPLE_RATE_HZ,
        RESOLUTION=PPG_RESOLUTION_BITS,
        CHANNELS=PPG_CHANNELS,
    )
    if err_code != 0:
        raise RuntimeError(f"Failed to start PPG stream: {err_code} {err_msg}")
    STATE.ppg_stream_started = True


async def stop_streaming_and_disconnect():
    pmd = STATE.pmd
    client = STATE.client

    if pmd is not None and client is not None and client.is_connected:
        if STATE.ppg_stream_started:
            try:
                await pmd.stop_streaming("PPG")
            except Exception as exc:
                syl.println(f"PPG stop failed: {exc}")
        if STATE.sdk_mode_started:
            try:
                await pmd.stop_streaming("SDK")
            except Exception as exc:
                syl.println(f"SDK stop failed: {exc}")

    if client is not None and client.is_connected:
        await client.disconnect()


def submit_batch(timestamps_us: list[int], rows: list[list[int]]):
    if not timestamps_us:
        return
    block = syl.IntSignalBlock()
    block.timestamps = np.array(timestamps_us, dtype=np.uint64)
    block.data = np.array(rows, dtype=np.int64)
    out.submit(block)
    timestamps_us.clear()
    rows.clear()


def append_ppg_frame_to_batch(frame, timestamps_us: list[int], rows: list[list[int]]):
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


def cleanup():
    client = STATE.client
    loop = STATE.loop

    if loop is not None and not loop.is_closed():
        if client and client.is_connected:
            loop.run_until_complete(stop_streaming_and_disconnect())

    if loop is not None and not loop.is_closed():
        loop.close()

    STATE.client = None
    STATE.pmd = None
    STATE.ppg_queue = None
    STATE.loop = None
    STATE.stop_requested = False
    STATE.ppg_stream_started = False
    STATE.sdk_mode_started = False
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
            try:
                frame = loop.run_until_complete(
                    asyncio.wait_for(ppg_queue.get(), timeout=PPG_QUEUE_TIMEOUT_SEC)
                )
            except asyncio.TimeoutError:
                submit_batch(timestamps_us, rows)
                syl.wait(5)
                continue

            append_ppg_frame_to_batch(frame, timestamps_us, rows)

            drained_frames = 0
            while True:
                try:
                    frame = ppg_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                append_ppg_frame_to_batch(frame, timestamps_us, rows)
                drained_frames += 1
                if len(timestamps_us) >= PPG_BATCH_SIZE:
                    submit_batch(timestamps_us, rows)
                if drained_frames >= PPG_MAX_FRAMES_PER_SLICE:
                    break

            if len(timestamps_us) >= PPG_BATCH_SIZE:
                submit_batch(timestamps_us, rows)

            # Mandatory for Syntalos Python modules: this lets Syntalos process IPC,
            # including stop() requests, while we are streaming in a tight Python loop.
            syl.wait(1)
    finally:
        submit_batch(timestamps_us, rows)
        cleanup()


def stop():
    STATE.stop_requested = True
