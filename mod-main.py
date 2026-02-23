import syntalos_mlink as syl

import time
import asyncio

from dataclasses import dataclass, asdict
import json

import numpy as np
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice

# Path to the UI file (same directory as this script)
UI_FILE_PATH = "settings.ui"

CHARACTERISTIC_BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"


@dataclass
class Settings:
    tbd: str = "tbd"


@dataclass
class State:
    settings: Settings | None = None
    stop_requested: bool = False

    loop: asyncio.AbstractEventLoop | None = None
    client: BleakClient | None = None


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


async def read_battery_level():
    client = STATE.client
    assert client is not None
    data = await client.read_gatt_char(CHARACTERISTIC_BATTERY)
    assert len(data) == 1, f"Expected 1 byte, {len(data) = }"
    return data[0]


def cleanup():
    client = STATE.client
    loop = STATE.loop

    if client and client.is_connected:
        assert loop is not None
        loop.run_until_complete(client.disconnect())

    if loop is not None and not loop.is_closed():
        loop.close()

    STATE.client = None
    STATE.loop = None
    STATE.stop_requested = False


# ## ###############################################################################################
# Syntalos interface
# ## ###############################################################################################


out = syl.get_output_port("packets")
out.set_metadata_value("signal_names", ["TIMESTAMP_RAW", "BATTERY"])
out.set_metadata_value("time_unit", "microseconds")
out.set_metadata_value("data_unit", ["nanoseconds", "%"])


def prepare():
    loop = asyncio.new_event_loop()
    STATE.loop = loop

    client = BleakClient(loop.run_until_complete(scan_for_device()))
    loop.run_until_complete(client.connect())
    syl.println(f"Connected to {client.address}")
    STATE.client = client
    return True


def start():
    pass


def run():
    loop = STATE.loop
    assert loop is not None
    t = time.time()
    l = 0
    try:
        while not STATE.stop_requested and syl.is_running():
            syl.println("Reading battery level...")
            level = loop.run_until_complete(read_battery_level())
            t1 = time.time()
            block = syl.IntSignalBlock()
            block.timestamps = np.array([int((t1 - t) * 1e6)], dtype=np.uint64)
            block.data = np.array([[int((t1 - t) * 1e9), level + l]], dtype=np.int64)
            l += 1
            out.submit(block)
            syl.wait_sec(1)
    finally:
        cleanup()


def stop():
    STATE.stop_requested = True
