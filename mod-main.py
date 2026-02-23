import syntalos_mlink as syl

import time
import asyncio

from dataclasses import dataclass, asdict
import json

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
    if client and client.is_connected:
        asyncio.run(client.disconnect())
    STATE.stop_requested = False


# ## ###############################################################################################
# Syntalos interface
# ## ###############################################################################################


out = syl.get_output_port("packets")
out.set_metadata_value("signal_names", ["TIMESTAMP_RAW", "BATTERY"])
out.set_metadata_value("time_unit", "microseconds")
out.set_metadata_value("data_unit", ["nanoseconds", "%"])


def prepare():
    client = BleakClient(asyncio.run(scan_for_device()))
    asyncio.run(client.connect())
    STATE.client = client
    return True


def start():
    pass


def run():
    try:
        while not STATE.stop_requested and syl.is_running():
            out.submit([int(time.time() * 1e6), asyncio.run(read_battery_level())])
            syl.wait(20)
    finally:
        cleanup()


def stop():
    STATE.stop_requested = True
