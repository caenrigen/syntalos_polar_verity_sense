#!/usr/bin/env python3

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import mean, median, stdev
from time import perf_counter_ns, time_ns

import click
from bleak import BleakClient
from bleakheart import PolarMeasurementData

POLAR_ERR_ALREADY_IN_STATE = 6
VALID_PPG_SAMPLE_RATES = [28, 44, 55, 135, 176]
DEFAULT_SAMPLE_RATE = 176
DEFAULT_RESOLUTION = 22
DEFAULT_CHANNELS = 4


@dataclass(slots=True)
class FirstPacketObservation:
    received_ns: int
    samples_in_packet: int


@dataclass(slots=True)
class TrialResult:
    trial_index: int
    latency_ns: int
    samples_in_first_packet: int


class TimedPolarMeasurementData(PolarMeasurementData):
    def __init__(
        self,
        client: BleakClient,
        *,
        first_packet_future: asyncio.Future[FirstPacketObservation],
    ) -> None:
        super().__init__(client, callback=lambda _payload: None)
        self.first_packet_future = first_packet_future
        self.ppg_start_write_ns: int | None = None

    async def _pmd_ctrl_request(self, request: bytearray):
        if not self._notifications_started:
            await self._start_notifications()

        async with self._ctrl_lock:
            self._ctrl_recv.clear()
            if self._is_ppg_start_request(request) and self.ppg_start_write_ns is None:
                self.ppg_start_write_ns = perf_counter_ns()
            await self.client.write_gatt_char(self.PMDCTRLPOINT, request)
            await asyncio.wait_for(self._ctrl_recv.wait(), timeout=10)
            response = self._ctrl_response

        return response

    async def _pmd_data_handler(self, characteristic, data: bytearray):
        received_ns = perf_counter_ns()
        meas = self.measurement_types[data[0]]
        timestamp = int.from_bytes(data[1:9], "little", signed=False)
        frametype = data[9]

        try:
            timestamp += self._time_offset
        except TypeError:
            self._time_offset = time_ns() - timestamp
            timestamp += self._time_offset

        if meas == "PPG" and frametype == 128:
            payload = self._decode_ppg_data(data)
            if payload and not self.first_packet_future.done():
                self.first_packet_future.set_result(
                    FirstPacketObservation(
                        received_ns=received_ns,
                        samples_in_packet=len(payload),
                    )
                )
            if self._ppg_callback_is_coro:
                await self._ppg_callback(("PPG", timestamp, payload))
            else:
                self._ppg_callback(("PPG", timestamp, payload))
            return

        await super()._pmd_data_handler(characteristic, data)

    def _is_ppg_start_request(self, request: bytearray) -> bool:
        return (
            len(request) >= 2
            and request[0] == self.op_codes["START"]
            and request[1] == self.measurement_types.index("PPG")
        )


async def ensure_not_streaming(pmd: PolarMeasurementData, measurement: str) -> None:
    err_code, err_msg = await pmd.stop_streaming(measurement)
    if err_code not in (0, POLAR_ERR_ALREADY_IN_STATE):
        raise RuntimeError(f"Failed to stop {measurement} before restart: {err_code} {err_msg}")


async def start_sdk_mode(pmd: PolarMeasurementData) -> None:
    await ensure_not_streaming(pmd, "PPG")
    await ensure_not_streaming(pmd, "SDK")
    err_code, err_msg, _ = await pmd.start_streaming("SDK")
    if err_code != 0:
        raise RuntimeError(f"Failed to start SDK mode: {err_code} {err_msg}")


async def stop_streaming(pmd: PolarMeasurementData) -> None:
    for measurement in ("PPG", "SDK"):
        try:
            await pmd.stop_streaming(measurement)
        except Exception:
            pass


async def run_trial(
    *,
    trial_index: int,
    mac_address: str,
    timeout_s: float,
    sample_rate: int,
) -> TrialResult:
    first_packet_future: asyncio.Future[FirstPacketObservation] = (
        asyncio.get_running_loop().create_future()
    )
    client = BleakClient(mac_address)
    pmd: TimedPolarMeasurementData | None = None

    try:
        await client.connect()
        pmd = TimedPolarMeasurementData(client, first_packet_future=first_packet_future)

        await start_sdk_mode(pmd)

        err_code, err_msg, _ = await pmd.start_streaming(
            "PPG",
            sample_rate=sample_rate,
            resolution=DEFAULT_RESOLUTION,
            channels=DEFAULT_CHANNELS,
        )
        if err_code != 0:
            raise RuntimeError(f"Failed to start PPG stream: {err_code} {err_msg}")
        if pmd.ppg_start_write_ns is None:
            raise RuntimeError("Failed to capture the PPG start control write timestamp")

        first_packet = await asyncio.wait_for(first_packet_future, timeout=timeout_s)
        return TrialResult(
            trial_index=trial_index,
            latency_ns=first_packet.received_ns - pmd.ppg_start_write_ns,
            samples_in_first_packet=first_packet.samples_in_packet,
        )
    finally:
        if pmd is not None:
            await stop_streaming(pmd)
        try:
            if client.is_connected:
                await client.disconnect()
        except EOFError:
            pass


async def run_trials(
    *,
    mac_address: str,
    trials: int,
    timeout_s: float,
    pause_s: float,
    sample_rate: int,
) -> int:
    results: list[TrialResult] = []
    failures = 0

    click.echo(
        "Measuring from the PMD PPG START control write to the first received PPG data packet."
    )
    click.echo(
        f"Device: {mac_address} | Trials: {trials} | Sample rate: {sample_rate} Hz | Timeout: {timeout_s:.1f}s"
    )

    for trial_index in range(1, trials + 1):
        try:
            result = await run_trial(
                trial_index=trial_index,
                mac_address=mac_address,
                timeout_s=timeout_s,
                sample_rate=sample_rate,
            )
        except Exception as exc:
            failures += 1
            click.echo(
                f"Trial {trial_index}/{trials}: FAILED ({exc.__class__.__name__}: {exc})",
                err=True,
            )
        else:
            results.append(result)
            click.echo(
                f"Trial {trial_index}/{trials}: "
                f"{result.latency_ns / 1_000_000:.3f} ms "
                f"({result.samples_in_first_packet} samples in first packet)"
            )

        if pause_s > 0 and trial_index < trials:
            await asyncio.sleep(pause_s)

    click.echo("")
    if not results:
        click.echo("No successful trials.", err=True)
        return 1

    latencies_ms = [result.latency_ns / 1_000_000 for result in results]
    mean_ms = mean(latencies_ms)
    latencies_ms = [l - mean_ms for l in latencies_ms]
    first_packet_sizes = [result.samples_in_first_packet for result in results]

    click.echo(f"Successful trials: {len(results)}/{trials}")
    click.echo(
        "Latency summary [ms]: "
        f"min={min(latencies_ms):.3f} "
        f"median={median(latencies_ms):.3f} "
        f"max={max(latencies_ms):.3f}"
        + (f" stdev={stdev(latencies_ms):.3f}" if len(latencies_ms) > 1 else "")
    )
    click.echo(
        "First packet size summary [samples]: "
        f"min={min(first_packet_sizes)} "
        f"mean={mean(first_packet_sizes):.1f} "
        f"max={max(first_packet_sizes)}"
    )

    return 0 if failures == 0 else 1


def validate_sample_rate(
    _ctx: click.Context,
    _param: click.Parameter,
    value: int,
) -> int:
    if value not in VALID_PPG_SAMPLE_RATES:
        raise click.BadParameter(
            f"must be one of {', '.join(str(rate) for rate in VALID_PPG_SAMPLE_RATES)}"
        )
    return value


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--mac",
    "mac_address",
    required=True,
    help="Polar device MAC address.",
)
@click.option(
    "--trials",
    required=True,
    type=click.IntRange(min=1),
    help="Number of full connect/start/stop/disconnect trials to run.",
)
@click.option(
    "--timeout",
    "timeout_s",
    default=10.0,
    show_default=True,
    type=click.FloatRange(min=0.1),
    help="Seconds to wait for the first PPG packet in each trial.",
)
@click.option(
    "--pause",
    "pause_s",
    default=0.0,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help="Seconds to wait between trials.",
)
@click.option(
    "--sample-rate",
    default=DEFAULT_SAMPLE_RATE,
    show_default=True,
    callback=validate_sample_rate,
    type=int,
    help="PPG sample rate in Hz.",
)
def main(
    mac_address: str,
    trials: int,
    timeout_s: float,
    pause_s: float,
    sample_rate: int,
) -> None:
    raise SystemExit(
        asyncio.run(
            run_trials(
                mac_address=mac_address,
                trials=trials,
                timeout_s=timeout_s,
                pause_s=pause_s,
                sample_rate=sample_rate,
            )
        )
    )


if __name__ == "__main__":
    main()
