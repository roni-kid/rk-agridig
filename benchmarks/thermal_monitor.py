#!/usr/bin/env python3
"""
thermal_monitor.py — RK AgriDig Phase 1, Task 1.4

Runs inference on the Phi-3-mini model in a loop for a configurable duration
(default 5+ minutes per the build plan), logging CPU temperature every 5
seconds via lm-sensors. Alerts if temperature crosses the ADTC disqualification
threshold (85°C) and reports avg/max/min at the end.

IMPORTANT — WSL2 users read this:
    lm-sensors CANNOT read real hardware temperatures inside WSL2. The
    `coretemp` kernel module (and the ISA/I2C interfaces lm-sensors needs)
    are not exposed by the WSL2 kernel — this is a hard platform limitation,
    not a configuration problem. `sensors-detect` will fail with something
    like "Module cpuid not found" no matter what you try.

    This script detects that condition and reports it honestly rather than
    fabricating numbers. For a real thermal reading against the ADTC 85°C
    threshold, this must be run on a bare-metal Ubuntu 22.04 boot (or a VM
    with real hardware passthrough), not WSL2.

Usage:
    python3 benchmarks/thermal_monitor.py
    python3 benchmarks/thermal_monitor.py --duration 300 --interval 5
    python3 benchmarks/thermal_monitor.py --dry-run   # test logging without running inference
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil
except ImportError:
    print(
        "ERROR: psutil is not installed.\nInstall it with: pip install psutil --break-system-packages\n",
        file=sys.stderr,
    )
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "phi3_mini_4k_instruct.gguf"
LLAMA_CLI = REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-cli"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
LOG_PATH = RESULTS_DIR / "thermal_logs.txt"
SUMMARY_JSON_PATH = RESULTS_DIR / "thermal_summary.json"

ADTC_DISQUALIFICATION_TEMP_C = 85.0
ADTC_WARNING_TEMP_C = 80.0  # success criteria in the build plan: stay <80C

TEST_PROMPT = (
    "<|user|>\nMy tomato leaves have dark spots with yellow rings around them. "
    "What disease is this and how do I treat it?<|end|>\n<|assistant|>"
)


@dataclass
class ThermalReading:
    timestamp: str
    elapsed_s: float
    temp_c: float | None  # None if sensors unavailable for this reading
    source: str  # "lm-sensors" | "unavailable"


@dataclass
class MonitorState:
    readings: list[ThermalReading] = field(default_factory=list)
    sensors_available: bool = True
    max_alert_fired: bool = False
    inference_error_count: int = 0


def check_sensors_available() -> tuple[bool, str]:
    """
    Determine whether lm-sensors can produce a real temperature reading.

    Returns (available, reason). `reason` is a human string used both in
    logs and in the final JSON so the person reading results later
    understands *why* readings might be missing, instead of assuming a bug.
    """
    if shutil.which("sensors") is None:
        return False, "lm-sensors is not installed (command 'sensors' not found)"

    try:
        result = subprocess.run(
            ["sensors", "-j"], capture_output=True, text=True, timeout=5
        )
    except subprocess.TimeoutExpired:
        return False, "'sensors' command timed out"
    except Exception as exc:  # noqa: BLE001
        return False, f"'sensors' command failed to run: {exc}"

    raw = result.stdout.strip()
    if not raw:
        return False, "'sensors -j' returned empty output — no sensors detected on this system"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Known lm-sensors bug: invalid JSON (dangling commas) when a feature
        # is unreadable. Try to salvage anything parseable before giving up.
        return False, "'sensors -j' returned malformed JSON (known lm-sensors issue with unreadable features)"

    if not data:
        return False, "'sensors -j' returned an empty object — no chips detected"

    # Look for anything that smells like a CPU package/core temperature.
    for chip_name, chip_data in data.items():
        if not isinstance(chip_data, dict):
            continue
        for feature_name, feature_data in chip_data.items():
            if "package" in feature_name.lower() or "core" in feature_name.lower() or "tdie" in feature_name.lower():
                if isinstance(feature_data, dict):
                    for k, v in feature_data.items():
                        if "input" in k and isinstance(v, (int, float)):
                            return True, f"found temperature feature '{feature_name}' on chip '{chip_name}'"

    return False, "sensors detected but no CPU package/core temperature feature found (common in WSL2 — see script docstring)"


def read_cpu_temp_c() -> float | None:
    """
    Read current CPU package temperature via `sensors -j`.
    Returns None if unavailable or unparseable for this specific call
    (transient failures shouldn't crash a 5-minute monitoring run).
    """
    try:
        result = subprocess.run(
            ["sensors", "-j"], capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)
    except Exception:  # noqa: BLE001 - any failure here just means "no reading this tick"
        return None

    candidates: list[float] = []
    for chip_name, chip_data in data.items():
        if not isinstance(chip_data, dict):
            continue
        for feature_name, feature_data in chip_data.items():
            fname_lower = feature_name.lower()
            if "package" in fname_lower or "tdie" in fname_lower:
                if isinstance(feature_data, dict):
                    for k, v in feature_data.items():
                        if "input" in k and isinstance(v, (int, float)):
                            candidates.append(float(v))
            elif "core" in fname_lower and "coretemp" in chip_name.lower():
                if isinstance(feature_data, dict):
                    for k, v in feature_data.items():
                        if "input" in k and isinstance(v, (int, float)):
                            candidates.append(float(v))

    if not candidates:
        return None
    # Package temp (if found) is most representative; otherwise max of cores
    # is the conservative choice for a throttling/safety check.
    return max(candidates)


def run_inference_loop(stop_event: threading.Event, state: MonitorState, dry_run: bool) -> None:
    """
    Runs llama-cli repeatedly against TEST_PROMPT until stop_event is set.
    Runs in a background thread so temperature sampling isn't blocked by
    each inference call.
    """
    if dry_run:
        # Just idle — useful for testing the logging/alerting logic without
        # burning 5 minutes of real CPU time or requiring the model file.
        while not stop_event.is_set():
            time.sleep(0.5)
        return

    if not LLAMA_CLI.exists():
        print(f"ERROR: llama-cli not found at {LLAMA_CLI}. Run setup.sh first.", file=sys.stderr)
        state.inference_error_count += 1
        stop_event.set()
        return

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}. Run models/download_model.py first.", file=sys.stderr)
        state.inference_error_count += 1
        stop_event.set()
        return

    # Use physical core count, not logical/hyperthreaded count. Task 1.3's
    # profiling (see REPORT.md) found 16 logical threads nearly halves
    # throughput vs 8 physical cores on this hardware due to hyperthread
    # contention — running the thermal test at the wrong thread count also
    # confounds the heat measurement with that same contention overhead,
    # rather than reflecting realistic sustained-load heat.
    physical_cores = psutil.cpu_count(logical=False)
    threads = str(physical_cores or psutil.cpu_count(logical=True) or 4)

    while not stop_event.is_set():
        try:
            subprocess.run(
                [
                    str(LLAMA_CLI),
                    "-m", str(MODEL_PATH),
                    "-p", TEST_PROMPT,
                    "-n", "128",
                    "-t", threads,
                    "--no-display-prompt",
                ],
                capture_output=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            state.inference_error_count += 1
        except Exception:  # noqa: BLE001
            state.inference_error_count += 1
        # small gap between runs so we're not purely CPU-load-testing;
        # this approximates realistic intermittent farmer usage
        time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Thermal monitor for RK AgriDig inference.")
    parser.add_argument("--duration", type=int, default=300, help="Monitoring duration in seconds (default: 300 = 5 min)")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between temperature samples (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Test logging/alerting without running real inference")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RK AgriDig — Thermal Monitor (Task 1.4)")
    print("=" * 70)
    print(f"Duration:  {args.duration}s")
    print(f"Interval:  {args.interval}s")
    print(f"ADTC disqualification threshold: {ADTC_DISQUALIFICATION_TEMP_C}°C")
    print(f"ADTC target ceiling (success criteria): {ADTC_WARNING_TEMP_C}°C")

    sensors_ok, reason = check_sensors_available()
    state = MonitorState(sensors_available=sensors_ok)

    if not sensors_ok:
        print(f"\n⚠ lm-sensors cannot provide real temperature readings: {reason}")
        print("  This is EXPECTED and UNAVOIDABLE under WSL2 (see script docstring).")
        print("  This run will still exercise the inference loop and log timing,")
        print("  but temperature values will be recorded as unavailable.")
        print("  For a real ADTC thermal result, run this on bare-metal Ubuntu 22.04.\n")
    else:
        print(f"\n✓ lm-sensors available: {reason}\n")

    if args.dry_run:
        print("(--dry-run: not actually running inference, just testing the monitor loop)\n")

    stop_event = threading.Event()
    inference_thread = threading.Thread(
        target=run_inference_loop, args=(stop_event, state, args.dry_run), daemon=True
    )
    inference_thread.start()

    start_time = time.monotonic()
    log_lines: list[str] = []
    header = f"# RK AgriDig Thermal Log — started {datetime.now(timezone.utc).isoformat()}"
    log_lines.append(header)
    log_lines.append(f"# sensors_available={sensors_ok} reason=\"{reason}\"")
    log_lines.append(f"# adtc_disqualification_threshold_c={ADTC_DISQUALIFICATION_TEMP_C}")
    log_lines.append("# timestamp_utc,elapsed_s,temp_c,source")

    try:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= args.duration:
                break
            if state.inference_error_count > 0 and not args.dry_run:
                # Inference thread hit a hard error (missing binary/model) —
                # no point continuing to "monitor" nothing.
                print("Stopping early: inference loop reported a fatal error (see above).")
                break

            temp = read_cpu_temp_c() if sensors_ok else None
            reading = ThermalReading(
                timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_s=round(elapsed, 1),
                temp_c=temp,
                source="lm-sensors" if temp is not None else "unavailable",
            )
            state.readings.append(reading)

            temp_str = f"{temp:.1f}°C" if temp is not None else "N/A"
            line = f"{reading.timestamp},{reading.elapsed_s},{temp_str},{reading.source}"
            log_lines.append(line)
            print(f"[{elapsed:6.1f}s] Temp: {temp_str}")

            if temp is not None:
                if temp >= ADTC_DISQUALIFICATION_TEMP_C and not state.max_alert_fired:
                    print(f"\n🔴 ALERT: Temperature {temp:.1f}°C EXCEEDS ADTC disqualification threshold ({ADTC_DISQUALIFICATION_TEMP_C}°C)!\n")
                    state.max_alert_fired = True
                elif temp >= ADTC_WARNING_TEMP_C:
                    print(f"   ⚠ Above target ceiling of {ADTC_WARNING_TEMP_C}°C")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user — writing partial results...")
    finally:
        stop_event.set()
        inference_thread.join(timeout=10)

    # --- Write raw log ---
    LOG_PATH.write_text("\n".join(log_lines) + "\n")

    # --- Compute summary ---
    valid_temps = [r.temp_c for r in state.readings if r.temp_c is not None]

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_requested_s": args.duration,
        "duration_actual_s": round(time.monotonic() - start_time, 1),
        "interval_s": args.interval,
        "sensors_available": sensors_ok,
        "sensors_unavailable_reason": None if sensors_ok else reason,
        "sample_count": len(state.readings),
        "valid_temp_sample_count": len(valid_temps),
        "avg_temp_c": round(sum(valid_temps) / len(valid_temps), 2) if valid_temps else None,
        "max_temp_c": round(max(valid_temps), 2) if valid_temps else None,
        "min_temp_c": round(min(valid_temps), 2) if valid_temps else None,
        "adtc_disqualification_threshold_c": ADTC_DISQUALIFICATION_TEMP_C,
        "adtc_target_ceiling_c": ADTC_WARNING_TEMP_C,
        "exceeded_disqualification_threshold": (max(valid_temps) >= ADTC_DISQUALIFICATION_TEMP_C) if valid_temps else None,
        "exceeded_target_ceiling": (max(valid_temps) >= ADTC_WARNING_TEMP_C) if valid_temps else None,
        "inference_error_count": state.inference_error_count,
        "recommendations": [],
    }

    if not sensors_ok:
        summary["recommendations"].append(
            "Re-run this script on bare-metal Ubuntu 22.04 (not WSL2) to get real thermal data for ADTC submission."
        )
    if valid_temps and max(valid_temps) >= ADTC_WARNING_TEMP_C:
        summary["recommendations"].append(
            "Peak temperature approached/exceeded the 80°C target. Consider: reducing CPU thread count "
            "(-t flag) to leave headroom, improving case/laptop airflow, or checking the CPU frequency "
            "governor (e.g. 'cpupower frequency-info') isn't stuck in a high-power mode."
        )
    if state.inference_error_count > 0:
        summary["recommendations"].append(
            f"{state.inference_error_count} inference call(s) failed or timed out during the run — investigate before relying on these thermal numbers."
        )

    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2))

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("Thermal Monitoring Summary")
    print("=" * 70)
    if valid_temps:
        print(f"  Avg temp: {summary['avg_temp_c']}°C")
        print(f"  Max temp: {summary['max_temp_c']}°C")
        print(f"  Min temp: {summary['min_temp_c']}°C")
        print(f"  Exceeded 85°C disqualification threshold: {summary['exceeded_disqualification_threshold']}")
        print(f"  Exceeded 80°C target ceiling: {summary['exceeded_target_ceiling']}")
    else:
        print("  No valid temperature samples were collected.")
        print(f"  Reason: {reason}")
    if state.inference_error_count:
        print(f"  ⚠ Inference errors during run: {state.inference_error_count}")
    print(f"\n  Raw log:     {LOG_PATH}")
    print(f"  Summary JSON: {SUMMARY_JSON_PATH}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())