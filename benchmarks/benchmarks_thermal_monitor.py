#!/usr/bin/env python3
"""
CPU Temperature Monitor for RK AgriDig

Monitors CPU core and package temperature during model inference to ensure
we stay below the 85°C threshold (ADTC disqualification penalty: -10 points).

This script:
- Reads CPU temperature via lm-sensors
- Logs temperature during inference
- Alerts if approaching or exceeding 85°C
- Provides thermal optimization recommendations
- Outputs results to JSON

Requirements:
    pip install psutil

Optional (for better temperature reading):
    sudo apt-get install lm-sensors

Usage:
    python benchmarks/thermal_monitor.py --duration 300 --model models/phi3_mini_4k_instruct.gguf
    python benchmarks/thermal_monitor.py --help

ADTC Critical:
    - CPU temp > 85°C = -10 point disqualification
    - Thermal throttling = -10 point penalty
    This tool ensures we avoid both.
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
import statistics

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed.")
    print("Install with: pip install psutil")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

ADTC_TEMP_THRESHOLD = 85.0  # Disqualification threshold (°C)
ADTC_THROTTLE_PENALTY = 10  # Points lost if throttling detected
WARNING_THRESHOLD = 75.0     # Issue warning at this temperature (°C)
CRITICAL_THRESHOLD = 80.0    # Critical warning at this temperature (°C)

# Logging
LOG_FILE = Path("benchmarks/results/thermal_logs.txt")
LOG_LEVEL = logging.INFO

# ============================================================================
# Logging Setup
# ============================================================================

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Utility Functions
# ============================================================================

def run_command(cmd: str, timeout: int = 10) -> str:
    """
    Run a shell command and return output.
    
    Args:
        cmd: Command to run
        timeout: Timeout in seconds
        
    Returns:
        Command stdout
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        logger.debug(f"Command failed: {e}")
        return ""

def get_cpu_temperature_psutil() -> Optional[Dict[str, float]]:
    """
    Get CPU temperature using psutil.
    
    Returns:
        Dictionary of sensor names to temperatures (°C), or None if unavailable
    """
    try:
        temps = psutil.sensors_temperatures()
        
        if not temps:
            return None
        
        # Prefer coretemp (Intel) or k10temp (AMD)
        preferred_sensors = ['coretemp', 'k10temp', 'zenpower', 'it8792', 'nct6798']
        
        # Each entry is a psutil shwtemp namedtuple: (label, current, high,
        # critical) — 4 fields, not 3. Unpacking as (_, temp, _) raises
        # ValueError on any real system that actually returns sensor data;
        # this went uncaught because dev/test ran under WSL2, where
        # sensors_temperatures() returns an empty dict and this code path
        # never executes.
        for sensor in preferred_sensors:
            if sensor in temps:
                return {f"{sensor}_{i}": entry.current for i, entry in enumerate(temps[sensor])}
        
        # Fallback: use first available sensor
        first_sensor = list(temps.keys())[0]
        return {f"{first_sensor}_{i}": entry.current for i, entry in enumerate(temps[first_sensor])}
    
    except Exception as e:
        logger.debug(f"psutil temperature reading failed: {e}")
        return None

def get_cpu_temperature_lmsensors() -> Optional[Dict[str, float]]:
    """
    Get CPU temperature using lm-sensors (more detailed).
    
    Returns:
        Dictionary of sensor names to temperatures (°C), or None if unavailable
    """
    try:
        output = run_command("sensors", timeout=5)
        
        if not output:
            return None
        
        temps = {}
        for line in output.split('\n'):
            # Parse lines like: "Core 0:       +45.0°C  (high = +80.0°C, crit = +95.0°C)"
            if '°C' in line and ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    sensor_name = parts[0].strip()
                    temp_str = parts[1].split('°C')[0].strip()
                    
                    try:
                        temp = float(temp_str.replace('+', ''))
                        temps[sensor_name] = temp
                    except ValueError:
                        pass
        
        return temps if temps else None
    
    except Exception as e:
        logger.debug(f"lm-sensors reading failed: {e}")
        return None

def get_cpu_temperature() -> Optional[float]:
    """
    Get average CPU temperature across all cores.
    
    Returns:
        Average temperature in °C, or None if unavailable
    """
    # Try lm-sensors first (more detailed)
    temps_dict = get_cpu_temperature_lmsensors()
    
    # Fallback to psutil
    if not temps_dict:
        temps_dict = get_cpu_temperature_psutil()
    
    if not temps_dict:
        return None
    
    # Filter out non-numeric or invalid readings
    temps = [t for t in temps_dict.values() if isinstance(t, (int, float)) and 0 < t < 150]
    
    if not temps:
        return None
    
    return statistics.mean(temps)

def check_thermal_throttling() -> bool:
    """
    Check if CPU thermal throttling is active.
    
    Returns:
        True if throttling detected, False otherwise
    """
    try:
        # Check for throttling via sysfs (Linux)
        throttle_file = Path("/sys/class/thermal/cooling_device0/cur_state")
        if throttle_file.exists():
            with open(throttle_file) as f:
                throttle_state = int(f.read().strip())
                if throttle_state > 0:
                    return True
        
        # Check CPU frequency scaling
        freq_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        max_freq_file = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        
        if freq_file.exists() and max_freq_file.exists():
            with open(freq_file) as f:
                current = int(f.read().strip())
            with open(max_freq_file) as f:
                max_freq = int(f.read().strip())
            
            # If current freq is significantly lower than max, likely throttling
            if current < max_freq * 0.9:
                return True
    
    except Exception as e:
        logger.debug(f"Throttling check failed: {e}")
    
    return False

def get_cpu_count() -> int:
    """Get number of CPU cores."""
    return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1

def get_cpu_usage() -> float:
    """Get average CPU utilization percentage."""
    return psutil.cpu_percent(interval=0.1)

# ============================================================================
# Monitoring
# ============================================================================

def monitor_temperature(
    duration_seconds: int = 300,
    poll_interval: float = 1.0,
    model_path: Optional[str] = None
) -> Dict:
    """
    Monitor CPU temperature over time.
    
    Args:
        duration_seconds: How long to monitor (seconds)
        poll_interval: How often to read temperature (seconds)
        model_path: Optional path to model for logging
        
    Returns:
        Dictionary with temperature statistics
    """
    logger.info("=" * 80)
    logger.info("CPU Temperature Monitoring (ADTC Thermal Safety)")
    logger.info("=" * 80)
    
    if model_path and Path(model_path).exists():
        model_size_gb = Path(model_path).stat().st_size / (1024 ** 3)
        logger.info(f"Model: {model_path} ({model_size_gb:.2f} GB)")
    
    logger.info(f"Duration: {duration_seconds} seconds")
    logger.info(f"Poll interval: {poll_interval} seconds")
    logger.info(f"ADTC Thresholds:")
    logger.info(f"  - Disqualification (>85°C): -10 points")
    logger.info(f"  - Warning ({WARNING_THRESHOLD}°C): Approach caution")
    logger.info(f"  - Critical ({CRITICAL_THRESHOLD}°C): Take action")
    logger.info("")
    
    temperatures = []
    cpu_usages = []
    start_time = time.time()
    throttle_detected = False
    num_polls = 0
    
    logger.info("Starting monitoring...")
    logger.info(f"{'Time':<10} {'Temp (°C)':<12} {'CPU Usage':<12} {'Status':<20}")
    logger.info("-" * 54)
    
    try:
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            
            # Read temperature
            temp = get_cpu_temperature()
            usage = get_cpu_usage()
            throttling = check_thermal_throttling()
            
            if throttling:
                throttle_detected = True
            
            # Store readings
            if temp is not None:
                temperatures.append(temp)
                cpu_usages.append(usage)
                num_polls += 1
                
                # Determine status
                if temp > ADTC_TEMP_THRESHOLD:
                    status = "🔴 CRITICAL - DISQUALIFICATION"
                    log_level = logger.error
                elif temp > CRITICAL_THRESHOLD:
                    status = "🟡 CRITICAL - Action needed"
                    log_level = logger.warning
                elif temp > WARNING_THRESHOLD:
                    status = "🟠 WARNING - Monitor"
                    log_level = logger.warning
                else:
                    status = "✓ OK"
                    log_level = logger.info
                
                log_level(f"{elapsed:6.1f}s   {temp:7.1f}°C      {usage:6.1f}%       {status}")
            
            time.sleep(poll_interval)
    
    except KeyboardInterrupt:
        logger.info("\nMonitoring interrupted by user")
    
    # ========================================================================
    # Calculate Statistics
    # ========================================================================
    
    logger.info("")
    logger.info("-" * 54)
    logger.info("")
    
    if not temperatures:
        logger.error("No temperature readings collected")
        return {}
    
    # Basic statistics
    avg_temp = statistics.mean(temperatures)
    max_temp = max(temperatures)
    min_temp = min(temperatures)
    stdev_temp = statistics.stdev(temperatures) if len(temperatures) > 1 else 0
    
    avg_usage = statistics.mean(cpu_usages) if cpu_usages else 0
    max_usage = max(cpu_usages) if cpu_usages else 0
    
    # Count readings by category
    safe_count = sum(1 for t in temperatures if t < WARNING_THRESHOLD)
    warning_count = sum(1 for t in temperatures if WARNING_THRESHOLD <= t < CRITICAL_THRESHOLD)
    critical_count = sum(1 for t in temperatures if CRITICAL_THRESHOLD <= t < ADTC_TEMP_THRESHOLD)
    disqualify_count = sum(1 for t in temperatures if t >= ADTC_TEMP_THRESHOLD)
    
    logger.info("Temperature Statistics:")
    logger.info(f"  Average: {avg_temp:.1f}°C")
    logger.info(f"  Maximum: {max_temp:.1f}°C")
    logger.info(f"  Minimum: {min_temp:.1f}°C")
    logger.info(f"  Std Dev: {stdev_temp:.1f}°C")
    logger.info("")
    logger.info("CPU Usage Statistics:")
    logger.info(f"  Average: {avg_usage:.1f}%")
    logger.info(f"  Maximum: {max_usage:.1f}%")
    logger.info("")
    
    # Safety assessment
    logger.info("Temperature Distribution:")
    logger.info(f"  Safe (<{WARNING_THRESHOLD}°C):           {safe_count:3d} readings ({100*safe_count/num_polls:.1f}%)")
    logger.info(f"  Warning ({WARNING_THRESHOLD}-{CRITICAL_THRESHOLD}°C):      {warning_count:3d} readings ({100*warning_count/num_polls:.1f}%)")
    logger.info(f"  Critical ({CRITICAL_THRESHOLD}-{ADTC_TEMP_THRESHOLD}°C):     {critical_count:3d} readings ({100*critical_count/num_polls:.1f}%)")
    logger.info(f"  Disqualify (>{ADTC_TEMP_THRESHOLD}°C):       {disqualify_count:3d} readings ({100*disqualify_count/num_polls:.1f}%)")
    logger.info("")
    
    # ADTC Assessment
    logger.info("ADTC Thermal Assessment:")
    if disqualify_count > 0:
        logger.error(f"  🔴 DISQUALIFICATION RISK: {disqualify_count} readings exceeded {ADTC_TEMP_THRESHOLD}°C")
        logger.error("  Penalty: -10 points (disqualification)")
        adtc_verdict = "FAIL"
    elif throttle_detected:
        logger.error("  🔴 THERMAL THROTTLING DETECTED")
        logger.error("  Penalty: -10 points")
        adtc_verdict = "FAIL"
    else:
        logger.info(f"  ✓ PASS: No disqualification-level temperatures")
        adtc_verdict = "PASS"
    
    # Recommendations
    logger.info("")
    logger.info("Optimization Recommendations:")
    
    if max_temp > CRITICAL_THRESHOLD:
        logger.warning("  1. Reduce batch size (-n-batch parameter)")
        logger.warning("  2. Reduce number of threads (-n-threads)")
        logger.warning("  3. Use smaller model (Q3_K or Q2_K quantization)")
        logger.warning("  4. Improve system cooling (check fans, ventilation)")
        logger.warning("  5. Disable CPU frequency turbo/boost")
    elif max_temp > WARNING_THRESHOLD:
        logger.warning("  - Monitor temperature during full inference")
        logger.warning("  - Consider reducing threads if > 75°C sustained")
    else:
        logger.info("  ✓ Temperature is well-managed")
    
    logger.info("")
    logger.info("=" * 80)
    
    # ========================================================================
    # Save Results to JSON
    # ========================================================================
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration_seconds,
        "num_readings": num_polls,
        "temperature": {
            "avg_celsius": round(avg_temp, 2),
            "max_celsius": round(max_temp, 2),
            "min_celsius": round(min_temp, 2),
            "stdev_celsius": round(stdev_temp, 2)
        },
        "cpu_usage": {
            "avg_percent": round(avg_usage, 2),
            "max_percent": round(max_usage, 2)
        },
        "distribution": {
            "safe": {"count": safe_count, "percent": round(100*safe_count/num_polls, 1)},
            "warning": {"count": warning_count, "percent": round(100*warning_count/num_polls, 1)},
            "critical": {"count": critical_count, "percent": round(100*critical_count/num_polls, 1)},
            "disqualify": {"count": disqualify_count, "percent": round(100*disqualify_count/num_polls, 1)}
        },
        "throttling_detected": throttle_detected,
        "adtc_verdict": adtc_verdict,
        "adtc_penalty_points": 0 if adtc_verdict == "PASS" else -10
    }
    
    # Save JSON
    results_file = Path("benchmarks/results/thermal_monitoring.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    return results

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monitor CPU temperature during inference (ADTC thermal safety)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Monitoring duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Poll interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (for logging)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = monitor_temperature(
        duration_seconds=args.duration,
        poll_interval=args.interval,
        model_path=args.model
    )
    
    # Exit with appropriate code
    if results.get("adtc_verdict") == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()