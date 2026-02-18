from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from socketserver import ThreadingMixIn
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import tinytuya
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

EXPORTER_VERSION = "1.1.0"

@dataclass
class DeviceConfig:
    device_id: str
    ip: str
    local_key: str
    name: str = ""
    version: Optional[float] = None
    dps_current: Optional[str] = None
    dps_power: Optional[str] = None
    dps_voltage: Optional[str] = None
    dps_relay: Optional[str] = None
    scale_current: Optional[float] = None
    scale_power: Optional[float] = None
    scale_voltage: Optional[float] = None

@dataclass
class ScrapePayload:
    ok: bool
    duration: float
    dps: Dict[str, Any]
    error: str = ""


@dataclass
class InferredLayout:
    dps_voltage: str
    dps_power: str
    dps_current: str
    scale_voltage: float
    scale_power: float
    scale_current: float
    confidence: float
    reason: str


@dataclass
class InferredRelay:
    dps_relay: str
    confidence: float
    reason: str


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True
    allow_reuse_address = True


class QuietHandler(WSGIRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s")


def parse_listen_address(s: str) -> Tuple[str, int]:
    s = s.strip()
    if s.startswith(":"):
        return "", int(s[1:])
    if ":" in s:
        host, port_s = s.rsplit(":", 1)
        return host, int(port_s)
    return "", int(s)


def load_config_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if path.lower().endswith(".json"):
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Config root must be an object")
        return data

    try:
        import yaml
    except Exception as e:
        raise RuntimeError("YAML config requires PyYAML. Install with: pip install pyyaml") from e

    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object")
    return data


def to_device_configs(cfg: Dict[str, Any], autodiscover_enabled: bool) -> List[DeviceConfig]:
    devices = cfg.get("devices", [])
    if not isinstance(devices, list) or not devices:
        raise ValueError("Config must include a non-empty 'devices' list")

    out: List[DeviceConfig] = []
    for item in devices:
        if not isinstance(item, dict):
            raise ValueError("Each device entry must be an object")

        dps = item.get("dps", {}) if isinstance(item.get("dps", {}), dict) else {}
        scale = item.get("scale", {}) if isinstance(item.get("scale", {}), dict) else {}

        dc = DeviceConfig(
            device_id=str(item["device_id"]),
            ip=str(item["ip"]),
            local_key=str(item["local_key"]),
            name=str(item.get("name", "")).strip(),
            version=float(item["version"]) if item.get("version") is not None else None,
            dps_current=str(dps["current"]) if dps.get("current") is not None else None,
            dps_power=str(dps["power"]) if dps.get("power") is not None else None,
            dps_voltage=str(dps["voltage"]) if dps.get("voltage") is not None else None,
            dps_relay=str(dps["relay"]) if dps.get("relay") is not None else None,
            scale_current=float(scale["current"]) if scale.get("current") is not None else None,
            scale_power=float(scale["power"]) if scale.get("power") is not None else None,
            scale_voltage=float(scale["voltage"]) if scale.get("voltage") is not None else None,
        )

        if not dc.name:
            dc.name = f"plug-{dc.ip}"

        if not autodiscover_enabled:
            missing = []
            if dc.dps_current is None:
                missing.append("dps.current")
            if dc.dps_power is None:
                missing.append("dps.power")
            if dc.dps_voltage is None:
                missing.append("dps.voltage")
            if dc.scale_current is None:
                missing.append("scale.current")
            if dc.scale_power is None:
                missing.append("scale.power")
            if dc.scale_voltage is None:
                missing.append("scale.voltage")
            if missing:
                raise ValueError(f"device {dc.name}: missing required fields with autodiscovery disabled: {', '.join(missing)}")

        out.append(dc)

    return out


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _mean(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    return sum(vals) / float(len(vals))


def _std(vals: List[float], mu: float) -> float:
    if len(vals) < 2:
        return 0.0
    v = sum((x - mu) ** 2 for x in vals) / float(len(vals) - 1)
    return math.sqrt(v)


def _cv(vals: List[float]) -> float:
    mu = _mean(vals)
    if not _finite(mu) or abs(mu) < 1e-9:
        return float("inf")
    sd = _std(vals, mu)
    return abs(sd / mu)


def _in_range(x: float, lo: float, hi: float) -> bool:
    return _finite(x) and lo <= x <= hi


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


def _to_float(x: Any) -> Optional[float]:
    if _is_number(x):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except Exception:
            return None
    return None


def _parse_relay(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, int) and x in (0, 1):
        return int(x)
    if isinstance(x, float) and x in (0.0, 1.0):
        return int(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("1", "true", "on", "yes"):
            return 1
        if v in ("0", "false", "off", "no"):
            return 0
    return None


def infer_layout_level2(
    samples: List[Dict[str, Any]],
    allowed_divisors_voltage: List[float],
    allowed_divisors_power: List[float],
    allowed_divisors_current: List[float],
    tol_rel: float,
    min_power_w: float,
    min_current_a: float,
) -> Optional[InferredLayout]:
    if len(samples) < 2:
        return None

    keys: List[str] = []
    seen = set()
    for s in samples:
        d = s.get("dps", {})
        if isinstance(d, dict):
            for k, v in d.items():
                if k not in seen and _is_number(v):
                    seen.add(k)
                    keys.append(str(k))

    if not keys:
        return None

    series: Dict[str, List[float]] = {}
    for k in keys:
        vals: List[float] = []
        for s in samples:
            d = s.get("dps", {})
            v = d.get(k)
            if _is_number(v):
                vals.append(float(v))
        if len(vals) >= 2:
            series[k] = vals

    if not series:
        return None

    v_candidates: List[Tuple[str, float, float, float]] = []
    for k, vals in series.items():
        cv = _cv(vals)
        if not _finite(cv):
            continue
        mu = _mean(vals)

        best = None
        for div in allowed_divisors_voltage:
            vv = mu / div
            if _in_range(vv, 80.0, 300.0):
                closeness = 1.0 - min(_rel_err(vv, 230.0), 1.0)
                stability = 1.0 - min(cv / 0.03, 1.0)
                score = 0.7 * stability + 0.3 * closeness
                cand = (k, div, score, vv)
                if best is None or cand[2] > best[2]:
                    best = cand
        if best is not None:
            v_candidates.append(best)

    v_candidates.sort(key=lambda x: x[2], reverse=True)
    if not v_candidates:
        return None

    v_key, v_div, v_score, _v_val = v_candidates[0]
    other_keys = [k for k in series.keys() if k != v_key]
    if not other_keys:
        return None

    def has_load_for_key(k: str, divs: List[float], min_val: float) -> bool:
        vals = series.get(k, [])
        if not vals:
            return False
        mx = max(vals)
        for div in divs:
            if _finite(mx / div) and (mx / div) >= min_val:
                return True
        return False

    any_load = False
    for k in other_keys:
        if has_load_for_key(k, allowed_divisors_power, min_power_w) or has_load_for_key(k, allowed_divisors_current, min_current_a):
            any_load = True
            break
    if not any_load:
        return InferredLayout(
            dps_voltage=v_key,
            dps_power="",
            dps_current="",
            scale_voltage=v_div,
            scale_power=0.0,
            scale_current=0.0,
            confidence=min(0.55, 0.35 + 0.4 * v_score),
            reason="no_load",
        )

    best_combo = None
    v_vals = series[v_key]
    v_scaled_series: List[float] = [float(x) / float(v_div) for x in v_vals]

    for p_key in other_keys:
        for c_key in other_keys:
            if p_key == c_key:
                continue

            p_vals = series.get(p_key, [])
            c_vals = series.get(c_key, [])
            n = min(len(p_vals), len(c_vals), len(v_scaled_series))
            if n < 2:
                continue

            for p_div in allowed_divisors_power:
                for c_div in allowed_divisors_current:
                    passes = 0
                    errs: List[float] = []
                    plausible = 0

                    for i in range(n):
                        V = v_scaled_series[i]
                        P = float(p_vals[i]) / float(p_div)
                        I = float(c_vals[i]) / float(c_div)

                        if _in_range(V, 80.0, 300.0) and _in_range(P, 0.0, 10000.0) and _in_range(I, 0.0, 60.0):
                            plausible += 1

                        if P < min_power_w and I < min_current_a:
                            continue

                        pred = V * I
                        e = _rel_err(P, pred)
                        errs.append(e)
                        if e <= tol_rel:
                            passes += 1

                    if not errs:
                        continue

                    pass_rate = passes / float(len(errs))
                    plausible_rate = plausible / float(n)
                    avg_err = sum(errs) / float(len(errs))

                    score = (0.55 * pass_rate) + (0.25 * plausible_rate) + (0.20 * (1.0 - min(avg_err / tol_rel, 1.0)))
                    cand = (score, pass_rate, plausible_rate, avg_err, p_key, p_div, c_key, c_div)
                    if best_combo is None or cand[0] > best_combo[0]:
                        best_combo = cand

    if best_combo is None:
        return InferredLayout(
            dps_voltage=v_key,
            dps_power="",
            dps_current="",
            scale_voltage=v_div,
            scale_power=0.0,
            scale_current=0.0,
            confidence=min(0.65, 0.40 + 0.5 * v_score),
            reason="insufficient_signal",
        )

    score, _pass_rate, plausible_rate, _avg_err, p_key, p_div, c_key, c_div = best_combo
    confidence = (0.35 * v_score) + (0.55 * score) + (0.10 * min(plausible_rate, 1.0))
    confidence = max(0.0, min(confidence, 0.999))

    return InferredLayout(
        dps_voltage=v_key,
        dps_power=p_key,
        dps_current=c_key,
        scale_voltage=float(v_div),
        scale_power=float(p_div),
        scale_current=float(c_div),
        confidence=float(confidence),
        reason="ok",
    )


def infer_relay_key(samples: List[Dict[str, Any]]) -> Optional[InferredRelay]:
    if len(samples) < 2:
        return None

    per_key: Dict[str, List[int]] = {}

    for s in samples:
        d = s.get("dps", {})
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            rv = _parse_relay(v)
            if rv is None:
                continue
            kk = str(k)
            per_key.setdefault(kk, []).append(int(rv))

    candidates: List[Tuple[float, str, str]] = []
    for k, vals in per_key.items():
        if len(vals) < 2:
            continue

        uniq = set(vals)
        transitions = 0
        for i in range(1, len(vals)):
            if vals[i] != vals[i - 1]:
                transitions += 1

        if uniq == {0, 1}:
            conf = 0.95
            reason = "both_values"
        elif uniq in ({0}, {1}):
            conf = 0.60
            reason = "single_value"
        else:
            continue

        score = conf + min(transitions, 3) * 0.02
        candidates.append((score, k, reason))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, k, reason = candidates[0]
    conf = min(0.999, max(0.0, score))
    return InferredRelay(dps_relay=k, confidence=float(conf), reason=reason)


class TuyaDeviceState:
    def __init__(self, cfg: DeviceConfig) -> None:
        self.cfg = cfg
        self.device: Optional[Any] = None
        self.working_version: Optional[float] = cfg.version
        self.needs_updatedps: bool = True

        self.last_scrape_ts: float = 0.0
        self.last_ok_ts: float = 0.0
        self.last_duration: float = 0.0
        self.last_error: str = ""
        self.last_poll_ok: bool = False

        self.scrapes_total: int = 0
        self.errors_total: int = 0

        self.discovery_ready: bool = False
        self.discovery_pending: bool = True
        self.discovery_confidence: float = 0.0
        self.discovery_reason: str = "init"
        self.discovery_attempts: int = 0

        self.relay_ready: bool = False
        self.relay_confidence: float = 0.0
        self.relay_reason: str = "init"
        self.relay_state: int = -1
        self.relay_state_ts: float = 0.0

        self.relay_inferred: int = 0
        self.relay_inferred_ts: float = 0.0

        self.dps_current: Optional[str] = cfg.dps_current
        self.dps_power: Optional[str] = cfg.dps_power
        self.dps_voltage: Optional[str] = cfg.dps_voltage
        self.dps_relay: Optional[str] = cfg.dps_relay

        self.scale_current: Optional[float] = cfg.scale_current
        self.scale_power: Optional[float] = cfg.scale_power
        self.scale_voltage: Optional[float] = cfg.scale_voltage

        self.last_current: float = 0.0
        self.last_power: float = 0.0
        self.last_voltage: float = 0.0
        self.last_values_ts: float = 0.0
        self.telemetry_ok: bool = False

        self.samples: List[Dict[str, Any]] = []

        if self.dps_current and self.dps_power and self.dps_voltage and self.scale_current and self.scale_power and self.scale_voltage:
            self.discovery_ready = True
            self.discovery_pending = False
            self.discovery_confidence = 1.0
            self.discovery_reason = "config"

        if self.dps_relay:
            self.relay_ready = True
            self.relay_confidence = 1.0
            self.relay_reason = "config"


class TuyaCollector:
    def __init__(
        self,
        devices: List[DeviceConfig],
        versions: List[float],
        timeout_seconds: float,
        max_parallel: int,
        stale_seconds: float,
        poll_interval_seconds: float,
        ready_grace_seconds: float,
        inferred_on_power_w: float,
        inferred_off_power_w: float,
        inferred_on_current_a: float,
        inferred_off_current_a: float,
        autodiscover_enabled: bool,
        autodiscover_threshold: float,
        autodiscover_samples: int,
        autodiscover_probe_dps: List[str],
        autodiscover_tol_rel: float,
        autodiscover_min_power_w: float,
        autodiscover_min_current_a: float,
        relay_threshold: float,
    ) -> None:
        self.versions = versions
        self.timeout_seconds = timeout_seconds
        self.max_parallel = max_parallel
        self.stale_seconds = stale_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.ready_grace_seconds = ready_grace_seconds

        self.inferred_on_power_w = float(inferred_on_power_w)
        self.inferred_off_power_w = float(inferred_off_power_w)
        self.inferred_on_current_a = float(inferred_on_current_a)
        self.inferred_off_current_a = float(inferred_off_current_a)

        self.autodiscover_enabled = autodiscover_enabled
        self.autodiscover_threshold = autodiscover_threshold
        self.autodiscover_samples = autodiscover_samples
        self.autodiscover_probe_dps = autodiscover_probe_dps
        self.autodiscover_tol_rel = autodiscover_tol_rel
        self.autodiscover_min_power_w = autodiscover_min_power_w
        self.autodiscover_min_current_a = autodiscover_min_current_a
        self.relay_threshold = float(relay_threshold)

        self.states: Dict[str, TuyaDeviceState] = {d.device_id: TuyaDeviceState(d) for d in devices}
        self.lock = Lock()
        self.poll_lock = Lock()
        self.stop_event = Event()

        self.last_scrape_error: int = 0
        self.last_scrape_duration: float = 0.0
        self.last_poll_cycle_ts: float = 0.0

        self.executor = ThreadPoolExecutor(max_workers=max(1, self.max_parallel))

    def start_polling(self) -> None:
        Thread(target=self._poll_forever, daemon=True).start()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def is_ready(self) -> bool:
        with self.lock:
            if self.last_poll_cycle_ts <= 0:
                return False
            return (time.time() - self.last_poll_cycle_ts) <= self.ready_grace_seconds

    def _ensure_device(self, st: TuyaDeviceState) -> Any:
        if st.device is None:
            d = tinytuya.OutletDevice(st.cfg.device_id, st.cfg.ip, st.cfg.local_key)
            d.set_socketTimeout(self.timeout_seconds)
            st.device = d
            st.needs_updatedps = True
        return st.device

    def _is_valid_response(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        if "Error" in data or "error" in data:
            return False
        return isinstance(data.get("dps"), dict)

    def _wanted_dps_list(self, st: TuyaDeviceState) -> List[str]:
        if st.discovery_ready and st.dps_current and st.dps_power and st.dps_voltage:
            keys = [st.dps_current, st.dps_power, st.dps_voltage]
        else:
            keys = self.autodiscover_probe_dps[:]

        if st.dps_relay:
            keys.append(st.dps_relay)

        seen = set()
        out: List[str] = []
        for k in keys:
            kk = str(k)
            if kk not in seen:
                seen.add(kk)
                out.append(kk)
        return out

    def _try_versions(self, st: TuyaDeviceState, dps_list: List[str]) -> Optional[float]:
        dev = self._ensure_device(st)
        for v in self.versions:
            try:
                dev.set_version(v)
                if dps_list:
                    dev.updatedps(dps_list)
                data = dev.status()
                if self._is_valid_response(data):
                    st.needs_updatedps = False
                    return v
            except Exception:
                continue
        return None

    def _scrape_device(self, device_id: str) -> ScrapePayload:
        t0 = time.time()
        st = self.states[device_id]
        try:
            dev = self._ensure_device(st)
            dps_list = self._wanted_dps_list(st)

            if st.working_version is None:
                found = self._try_versions(st, dps_list)
                if found is None:
                    raise RuntimeError("version_detection_failed")
                st.working_version = found

            dev.set_version(st.working_version)
            if st.needs_updatedps and dps_list:
                dev.updatedps(dps_list)
                st.needs_updatedps = False

            data = dev.status()
            if not self._is_valid_response(data):
                raise RuntimeError("invalid_response")

            dps = data.get("dps", {})
            if not isinstance(dps, dict):
                raise RuntimeError("missing_dps")

            dt = time.time() - t0
            return ScrapePayload(True, dt, dps, "")
        except Exception as e:
            dt = time.time() - t0
            st.device = None
            st.needs_updatedps = True
            if st.working_version is not None and st.cfg.version is None:
                st.working_version = None
            return ScrapePayload(False, dt, {}, str(e))

    def _scaled_consumption(self, st: TuyaDeviceState, dps: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        if not st.discovery_ready:
            return None
        if not (st.dps_current and st.dps_power and st.dps_voltage):
            return None

        sc = float(st.scale_current) if st.scale_current is not None else 1.0
        sp = float(st.scale_power) if st.scale_power is not None else 1.0
        sv = float(st.scale_voltage) if st.scale_voltage is not None else 1.0

        rc = _to_float(dps.get(st.dps_current))
        rp = _to_float(dps.get(st.dps_power))
        rv = _to_float(dps.get(st.dps_voltage))
        if rc is None or rp is None or rv is None:
            return None

        current = (float(rc) / sc) / 10.0
        power = (float(rp) / sp) / 10.0
        voltage = float(rv) / sv

        if not (_in_range(voltage, 60.0, 320.0) and _in_range(current, 0.0, 80.0) and _in_range(power, 0.0, 20000.0)):
            return None

        return float(current), float(power), float(voltage)

    def _update_relay_state_from_dps(self, st: TuyaDeviceState, dps: Dict[str, Any], now: float) -> None:
        if not st.dps_relay:
            return
        v = _parse_relay(dps.get(st.dps_relay))
        if v is None:
            return
        st.relay_state = int(v)
        st.relay_state_ts = now
        st.relay_ready = True

    def _update_relay_inferred(self, st: TuyaDeviceState, now: float, current: float, power: float) -> None:
        cur = float(current)
        pwr = float(power)

        prev = int(st.relay_inferred)
        nxt = prev

        on = (pwr >= self.inferred_on_power_w) or (cur >= self.inferred_on_current_a)
        off = (pwr <= self.inferred_off_power_w) and (cur <= self.inferred_off_current_a)

        if on:
            nxt = 1
        elif off:
            nxt = 0

        st.relay_inferred = int(nxt)
        st.relay_inferred_ts = now

    def _maybe_autodiscover(self, st: TuyaDeviceState, payload: ScrapePayload) -> None:
        if not self.autodiscover_enabled:
            return
        if not payload.ok:
            return

        st.samples.append({"ts": time.time(), "dps": payload.dps})
        if len(st.samples) > max(2, self.autodiscover_samples):
            st.samples = st.samples[-self.autodiscover_samples :]

        if len(st.samples) < 2:
            return

        st.discovery_attempts += 1

        if not st.discovery_ready:
            inferred = infer_layout_level2(
                samples=st.samples,
                allowed_divisors_voltage=[1.0, 10.0, 100.0],
                allowed_divisors_power=[1.0, 10.0, 100.0],
                allowed_divisors_current=[1.0, 10.0, 100.0, 1000.0],
                tol_rel=self.autodiscover_tol_rel,
                min_power_w=self.autodiscover_min_power_w,
                min_current_a=self.autodiscover_min_current_a,
            )

            if inferred is not None:
                st.discovery_confidence = float(inferred.confidence)
                st.discovery_reason = inferred.reason

                if inferred.reason == "ok" and inferred.confidence >= self.autodiscover_threshold:
                    st.dps_voltage = inferred.dps_voltage
                    st.dps_power = inferred.dps_power
                    st.dps_current = inferred.dps_current
                    st.scale_voltage = inferred.scale_voltage
                    st.scale_power = inferred.scale_power
                    st.scale_current = inferred.scale_current

                    st.discovery_ready = True
                    st.discovery_pending = False
                    st.needs_updatedps = True

        if not st.dps_relay:
            rinf = infer_relay_key(st.samples)
            if rinf is not None:
                st.relay_confidence = float(rinf.confidence)
                st.relay_reason = rinf.reason
                if rinf.confidence >= self.relay_threshold:
                    st.dps_relay = rinf.dps_relay
                    st.relay_ready = True
                    st.needs_updatedps = True

    def _parallel_scrape(self) -> Dict[str, ScrapePayload]:
        results: Dict[str, ScrapePayload] = {}
        ids = list(self.states.keys())

        futs = {self.executor.submit(self._scrape_device, device_id): device_id for device_id in ids}
        for fut in as_completed(futs):
            device_id = futs[fut]
            try:
                results[device_id] = fut.result()
            except Exception as e:
                results[device_id] = ScrapePayload(False, 0.0, {}, str(e))

        return results

    def _poll_once(self) -> None:
        if not self.poll_lock.acquire(blocking=False):
            return
        try:
            t0 = time.time()
            results = self._parallel_scrape()
            dt = time.time() - t0

            now = time.time()
            any_error = 0

            with self.lock:
                self.last_scrape_duration = float(dt)
                self.last_poll_cycle_ts = now

                for device_id, st in self.states.items():
                    st.scrapes_total += 1
                    st.last_scrape_ts = now

                    r = results.get(device_id)
                    if r is None:
                        st.errors_total += 1
                        st.last_error = "missing_result"
                        st.last_poll_ok = False
                        st.telemetry_ok = False
                        any_error = 1
                        continue

                    st.last_duration = float(r.duration)

                    if r.ok:
                        st.last_ok_ts = now
                        st.last_error = ""
                        st.last_poll_ok = True

                        self._maybe_autodiscover(st, r)
                        self._update_relay_state_from_dps(st, r.dps, now)

                        scaled = self._scaled_consumption(st, r.dps)
                        if scaled is not None:
                            cur, pwr, vol = scaled
                            st.last_current = float(cur)
                            st.last_power = float(pwr)
                            st.last_voltage = float(vol)
                            st.last_values_ts = now
                            st.telemetry_ok = True
                            self._update_relay_inferred(st, now, float(cur), float(pwr))
                        else:
                            st.telemetry_ok = False
                    else:
                        st.errors_total += 1
                        st.last_error = r.error or "error"
                        st.last_poll_ok = False
                        st.telemetry_ok = False
                        any_error = 1

                self.last_scrape_error = int(any_error)
        finally:
            self.poll_lock.release()

    def _poll_forever(self) -> None:
        while not self.stop_event.is_set():
            start = time.time()
            try:
                self._poll_once()
            except Exception:
                logging.exception("poll cycle failed")
            elapsed = time.time() - start
            sleep_s = self.poll_interval_seconds - elapsed
            if sleep_s < 0.0:
                sleep_s = 0.0
            self.stop_event.wait(timeout=sleep_s)

    def collect(self):
        now = time.time()

        cur = GaugeMetricFamily("tuya_consumption_current", "Tuya current in amps.", labels=["device_id", "device_name"])
        powm = GaugeMetricFamily("tuya_consumption_power", "Tuya power in watts.", labels=["device_id", "device_name"])
        vol = GaugeMetricFamily("tuya_consumption_voltage", "Tuya voltage in volts.", labels=["device_id", "device_name"])

        relay_state = GaugeMetricFamily("tuya_relay_state", "Relay state from device DPS (1 on, 0 off, -1 unknown).", labels=["device_id", "device_name"])
        relay_inferred = GaugeMetricFamily("tuya_relay_inferred", "Relay inferred from consumption (1 on, 0 off).", labels=["device_id", "device_name"])
        relay_effective = GaugeMetricFamily("tuya_relay_effective", "Relay effective (uses DPS if known else inferred).", labels=["device_id", "device_name"])

        telemetry_ok = GaugeMetricFamily("tuya_telemetry_ok", "Telemetry valid on last poll (1 ok, 0 not ok).", labels=["device_id", "device_name"])
        up = GaugeMetricFamily("tuya_up", "Tuya device scrape ok on last poll (1 ok, 0 fail).", labels=["device_id", "device_name"])
        last_ok = GaugeMetricFamily("tuya_last_success_timestamp", "Unix timestamp of the last successful device scrape.", labels=["device_id", "device_name"])
        last_telemetry = GaugeMetricFamily("tuya_last_telemetry_timestamp", "Unix timestamp of the last valid telemetry sample.", labels=["device_id", "device_name"])
        dur = GaugeMetricFamily("tuya_device_scrape_duration_seconds", "Device scrape duration in seconds.", labels=["device_id", "device_name"])
        stale = GaugeMetricFamily("tuya_stale_seconds", "Seconds since last valid telemetry sample (-1 never).", labels=["device_id", "device_name"])

        ready = GaugeMetricFamily("tuya_autodiscovery_ready", "Autodiscovery ready (1) or not (0).", labels=["device_id", "device_name"])
        pending = GaugeMetricFamily("tuya_autodiscovery_pending", "Autodiscovery pending (1) or not (0).", labels=["device_id", "device_name"])
        confidence = GaugeMetricFamily("tuya_autodiscovery_confidence", "Autodiscovery confidence score (0..1).", labels=["device_id", "device_name"])
        attempts = CounterMetricFamily("tuya_autodiscovery_attempts_total", "Autodiscovery attempts per device.", labels=["device_id", "device_name"])

        relay_conf = GaugeMetricFamily("tuya_autodiscovery_relay_confidence", "Relay autodiscovery confidence score (0..1).", labels=["device_id", "device_name"])
        relay_ready = GaugeMetricFamily("tuya_autodiscovery_relay_ready", "Relay autodiscovery ready (1) or not (0).", labels=["device_id", "device_name"])

        err_total = CounterMetricFamily("tuya_errors_total", "Total device scrape errors per device.", labels=["device_id", "device_name"])
        scr_total = CounterMetricFamily("tuya_scrapes_total", "Total device scrapes per device.", labels=["device_id", "device_name"])

        scrape_err = GaugeMetricFamily("tuya_last_scrape_error", "Last poll cycle had any error (1) or not (0).")
        scrape_dur = GaugeMetricFamily("tuya_last_scrape_duration_seconds", "Duration of the last poll cycle over all devices.")
        build = GaugeMetricFamily("tuya_exporter_build_info", "Exporter build information.", labels=["version", "python"])

        with self.lock:
            for device_id, st in self.states.items():
                name = st.cfg.name

                up.add_metric([device_id, name], 1.0 if st.last_poll_ok else 0.0)
                telemetry_ok.add_metric([device_id, name], 1.0 if st.telemetry_ok else 0.0)

                last_ok.add_metric([device_id, name], float(st.last_ok_ts) if st.last_ok_ts > 0 else -1.0)
                last_telemetry.add_metric([device_id, name], float(st.last_values_ts) if st.last_values_ts > 0 else -1.0)
                dur.add_metric([device_id, name], float(st.last_duration))

                if st.last_values_ts > 0:
                    stale_s = now - st.last_values_ts
                    stale.add_metric([device_id, name], float(stale_s) if _finite(float(stale_s)) else -1.0)
                else:
                    stale.add_metric([device_id, name], -1.0)

                ready.add_metric([device_id, name], 1.0 if st.discovery_ready else 0.0)
                pending.add_metric([device_id, name], 1.0 if st.discovery_pending else 0.0)
                confidence.add_metric([device_id, name], float(st.discovery_confidence))
                attempts.add_metric([device_id, name], float(st.discovery_attempts))

                relay_conf.add_metric([device_id, name], float(st.relay_confidence))
                relay_ready.add_metric([device_id, name], 1.0 if st.relay_ready else 0.0)

                err_total.add_metric([device_id, name], float(st.errors_total))
                scr_total.add_metric([device_id, name], float(st.scrapes_total))

                relay_state.add_metric([device_id, name], float(st.relay_state))
                relay_inferred.add_metric([device_id, name], float(int(st.relay_inferred)))

                if st.relay_state in (0, 1):
                    relay_effective.add_metric([device_id, name], float(st.relay_state))
                else:
                    relay_effective.add_metric([device_id, name], float(int(st.relay_inferred)))

                fresh = st.last_values_ts > 0 and (now - st.last_values_ts) <= self.stale_seconds
                if fresh:
                    cur.add_metric([device_id, name], float(st.last_current))
                    powm.add_metric([device_id, name], float(st.last_power))
                    vol.add_metric([device_id, name], float(st.last_voltage))

            scrape_err.add_metric([], float(self.last_scrape_error))
            scrape_dur.add_metric([], float(self.last_scrape_duration))

        build.add_metric([EXPORTER_VERSION, sys.version.split()[0]], 1.0)

        yield cur
        yield powm
        yield vol
        yield relay_state
        yield relay_inferred
        yield relay_effective
        yield telemetry_ok
        yield up
        yield last_ok
        yield last_telemetry
        yield dur
        yield stale
        yield ready
        yield pending
        yield confidence
        yield attempts
        yield relay_conf
        yield relay_ready
        yield err_total
        yield scr_total
        yield scrape_err
        yield scrape_dur
        yield build


def make_app(registry: CollectorRegistry, telemetry_path: str, collector: TuyaCollector):
    def app(environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path == telemetry_path or path == "/":
            output = generate_latest(registry)
            start_response("200 OK", [("Content-Type", CONTENT_TYPE_LATEST)])
            return [output]
        if path in ("/-/healthy", "/healthz"):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"ok"]
        if path in ("/-/ready", "/readyz"):
            if collector.is_ready():
                start_response("200 OK", [("Content-Type", "text/plain")])
                return [b"ready"]
            start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
            return [b"not_ready"]
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]

    return app


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config.file", dest="config_file", default=None)
    p.add_argument("--web.listen-address", dest="web_listen_address", default=None)
    p.add_argument("--web.telemetry-path", dest="web_telemetry_path", default=None)
    p.add_argument("--log.level", dest="log_level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = p.parse_args()
    
    setup_logging(args.log_level)
    
    cfg_path = args.config_file or os.environ.get("TUYA_EXPORTER_CONFIG", "").strip() or None
    
    if cfg_path is None:
        candidates = [
            Path.cwd() / "config.yaml",
            Path(__file__).resolve().parent / "config.yaml",
            Path("/config/config.yaml"),
        ]
        for c in candidates:
            if c.is_file():
                cfg_path = str(c)
                break
    
    if not cfg_path:
        raise SystemExit(
            "missing config file: use --config.file=PATH, set TUYA_EXPORTER_CONFIG=PATH, "
            "or place config.yaml in the current directory, script directory, or /config/config.yaml"
        )
    
    cfg = load_config_file(cfg_path)
    logging.info("config_file=%s", cfg_path)

    web_cfg = cfg.get("web", {}) if isinstance(cfg.get("web", {}), dict) else {}
    scrape_cfg = cfg.get("scrape", {}) if isinstance(cfg.get("scrape", {}), dict) else {}
    tuya_cfg = cfg.get("tuya", {}) if isinstance(cfg.get("tuya", {}), dict) else {}
    autod_cfg = cfg.get("autodiscovery", {}) if isinstance(cfg.get("autodiscovery", {}), dict) else {}

    listen_default = str(web_cfg.get("listen_address", "0.0.0.0:9999"))
    telemetry_default = str(web_cfg.get("telemetry_path", "/metrics"))

    listen = args.web_listen_address or listen_default
    telemetry_path = args.web_telemetry_path or telemetry_default
    host, port = parse_listen_address(listen)

    versions = tuya_cfg.get("versions", [3.4, 3.3, 3.2, 3.1, 3.0])
    versions = [float(v) for v in versions]

    timeout_seconds = float(scrape_cfg.get("timeout_seconds", 3.0))
    max_parallel = int(scrape_cfg.get("max_parallel", 8))
    stale_seconds = float(scrape_cfg.get("stale_seconds", 300.0))
    poll_interval_seconds = float(scrape_cfg.get("poll_interval_seconds", 10.0))
    ready_grace_seconds = float(scrape_cfg.get("ready_grace_seconds", max(30.0, 3.0 * poll_interval_seconds)))

    inferred_on_power_w = float(scrape_cfg.get("inferred_on_power_w", 3.0))
    inferred_off_power_w = float(scrape_cfg.get("inferred_off_power_w", 1.5))
    inferred_on_current_a = float(scrape_cfg.get("inferred_on_current_a", 0.03))
    inferred_off_current_a = float(scrape_cfg.get("inferred_off_current_a", 0.015))

    autodiscover_enabled = bool(autod_cfg.get("enabled", True))
    autodiscover_threshold = float(autod_cfg.get("threshold", 0.85))
    autodiscover_samples = int(autod_cfg.get("samples", 4))
    autodiscover_tol_rel = float(autod_cfg.get("tol_rel", 0.30))
    autodiscover_min_power_w = float(autod_cfg.get("min_power_w", 8.0))
    autodiscover_min_current_a = float(autod_cfg.get("min_current_a", 0.05))
    relay_threshold = float(autod_cfg.get("relay_threshold", 0.60))

    probe_dps = autod_cfg.get("probe_dps", None)
    if isinstance(probe_dps, list) and probe_dps:
        autodiscover_probe_dps = [str(x) for x in probe_dps]
    else:
        autodiscover_probe_dps = [str(x) for x in range(1, 26)]

    devices = to_device_configs(cfg, autodiscover_enabled=autodiscover_enabled)

    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)

    collector = TuyaCollector(
        devices=devices,
        versions=versions,
        timeout_seconds=timeout_seconds,
        max_parallel=max_parallel,
        stale_seconds=stale_seconds,
        poll_interval_seconds=poll_interval_seconds,
        ready_grace_seconds=ready_grace_seconds,
        inferred_on_power_w=inferred_on_power_w,
        inferred_off_power_w=inferred_off_power_w,
        inferred_on_current_a=inferred_on_current_a,
        inferred_off_current_a=inferred_off_current_a,
        autodiscover_enabled=autodiscover_enabled,
        autodiscover_threshold=autodiscover_threshold,
        autodiscover_samples=autodiscover_samples,
        autodiscover_probe_dps=autodiscover_probe_dps,
        autodiscover_tol_rel=autodiscover_tol_rel,
        autodiscover_min_power_w=autodiscover_min_power_w,
        autodiscover_min_current_a=autodiscover_min_current_a,
        relay_threshold=relay_threshold,
    )
    registry.register(collector)

    app = make_app(registry, telemetry_path, collector)

    httpd = make_server(
        host,
        port,
        app,
        server_class=ThreadingWSGIServer,
        handler_class=QuietHandler,
    )

    def _sig(*_):
        Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    collector.start_polling()

    logging.info(
        "listening=%s:%s telemetry_path=%s devices=%s poll_interval=%.1fs stale=%.0fs timeout=%.1fs parallel=%s autodiscovery=%s threshold=%.2f relay_threshold=%.2f",
        host if host else "0.0.0.0",
        port,
        telemetry_path,
        len(devices),
        poll_interval_seconds,
        stale_seconds,
        timeout_seconds,
        max_parallel,
        1 if autodiscover_enabled else 0,
        autodiscover_threshold,
        relay_threshold,
    )

    try:
        httpd.serve_forever()
    finally:
        collector.stop()
        try:
            httpd.server_close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

