#!/usr/bin/env python3
"""Generate a synthetic RF dataset for baseline drone/non-drone model training.

IMPORTANT:
- Synthetic data is only for bootstrapping pipelines and tests.
- Real-world deployment requires field-collected, labeled captures.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SynthConfig:
    sample_rate_hz: float = 10e6
    sample_count: int = 262144


def _complex_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.normal(0.0, sigma, n) + 1j * rng.normal(0.0, sigma, n)).astype(np.complex64)


def _tone(n: int, fs: float, f0: float, amp: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    phase = rng.uniform(0, 2 * np.pi)
    return (amp * np.exp(1j * (2 * np.pi * f0 * t + phase))).astype(np.complex64)


def _linear_chirp(n: int, fs: float, f_start: float, f_end: float, amp: float) -> np.ndarray:
    t = np.arange(n) / fs
    k = (f_end - f_start) / max(t[-1], 1e-12)
    phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _bursty_envelope(n: int, rng: np.random.Generator, min_len: int = 2000, max_len: int = 10000) -> np.ndarray:
    env = np.zeros(n, dtype=np.float32)
    idx = 0
    while idx < n:
        off_len = int(rng.integers(500, 4000))
        idx += off_len
        if idx >= n:
            break
        on_len = int(rng.integers(min_len, max_len))
        env[idx : min(n, idx + on_len)] = 1.0
        idx += on_len
    # soften edges
    kernel = np.hanning(65)
    kernel /= np.sum(kernel)
    smoothed = np.convolve(env, kernel, mode="same")
    return smoothed.astype(np.float32)


def make_non_drone_iq(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    n = cfg.sample_count
    fs = cfg.sample_rate_hz

    iq = _complex_noise(n, sigma=rng.uniform(0.04, 0.12), rng=rng)

    # Stationary interferers typical of crowded ISM-like bands
    for _ in range(int(rng.integers(1, 4))):
        f0 = rng.uniform(-0.4 * fs, 0.4 * fs)
        amp = rng.uniform(0.02, 0.12)
        iq += _tone(n, fs, f0, amp, rng)

    # Occasional broadband-like block
    if rng.uniform() < 0.4:
        bw_noise = _complex_noise(n, sigma=rng.uniform(0.01, 0.05), rng=rng)
        env = _bursty_envelope(n, rng, min_len=5000, max_len=20000)
        iq += (env * bw_noise).astype(np.complex64)

    return iq.astype(np.complex64)


def make_drone_like_iq(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    n = cfg.sample_count
    fs = cfg.sample_rate_hz

    iq = _complex_noise(n, sigma=rng.uniform(0.03, 0.1), rng=rng)

    # Bursty control-like channels
    for _ in range(int(rng.integers(1, 3))):
        f0 = rng.uniform(-0.35 * fs, 0.35 * fs)
        carrier = _tone(n, fs, f0, amp=rng.uniform(0.08, 0.2), rng=rng)
        env = _bursty_envelope(n, rng)
        iq += (env * carrier).astype(np.complex64)

    # Frequency-agile/chirp component (rough proxy for hopping/telemetry dynamics)
    if rng.uniform() < 0.8:
        f_start = rng.uniform(-0.3 * fs, 0.0)
        f_end = rng.uniform(0.0, 0.3 * fs)
        chirp_sig = _linear_chirp(n, fs, f_start, f_end, amp=rng.uniform(0.04, 0.12))
        env = _bursty_envelope(n, rng, min_len=6000, max_len=25000)
        iq += (env * chirp_sig).astype(np.complex64)

    return iq.astype(np.complex64)


def generate_dataset(
    out_dir: str,
    manifest_path: str,
    per_class: int,
    cfg: SynthConfig,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    root = Path(out_dir)
    drone_dir = root / "drone"
    non_dir = root / "non_drone"
    drone_dir.mkdir(parents=True, exist_ok=True)
    non_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, str]] = []

    for i in range(per_class):
        d = make_drone_like_iq(cfg, rng)
        d_path = drone_dir / f"drone_{i:05d}.npy"
        np.save(d_path, d)
        rows.append((str(d_path), "drone"))

        n = make_non_drone_iq(cfg, rng)
        n_path = non_dir / f"non_drone_{i:05d}.npy"
        np.save(n_path, n)
        rows.append((str(n_path), "non_drone"))

    mpath = Path(manifest_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with open(mpath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for row in rows:
            writer.writerow(row)

    print(f"Generated {2 * per_class} files into: {root}")
    print(f"Manifest: {mpath}")
    print("NOTE: Synthetic dataset is for pipeline bootstrap only, not mission-grade deployment.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic RF dataset generator")
    p.add_argument("--out-dir", default="data/synth", help="Output directory for generated IQ files")
    p.add_argument("--manifest", default="data/synth_manifest.csv", help="Output manifest CSV")
    p.add_argument("--per-class", type=int, default=200, help="Number of samples per class")
    p.add_argument("--sample-rate", type=float, default=10e6, help="Sample rate in Hz")
    p.add_argument("--sample-count", type=int, default=262144, help="IQ samples per file")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SynthConfig(sample_rate_hz=args.sample_rate, sample_count=args.sample_count)
    generate_dataset(args.out_dir, args.manifest, args.per_class, cfg, args.seed)


if __name__ == "__main__":
    main()
