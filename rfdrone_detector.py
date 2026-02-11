#!/usr/bin/env python3
"""Defensive RF drone detection toolkit.

Pipeline:
1) Load complex IQ samples from files or live SDR.
2) Build spectrogram.
3) Extract robust spectral-temporal features.
4) Train/predict with supervised model.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DSPConfig:
    sample_rate_hz: float = 10e6
    nperseg: int = 1024
    noverlap: int = 512


@dataclass(frozen=True)
class LiveConfig:
    sample_rate_hz: float
    center_freq_hz: float
    chunk_size: int
    gain_db: float = 20.0


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def load_iq_file(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".wav":
        _, data = wavfile.read(path)
        data = np.asarray(data)
    else:
        raise ValueError("Unsupported format. Use .npy or .wav IQ files.")

    data = np.asarray(data)
    if np.iscomplexobj(data):
        iq = data.astype(np.complex64)
    elif data.ndim == 2 and data.shape[1] >= 2:
        i = data[:, 0].astype(np.float32)
        q = data[:, 1].astype(np.float32)
        iq = i + 1j * q
    else:
        raise ValueError("Could not interpret IQ data. Expected complex array or 2-channel I/Q.")

    if iq.size < 4096:
        raise ValueError("IQ data too short; provide at least 4096 samples.")
    return iq


def compute_spectrogram(iq: np.ndarray, cfg: DSPConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, sxx = signal.spectrogram(
        iq,
        fs=cfg.sample_rate_hz,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        scaling="density",
        mode="magnitude",
        return_onesided=False,
    )
    sxx = np.fft.fftshift(sxx, axes=0)
    f = np.fft.fftshift(f)
    sxx_db = 20 * np.log10(sxx + 1e-12)
    return f, t, sxx_db


def _bandwidth_occupied(power_db: np.ndarray, threshold_db: float = -35.0) -> float:
    mask = power_db > threshold_db
    return float(np.mean(mask))


def extract_features(iq: np.ndarray, cfg: DSPConfig) -> np.ndarray:
    _, _, sxx_db = compute_spectrogram(iq, cfg)

    freq_profile = np.mean(sxx_db, axis=1)
    time_profile = np.mean(sxx_db, axis=0)

    f_mean = float(np.mean(sxx_db))
    f_std = float(np.std(sxx_db))
    f_max = float(np.max(sxx_db))
    f_min = float(np.min(sxx_db))
    f_p95 = float(np.percentile(sxx_db, 95))
    f_p5 = float(np.percentile(sxx_db, 5))

    spectral_flatness = float(np.exp(np.mean(np.log(np.maximum(np.abs(freq_profile), 1e-9)))) / np.mean(np.abs(freq_profile)))
    spectral_kurtosis = float(kurtosis(freq_profile, fisher=True, bias=False))

    time_var = float(np.var(time_profile))
    time_peaks = float(np.mean(time_profile > (np.mean(time_profile) + 2 * np.std(time_profile))))

    occ_low = _bandwidth_occupied(sxx_db, threshold_db=-45.0)
    occ_mid = _bandwidth_occupied(sxx_db, threshold_db=-35.0)
    occ_high = _bandwidth_occupied(sxx_db, threshold_db=-25.0)

    centroid = float(np.sum(np.arange(freq_profile.size) * np.maximum(freq_profile - np.min(freq_profile), 0.0)) /
                     (np.sum(np.maximum(freq_profile - np.min(freq_profile), 0.0)) + 1e-12))

    features = np.array(
        [
            f_mean,
            f_std,
            f_max,
            f_min,
            f_p95,
            f_p5,
            spectral_flatness,
            spectral_kurtosis,
            time_var,
            time_peaks,
            occ_low,
            occ_mid,
            occ_high,
            centroid,
        ],
        dtype=np.float32,
    )
    return features


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def load_manifest(manifest_path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], row["label"]))
    if not rows:
        raise ValueError("Manifest is empty.")
    return rows


def build_dataset(manifest_rows: Sequence[Tuple[str, str]], dsp: DSPConfig) -> Tuple[np.ndarray, np.ndarray]:
    x_rows: List[np.ndarray] = []
    y_rows: List[str] = []

    for path, label in manifest_rows:
        iq = load_iq_file(path)
        feats = extract_features(iq, dsp)
        x_rows.append(feats)
        y_rows.append(label)

    return np.vstack(x_rows), np.array(y_rows)


def train(manifest: str, model_out: str, sample_rate: float) -> None:
    dsp = DSPConfig(sample_rate_hz=sample_rate)
    rows = load_manifest(manifest)
    x, y = build_dataset(rows, dsp)

    clf = build_model()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, x, y, cv=cv, n_jobs=-1)

    report = classification_report(y, y_pred, digits=4)
    matrix = confusion_matrix(y, y_pred)

    print("=== Cross-Validation Report ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(matrix)

    clf.fit(x, y)

    model_blob: Dict[str, object] = {
        "model": clf,
        "sample_rate_hz": sample_rate,
        "labels": sorted(set(y.tolist())),
        "feature_count": int(x.shape[1]),
        "created_unix": int(time.time()),
    }

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_blob, out_path)
    print(f"Saved model to: {out_path}")


def detect_from_file(model_path: str, input_path: str) -> Dict[str, object]:
    blob = joblib.load(model_path)
    clf: Pipeline = blob["model"]
    sample_rate = float(blob["sample_rate_hz"])

    dsp = DSPConfig(sample_rate_hz=sample_rate)
    iq = load_iq_file(input_path)
    feats = extract_features(iq, dsp).reshape(1, -1)

    pred = clf.predict(feats)[0]
    probs = clf.predict_proba(feats)[0]
    classes = clf.classes_

    confidence = float(np.max(probs))
    score_map = {str(c): float(p) for c, p in zip(classes, probs)}

    result = {
        "prediction": str(pred),
        "confidence": confidence,
        "class_probabilities": score_map,
        "input": input_path,
    }
    return result


def _open_soapy_device(sample_rate_hz: float, center_freq_hz: float, gain_db: float):
    if not _has_module("SoapySDR"):
        raise RuntimeError(
            "SoapySDR module not found. Install SoapySDR Python bindings for live mode."
        )

    SoapySDR = importlib.import_module("SoapySDR")
    SOAPY_SDR_RX = SoapySDR.SOAPY_SDR_RX
    SOAPY_SDR_CF32 = SoapySDR.SOAPY_SDR_CF32

    dev = SoapySDR.Device(dict(driver="hackrf"))
    dev.setSampleRate(SOAPY_SDR_RX, 0, sample_rate_hz)
    dev.setFrequency(SOAPY_SDR_RX, 0, center_freq_hz)
    dev.setGain(SOAPY_SDR_RX, 0, gain_db)

    stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    dev.activateStream(stream)
    return dev, stream


def run_live_detection(model_path: str, cfg: LiveConfig, alert_log: str | None, loops: int) -> None:
    blob = joblib.load(model_path)
    clf: Pipeline = blob["model"]

    dsp = DSPConfig(sample_rate_hz=cfg.sample_rate_hz)
    dev, stream = _open_soapy_device(cfg.sample_rate_hz, cfg.center_freq_hz, cfg.gain_db)

    print("Starting live detection. Press Ctrl+C to stop.")

    event_fh = None
    if alert_log:
        Path(alert_log).parent.mkdir(parents=True, exist_ok=True)
        event_fh = open(alert_log, "a", encoding="utf-8")

    try:
        for _ in range(loops):
            buff = np.empty(cfg.chunk_size, np.complex64)
            sr = dev.readStream(stream, [buff], cfg.chunk_size)
            if sr.ret <= 0:
                continue

            feats = extract_features(buff[: sr.ret], dsp).reshape(1, -1)
            pred = str(clf.predict(feats)[0])
            probs = clf.predict_proba(feats)[0]
            confidence = float(np.max(probs))

            ts = int(time.time())
            event = {
                "timestamp": ts,
                "prediction": pred,
                "confidence": confidence,
                "center_freq_hz": cfg.center_freq_hz,
                "sample_rate_hz": cfg.sample_rate_hz,
            }

            print(json.dumps(event))
            if event_fh and pred.lower() == "drone":
                event_fh.write(json.dumps(event) + "\n")
                event_fh.flush()
    finally:
        if event_fh:
            event_fh.close()
        dev.deactivateStream(stream)
        dev.closeStream(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Defensive RF drone detection toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train model from labeled manifest CSV")
    p_train.add_argument("--manifest", required=True, help="CSV with columns: path,label")
    p_train.add_argument("--model-out", required=True, help="Output .pkl path")
    p_train.add_argument("--sample-rate", type=float, default=10e6, help="Sample rate in Hz")

    p_detect = sub.add_parser("detect", help="Detect class for an IQ file")
    p_detect.add_argument("--model", required=True, help="Model .pkl path")
    p_detect.add_argument("--input", required=True, help="Input IQ file (.npy or .wav)")

    p_live = sub.add_parser("live", help="Run live inference from SDR stream")
    p_live.add_argument("--model", required=True, help="Model .pkl path")
    p_live.add_argument("--sample-rate", type=float, default=10e6)
    p_live.add_argument("--center-freq", type=float, default=2.45e9)
    p_live.add_argument("--chunk-size", type=int, default=262144)
    p_live.add_argument("--gain", type=float, default=20.0)
    p_live.add_argument("--alert-log", default=None, help="Optional JSONL file for drone alerts")
    p_live.add_argument("--loops", type=int, default=1000000, help="Number of read loops before exit")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        train(args.manifest, args.model_out, args.sample_rate)
    elif args.command == "detect":
        result = detect_from_file(args.model, args.input)
        print(json.dumps(result, indent=2))
    elif args.command == "live":
        cfg = LiveConfig(
            sample_rate_hz=args.sample_rate,
            center_freq_hz=args.center_freq,
            chunk_size=args.chunk_size,
            gain_db=args.gain,
        )
        run_live_detection(args.model, cfg, args.alert_log, args.loops)


if __name__ == "__main__":
    main()
