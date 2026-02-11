# RF Drone Signal Analysis and Defensive Detection Toolkit

This repository provides a **Python 3 baseline implementation** for lawful, defensive RF monitoring workflows aimed at identifying drone-like radio activity in crowded spectrum environments.

## What was analyzed

Based on the references you provided:

1. **RFUAV repository (`kitoweeknd/RFUAV`)**
   - Practical signal-processing pipeline ideas for UAV RF capture and classification.
   - Typical use of waterfall/spectrogram-like representations and supervised learning.

2. **RTL-SDR article on using HackRF as a wideband spectrum analyzer**
   - Confirms practical field setup constraints: wide instantaneous bandwidth, short dwell, and noisy ISM bands.
   - Reinforces use of time-frequency analysis rather than raw power-only thresholds.

3. **MDPI Sensors paper (2024, 24(1):125)**
   - Supports feature-based ML for drone RF detection/classification.
   - Highlights value of robust preprocessing, class balancing, and validation under realistic interference.

4. **RF Toolbox PDF for Drone Detection and Classification**
   - Motivates complete pipeline design: acquisition → preprocessing → feature extraction → model inference → alerting.

## Key answer to your question: do we need trained data?

Yes — for supervised ML (`train` command), you need labeled data.

- This repo now includes a **synthetic dataset generator** (`synthetic_iq_dataset.py`) so you can bootstrap training immediately.
- Synthetic data is useful for:
  - validating the pipeline,
  - CI checks,
  - initial threshold/model tuning.
- Synthetic data is **not enough** for production defense use. For real deployment, you still need field captures collected with your SDR setup in your operational RF environment.

### About public drone RF datasets

Public, high-quality, reusable drone RF datasets are limited and often constrained by:
- licensing,
- narrow device coverage,
- missing metadata,
- non-representative environments.

So in practice, teams usually combine:
1. limited public/research data,
2. synthetic augmentation,
3. their own labeled field collections.

## Design choices implemented here

This implementation focuses on **defensive spectrum monitoring**:

- Ingests IQ data from files (`.npy`, `.wav`) and optional live SDR stream (SoapySDR-compatible devices such as HackRF).
- Extracts spectrogram-derived statistical features that are robust under noise and variable SNR.
- Trains a supervised model (`RandomForestClassifier`) with standardized features.
- Performs:
  - binary detection (`drone` / `non_drone`),
  - confidence scoring,
  - optional event logging for SOC/SIEM integration.

## Safety, legal, and operational notes

- This code is for **legal defensive monitoring** only.
- You must comply with local RF laws, privacy regulations, and organizational policy.
- This is a deployable baseline, not a certified military system. For real deployment, perform:
  - site-specific calibration,
  - adversarial/red-team testing,
  - EMC/EMI assessment,
  - MLOps drift monitoring.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Option A: bootstrap with synthetic training data

Generate labeled synthetic IQ samples and a manifest:

```bash
python3 synthetic_iq_dataset.py \
  --out-dir data/synth \
  --manifest data/synth_manifest.csv \
  --per-class 300
```

Train on that manifest:

```bash
python3 rfdrone_detector.py train \
  --manifest data/synth_manifest.csv \
  --model-out models/rfdrone_rf.pkl
```

## Option B: train with your real captures

Create a CSV with:

```text
path,label
/path/to/sample1.npy,drone
/path/to/sample2.npy,non_drone
```

Then:

```bash
python3 rfdrone_detector.py train \
  --manifest data_manifest.csv \
  --model-out models/rfdrone_rf.pkl
```

## Detect from file

```bash
python3 rfdrone_detector.py detect \
  --model models/rfdrone_rf.pkl \
  --input capture.npy
```

## (Optional) Live detection from SDR

```bash
python3 rfdrone_detector.py live \
  --model models/rfdrone_rf.pkl \
  --sample-rate 10000000 \
  --center-freq 2450000000 \
  --chunk-size 262144
```

`live` mode requires SoapySDR Python bindings and a compatible SDR runtime.

## Repository structure

- `rfdrone_detector.py` – end-to-end CLI for train/detect/live workflows.
- `synthetic_iq_dataset.py` – synthetic labeled IQ dataset generator.
- `requirements.txt` – Python dependencies.
