# Local Spherical DOA

Simulation and benchmarking of a local 3D acoustic
direction-of-arrival estimator using an eight-microphone
spherical array.

The system estimates:

- Azimuth
- Elevation
- Angular error

inside a bounded 45-degree local field of view.

## Project status

Current status: simulation baseline complete.

The simulation pipeline is complete and reproducible.
Hardware acquisition, microphone calibration, and PCB
implementation are planned future stages.

This repository does not claim anatomical localization,
tongue localization, or physical localization inside the mouth.

## Method

The estimator uses:

1. An eight-microphone spherical array.
2. GCC-PHAT delay estimation.
3. Sub-sample peak interpolation.
4. Coarse-to-fine angular search.
5. Continuous local refinement.
6. Angular-error evaluation on the unit sphere.

Default configuration:

```text
Microphones: 8
Array radius: 25 mm
Sample rate: 192 kHz
Azimuth field of view: 45 degrees
Elevation field of view: 45 degrees
Azimuth center: 125 degrees
Elevation center: 25 degrees
Estimator: GCC-PHAT
Coarse grid step: 4 degrees
Fine grid step: 0.5 degrees
SNR: 30 dB
```

## Results

### 30 dB SNR

```text
Trials: 1000
Successful trials: 1000
Failures: 0
Mean angular error: 0.386 degrees
Median angular error: 0.371 degrees
P90 error: 0.674 degrees
P95 error: 0.771 degrees
Maximum error: 1.121 degrees
Within 0.5 degrees: 74.0%
Within 1 degree: 99.7%
```

### 20 dB SNR

```text
Trials: 1000
Successful trials: 1000
Failures: 0
Mean angular error: 0.286 degrees
Median angular error: 0.263 degrees
P90 error: 0.499 degrees
P95 error: 0.584 degrees
Maximum error: 1.070 degrees
Within 0.5 degrees: 90.0%
Within 1 degree: 99.9%
```

### 10 dB SNR

```text
Trials: 1000
Successful trials: 1000
Failures: 0
Mean angular error: 21.105 degrees
Median error: 16.752 degrees
P90 error: 39.520 degrees
P95 error: 50.415 degrees
Maximum error: 136.943 degrees
Within 1 degree: 0.3%
```

The 10 dB result is treated as a stress-test failure region.
At this SNR, GCC-PHAT peak selection becomes unreliable in
the current idealized model.

The 20 dB and 30 dB results must be compared using paired
trials before drawing conclusions about SNR monotonicity.

## Important limitations

These results are simulation-only.

The current model does not include:

- Room reverberation.
- Early reflections.
- Multiple simultaneous sources.
- Microphone gain mismatch.
- Microphone phase mismatch.
- ADC timing skew.
- Microphone-position error.
- Clock jitter.
- Temperature-dependent speed of sound.
- Mechanical vibration.
- Real microphone frequency-response variation.
- Hardware measurements.

The estimator determines acoustic direction of arrival.
It does not determine distance and does not prove anatomical
or articulatory localization.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

## Run one simulation

```bash
python scripts/simulate_single_trial.py
```

## Run a benchmark

```bash
python scripts/benchmark.py \
  --trials 1000 \
  --snr-db 30
```

For 20 dB:

```bash
python scripts/benchmark.py \
  --trials 1000 \
  --snr-db 20
```

For 10 dB:

```bash
python scripts/benchmark.py \
  --trials 1000 \
  --snr-db 10
```

The benchmark writes its JSON result to:

```text
results/benchmark_local_spherical_doa.json
```

## Reproducibility

The benchmark uses a fixed seed by default:

```text
20260816
```

For a new run:

```bash
python scripts/benchmark.py \
  --trials 1000 \
  --snr-db 30 \
  --seed 12345
```

## Testing

Run:

```bash
python -m pytest
```

## Hardware roadmap

Planned hardware stages:

1. Three- or eight-channel synchronized ADC prototype.
2. Controlled loudspeaker validation.
3. Channel-delay calibration.
4. Microphone-position calibration.
5. Real-room evaluation.
6. PCB revision.
7. Real speech recordings.

The hardware is not included in the current validation claim.

## License

This project is released under the MIT License.

## Citation

If you use this software, cite it using the information in
`CITATION.cff`.

## Disclaimer

This is an academic simulation and research prototype.
It is not a medical device, diagnostic system, or anatomical
localization instrument.