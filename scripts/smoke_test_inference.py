#!/usr/bin/env python3
"""Baseline Inference Smoke Test for AASIST-L (Week 1, Task 6).

Executes end-to-end forward pass inference using documented preprocessing,
verifying tensor shapes, numerical stability, execution timing, and distinguishing
raw scores from softmax probabilities.

Usage:
    # Run with synthetic test audio (input-pipeline test):
    python scripts/smoke_test_inference.py --synthetic

    # Run with a real audio file:
    python scripts/smoke_test_inference.py --audio /path/to/sample.flac
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Ensure src/ is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prj55.audio import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
    pad_or_truncate_waveform,
    preprocess_audio,
)
from prj55.models.aasist import load_aasist_l  # noqa: E402


def generate_synthetic_audio(
    duration_sec: float = 4.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
    freq_hz: float = 440.0,
) -> np.ndarray:
    """Generate a clean synthetic test tone (sine wave) for input plumbing verification."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return waveform


def run_smoke_test(
    audio_path: Path | None = None,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
    use_synthetic: bool = False,
) -> dict[str, object]:
    """Execute AASIST-L inference smoke test and return structured audit results."""
    result: dict[str, object] = {
        "status": "INIT",
        "device": device,
        "model_architecture": "AASIST-L",
        "preprocessing_config": {
            "expected_sample_rate_hz": TARGET_SAMPLE_RATE,
            "target_length_samples": TARGET_SAMPLES,
            "channel_mode": "mono_downmix",
            "padding_mode": "repeat_and_truncate",
        },
    }

    # Step 1: Resolve checkpoint
    resolved_ckpt = checkpoint_path
    if resolved_ckpt is None:
        default_ckpt = REPO_ROOT / "models" / "weights" / "AASIST-L.pth"
        if default_ckpt.is_file():
            resolved_ckpt = default_ckpt

    if resolved_ckpt and resolved_ckpt.is_file():
        result["checkpoint_path"] = str(resolved_ckpt)
        result["checkpoint_found"] = True
    else:
        result["checkpoint_path"] = str(resolved_ckpt) if resolved_ckpt else "None"
        result["checkpoint_found"] = False

    # Step 2: Load model
    print(f"[INFO] Initializing AASIST-L baseline model (device: {device})...")
    load_start = time.perf_counter()
    model = load_aasist_l(
        checkpoint_path=resolved_ckpt if (resolved_ckpt and resolved_ckpt.is_file()) else None,
        device=device,
    )
    model.eval()
    load_time_ms = (time.perf_counter() - load_start) * 1000.0
    result["model_load_time_ms"] = round(load_time_ms, 2)
    result["parameter_count"] = sum(p.numel() for p in model.parameters())

    # Step 3: Prepare audio tensor
    if use_synthetic or audio_path is None:
        print("[INFO] Generating synthetic audio fixture (4.0s sine wave at 16 kHz)...")
        synth_data = generate_synthetic_audio(duration_sec=4.0)
        input_tensor = pad_or_truncate_waveform(synth_data, target_length=TARGET_SAMPLES)
        result["input_source"] = "SYNTHETIC_SIGNAL (Input pipeline test only - not real benchmark)"
        result["input_samples_before_framing"] = len(synth_data)
    else:
        print(f"[INFO] Preprocessing audio file: {audio_path}")
        input_tensor = preprocess_audio(audio_path, expected_sr=TARGET_SAMPLE_RATE)
        result["input_source"] = str(audio_path)

    input_tensor = input_tensor.to(device)
    result["input_tensor_shape"] = list(input_tensor.shape)

    # Step 4: Run inference with gradient disabled
    print("[INFO] Executing forward pass (torch.no_grad, eval mode)...")
    infer_start = time.perf_counter()
    with torch.no_grad():
        last_hidden, logits = model(input_tensor)
    infer_time_ms = (time.perf_counter() - infer_start) * 1000.0
    result["inference_latency_ms"] = round(infer_time_ms, 2)

    # Step 5: Validate output shapes and numerical stability
    hidden_shape = list(last_hidden.shape)
    logits_shape = list(logits.shape)
    result["hidden_embedding_shape"] = hidden_shape
    result["logits_shape"] = logits_shape

    is_hidden_finite = bool(torch.all(torch.isfinite(last_hidden)).item())
    is_logits_finite = bool(torch.all(torch.isfinite(logits)).item())
    result["numerical_integrity_finite"] = is_hidden_finite and is_logits_finite

    if not result["numerical_integrity_finite"]:
        result["status"] = "FAIL"
        result["error"] = "Model produced non-finite values (NaN or Inf)"
        return result

    # Step 6: Extract raw scores and normalized probabilities
    # Index 0: spoof logit; Index 1: bonafide logit
    raw_spoof_score = float(logits[0, 0].item())
    raw_bonafide_score = float(logits[0, 1].item())
    probs = F.softmax(logits, dim=1)
    prob_spoof = float(probs[0, 0].item())
    prob_bonafide = float(probs[0, 1].item())

    result["scores"] = {
        "raw_logits": {
            "spoof_index_0": round(raw_spoof_score, 5),
            "bonafide_index_1": round(raw_bonafide_score, 5),
        },
        "calibrated_probabilities_softmax": {
            "prob_spoof": round(prob_spoof, 5),
            "prob_bonafide": round(prob_bonafide, 5),
        },
        "note": "Raw bonafide logit is the primary countermeasure (CM) score in ASVspoof protocol.",
    }

    result["status"] = "PASS"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PRJ_55 Baseline Inference Smoke Test for AASIST-L"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        default=None,
        help="Path to an input audio file (.flac or .wav).",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to AASIST-L pretrained weights (.pth).",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "mps"],
        default="cpu",
        help="Compute device for execution (default: cpu).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicitly use generated synthetic test signal.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=Path,
        default=None,
        help="Path to save JSON execution results.",
    )

    args = parser.parse_args()

    # If MPS requested, verify availability
    device_to_use = args.device
    if device_to_use == "mps" and not torch.backends.mps.is_available():
        print("[WARN] MPS device requested but unavailable; falling back to CPU.", file=sys.stderr)
        device_to_use = "cpu"

    try:
        results = run_smoke_test(
            audio_path=args.audio,
            checkpoint_path=args.checkpoint,
            device=device_to_use,
            use_synthetic=args.synthetic,
        )
    except Exception as e:
        print(f"[ERROR] Inference smoke test failed with exception: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("           BASELINE INFERENCE SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"Status:                      {results['status']}")
    print(f"Model Architecture:          {results['model_architecture']}")
    print(f"Parameter Count:             {results.get('parameter_count', 'N/A'):,}")
    print(
        f"Checkpoint Loaded:           {results['checkpoint_found']} ({results['checkpoint_path']})"
    )
    print(f"Compute Device:              {results['device']}")
    print(f"Inference Latency:           {results.get('inference_latency_ms', 'N/A')} ms")
    print(f"Input Source:                {results['input_source']}")
    print(f"Input Shape:                 {results['input_tensor_shape']}")
    print(f"Logits Shape:                {results['logits_shape']}")
    if "scores" in results and isinstance(results["scores"], dict):
        scores = results["scores"]
        print("\nModel Output Scores:")
        print(f"  Raw Logit (Spoof [0]):     {scores['raw_logits']['spoof_index_0']}")
        print(f"  Raw Logit (Bonafide [1]):  {scores['raw_logits']['bonafide_index_1']}")
        prob_sp = scores["calibrated_probabilities_softmax"]["prob_spoof"]
        prob_bf = scores["calibrated_probabilities_softmax"]["prob_bonafide"]
        print(f"  Softmax Prob (Spoof):      {prob_sp}")
        print(f"  Softmax Prob (Bonafide):   {prob_bf}")
    print("=" * 60)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[INFO] Detailed JSON summary saved to: {args.output_json}")

    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
