#!/usr/bin/env python3
"""PRJ_55 — Phase 3 Channel Degradation Experiment Runner.

Applies configured audio degradation transforms (telephony, noise, Opus) to
an input audio file or a synthetic test signal, runs AASIST-L inference on
each condition, and records structured results.

Usage examples:

    # Run on a synthetic test signal (software-only verification):
    python scripts/run_degradation_experiment.py --synthetic

    # Run on a real audio file with AASIST-L checkpoint:
    python scripts/run_degradation_experiment.py \\
        --audio /path/to/sample.flac \\
        --checkpoint models/weights/AASIST-L.pth \\
        --output-dir reports/phase3_experiment_results

    # Use custom config:
    python scripts/run_degradation_experiment.py \\
        --synthetic \\
        --config configs/degradation_config.yaml \\
        --output-dir /Volumes/ExternalSSD/prj55/phase3_results

IMPORTANT: When run with synthetic signals, the output reports software
           behavior only — NOT deepfake detection research results.
           Do not present synthetic-signal scores as benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure src/ is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prj55.audio import TARGET_SAMPLE_RATE, TARGET_SAMPLES, pad_or_truncate_waveform  # noqa: E402
from prj55.degradation import (  # noqa: E402
    DegradationError,
    DegradationResult,
    DegradationUnavailableError,
    NoiseConfig,
    OpusConfig,
    TelephonyConfig,
    apply_noise,
    apply_opus,
    apply_telephony,
    codec_probe_report,
    opus_available,
)
from prj55.models.aasist import load_aasist_l  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_synthetic_waveform(
    duration_sec: float = 4.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
    freq_hz: float = 440.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a deterministic sine-wave test tone at 16 kHz."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Mix of two harmonics for slightly richer signal
    waveform = 0.4 * np.sin(2 * np.pi * freq_hz * t) + 0.1 * rng.standard_normal(len(t))
    return waveform.astype(np.float32)


def _run_inference(
    model: torch.nn.Module,
    waveform: np.ndarray,
    device: str = "cpu",
) -> tuple[float, float, float]:
    """Run AASIST-L on a waveform and return (spoof_logit, bonafide_logit, latency_ms)."""
    tensor = pad_or_truncate_waveform(waveform, target_length=TARGET_SAMPLES).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        _hidden, logits = model(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    spoof_logit = float(logits[0, 0].item())
    bonafide_logit = float(logits[0, 1].item())
    return spoof_logit, bonafide_logit, latency_ms


def _apply_transform(
    name: str,
    waveform: np.ndarray,
    transform_fn,
    transform_params: dict,
    model: torch.nn.Module | None,
    device: str,
    original_bonafide: float | None,
) -> DegradationResult:
    """Apply a single transform and optionally run inference. Returns DegradationResult."""
    result = DegradationResult(
        transform_name=name,
        transform_params=transform_params,
    )
    try:
        degraded = transform_fn(waveform)
    except DegradationUnavailableError as e:
        result.error = f"UNAVAILABLE: {e}"
        logger.warning("[SKIP] %s: %s", name, e)
        return result
    except DegradationError as e:
        result.error = f"DEGRADATION_ERROR: {e}"
        logger.error("[FAIL] %s: %s", name, e)
        return result

    result.degraded_waveform = degraded

    if model is not None:
        try:
            sp_logit, bf_logit, lat_ms = _run_inference(model, degraded, device)
            result.degraded_score_bonafide = bf_logit
            result.original_score_bonafide = original_bonafide
            if original_bonafide is not None:
                result.score_delta = bf_logit - original_bonafide
            result.inference_latency_ms = lat_ms
            result.metadata["spoof_logit"] = sp_logit
        except Exception as exc:
            result.error = f"INFERENCE_ERROR: {exc}"
            logger.error("[FAIL] Inference for %s: %s", name, exc)

    logger.info("[OK] %s | bonafide logit: %.4f | Δ: %s | latency: %.1f ms",
                name,
                result.degraded_score_bonafide or float("nan"),
                f"{result.score_delta:+.4f}" if result.score_delta is not None else "N/A",
                result.inference_latency_ms or 0.0)
    return result


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    waveform: np.ndarray,
    source_label: str,
    config: dict,
    model: torch.nn.Module | None,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> dict:
    """Run all configured degradation conditions and return a results dict."""
    seed = config.get("global_seed", 42)
    results = {
        "experiment_name": config.get("experiment_name", "phase3_robustness"),
        "experiment_version": config.get("experiment_version", "1.0.0"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_label": source_label,
        "codec_probe": codec_probe_report(),
        "conditions": [],
        "warnings": [],
    }

    # Baseline inference on clean signal
    original_bonafide: float | None = None
    if model is not None:
        sp_logit, bf_logit, lat_ms = _run_inference(model, waveform, device)
        original_bonafide = bf_logit
        results["baseline"] = {
            "source": source_label,
            "bonafide_logit": bf_logit,
            "spoof_logit": sp_logit,
            "inference_latency_ms": lat_ms,
        }
        logger.info("[BASELINE] bonafide logit: %.4f (spoof: %.4f) in %.1f ms",
                    bf_logit, sp_logit, lat_ms)

    # ── Telephony conditions ────────────────────────────────────────────────
    tel_cfg_block = config.get("telephony", {})
    if tel_cfg_block.get("enabled", True):
        for cond in tel_cfg_block.get("conditions", []):
            params = {
                "intermediate_rate": cond.get("intermediate_rate", 8000),
                "upsample_order": cond.get("upsample_order", 16),
            }
            tel_config = TelephonyConfig(**params)
            result = _apply_transform(
                name=cond["name"],
                waveform=waveform,
                transform_fn=lambda w, c=tel_config: apply_telephony(w, c),
                transform_params={"type": "telephony", **params},
                model=model,
                device=device,
                original_bonafide=original_bonafide,
            )
            results["conditions"].append(_result_to_dict(result))

    # ── Noise conditions ────────────────────────────────────────────────────
    noise_cfg_block = config.get("noise", {})
    if noise_cfg_block.get("enabled", True):
        for cond in noise_cfg_block.get("conditions", []):
            params = {
                "snr_db": float(cond.get("snr_db", 10.0)),
                "seed": int(cond.get("seed", seed)),
            }
            noise_config = NoiseConfig(**params)
            result = _apply_transform(
                name=cond["name"],
                waveform=waveform,
                transform_fn=lambda w, c=noise_config: apply_noise(w, c),
                transform_params={"type": "noise", **params},
                model=model,
                device=device,
                original_bonafide=original_bonafide,
            )
            results["conditions"].append(_result_to_dict(result))

    # ── Opus conditions ─────────────────────────────────────────────────────
    opus_cfg_block = config.get("opus", {})
    if opus_cfg_block.get("enabled", True):
        if not opus_available():
            msg = (
                "Opus conditions are configured but FFmpeg/libopus was not found. "
                "Install with 'brew install ffmpeg'. Skipping Opus conditions."
            )
            results["warnings"].append(msg)
            logger.warning(msg)
        else:
            for cond in opus_cfg_block.get("conditions", []):
                params = {
                    "bitrate_kbps": int(cond.get("bitrate_kbps", 24)),
                    "application": cond.get("application", "voip"),
                }
                opus_config = OpusConfig(**params)
                result = _apply_transform(
                    name=cond["name"],
                    waveform=waveform,
                    transform_fn=lambda w, c=opus_config: apply_opus(w, c),
                    transform_params={"type": "opus", **params},
                    model=model,
                    device=device,
                    original_bonafide=original_bonafide,
                )
                results["conditions"].append(_result_to_dict(result))

    # ── Save results ────────────────────────────────────────────────────────
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_file = output_dir / f"results_{ts}.json"
        out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results saved: %s", out_file)
        results["output_file"] = str(out_file)

    return results


def _result_to_dict(r: DegradationResult) -> dict:
    """Serialise a DegradationResult to a JSON-safe dict."""
    return {
        "condition_name": r.transform_name,
        "transform_params": r.transform_params,
        "success": r.success,
        "error": r.error or None,
        "original_bonafide_logit": r.original_score_bonafide,
        "degraded_bonafide_logit": r.degraded_score_bonafide,
        "score_delta": r.score_delta,
        "inference_latency_ms": r.inference_latency_ms,
        "extra": r.metadata,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None) -> dict:
    """Load YAML config or return empty dict for defaults."""
    if config_path is None:
        default = REPO_ROOT / "configs" / "degradation_config.yaml"
        if default.is_file():
            config_path = default
        else:
            return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PRJ_55 Phase 3: Channel Degradation Robustness Experiment Runner"
    )
    parser.add_argument(
        "--audio", "-a", type=Path, default=None,
        help="Path to input audio file (.flac or .wav at 16 kHz).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate and use a synthetic test tone (software pipeline test only).",
    )
    parser.add_argument(
        "--checkpoint", "-c", type=Path, default=None,
        help="Path to AASIST-L pretrained weights (.pth). Omit to skip inference.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "mps"], default="cpu",
        help="PyTorch compute device (default: cpu).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to degradation YAML config (default: configs/degradation_config.yaml).",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Directory for result JSON files.",
    )

    args = parser.parse_args()

    if not args.synthetic and args.audio is None:
        parser.error("Provide --audio <file> or --synthetic.")

    cfg = _load_config(args.config)

    # Load model
    model_cfg = cfg.get("model", {})
    ckpt_path = args.checkpoint or (
        Path(model_cfg.get("checkpoint", "models/weights/AASIST-L.pth"))
    )
    device = args.device or model_cfg.get("device", "cpu")

    model: torch.nn.Module | None = None
    if ckpt_path.is_file():
        logger.info("Loading AASIST-L checkpoint: %s", ckpt_path)
        model = load_aasist_l(checkpoint_path=ckpt_path, device=device)
        model.eval()
    else:
        logger.warning("Checkpoint not found at %s — inference disabled.", ckpt_path)

    # Prepare waveform
    if args.synthetic:
        logger.info("Using synthetic test tone (440 Hz sine, 4 s, 16 kHz).")
        waveform = _generate_synthetic_waveform(seed=cfg.get("global_seed", 42))
        source_label = "SYNTHETIC_SIGNAL — software verification only, not benchmark data"
    else:
        import soundfile as sf
        logger.info("Loading audio: %s", args.audio)
        data, sr = sf.read(str(args.audio), dtype="float32")
        if sr != TARGET_SAMPLE_RATE:
            logger.error(
                "Audio sample rate %d Hz != required %d Hz. Aborting.",
                sr,
                TARGET_SAMPLE_RATE,
            )
            return 1
        if data.ndim == 2:
            data = np.mean(data, axis=1).astype(np.float32)
        waveform = data
        source_label = str(args.audio)

    output_dir = args.output_dir or (
        REPO_ROOT / "reports" / "phase3_experiment_results"
    )

    results = run_experiment(
        waveform=waveform,
        source_label=source_label,
        config=cfg,
        model=model,
        device=device,
        output_dir=output_dir,
    )

    # Print summary table
    print("\n" + "=" * 72)
    print("  PRJ_55 Phase 3 — Channel Degradation Experiment Summary")
    print("=" * 72)
    print(f"  Source:   {results['source_label'][:60]}")
    print(f"  Time:     {results['timestamp']}")
    print(f"  FFmpeg/Opus: {results['codec_probe']['libopus_available']}")
    if "baseline" in results:
        bl = results["baseline"]
        print(f"  Baseline bonafide logit: {bl['bonafide_logit']:.4f} "
              f"({bl['inference_latency_ms']:.1f} ms)")
    print("-" * 72)
    print(f"  {'Condition':<35} {'BF logit':>9} {'Δ logit':>9} {'Status':<12}")
    print("-" * 72)
    for cond in results["conditions"]:
        bf = (
            f"{cond['degraded_bonafide_logit']:+.4f}"
            if cond["degraded_bonafide_logit"] is not None
            else "   N/A"
        )
        delta = (
            f"{cond['score_delta']:+.4f}"
            if cond["score_delta"] is not None
            else "    N/A"
        )
        status = (
            "OK"
            if cond["success"]
            else f"FAIL:{cond['error'][:20] if cond['error'] else '?'}"
        )
        print(f"  {cond['condition_name']:<35} {bf:>9} {delta:>9} {status}")
    print("=" * 72)

    if results.get("warnings"):
        print("\nWarnings:")
        for w in results["warnings"]:
            print(f"  ⚠  {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
