"""Reproducible performance and latency benchmark for AASIST-L on Audio Authenticity Lab.

Measures:
1. Model forward pass on Apple M2 CPU and MPS (if available)
2. Preprocessing / decoding latency
3. Postprocessing / dossier generation latency
4. End-to-end ForensicsService latency
5. Concurrency throughput under 1 and 4 concurrent workers on Apple M2
"""

import concurrent.futures
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from prj55.models.aasist import load_aasist_l  # noqa: E402
from prj55.web.service import ForensicsService  # noqa: E402

CKPT_PATH = REPO_ROOT / "models" / "weights" / "AASIST-L.pth"


def run_model_benchmark(device_str: str, trials: int = 30, warmup: int = 5):
    device = torch.device(device_str)
    model = load_aasist_l(CKPT_PATH, device=device)
    model.eval()

    # Standard 4.0375s audio @ 16 kHz = 64,600 samples
    x = torch.randn(1, 64600, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device_str == "mps":
                torch.mps.synchronize()

    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = model(x)
            if device_str == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    median_ms = statistics.median(latencies_ms)
    mean_ms = statistics.mean(latencies_ms)
    p95_ms = float(np.percentile(latencies_ms, 95))
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    throughput_rps = 1000.0 / median_ms

    return {
        "device": device_str,
        "trials": trials,
        "warmup": warmup,
        "median_ms": round(median_ms, 2),
        "mean_ms": round(mean_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "throughput_rps": round(throughput_rps, 2),
    }


def run_service_end_to_end_benchmark(
    trials: int = 30, warmup: int = 5, device_str: str = "cpu"
):
    # Initialize service
    service = ForensicsService(device_str=device_str, checkpoint_path=CKPT_PATH)
    service.load_model()

    # 4.0-second dummy audio waveform (16 kHz mono)
    sr = 16000
    duration_s = 4.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Warmup
    for _ in range(warmup):
        _ = service.analyze_waveform(
            audio,
            metadata={"filename": "benchmark_warmup.wav", "duration": duration_s, "sr": sr},
        )
        if device_str == "mps":
            torch.mps.synchronize()

    e2e_ms: list[float] = []
    model_ms: list[float] = []
    post_ms: list[float] = []

    for i in range(trials):
        meta = {"filename": f"benchmark_{i}.wav", "duration": duration_s, "sr": sr}
        t0 = time.perf_counter()
        resp = service.analyze_waveform(
            audio,
            metadata=meta,
            threshold=0.0,
        )
        if device_str == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        total_time = (t1 - t0) * 1000.0
        m_time = resp["performance"]["latency_ms"]
        e2e_ms.append(total_time)
        model_ms.append(m_time)
        post_ms.append(total_time - m_time)

    median_e2e = statistics.median(e2e_ms)
    p95_e2e = float(np.percentile(e2e_ms, 95))
    median_model = statistics.median(model_ms)
    median_post = statistics.median(post_ms)

    return {
        "device": device_str,
        "trials": trials,
        "median_e2e_ms": round(median_e2e, 2),
        "p95_e2e_ms": round(p95_e2e, 2),
        "median_model_ms": round(median_model, 2),
        "median_post_ms": round(median_post, 2),
        "single_thread_throughput_rps": round(1000.0 / median_e2e, 2),
    }


def run_concurrent_model_benchmark(concurrency: int = 4, total_requests: int = 20):
    device = torch.device("cpu")
    model = load_aasist_l(CKPT_PATH, device=device)
    model.eval()
    x = torch.randn(1, 64600, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    def worker_task(_: int):
        with torch.no_grad():
            t0 = time.perf_counter()
            _ = model(x)
            return time.perf_counter() - t0

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker_task, i) for i in range(total_requests)]
        latencies = [f.result() * 1000.0 for f in concurrent.futures.as_completed(futures)]
    total_elapsed = time.perf_counter() - t_start

    throughput = total_requests / total_elapsed
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_elapsed_s": round(total_elapsed, 2),
        "throughput_rps": round(throughput, 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
    }


def run_concurrent_throughput_benchmark(concurrency: int = 4, total_requests: int = 20):
    service = ForensicsService(device_str="cpu", checkpoint_path=CKPT_PATH)
    service.load_model()
    sr = 16000
    duration_s = 4.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Warmup
    service.analyze_waveform(
        audio, metadata={"filename": "warmup.wav", "duration": duration_s, "sr": sr}
    )

    def worker_task(idx: int):
        t0 = time.perf_counter()
        _ = service.analyze_waveform(
            audio,
            metadata={"filename": f"conc_{idx}.wav", "duration": duration_s, "sr": sr},
        )
        return time.perf_counter() - t0

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker_task, i) for i in range(total_requests)]
        latencies = [f.result() * 1000.0 for f in concurrent.futures.as_completed(futures)]
    total_elapsed = time.perf_counter() - t_start

    throughput = total_requests / total_elapsed
    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "total_elapsed_s": round(total_elapsed, 2),
        "throughput_rps": round(throughput, 2),
        "median_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
    }


if __name__ == "__main__":
    print("==================================================")
    print("AASIST-L EMPIRICAL BENCHMARK")
    print(f"Platform: {platform.system()} {platform.machine()} ({platform.processor()})")
    print("==================================================")

    # 1. Model Forward Pass on CPU
    cpu_res = run_model_benchmark("cpu", trials=30, warmup=5)
    print("\n1. Model-Only Forward Pass (Apple M2 CPU, batch=1, 64,600 samples = 4.0375s):")
    print(
        f"   Median: {cpu_res['median_ms']} ms | Mean: {cpu_res['mean_ms']} ms | "
        f"p95: {cpu_res['p95_ms']} ms"
    )
    print(f"   Min: {cpu_res['min_ms']} ms | Max: {cpu_res['max_ms']} ms")
    print(f"   Model Forward Throughput: {cpu_res['throughput_rps']} inferences/sec")

    # 2. MPS if available
    mps_available = torch.backends.mps.is_available()
    if mps_available:
        mps_res = run_model_benchmark("mps", trials=30, warmup=5)
        print("\n2. Model-Only Forward Pass (Apple Silicon MPS Metal GPU, batch=1):")
        print(
            f"   Median: {mps_res['median_ms']} ms | Mean: {mps_res['mean_ms']} ms | "
            f"p95: {mps_res['p95_ms']} ms"
        )
        print(f"   Model Forward Throughput (MPS): {mps_res['throughput_rps']} inferences/sec")

    # 3. End-to-End Service Latency (Apple M2 CPU)
    e2e_res = run_service_end_to_end_benchmark(trials=20, warmup=3, device_str="cpu")
    print("\n3. ForensicsService End-to-End Request Pipeline (Apple M2 CPU):")
    print(f"   Median E2E: {e2e_res['median_e2e_ms']} ms | p95 E2E: {e2e_res['p95_e2e_ms']} ms")
    print(f"   - Model Forward: {e2e_res['median_model_ms']} ms")
    print(f"   - Pre/Postprocessing, Timeline, DB: {e2e_res['median_post_ms']} ms")
    print(
        f"   Single-thread E2E Throughput: {e2e_res['single_thread_throughput_rps']} analyses/sec"
    )

    # 4. End-to-End Service Latency (MPS if available)
    if mps_available:
        e2e_mps_res = run_service_end_to_end_benchmark(trials=20, warmup=3, device_str="mps")
        print("\n4. ForensicsService End-to-End Request Pipeline (Apple Silicon MPS Metal GPU):")
        print(
            f"   Median E2E: {e2e_mps_res['median_e2e_ms']} ms | "
            f"p95 E2E: {e2e_mps_res['p95_e2e_ms']} ms"
        )
        print(f"   - Model Forward (MPS): {e2e_mps_res['median_model_ms']} ms")
        print(f"   - Pre/Postprocessing, Timeline, DB: {e2e_mps_res['median_post_ms']} ms")
        print(
            f"   Single-thread E2E Throughput (MPS): "
            f"{e2e_mps_res['single_thread_throughput_rps']} analyses/sec"
        )

    # 5. Concurrent Model Forward Throughput (4 workers, isolated neural inference)
    conc_model_res = run_concurrent_model_benchmark(concurrency=4, total_requests=20)
    print("\n5. Concurrent Model Forward Throughput (4 Workers, Apple M2 CPU):")
    print(f"   Total Elapsed: {conc_model_res['total_elapsed_s']} s for 20 requests")
    print(f"   Effective Forward Throughput: {conc_model_res['throughput_rps']} inferences/sec")
    print(
        f"   Median Latency under load: {conc_model_res['median_latency_ms']} ms | "
        f"p95: {conc_model_res['p95_latency_ms']} ms"
    )

    # 6. Concurrent ForensicsService Pipeline Throughput (4 workers, full pipeline + SQLite)
    conc_res = run_concurrent_throughput_benchmark(concurrency=4, total_requests=20)
    print(
        "\n6. Concurrent ForensicsService Pipeline Throughput "
        "(4 Workers, Apple M2 CPU + SQLite):"
    )
    print(f"   Total Elapsed: {conc_res['total_elapsed_s']} s for 20 requests")
    print(f"   Effective Pipeline Throughput: {conc_res['throughput_rps']} analyses/sec")
    print(
        f"   Median Latency under load: {conc_res['median_latency_ms']} ms | "
        f"p95: {conc_res['p95_latency_ms']} ms"
    )
    print("==================================================")
