"""Spike 2: is CPU-only faster-whisper fast enough for real-time Spanish?

The whole streaming design hinges on the real-time factor (RTF = processing
time / audio duration). RTF must stay well under 1.0 with headroom for VAD,
diarization and two concurrent tracks (mic + loopback).

Generates a deterministic synthetic speech-like signal locally (no network,
no sample files), then benchmarks each model size on it. The transcript text
is meaningless for synthetic audio — we are measuring THROUGHPUT, and the
audio is shaped so the encoder does the same work it would on real speech.
"""
from __future__ import annotations

import os
import time

import numpy as np

SR = 16000
DURATION = 30.0  # whisper's native window


def synth_speech_like(seconds: float, sr: int = SR, seed: int = 7) -> np.ndarray:
    """Formant-ish babble: voiced pitch sweeps + noise bursts + pauses.

    Not intelligible, but spectrally in the speech band, so the encoder's
    workload matches a real utterance. Deterministic via the seed.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)

    pos = 0
    while pos < n:
        seg = int(rng.uniform(0.25, 0.9) * sr)          # an utterance chunk
        end = min(pos + seg, n)
        tt = t[pos:end] - t[pos]
        f0 = rng.uniform(95, 190)                        # speaker pitch
        f0_track = f0 * (1.0 + 0.08 * np.sin(2 * np.pi * 2.3 * tt))
        phase = 2 * np.pi * np.cumsum(f0_track) / sr
        voiced = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 12))
        for fc, bw, amp in ((650, 90, 1.0), (1180, 120, 0.6), (2600, 180, 0.35)):
            voiced += amp * np.sin(2 * np.pi * fc * tt) * np.exp(-bw * tt % 1.0)
        env = np.hanning(end - pos)
        sig[pos:end] += voiced * env
        pos = end + int(rng.uniform(0.05, 0.35) * sr)    # inter-word pause

    sig += 0.005 * rng.standard_normal(n)                # room noise floor
    sig /= max(np.max(np.abs(sig)), 1e-9)
    return (0.7 * sig).astype(np.float32)


def bench(model_name: str, audio: np.ndarray, threads: int) -> dict | None:
    from faster_whisper import WhisperModel

    print(f"\n--- {model_name}  (int8, cpu_threads={threads}) ---")
    try:
        t0 = time.perf_counter()
        model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
        load_s = time.perf_counter() - t0
        print(f"  load: {load_s:6.2f}s")

        # warm-up so the first-call graph build doesn't pollute the measurement
        list(model.transcribe(audio[: SR * 3], language="es", beam_size=1)[0])

        t0 = time.perf_counter()
        segments, info = model.transcribe(
            audio, language="es", beam_size=1, vad_filter=False, condition_on_previous_text=False
        )
        segs = list(segments)
        proc_s = time.perf_counter() - t0

        rtf = proc_s / DURATION
        budget = "OK" if rtf < 0.35 else ("TIGHT" if rtf < 0.7 else "TOO SLOW")
        print(f"  process: {proc_s:6.2f}s for {DURATION:.0f}s audio")
        print(f"  RTF: {rtf:.3f}   ->  {1/rtf:5.1f}x faster than real time   [{budget}]")
        print(f"  segments: {len(segs)}")
        return {"model": model_name, "load_s": load_s, "proc_s": proc_s, "rtf": rtf}
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED -> {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    threads = min(8, os.cpu_count() or 4)
    print("=" * 74)
    print(f"Spike 2 - faster-whisper CPU throughput   (cpu_threads={threads})")
    print(f"audio: {DURATION:.0f}s synthetic speech-like @ {SR} Hz")
    print("=" * 74)

    audio = synth_speech_like(DURATION)
    print(f"generated {audio.size} samples, peak={np.max(np.abs(audio)):.3f}")

    results = []
    for name in ("tiny", "base", "small"):
        r = bench(name, audio, threads)
        if r:
            results.append(r)

    print("\n" + "=" * 74)
    print("VERDICT  (need RTF < ~0.35 to run mic + loopback concurrently)")
    print("=" * 74)
    print(f"{'model':<10}{'load':>9}{'proc/30s':>11}{'RTF':>9}{'speed':>10}   fit")
    for r in results:
        fit = "yes" if r["rtf"] < 0.35 else ("marginal" if r["rtf"] < 0.7 else "no")
        print(f"{r['model']:<10}{r['load_s']:>8.2f}s{r['proc_s']:>10.2f}s"
              f"{r['rtf']:>9.3f}{1/r['rtf']:>9.1f}x   {fit}")
