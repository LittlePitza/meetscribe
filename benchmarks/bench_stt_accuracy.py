"""Spike 2b: the HONEST faster-whisper benchmark, on real Spanish speech.

Spike 2 measured RTF against synthetic babble and Whisper returned `segments: 0` -
it classified the audio as non-speech, so the autoregressive decoder never ran and the
numbers only reflected the encoder. Those figures were a floor, not an estimate.

This run uses SAPI-generated Spanish meeting speech with known ground truth, so both
THROUGHPUT (decoder included) and ACCURACY are measured. Accuracy matters as much as
speed: a model twice as fast that mangles names and numbers is not a saving.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures_3spk"
TARGET_SR = 16000


# ---------------------------------------------------------------- audio io
def read_wav_mono16k(path: Path) -> np.ndarray:
    """Read a PCM wav, downmix to mono, resample to 16 kHz float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path.name}: expected 16-bit PCM, got {sw*8}-bit")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != TARGET_SR:
        # Polyphase rational resampling - the same path the real capture stage will use.
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def build_meeting(gap_s: float = 0.45) -> tuple[np.ndarray, list[dict]]:
    """Concatenate the turns into one continuous meeting with realistic pauses."""
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8-sig"))
    if isinstance(manifest, dict):
        manifest = [manifest]
    gap = np.zeros(int(gap_s * TARGET_SR), dtype=np.float32)
    parts, timeline, t = [], [], 0.0
    for entry in manifest:
        x = read_wav_mono16k(FIXTURES / entry["file"])
        dur = len(x) / TARGET_SR
        timeline.append({**entry, "start": t, "end": t + dur})
        parts.extend([x, gap])
        t += dur + gap_s
    return np.concatenate(parts), timeline


# ------------------------------------------------------------- accuracy
def normalize(s: str) -> str:
    """Fold case, accents and punctuation so scoring measures words, not typography."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def wer(ref: str, hyp: str) -> float:
    """Word error rate via Levenshtein over word tokens."""
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


# ------------------------------------------------------------- benchmark
def bench(model_name: str, audio: np.ndarray, ref_text: str, threads: int) -> dict | None:
    from faster_whisper import WhisperModel

    dur = len(audio) / TARGET_SR
    print(f"\n--- {model_name} (int8, {threads} threads) ---")
    try:
        t0 = time.perf_counter()
        model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads)
        load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        segments, info = model.transcribe(
            audio,
            language="es",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )
        segs = list(segments)
        proc_s = time.perf_counter() - t0

        hyp = " ".join(s.text.strip() for s in segs)
        err = wer(ref_text, hyp)
        rtf = proc_s / dur
        print(f"  load {load_s:5.1f}s | process {proc_s:5.2f}s for {dur:.1f}s audio")
        print(f"  RTF {rtf:.3f} ({1/rtf:.1f}x real time) | segments {len(segs)} | WER {err*100:5.1f}%")
        print(f"  first segment: {segs[0].text.strip()[:90] if segs else '(none)'}")
        return {"model": model_name, "load_s": load_s, "proc_s": proc_s, "rtf": rtf,
                "segments": len(segs), "wer": err, "hyp": hyp}
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED -> {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    threads = min(8, os.cpu_count() or 4)
    audio, timeline = build_meeting()
    ref = " ".join(t["text"] for t in timeline)
    dur = len(audio) / TARGET_SR

    print("=" * 78)
    print("Spike 2b - faster-whisper on REAL Spanish speech (decoder included)")
    print("=" * 78)
    print(f"meeting: {dur:.1f}s | {len(timeline)} turns | {len(ref.split())} reference words")
    print(f"speakers in fixture: {sorted({t['speaker'] for t in timeline})}")

    results = [r for m in ("small", "medium", "large-v3-turbo") if (r := bench(m, audio, ref, threads))]

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"{'model':<8}{'RTF':>8}{'speed':>9}{'WER':>9}{'segs':>7}   veredicto")
    for r in results:
        # Two concurrent tracks + VAD + diarization must all fit in the CPU budget.
        fit = "descartado" if r["wer"] > 0.35 else ("holgado" if r["rtf"] < 0.15 else
              ("viable" if r["rtf"] < 0.35 else "ajustado"))
        print(f"{r['model']:<8}{r['rtf']:>8.3f}{1/r['rtf']:>8.1f}x{r['wer']*100:>8.1f}%{r['segments']:>7}   {fit}")

    if results:
        best = min((r for r in results if r["wer"] <= 0.35), key=lambda r: r["wer"], default=None)
        if best:
            print(f"\nmejor precision aceptable: {best['model']} (WER {best['wer']*100:.1f}%, RTF {best['rtf']:.3f})")
            print(f"\ntranscripcion [{best['model']}]:\n  {best['hyp'][:400]}")
