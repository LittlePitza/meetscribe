"""Spike 3: does CPU diarization work with sherpa-onnx, and how fast?

Two questions, both load-bearing for the design:
  1. VAD  - does the bundled Silero VAD segment the meeting into utterances correctly,
     and cheaply enough to run on two live tracks?
  2. SPEAKER EMBEDDINGS - do TitaNet embeddings separate talkers well enough for
     online clustering to assign stable "Participante N" labels?

Honesty note on the fixture: all turns are the same SAPI voice with different prosody
(pitch/rate). That is a DELIBERATELY HARD case - same vocal tract, so the embedder has
only prosody to go on. Real meetings have genuinely different speakers and are easier.
Treat separation here as a lower bound, and a failure as inconclusive rather than fatal.
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures_2spk"
MODELS = HERE / "models"
SR = 16000


def read_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8-sig"))
if isinstance(manifest, dict):
    manifest = [manifest]

# ---------------------------------------------------------------- 1. VAD
hr("1. Silero VAD via sherpa-onnx  (segmentation into utterances)")
gap = np.zeros(int(0.45 * SR), dtype=np.float32)
parts, truth, t = [], [], 0.0
for e in manifest:
    x = read_wav_mono16k(FIXTURES / e["file"])
    d = len(x) / SR
    truth.append({"spk": e["speaker"], "start": t, "end": t + d})
    parts += [x, gap]
    t += d + 0.45
meeting = np.concatenate(parts)
print(f"meeting: {len(meeting)/SR:.1f}s, {len(truth)} true turns, speakers {sorted({x['spk'] for x in truth})}")

cfg = sherpa_onnx.VadModelConfig()
cfg.silero_vad.model = str(MODELS / "silero_vad.onnx")
cfg.silero_vad.threshold = 0.5
cfg.silero_vad.min_silence_duration = 0.35
cfg.silero_vad.min_speech_duration = 0.25
cfg.sample_rate = SR
vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=100)

t0 = time.perf_counter()
window, detected = 512, []
for i in range(0, len(meeting) - window, window):
    vad.accept_waveform(meeting[i : i + window])
    while not vad.empty():
        seg = vad.front
        detected.append({"start": seg.start / SR, "end": (seg.start + len(seg.samples)) / SR})
        vad.pop()
vad.flush()
while not vad.empty():
    seg = vad.front
    detected.append({"start": seg.start / SR, "end": (seg.start + len(seg.samples)) / SR})
    vad.pop()
vad_s = time.perf_counter() - t0

print(f"VAD found {len(detected)} speech segments in {vad_s:.2f}s "
      f"(RTF {vad_s/(len(meeting)/SR):.4f})")
for d in detected[:6]:
    print(f"   {d['start']:6.2f}s -> {d['end']:6.2f}s  ({d['end']-d['start']:.2f}s)")
if len(detected) > 6:
    print(f"   ... +{len(detected)-6} more")

# ------------------------------------------------- 2. speaker embeddings
hr("2. TitaNet speaker embeddings  (can talkers be separated?)")
ecfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
    model=str(MODELS / "nemo_en_titanet_small.onnx"), num_threads=4
)
ext = sherpa_onnx.SpeakerEmbeddingExtractor(ecfg)
print(f"embedding dimension: {ext.dim}")

t0 = time.perf_counter()
embs, labels, total_audio = [], [], 0.0
for e in manifest:
    x = read_wav_mono16k(FIXTURES / e["file"])
    total_audio += len(x) / SR
    stream = ext.create_stream()
    stream.accept_waveform(sample_rate=SR, waveform=x)
    stream.input_finished()
    embs.append(np.array(ext.compute(stream), dtype=np.float32))
    labels.append(e["speaker"])
emb_s = time.perf_counter() - t0

E = np.stack(embs)
E /= np.linalg.norm(E, axis=1, keepdims=True)
print(f"embedded {len(E)} utterances ({total_audio:.1f}s audio) in {emb_s:.2f}s "
      f"(RTF {emb_s/total_audio:.4f})")

S = E @ E.T
labels = np.array(labels)
same = S[(labels[:, None] == labels[None, :]) & ~np.eye(len(E), dtype=bool)]
diff = S[labels[:, None] != labels[None, :]]
print(f"\ncosine similarity")
print(f"  same speaker : mean {same.mean():.3f}  min {same.min():.3f}  n={same.size}")
print(f"  diff speaker : mean {diff.mean():.3f}  max {diff.max():.3f}  n={diff.size}")
margin = same.mean() - diff.mean()
print(f"  separation margin: {margin:+.3f}")

overlap = (diff.max() >= same.min())
print(f"  distributions overlap: {'YES - a single global threshold cannot split them' if overlap else 'NO - cleanly separable'}")

hr("3. Agglomerative clustering at a candidate operating threshold")
try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D = np.clip(1.0 - S, 0, 2)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    for thr in (0.25, 0.35, 0.45, 0.55):
        pred = fcluster(Z, t=thr, criterion="distance")
        n_found = len(set(pred))
        # purity: fraction of utterances landing in their cluster's majority true speaker
        purity = sum(
            max(np.sum(labels[pred == c] == s) for s in set(labels)) for c in set(pred)
        ) / len(labels)
        print(f"  thr={thr:.2f}  clusters={n_found} (true=3)  purity={purity*100:5.1f}%  {list(pred)}")
except Exception as e:  # noqa: BLE001
    print(f"  clustering skipped: {type(e).__name__}: {e}")

hr("VEREDICTO")
print(f"VAD      : {'OK' if len(detected) >= len(truth) * 0.7 else 'REVISAR'} "
      f"- {len(detected)} segmentos vs {len(truth)} turnos reales, RTF {vad_s/(len(meeting)/SR):.4f}")
print(f"Embedder : RTF {emb_s/total_audio:.4f}, dim {ext.dim}, 38 MB, sin PyTorch")
print(f"Separacion: margen {margin:+.3f} "
      f"({'suficiente' if margin > 0.15 else 'debil - fixture de una sola voz, no concluyente'})")
