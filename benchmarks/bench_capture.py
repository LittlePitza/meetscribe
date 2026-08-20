"""Spike 1: can we actually get WASAPI loopback + mic on this machine?

Proves (or disproves) the core capture assumption before any design work:
  - enumerate the default render endpoint and its loopback companion
  - enumerate the default capture (mic) endpoint
  - pull real frames from BOTH and report level, so a silent-but-"working"
    device can't masquerade as success.
"""
from __future__ import annotations

import sys
import time

import numpy as np


def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -999.0
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(r, 1e-12))


# --------------------------------------------------------------------------
hr("A. PyAudioWPatch — WASAPI host API + loopback discovery")
try:
    import pyaudiowpatch as pyaudio

    p = pyaudio.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        print(f"WASAPI host api index={wasapi['index']} devices={wasapi['deviceCount']}")

        default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        default_in = p.get_device_info_by_index(wasapi["defaultInputDevice"])
        print(f"\ndefault RENDER : [{default_out['index']}] {default_out['name']}")
        print(f"  rate={int(default_out['defaultSampleRate'])} out_ch={default_out['maxOutputChannels']}")
        print(f"default CAPTURE: [{default_in['index']}] {default_in['name']}")
        print(f"  rate={int(default_in['defaultSampleRate'])} in_ch={default_in['maxInputChannels']}")

        print("\nloopback endpoints exposed by the patch:")
        loop_for_default = None
        for lb in p.get_loopback_device_info_generator():
            mark = ""
            if default_out["name"] in lb["name"]:
                loop_for_default = lb
                mark = "  <-- matches default render"
            print(f"  [{lb['index']}] {lb['name']}  ch={lb['maxInputChannels']} "
                  f"rate={int(lb['defaultSampleRate'])}{mark}")

        if loop_for_default is None:
            print("\n!! no loopback endpoint matched the default render device")
        else:
            hr("B. Live capture from the loopback endpoint (3 s)")
            ch = int(loop_for_default["maxInputChannels"])
            sr = int(loop_for_default["defaultSampleRate"])
            frames: list[np.ndarray] = []
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=ch,
                rate=sr,
                frames_per_buffer=1024,
                input=True,
                input_device_index=loop_for_default["index"],
            )
            t0 = time.time()
            while time.time() - t0 < 3.0:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            stream.stop_stream()
            stream.close()
            buf = np.concatenate(frames) if frames else np.array([], dtype=np.float32)
            secs = buf.size / max(ch, 1) / sr
            print(f"captured {buf.size} samples  ch={ch} rate={sr}  ~{secs:.2f}s")
            print(f"level: {rms_dbfs(buf):.1f} dBFS   peak: {float(np.max(np.abs(buf))) if buf.size else 0:.4f}")
            print("(near -999/0.0000 just means nothing was playing — the stream still opened)")

            hr("C. Live capture from the microphone (2 s)")
            mch = min(2, int(default_in["maxInputChannels"])) or 1
            msr = int(default_in["defaultSampleRate"])
            mframes: list[np.ndarray] = []
            mstream = p.open(
                format=pyaudio.paFloat32,
                channels=mch,
                rate=msr,
                frames_per_buffer=1024,
                input=True,
                input_device_index=default_in["index"],
            )
            t0 = time.time()
            while time.time() - t0 < 2.0:
                mframes.append(np.frombuffer(mstream.read(1024, exception_on_overflow=False), dtype=np.float32))
            mstream.stop_stream()
            mstream.close()
            mbuf = np.concatenate(mframes) if mframes else np.array([], dtype=np.float32)
            print(f"captured {mbuf.size} samples  ch={mch} rate={msr}")
            print(f"level: {rms_dbfs(mbuf):.1f} dBFS   peak: {float(np.max(np.abs(mbuf))) if mbuf.size else 0:.4f}")

        print("\nRESULT A/B/C: PyAudioWPatch loopback + mic = WORKS")
    finally:
        p.terminate()
except Exception as e:  # noqa: BLE001 — a spike reports, never crashes the run
    print(f"RESULT A/B/C: FAILED -> {type(e).__name__}: {e}")

# --------------------------------------------------------------------------
hr("D. soundcard — independent second opinion on loopback")
try:
    import soundcard as sc

    spk = sc.default_speaker()
    mic = sc.default_microphone()
    print(f"default speaker  : {spk.name}")
    print(f"default mic      : {mic.name}")
    loop = sc.get_microphone(id=str(spk.name), include_loopback=True)
    print(f"loopback handle  : {loop.name}")
    with loop.recorder(samplerate=48000, channels=2, blocksize=1024) as rec:
        data = rec.record(numframes=48000)  # 1 s
    print(f"captured shape={data.shape}  level={rms_dbfs(data):.1f} dBFS")
    print("RESULT D: soundcard loopback = WORKS")
except Exception as e:  # noqa: BLE001
    print(f"RESULT D: FAILED -> {type(e).__name__}: {e}")

print("\n" + "=" * 74)
print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
print("=" * 74)
