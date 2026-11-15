#!/usr/bin/env python
"""An original 29.5-second track for the demo video, synthesised from scratch.

    python make_music.py music.wav

120 beats a minute in D minor; a bar is two seconds, and the video's cuts sit on bar
lines. A quiet opening on pads (0-6 s); the beat arrives with the headset footage (6 s)
and builds through the live training (10-14 s); everything is in while the view turns
through the room and switches (14-20 s); the beat drops out and climbs back as the
outdoor scene forms (20-24 s), landing when it is finished (24 s); one held chord under
the end card (26 s). A soft swoosh marks the fade into the headset and every switch
between reality and the reconstruction. Needs numpy and scipy.
"""
import sys
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter

SR, LENGTH, BEAT = 44100, 29.5, 0.5
N = int(SR * LENGTH)
T = np.arange(N) / SR
rng = np.random.default_rng(7)

DM, BB, F, C = (50, 53, 57, 62), (46, 50, 53, 58), (41, 53, 57, 60), (48, 52, 55, 60)
BARS = [DM, BB, C, DM, BB, F, C, DM, BB, F, C, DM, F, BB, DM]   # one chord a bar
ROOT = {DM: 38, BB: 34, F: 41, C: 36}
# The fade into the headset, and the switches of the compare stretch (make_demo.py prints them).
SWITCHES = [3.0] + [14.0 + t for t in (0.29, 1.45, 2.17, 3.07, 3.69, 4.55, 5.24)]


def hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def filt(x, kind, cutoff, order=2):
    b, a = butter(order, np.array(cutoff) / (SR / 2), kind)
    return lfilter(b, a, x, axis=0)


def place(track, sound, at, gain=1.0):
    i = int(at * SR)
    if i >= N:
        return
    n = min(len(sound), N - i)
    track[i:i + n] += sound[:n] * gain


def stereo(mono, pan=0.0):
    return np.stack([mono * (1 - max(0, pan)), mono * (1 + min(0, pan))], axis=1)


# Pads: two detuned saws a note, one chord a bar, each bar fading into the next.
pad = np.zeros((N, 2))
for bar, chord in enumerate(BARS):
    start, n = bar * 2.0, int(2.4 * SR)
    t = np.arange(n) / SR
    env = np.clip(t / 0.35, 0, 1) * np.clip((2.4 - t) / 0.4, 0, 1)
    for note in chord:
        for side, detune in ((0, -0.07), (1, 0.07)):
            f = hz(note + detune)
            saw = 2 * ((f * t + rng.random()) % 1) - 1
            i = int(start * SR)
            m = min(n, N - i)
            if m > 0:
                pad[i:i + m, side] += saw[:m] * env[:m] * 0.2
# Darker at the start, opening up when the outdoor scene forms.
soft, bright = filt(pad, "low", 1100), filt(pad, "low", 3400)
mix = np.interp(T, [0, 13, 14.2, 20, 20.3, 24, 27, LENGTH], [0, 0, 0.75, 0.75, 0.15, 0.85, 0.6, 0.3])[:, None]
pad = soft * (1 - mix) + bright * mix
# Quiet under the opening, a step up when the beat arrives, full when the scene forms,
# back down for the comparison.
pad *= np.interp(T, [0, 5.9, 6.0, 13.9, 14.0, 19.9, 20.0, 23.9, 24.0, 25.9, 26.0, LENGTH],
                 [0.3, 0.42, 0.5, 0.62, 1.0, 1.0, 0.45, 0.8, 1.0, 1.0, 0.9, 0.75])[:, None]

# The drums.
def kick():
    t = np.arange(int(0.32 * SR)) / SR
    phase = 2 * np.pi * np.cumsum(46 + 120 * np.exp(-t / 0.03)) / SR
    return np.sin(phase) * np.exp(-t / 0.13) + 0.25 * np.exp(-t / 0.004) * rng.standard_normal(len(t)) * 0.3


def burst(seconds, decay, low, high):
    t = np.arange(int(seconds * SR)) / SR
    return filt(rng.standard_normal(len(t)), "band", [low, high]) * np.exp(-t / decay)


drums, duck = np.zeros(N), np.ones(N)
kicks = list(np.arange(6.0, 20.0, BEAT)) + list(np.arange(22.0, 26.0, BEAT))  # out for two bars at 20 s
for at in kicks:
    place(drums, kick(), at, 0.36 if at < 10 else 0.45 if at < 14 else 0.56 if at < 20 else 0.42 if at < 24 else 0.58)
    i = int(at * SR)
    n = min(int(0.3 * SR), N - i)
    duck[i:i + n] = np.minimum(duck[i:i + n], 1 - 0.55 * np.exp(-np.arange(n) / SR / 0.09))
hats = np.zeros(N)
for at in list(np.arange(10.0, 20.0, BEAT)) + list(np.arange(22.0, 26.0, BEAT)):   # on the off-beats
    place(hats, burst(0.06, 0.018, 7000, 14000), at + BEAT / 2, 0.30 if 14 <= at < 20 or at >= 24 else 0.18)
for at in list(np.arange(14.0, 20.0, BEAT / 2)) + list(np.arange(24.0, 26.0, BEAT / 2)):   # sixteenths when everything is in
    place(hats, burst(0.04, 0.01, 8000, 15000), at + BEAT / 4, 0.07)
claps = np.zeros(N)
for at in list(np.arange(14.5, 20.0, 1.0)) + [24.5, 25.5]:   # beats two and four
    place(claps, burst(0.2, 0.05, 900, 3200), at, 0.5)
roll = np.concatenate([np.arange(22.0, 23.0, 0.25), np.arange(23.0, 23.5, 0.125), np.arange(23.5, 24.0, 0.0625)])
for i, at in enumerate(roll):                    # a snare roll into the finished scene
    place(claps, burst(0.09, 0.03, 1200, 5000), at, 0.12 + 0.3 * i / len(roll))

# Bass: the root of each chord, ducked under the kick.
bass = np.zeros(N)
for bar, chord in enumerate(BARS):
    if 3 <= bar < 10 or 11 <= bar < 13:
        t = np.arange(int(2.0 * SR)) / SR
        note = np.sin(2 * np.pi * hz(ROOT[chord]) * t) + 0.3 * np.sin(4 * np.pi * hz(ROOT[chord]) * t)
        place(bass, note * np.clip(t / 0.02, 0, 1) * np.clip((2.0 - t) / 0.05, 0, 1), bar * 2.0, 0.15)

# The arpeggio: sixteenth-note plucks up and down the chord, with an echo.
arp = np.zeros(N)
order = [0, 1, 2, 3, 2, 1, 3, 2]
for bar, chord in enumerate(BARS):
    level = [0, 0, 0, 0.10, 0.10, 0.22, 0.22, 0.42, 0.42, 0.42, 0.20, 0.30, 0.44, 0.12, 0.0][bar]
    if not level:
        continue
    for step in range(16):
        f = hz(chord[order[step % 8]] + 12)
        t = np.arange(int(0.22 * SR)) / SR
        pluck = (2 * ((f * t) % 1) - 1) * np.exp(-t / 0.06) + 0.5 * np.sin(2 * np.pi * f * t) * np.exp(-t / 0.1)
        place(arp, pluck, bar * 2.0 + step * BEAT / 4, level * (1.0 if step % 4 == 0 else 0.7))
arp = filt(arp, "low", 3800)
echo = np.zeros(N)
delay = int(0.375 * SR)
echo[delay:] = arp[:-delay] * 0.4
echo[2 * delay:] += arp[:-2 * delay] * 0.18
arp = np.stack([arp + echo * 0.3, arp * 0.85 + echo], axis=1)

# Risers into the two arrivals, a deep hit on each arrival, and a swoosh on each switch.
fx = np.zeros(N)
for start, end, level in ((4.0, 6.0, 0.16), (12.0, 14.0, 0.24), (20.0, 24.0, 0.30)):
    t = np.arange(int((end - start) * SR)) / SR
    grow = (t / (end - start)) ** 2.2
    noise = filt(rng.standard_normal(len(t)), "high", 2500) * grow
    sweep = np.sin(2 * np.pi * np.cumsum(180 + 900 * grow) / SR) * grow * 0.4
    place(fx, noise + sweep, start, level)
for at, level in ((6.0, 0.22), (14.0, 0.42), (24.0, 0.46), (26.0, 0.3)):
    t = np.arange(int(1.6 * SR)) / SR
    boom = np.sin(2 * np.pi * np.cumsum(38 + 40 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.45)
    place(fx, boom + filt(rng.standard_normal(len(t)), "low", 1800) * np.exp(-t / 0.25) * 0.35, at, level)
for at in SWITCHES:
    t = np.arange(int(0.7 * SR)) / SR
    swoosh = filt(rng.standard_normal(len(t)), "band", [500, 5000]) * np.exp(-0.5 * ((t - 0.35) / 0.11) ** 2)
    place(fx, swoosh, at - 0.35, 0.16 if at < 10 else 0.10)

# Mix: everything but the drums and the bass gets some room around it.
dry = pad * duck[:, None] + stereo(bass * duck) + arp + stereo(claps) * 0.9 + stereo(fx)
wet = np.zeros((N, 2))
t = np.arange(int(1.8 * SR)) / SR
for side in (0, 1):
    room = rng.standard_normal(len(t)) * np.exp(-t / 0.5)
    wet[:, side] = fftconvolve((pad + arp + stereo(claps) + stereo(fx))[:, side], room)[:N] * 0.012
track = dry + filt(wet, "high", 250) + stereo(drums) + stereo(hats, 0.25)
track = filt(track, "high", 28)
track *= np.clip(T / 0.08, 0, 1)[:, None] * np.clip((LENGTH - T) / 1.6, 0, 1)[:, None] ** 1.5
track = np.tanh(track * 0.8)
track *= 0.89 / np.abs(track).max()

with wave.open(sys.argv[1], "wb") as out:
    out.setnchannels(2)
    out.setsampwidth(2)
    out.setframerate(SR)
    out.writeframes((track * 32767).astype("<i2").tobytes())
rms = np.sqrt((track ** 2).mean())
print(f"{sys.argv[1]}: {LENGTH} s, peak {np.abs(track).max():.2f}, RMS {20 * np.log10(rms):.1f} dBFS; "
      f"per 2 s: {[round(float(20 * np.log10(np.sqrt((track[int(a * SR):int((a + 2) * SR)] ** 2).mean()) + 1e-9))) for a in range(0, 28, 2)]}")
