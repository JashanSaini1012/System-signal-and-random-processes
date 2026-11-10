"""
Project 14: BioRadio -> EMG processing -> Unity game (UDP).

Reads the BioRadio with BioRadioSDK.dll (same SDK calls as the Vizard script),
cleans the EMG, builds the envelope with the filter you pick, scales it to
% of maximum (MVC) and sends it to the game 100 times a second:

    biceps,triceps,unix_ms      (biceps and triceps as fractions of MVC, 0-1)

In Unity choose the EMG (UDP 5005) control. Pick the same filter on the game's
setup screen as the one running here (shown in the status line).

Stages:
  1. Find the BioRadio and connect (Close BioCapture first.)
  2. Channel check: live level of every channel, so you can see which is biceps/triceps
  3. Calibration: 5 s relax, 3 s max biceps, 3 s max triceps (saved per subject, reused next time)
  4. Streaming to the game. Keys: 0-7 switch filter, R start/stop recording, Q quit

Processing chain (per channel):
  raw (mV) -> 4th-order Butterworth band-pass 20-450 Hz -> 50 Hz notch (Q=30)
  -> full-wave rectify -> envelope filter F0..F7 -> (env - rest) / (MVC - rest)

  F0 rectified raw (no smoothing)     F3 2nd-order Butterworth low-pass 4 Hz
  F1 moving average 100 ms            F4 exponential moving average (time constant 50 ms)
  F2 moving RMS 100 ms                F5 Kalman filter on the 100 ms RMS (team member's design)
  F6 2nd-order Chebyshev type I low-pass, 1 dB ripple    } all three low-pass filters
  F7 2nd-order Chebyshev type II low-pass, 40 dB stopband } have their -3 dB point at 4 Hz
Filters compared in Project 14: F0 (raw baseline), F3, F5, F6, F7. F1/F2 are kept as MAV/RMS features.
All filters run all the time on the same data, so switching filters is instant and
fair. Calibration is computed separately for each filter.

Setup (Windows, once):
  pip install pythonnet numpy scipy
  Put BioRadioSDK.dll in the same folder as this script (or pass --dll).

Usage:
  python bioradio_emg.py --subject S1 --biceps 1 --triceps 2
  python bioradio_emg.py --subject S1 --recalibrate
  python bioradio_emg.py --simulate            # no hardware: fake EMG, to test the game link
"""
import argparse
import csv
import json
import math
import os
import socket
import sys
import time
from datetime import datetime

import numpy as np
from scipy import signal

try:
    import msvcrt  # Windows keyboard, non-blocking
except ImportError:
    msvcrt = None

HERE = os.path.dirname(os.path.abspath(__file__))
TO_MV = 1000.0  # SDK gives volts (as in the Vizard script); confirm against BioCapture
FILTERS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]
CAL_VERSION = "v3"   # bump when a filter changes, so old calibration files are not reused
FILTER_NAMES = {
    "F0": "rectified raw",
    "F1": "moving average 100 ms",
    "F2": "moving RMS 100 ms",
    "F3": "Butterworth LP 4 Hz",
    "F4": "exp. moving average 50 ms",
    "F5": "Kalman on RMS",
    "F6": "Chebyshev I LP 4 Hz",
    "F7": "Chebyshev II LP 4 Hz",
}


# --------------------------------------------------------------------------- keyboard

def key_pressed():
    """Returns the key pressed (lower case) or None. Never blocks."""
    if msvcrt is None or not msvcrt.kbhit():
        return None
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):  # arrow/function keys send two codes
        msvcrt.getwch()
        return None
    return ch.lower()


def wait_enter(prompt, auto):
    if auto:
        print(prompt + " (auto)")
        return
    input(prompt)


# --------------------------------------------------------------------------- devices

def normalize_mac(value):
    return str(value).replace(":", "").replace("-", "").replace(" ", "").upper()


class BioRadio:
    """BioRadio through BioRadioSDK.dll (pythonnet). Same calls as the Vizard script."""

    def __init__(self, dll_path):
        if not os.path.isfile(dll_path):
            raise RuntimeError("BioRadioSDK.dll not found at " + dll_path)
        try:
            from pythonnet import load
            load("netfx")  # Windows .NET Framework, like Vizard
        except Exception:
            pass  # already loaded, or an older pythonnet
        # A DLL downloaded or received over chat is "blocked" by Windows (Zone.Identifier),
        # and .NET refuses to load it ("network location"). Remove the block, like
        # Properties > Unblock does.
        try:
            os.remove(dll_path + ":Zone.Identifier")
        except OSError:
            pass
        import clr  # noqa: F401  (pythonnet)
        clr.AddReference(dll_path)
        from GLNeuroTech.Devices.BioRadio import BioRadioDeviceManager
        self.manager = BioRadioDeviceManager()
        self.device = None
        self.signals = []
        self.fs = 0.0

    def discover(self):
        devices = self.manager.DiscoverBluetoothDevices()
        found = []
        for i in range(devices.Length):
            item = devices[i]
            mac = normalize_mac(item.MacId)
            if mac and not any(d["mac"] == mac for d in found):
                found.append({"name": str(item.DeviceId), "mac": mac})
        return found

    def connect(self, mac):
        candidate = self.manager.GetBluetoothDevice(int(mac, 16))
        try:
            group = candidate.BioPotentialSignals
            count = int(group.Count)
            if count < 1:
                raise RuntimeError("No enabled biopotential channels. Configure them in BioCapture.")
            fs = float(group.SamplesPerSecond)
            if fs <= 0:
                raise RuntimeError("Invalid sampling rate: %s" % fs)
            self.signals = [group[i] for i in range(count)]  # group[i] works; group.Item(i) does not
            candidate.StartAcquisition()
        except Exception:
            try:
                candidate.Disconnect()
            except Exception:
                pass
            raise
        self.device = candidate
        self.fs = fs
        return count, fs

    def read(self):
        """New samples since the last call, one numpy array (mV) per channel."""
        out = []
        for s in self.signals:
            batch = s.GetScaledValueArray()
            if batch is None or batch.Length == 0:
                out.append(np.empty(0))
            else:
                out.append(np.fromiter((float(v) for v in batch), dtype=float, count=batch.Length) * TO_MV)
        return out

    def close(self):
        if self.device is None:
            return
        for name in ("StopAcquisition", "Disconnect"):
            try:
                getattr(self.device, name)()
            except Exception as e:
                print("%s: %r" % (name, e))
        self.device = None


class FakeRadio:
    """Synthetic 2-channel EMG for testing without hardware.
    `cue` is set by the calibration ('rest', 'biceps', 'triceps'); otherwise it
    repeats: biceps on 4 s, rest, triceps burst 0.4 s (like udp_test_sender.py)."""

    def __init__(self, fs=1000.0):
        self.fs = fs
        self.cue = None
        self.rng = np.random.default_rng(1)

    def discover(self):
        return [{"name": "Simulated BioRadio", "mac": "000000000000"}]

    def connect(self, mac):
        self.t0 = time.perf_counter()
        self.n = 0
        return 2, self.fs

    def _activation(self, t):
        if self.cue == "rest":
            return 0.0, 0.0
        if self.cue == "biceps":
            return 1.0, 0.08
        if self.cue == "triceps":
            return 0.08, 1.0
        ph = t % 6.0
        b = 0.45 if ph < 4.0 else 0.0
        tr = 0.7 if 4.2 < ph < 4.6 else 0.0
        return b, tr

    def read(self):
        target = int((time.perf_counter() - self.t0) * self.fs)
        k = target - self.n
        if k <= 0:
            return [np.empty(0), np.empty(0)]
        t = (self.n + np.arange(k)) / self.fs
        self.n = target
        chans = []
        for c in range(2):
            act = np.array([self._activation(x)[c] for x in t])
            emg = self.rng.normal(0, 1, k) * (0.01 + 1.0 * act)          # mV, amplitude-modulated noise
            hum = 0.05 * np.sin(2 * np.pi * 50 * t)                       # mains interference
            drift = 0.2 * np.sin(2 * np.pi * 0.3 * t)                     # baseline wander
            chans.append(emg + hum + drift)
        return chans

    def close(self):
        pass


# --------------------------------------------------------------------------- processing

class Cleaner:
    """Band-pass 20-450 Hz (Butterworth, order 4) + 50 Hz notch, with carried state."""

    def __init__(self, fs, notch=True):
        hi = min(450.0, 0.45 * fs)
        if hi > 40.0:
            sos = signal.butter(4, [20.0, hi], btype="bandpass", fs=fs, output="sos")
        else:
            sos = signal.butter(4, 20.0, btype="highpass", fs=fs, output="sos")
        if notch and fs > 120:
            b, a = signal.iirnotch(50.0, 30.0, fs=fs)
            sos = np.vstack([sos, signal.tf2sos(b, a)])
        self.sos = sos
        self.zi = None

    def process(self, x):
        if len(x) == 0:
            return x
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos) * x[0]
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


class LFilter:
    """Streaming lfilter with carried state."""

    def __init__(self, b, a=(1.0,), post=None):
        self.b = np.asarray(b, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)
        self.post = post

    def process(self, x):
        if len(x) == 0:
            return x
        y, self.zi = signal.lfilter(self.b, self.a, x, zi=self.zi)
        return self.post(y) if self.post else y


class Identity:
    def process(self, x):
        return x


class Kalman:
    """1-D Kalman filter, random-walk state: x_k = x_{k-1} + w (Q), z_k = x_k + v (R).
    (Its steady-state gain makes it behave like an EMA; worth noting in the report.)"""

    def __init__(self, q=1e-4, r=1.0):
        self.q, self.r = q, r
        self.x, self.p = 0.0, 1.0

    def process(self, z):
        out = np.empty(len(z))
        x, p, q, r = self.x, self.p, self.q, self.r
        for i, zi in enumerate(z):
            p += q
            k = p / (p + r)
            x += k * (zi - x)
            p *= (1.0 - k)
            out[i] = x
        self.x, self.p = x, p
        return out


class MovingRms:
    def __init__(self, n):
        self.ma = LFilter(np.ones(n) / n)

    def process(self, x):
        if len(x) == 0:
            return x
        return np.sqrt(np.maximum(self.ma.process(x * x), 0.0))


class KalmanRms:
    """Team member's F5: 100 ms moving RMS, then a random-walk Kalman filter.
    His MATLAB sets Q = 1e-4*var and R = 1e-2*var of the RMS signal. Only Q/R matters (the variance
    cancels out of the Kalman gain), so Q/R = 0.01 gives exactly the same output and runs in real
    time. Starts from the first RMS value with P0 = R, as in his code."""

    def __init__(self, n, q=1e-2, r=1.0):
        self.rms = MovingRms(n)
        self.q, self.r = q, r
        self.x, self.p = None, r

    def process(self, x):
        z = self.rms.process(x)
        out = np.empty(len(z))
        if len(z) == 0:
            return out
        i0 = 0
        if self.x is None:
            self.x = float(z[0])
            out[0] = self.x
            i0 = 1
        xx, p, q, r = self.x, self.p, self.q, self.r
        for i in range(i0, len(z)):
            p += q
            k = p / (p + r)
            xx += k * (z[i] - xx)
            p *= (1.0 - k)
            out[i] = xx
        self.x, self.p = xx, p
        return out


def make_envelope(name, fs):
    n100 = max(1, int(round(0.100 * fs)))
    if name == "F0":
        return Identity()
    if name == "F1":
        return LFilter(np.ones(n100) / n100)
    if name == "F2":
        return MovingRms(n100)
    if name == "F3":
        b, a = signal.butter(2, 4.0, btype="lowpass", fs=fs)
        return LFilter(b, a)
    if name == "F4":
        alpha = 1.0 - math.exp(-1.0 / (0.050 * fs))
        return LFilter([alpha], [1.0, -(1.0 - alpha)])
    if name == "F5":
        return KalmanRms(n100)
    # F3, F6 and F7 are all 2nd order with their -3 dB point at 4 Hz, so they compare fairly.
    # (Chebyshev I's design frequency is where the gain is -1 dB; Chebyshev II's is where
    # the 40 dB stopband starts. 3.285 Hz and 28.35 Hz put both -3 dB points at 4 Hz.)
    if name == "F6":
        b, a = signal.cheby1(2, 1.0, 3.285, btype="lowpass", fs=fs)
        return LFilter(b, a)
    if name == "F7":
        b, a = signal.cheby2(2, 40.0, 28.35, btype="lowpass", fs=fs)
        return LFilter(b, a)
    raise ValueError(name)


class Channel:
    """One muscle: cleaner + all envelope filters running side by side."""

    def __init__(self, fs, notch=True):
        self.fs = fs
        self.cleaner = Cleaner(fs, notch)
        self.env = {f: make_envelope(f, fs) for f in FILTERS}

    def process(self, raw):
        rect = np.abs(self.cleaner.process(raw))
        return rect, {f: self.env[f].process(rect) for f in FILTERS}


# --------------------------------------------------------------------------- calibration

def collect(radio, seconds, idx, label, fake_cue=None, bar_channels=None):
    """Read `seconds` of raw data for channel indices idx. Shows a countdown."""
    if isinstance(radio, FakeRadio):
        radio.cue = fake_cue
    chunks = [[] for _ in idx]
    t_end = time.perf_counter() + seconds
    while True:
        left = t_end - time.perf_counter()
        if left <= 0:
            break
        batches = radio.read()
        for j, c in enumerate(idx):
            if len(batches[c]):
                chunks[j].append(batches[c])
        sys.stdout.write("\r  %-34s %4.1f s left " % (label, left))
        sys.stdout.flush()
        time.sleep(0.02)
    sys.stdout.write("\r  %-34s done          \n" % label)
    if isinstance(radio, FakeRadio):
        radio.cue = None
    return [np.concatenate(ch) if ch else np.empty(0) for ch in chunks]


def calibrate(radio, fs, bi, tri, notch, auto):
    print("\nCalibration. Keep the arm still and follow the prompts.")
    segs = []
    wait_enter("  Relax both muscles completely, then press Enter...", auto)
    segs.append(("rest", collect(radio, 5.0, [bi, tri], "RELAX", "rest")))
    wait_enter("  Get ready to flex the BICEPS as hard as you can, press Enter to start...", auto)
    segs.append(("biceps", collect(radio, 3.0, [bi, tri], "BICEPS: MAXIMUM!", "biceps")))
    wait_enter("  Relax. Get ready to push with the TRICEPS as hard as you can, press Enter...", auto)
    segs.append(("triceps", collect(radio, 3.0, [bi, tri], "TRICEPS: MAXIMUM!", "triceps")))

    # Run each filter over the whole calibration recording (continuous), per muscle.
    calib = {}
    for m, muscle in enumerate(("biceps", "triceps")):
        raw = np.concatenate([s[1][m] for s in segs])
        bounds, start = {}, 0
        for name, data in segs:
            bounds[name] = (start, start + len(data[m]))
            start += len(data[m])
        ch = Channel(fs, notch)
        _, envs = ch.process(raw)
        calib[muscle] = {}
        for f in FILTERS:
            e = envs[f]
            r0, r1 = bounds["rest"]
            m0, m1 = bounds[muscle]
            rest_part = e[r0 + int(1.0 * fs): r1]          # skip the filter start-up
            mvc_part = e[m0 + int(0.5 * fs): m1]           # skip the ramp-up
            if len(rest_part) == 0 or len(mvc_part) == 0:
                raise RuntimeError("Calibration got no data. Is the BioRadio streaming?")
            rest = float(np.median(rest_part))
            mvc = float(np.percentile(mvc_part, 95))
            calib[muscle][f] = {"rest": rest, "mvc": mvc}
        r, v = calib[muscle]["F2"]["rest"], calib[muscle]["F2"]["mvc"]
        print("  %-7s (F2) rest %.4f mV, max %.4f mV, ratio %.1fx" % (muscle, r, v, v / max(r, 1e-9)))
        if v < 2.0 * r:
            print("  WARNING: %s barely changed. Check the electrodes and the channel number." % muscle)
    return calib


# --------------------------------------------------------------------------- live stages

def bar(x, width=20):
    x = max(0.0, min(1.0, x))
    n = int(round(x * width))
    return "#" * n + "." * (width - n)


def channel_check(radio, count, fs, notch, auto):
    print("\nChannel check: flex each muscle and watch which channel jumps.")
    print("Press Enter (in this window) to continue." if msvcrt else "Showing for 10 s.")
    chans = [Channel(fs, notch) for _ in range(count)]
    peak = [1e-6] * count
    t_end = time.perf_counter() + (3.0 if auto else (1e9 if msvcrt else 10.0))
    while time.perf_counter() < t_end:
        batches = radio.read()
        levels = []
        for c in range(count):
            if len(batches[c]):
                _, envs = chans[c].process(batches[c])
                lvl = float(envs["F2"][-1])
                peak[c] = max(peak[c] * 0.999, lvl)
            else:
                lvl = 0.0
            levels.append(lvl)
        txt = "  ".join("CH%d %s %.3f" % (c + 1, bar(levels[c] / peak[c], 10), levels[c]) for c in range(min(count, 4)))
        sys.stdout.write("\r" + txt + "  mV ")
        sys.stdout.flush()
        k = key_pressed()
        if k in ("\r", "\n"):
            break
        time.sleep(0.05)
    print()


def stream(radio, fs, bi, tri, calib, args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    active = args.filter
    chans = {"biceps": Channel(fs, not args.no_notch), "triceps": Channel(fs, not args.no_notch)}
    norm = {"biceps": 0.0, "triceps": 0.0}

    rec = None
    rec_file = ""
    pend = {"biceps": [], "triceps": []}
    sample_no = 0
    unix0 = None

    def scale(muscle, value):
        c = calib[muscle][active]
        span = max(c["mvc"] - c["rest"], 1e-9)
        return min(1.5, max(0.0, (value - c["rest"]) / span))

    def start_recording():
        nonlocal rec, rec_file
        os.makedirs(args.out, exist_ok=True)
        rec_file = os.path.join(args.out, "%s_%s_game_%s.csv" % (
            args.subject, active, datetime.now().strftime("%Y%m%d_%H%M%S")))
        f = open(rec_file, "w", newline="")
        w = csv.writer(f)
        w.writerow(["sample", "t_s", "unix_ms", "biceps_raw_mV", "triceps_raw_mV",
                    "filter", "biceps_env", "triceps_env"])
        rec = (f, w)
        pend["biceps"].clear()
        pend["triceps"].clear()
        print("\n  Recording to " + rec_file)

    def stop_recording():
        nonlocal rec
        if rec:
            rec[0].close()
            print("\n  Saved " + rec_file)
        rec = None

    print("\nStreaming to Unity at %s:%d, %d Hz. In the game choose EMG (UDP %d)." % (
        args.host, args.port, args.rate, args.port))
    print("Keys: 0-7 filter   R record on/off   Q quit")
    if args.record:
        start_recording()

    period = 1.0 / args.rate
    next_tick = time.perf_counter()
    last_print = 0.0
    t_stop = time.perf_counter() + args.duration if args.duration else None
    try:
        while True:
            batches = radio.read()
            now_ms = int(time.time() * 1000)
            for muscle, c in (("biceps", bi), ("triceps", tri)):
                x = batches[c]
                if len(x) == 0:
                    continue
                _, envs = chans[muscle].process(x)
                norm[muscle] = scale(muscle, float(envs[active][-1]))
                if rec:
                    c_ = calib[muscle][active]
                    span = max(c_["mvc"] - c_["rest"], 1e-9)
                    pend[muscle].append((x, np.clip((envs[active] - c_["rest"]) / span, 0.0, 1.5)))

            if rec:
                # Write the samples both channels have, keep the rest for next time.
                braw = np.concatenate([p[0] for p in pend["biceps"]]) if pend["biceps"] else np.empty(0)
                benv = np.concatenate([p[1] for p in pend["biceps"]]) if pend["biceps"] else np.empty(0)
                traw = np.concatenate([p[0] for p in pend["triceps"]]) if pend["triceps"] else np.empty(0)
                tenv = np.concatenate([p[1] for p in pend["triceps"]]) if pend["triceps"] else np.empty(0)
                n = min(len(braw), len(traw))
                if n:
                    if unix0 is None:
                        unix0 = now_ms - 1000.0 * n / fs
                    w = rec[1]
                    for i in range(n):
                        t = sample_no / fs
                        w.writerow([sample_no, "%.4f" % t, "%.0f" % (unix0 + 1000.0 * t),
                                    "%.5f" % braw[i], "%.5f" % traw[i], active,
                                    "%.4f" % benv[i], "%.4f" % tenv[i]])
                        sample_no += 1
                pend["biceps"] = [(braw[n:], benv[n:])] if len(braw) > n else []
                pend["triceps"] = [(traw[n:], tenv[n:])] if len(traw) > n else []

            msg = "%.4f,%.4f,%d" % (norm["biceps"], norm["triceps"], now_ms)
            sock.sendto(msg.encode("ascii"), addr)

            k = key_pressed()
            if k and k in "01234567":
                active = "F" + k
                print("\n  Filter -> %s (%s). Set the same filter in the game." % (active, FILTER_NAMES[active]))
            elif k == "r":
                stop_recording() if rec else start_recording()
            elif k == "q":
                break

            t = time.perf_counter()
            if t - last_print > 0.25:
                last_print = t
                sys.stdout.write("\r  %s %-26s biceps %s %4.0f%%   triceps %s %4.0f%%  %s" % (
                    active, FILTER_NAMES[active], bar(norm["biceps"]), 100 * norm["biceps"],
                    bar(norm["triceps"]), 100 * norm["triceps"], "REC" if rec else "   "))
                sys.stdout.flush()
            if t_stop and t >= t_stop:
                break

            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    finally:
        stop_recording()
        print()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="BioRadio EMG -> Unity game")
    ap.add_argument("--subject", default="S1", help="subject ID, used for calibration and file names")
    ap.add_argument("--biceps", type=int, default=1, help="BioRadio channel number of the biceps (1-based)")
    ap.add_argument("--triceps", type=int, default=2, help="BioRadio channel number of the triceps (1-based)")
    ap.add_argument("--filter", default="F2", choices=FILTERS, help="starting envelope filter")
    ap.add_argument("--recalibrate", action="store_true", help="ignore a saved calibration")
    ap.add_argument("--record", action="store_true", help="start recording immediately")
    ap.add_argument("--no-notch", action="store_true", help="skip the 50 Hz notch")
    ap.add_argument("--dll", default=os.path.join(HERE, "BioRadioSDK.dll"))
    ap.add_argument("--mac", default="", help="connect to this MAC without asking")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--rate", type=int, default=100, help="packets per second to the game")
    ap.add_argument("--out", default=os.path.join(HERE, "data"), help="folder for calibration and recordings")
    ap.add_argument("--simulate", action="store_true", help="fake EMG, no hardware")
    ap.add_argument("--auto", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--duration", type=float, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    radio = FakeRadio() if args.simulate else BioRadio(args.dll)
    try:
        print("Scanning for BioRadio devices (close BioCapture first)...")
        found = radio.discover()
        if not found:
            print("No BioRadio detected. Check power and Bluetooth, then run again.")
            return 1
        for i, d in enumerate(found):
            print("  [%d] %s | %s" % (i + 1, d["name"], d["mac"]))
        if args.mac:
            mac = normalize_mac(args.mac)
        elif len(found) == 1 or args.auto:
            mac = found[0]["mac"]
        else:
            pick = int(input("Which one? Type the number: ")) - 1
            mac = found[pick]["mac"]

        print("Connecting to %s..." % mac)
        count, fs = radio.connect(mac)
        print("CONNECTED: %s | %d channel(s) | %.0f Hz" % (mac, count, fs))
        if fs < 900:
            print("Note: below 1 kHz the band-pass top is lowered to %.0f Hz." % min(450, 0.45 * fs))

        bi, tri = args.biceps - 1, args.triceps - 1
        if not (0 <= bi < count and 0 <= tri < count) or bi == tri:
            print("Biceps/triceps channels must be different and between 1 and %d." % count)
            return 1

        channel_check(radio, count, fs, not args.no_notch, args.auto)

        os.makedirs(args.out, exist_ok=True)
        calib_path = os.path.join(args.out, "%s_calibration.json" % args.subject)
        calib = None
        if os.path.isfile(calib_path) and not args.recalibrate:
            with open(calib_path) as f:
                saved = json.load(f)
            has_all = all(f in saved.get("levels", {}).get(m, {}) for m in ("biceps", "triceps") for f in FILTERS)
            if not has_all or saved.get("version") != CAL_VERSION:
                print("\nSaved calibration is from an older filter version. Calibrating again.")
            elif saved.get("fs") == fs and saved.get("biceps_ch") == args.biceps and saved.get("triceps_ch") == args.triceps:
                calib = saved["levels"]
                print("\nUsing saved calibration %s (from %s). Use --recalibrate after moving electrodes." % (
                    os.path.basename(calib_path), saved.get("time", "?")))
        if calib is None:
            calib = calibrate(radio, fs, bi, tri, not args.no_notch, args.auto)
            with open(calib_path, "w") as f:
                json.dump({"subject": args.subject, "version": CAL_VERSION,
                           "time": datetime.now().isoformat(timespec="seconds"),
                           "fs": fs, "biceps_ch": args.biceps, "triceps_ch": args.triceps,
                           "levels": calib}, f, indent=1)
            print("  Saved " + calib_path)

        stream(radio, fs, bi, tri, calib, args)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        radio.close()
        print("Disconnected.")


if __name__ == "__main__":
    sys.exit(main())
