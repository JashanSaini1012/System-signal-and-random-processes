"""
game_analysis.py - Project 14 (SSRP): game results for every smoothing filter.

Reads the Unity game logs (GameLogs/) and the Python EMG recordings (data/), and makes the tables
and figures for the game part of the report. Put this file next to bioradio_emg.py and run:

    python game_analysis.py --data data --logs GameLogs --out results_game

What it does
 1. Labels: finds which filter Python was REALLY running during each game round (the game's Filter
    box is only a label and can be wrong) and lists the corrected labels.
 2. Football results per filter (goals, shots, tackles, ...) from the game logs.
 3. Smoothness of the player's run command per filter, from the per-frame game logs.
 4. Response time: each filter's own delay (step test, same code as the game) plus the measured
    reaction test; reaction times for filters that were not tested are ESTIMATED from these.
 5. Offline game simulation: every filter, including F0 Raw, applied to the SAME recorded raw EMG
    and passed through the game's run/kick rules. This gives the raw-vs-processed comparison.
"""
import argparse
import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bioradio_emg as be

FS = 1000
NAMES = {"F0": "F0 Raw", "F1": "F1 MAV", "F2": "F2 Moving RMS", "F3": "F3 Butterworth",
         "F4": "F4 EMA", "F5": "F5 Kalman (RMS)", "F6": "F6 Chebyshev I", "F7": "F7 Chebyshev II"}
SIM_FILTERS = ["F0", "F2", "F3", "F5", "F6", "F7"]
COLORS = {"F0": "#9e9e9e", "F2": "#8c564b", "F3": "#1f77b4", "F5": "#2ca02c",
          "F6": "#d62728", "F7": "#9467bd"}
G = dict(run_on=0.10, run_full=0.60, kick_on=0.30, kick_off=0.20, kick_win=0.30, lockout=0.70)


# ----------------------------------------------------------------------------- reading files
def stamp(name):
    m = re.search(r"(\d{8}_\d{6})", name)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None


def summary(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("# subject="):
                return dict(kv.split("=", 1) for kv in line[2:].strip().split(" ") if "=" in kv)
    return None


def load_python(data_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(data_dir, "S*_*_game_*.csv"))):
        d = pd.read_csv(p, usecols=["t_s", "filter"])
        rows.append(dict(path=p, file=os.path.basename(p), subject=os.path.basename(p).split("_")[0],
                         start=stamp(p), dur=d.t_s.iloc[-1] - d.t_s.iloc[0], filt=d["filter"].values))
    return rows


def actual_filter(py_rows, subject, start, seconds):
    """Filter Python was running during [start, start+seconds] (wall clock), or None."""
    for r in py_rows:
        if r["subject"] != subject or r["start"] is None:
            continue
        off = (start - r["start"]).total_seconds()
        if -5 <= off <= r["dur"]:
            a = int(max(off, 0) * FS)
            b = int(min(off + seconds, r["dur"]) * FS)
            seg = r["filt"][a:max(b, a + 1)]
            vals, cnt = np.unique(seg, return_counts=True)
            return vals[np.argmax(cnt)], r["file"]
    return None, None


# ----------------------------------------------------------------------------- game rules
def kick_sim(tri, fs):
    k, st, tmr, dt = [], 0, 0.0, 1.0 / fs
    for i, v in enumerate(tri):
        if st == 0:
            if v >= G["kick_on"]:
                st, tmr = 1, 0.0
        elif st == 1:
            tmr += dt
            if tmr >= G["kick_win"] or v < G["kick_off"]:
                k.append(i)
                st, tmr = 2, 0.0
        else:
            tmr += dt
            if tmr >= G["lockout"] and v < G["kick_off"]:
                st = 0
    return np.array(k, dtype=int)


def run_stats(on, fs, minutes):
    """Run starts per minute and 'flickers' (stops shorter than 0.2 s) per minute."""
    on = np.asarray(on, bool)
    d = np.diff(np.r_[0, on.astype(int), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    gaps = starts[1:] - ends[:-1]
    flick = int(np.sum(gaps < 0.2 * fs))
    return len(starts) / minutes, flick / minutes


def ripple(x, on, fs):
    """Fast wobble (above 3 Hz) of a signal while it is on, as % of its mean."""
    if on.sum() < fs:
        return np.nan
    bh, ah = signal.butter(2, 3, "high", fs=fs)
    hp = signal.filtfilt(bh, ah, x)
    return 100 * hp[on].std() / max(x[on].mean(), 1e-9)


# ----------------------------------------------------------------------------- 1-3: game logs
def game_tables(logs_dir, py_rows):
    lab, foot, smooth, rt = [], [], [], []
    for p in sorted(glob.glob(os.path.join(logs_dir, "S*_*.csv"))):
        name = os.path.basename(p)
        s = summary(p)
        if s is None or s.get("played_s", s.get("cues_done", "")) == "":
            continue
        is_rt = "_RT_" in name
        start = stamp(name)
        length = 110 if is_rt else float(s.get("round_s", 300))
        act, pyf = actual_filter(py_rows, s["subject"], start, length)
        logged = s["filter"]
        act = act or logged
        fixed = name.replace("_%s_" % logged, "_%s_" % act, 1) if act != logged else name
        lab.append(dict(game_log=name, kind="reaction test" if is_rt else "football", subject=s["subject"],
                        logged_filter=logged, actual_filter=act, python_recording=pyf,
                        corrected_name=fixed, start=start))
        if is_rt:
            rt.append(dict(subject=s["subject"], filter=act, **{k: float(s[k]) for k in s if k.endswith("_ms")},
                           misses=int(s["misses"]), false_starts=int(s["false_starts"]),
                           false_kicks=int(s["false_kicks"]), crosstalk=int(s["crosstalk"]), file=name))
            continue
        g = pd.read_csv(p, comment="#")
        kp = [float(e.split(":")[1]) for e in g.event.dropna() if str(e).startswith("kick:")]
        goals, shots = int(s["goals"]), int(s["shots"])
        foot.append(dict(subject=s["subject"], filter=act, played_s=float(s["played_s"]), goals=goals,
                         shots=shots, shooting_pct=100 * goals / max(shots, 1), saves=int(s["saves"]),
                         misses=int(s["misses"]), tackles=int(s["tackles"]), attacks=int(s["attacks"]),
                         kicks_without_ball=int(s["kicks_without_ball"]),
                         avg_time_to_goal_s=float(s["avg_time_to_goal_s"]),
                         mean_kick_power=np.mean(kp) if kp else np.nan, start=start, file=name))
        # smoothness of the run command (resampled to 100 Hz, only while the player has the ball)
        t = g.t_s.values
        tt = np.arange(t[0], t[-1], 0.01)
        run = np.interp(tt, t, g.run.values)
        poss = np.interp(tt, t, (g.state == "Possession").values.astype(float)) > 0.5
        on = (run > 0) & poss
        starts, flick = run_stats(on, 100, (tt[-1] - tt[0]) / 60)
        smooth.append(dict(subject=s["subject"], filter=act, run_starts_per_min=starts,
                           flickers_per_min=flick, running_pct=100 * on.mean(),
                           run_ripple_pct=ripple(run, on, 100), file=name, t=tt, run=run))
    lab = pd.DataFrame(lab).sort_values("start")
    foot = pd.DataFrame(foot).sort_values("start")
    for sub, idx in foot.groupby("subject").groups.items():
        foot.loc[idx, "order_played"] = np.arange(1, len(idx) + 1)
    return lab, foot, smooth, pd.DataFrame(rt)


# ----------------------------------------------------------------------------- 4: response time
def step_delays():
    t0, t1 = 3 * FS, 4 * FS
    u = np.r_[np.zeros(t0), 0.5 * np.ones(t1 - t0), np.zeros(2 * FS)]
    out = []
    for f in SIM_FILTERS:
        y = be.make_envelope(f, FS).process(u)
        y = y / (y[t1 - 1] / 0.5)
        out.append(dict(filter=f, run_start_delay_ms=float(np.argmax(y[t0:] >= G["run_on"])),
                        run_stop_delay_ms=float(np.argmax(y[t1:] <= G["run_on"])),
                        kick_delay_ms=float(np.argmax(y[t0:] >= G["kick_on"])) + 1000 * G["kick_win"],
                        overshoot_pct=100 * (y[t0:t1].max() / 0.5 - 1)))
    return pd.DataFrame(out)


def response_table(steps, rt):
    r = steps.copy()
    r["measured"] = False
    if rt.empty:
        return r, None
    ref = rt.iloc[0]
    st = steps.set_index("filter").loc[ref["filter"]]
    human = dict(start=ref["run_onset_median_ms"] - st.run_start_delay_ms,
                 stop=ref["run_release_median_ms"] - st.run_stop_delay_ms,
                 kick=ref["kick_median_ms"] - st.kick_delay_ms)
    r["start_reaction_ms"] = human["start"] + r.run_start_delay_ms
    r["stop_reaction_ms"] = human["stop"] + r.run_stop_delay_ms
    r["kick_reaction_ms"] = human["kick"] + r.kick_delay_ms
    r.loc[r["filter"] == ref["filter"], "measured"] = True
    return r, human


# ----------------------------------------------------------------------------- 5: offline simulation
def offline_sim(data_dir, subjects):
    rows, example = [], None
    for sub in subjects:
        calp = os.path.join(data_dir, "%s_calibration.json" % sub)
        files = sorted(glob.glob(os.path.join(data_dir, "%s_*_game_*.csv" % sub)))
        if not os.path.isfile(calp) or not files:
            continue
        cal = json.load(open(calp))["levels"]
        acc = {f: {"b": [], "t": []} for f in SIM_FILTERS}
        for p in files:
            d = pd.read_csv(p, usecols=["biceps_raw_mV", "triceps_raw_mV"])
            for m, col in (("biceps", "b"), ("triceps", "t")):
                ch = be.Channel(FS, notch=True)
                _, envs = ch.process(d["%s_raw_mV" % m].values.astype(float))
                for f in SIM_FILTERS:
                    L = cal[m][f]
                    n = np.clip((envs[f][FS:] - L["rest"]) / max(L["mvc"] - L["rest"], 1e-9), 0, 1.5)
                    acc[f][col].append(n)
        for f in SIM_FILTERS:
            b, t = np.concatenate(acc[f]["b"]), np.concatenate(acc[f]["t"])
            mins = len(b) / FS / 60
            on = b > G["run_on"]
            starts, flick = run_stats(on, FS, mins)
            kicks = kick_sim(t, FS)
            rows.append(dict(subject=sub, filter=f, minutes=mins, run_starts_per_min=starts,
                             flickers_per_min=flick, running_pct=100 * on.mean(),
                             kicks_per_min=len(kicks) / mins, biceps_ripple_pct=ripple(b, on, FS)))
            if sub == "S2" and f == "F0":
                example = {ff: np.concatenate(acc[ff]["b"]) for ff in SIM_FILTERS}
    return pd.DataFrame(rows), example


# ----------------------------------------------------------------------------- figures
def bar(ax, labels, vals, colors, fmt="%.0f"):
    bars = ax.bar(labels, vals, color=colors)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), fmt % v, ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=.3)


def figures(out, foot, smooth, resp, human, rt_raw, sim, example):
    f2 = foot[foot.subject == "S2"].sort_values("order_played")
    lab = ["%s\n(%d%s)" % (NAMES[f], o, {1: "st", 2: "nd", 3: "rd"}.get(o, "th"))
           for f, o in zip(f2["filter"], f2.order_played.astype(int))]
    col = [COLORS[f] for f in f2["filter"]]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    bar(ax[0, 0], lab, f2.goals, col); ax[0, 0].set_title("Goals in 5 minutes")
    bar(ax[0, 1], lab, f2.shooting_pct, col); ax[0, 1].set_title("Shooting accuracy (goals / shots, %)")
    bar(ax[1, 0], lab, f2.tackles, col); ax[1, 0].set_title("Tackles (player failed to stop in time)")
    bar(ax[1, 1], lab, f2.avg_time_to_goal_s, col, "%.1f"); ax[1, 1].set_title("Average time to score (s)")
    for a in ax.flat:
        a.tick_params(axis="x", labelsize=8)
    fig.suptitle("S2 football results per smoothing filter (order played in brackets)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "FigG1_football_results.png"), dpi=200); plt.close(fig)

    s2 = [s for s in smooth if s["subject"] == "S2"]
    s2.sort(key=lambda s: f2.set_index("file").loc[s["file"], "order_played"])
    fig, ax = plt.subplots(len(s2) + 1, 1, figsize=(12, 2.0 * (len(s2) + 1)))
    names = [NAMES[s["filter"]] for s in s2]
    w = np.arange(len(s2))
    ax[0].bar(w - 0.2, [s["run_starts_per_min"] for s in s2], 0.4, label="run starts / min")
    ax[0].bar(w + 0.2, [s["flickers_per_min"] for s in s2], 0.4, label="flickers / min (stops < 0.2 s)")
    ax[0].set_xticks(w); ax[0].set_xticklabels(names, fontsize=8); ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", alpha=.3); ax[0].set_title("Smoothness of the run command (S2, from the game logs)")
    for i, s in enumerate(s2):
        a = ax[i + 1]
        m = (s["t"] >= 60) & (s["t"] < 80)
        a.plot(s["t"][m] - 60, s["run"][m], color=COLORS[s["filter"]])
        a.set_ylim(0, 1.05); a.set_ylabel(NAMES[s["filter"]], fontsize=8); a.grid(alpha=.3)
    ax[-1].set_xlabel("Time (s), 20 s of each round")
    fig.tight_layout(); fig.savefig(os.path.join(out, "FigG2_smoothness.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    lbl = [NAMES[f] for f in resp["filter"]]
    x = np.arange(len(resp))
    ax[0].bar(x - 0.2, resp.run_start_delay_ms, 0.4, label="start running")
    ax[0].bar(x + 0.2, resp.run_stop_delay_ms, 0.4, label="stop running")
    ax[0].set_xticks(x); ax[0].set_xticklabels(lbl, fontsize=8, rotation=15)
    ax[0].set_ylabel("ms"); ax[0].set_title("Delay added by the filter alone (step test)")
    ax[0].legend(); ax[0].grid(axis="y", alpha=.3)
    if human is not None:
        ax[1].bar(x - 0.27, resp.start_reaction_ms, 0.27, label="start")
        ax[1].bar(x, resp.stop_reaction_ms, 0.27, label="stop")
        ax[1].bar(x + 0.27, resp.kick_reaction_ms, 0.27, label="kick")
        for i, m in enumerate(resp.measured):
            ax[1].text(i, 20, "measured" if m else "estimated", ha="center", fontsize=7, rotation=90, color="w")
        ax[1].set_xticks(x); ax[1].set_xticklabels(lbl, fontsize=8, rotation=15)
        ax[1].set_ylabel("ms"); ax[1].legend(loc="upper left", fontsize=8); ax[1].grid(axis="y", alpha=.3)
        ax[1].set_ylim(0, 1.15 * np.nanmax(resp.kick_reaction_ms))
        ax[1].set_title("Reaction time in the game: cue to action (person + filter)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "FigG3_response_time.png"), dpi=200); plt.close(fig)

    if rt_raw is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        r = rt_raw
        run = r[r.type == "run"]; kick = r[r.type == "kick"]
        ax.plot(run.cue, run.onset_ms, "o", label="start running (FLEX cue)")
        ax.plot(run.cue, run.release_ms, "s", label="stop running (RELAX cue)")
        ax.plot(kick.cue, kick.onset_ms, "^", label="kick (KICK cue)")
        ax.set_xlabel("Cue number"); ax.set_ylabel("ms"); ax.grid(alpha=.3); ax.legend()
        ax.set_title("Reaction test, S2 with %s: every cue" % NAMES[rt_raw.attrs["filter"]])
        fig.tight_layout(); fig.savefig(os.path.join(out, "FigG5_reaction_test_cues.png"), dpi=200); plt.close(fig)

    if not sim.empty:
        m = sim.groupby("filter").mean(numeric_only=True).reindex(SIM_FILTERS)
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        lbl = [NAMES[f] for f in SIM_FILTERS]
        col = [COLORS[f] for f in SIM_FILTERS]
        bar(ax[0, 0], lbl, m.flickers_per_min, col, "%.1f"); ax[0, 0].set_title("Flickers per minute (run stops < 0.2 s), log scale")
        bar(ax[0, 1], lbl, m.run_starts_per_min, col, "%.1f"); ax[0, 1].set_title("Run starts per minute, log scale")
        ax[0, 0].set_yscale("log"); ax[0, 1].set_yscale("log")
        bar(ax[1, 0], lbl, m.kicks_per_min, col, "%.1f"); ax[1, 0].set_title("Kicks per minute")
        bar(ax[1, 1], lbl, m.biceps_ripple_pct, col, "%.0f"); ax[1, 1].set_title("Ripple of the biceps level (%)")
        for a in ax.flat:
            a.tick_params(axis="x", labelsize=8, rotation=15)
        subs = ", ".join(sorted(sim.subject.unique()))
        fig.suptitle("Offline game simulation: every filter on the SAME recorded EMG (%s, mean)" % subs)
        fig.tight_layout(); fig.savefig(os.path.join(out, "FigG4_simulation_all_filters.png"), dpi=200); plt.close(fig)
    if example is not None:
        b = example["F3"]
        i_end = int(np.nanargmax(pd.Series(b).rolling(4 * FS).mean().values))
        i0 = max(0, i_end - 6 * FS)
        sl = slice(i0, i0 + 8 * FS)
        tt = np.arange(8 * FS) / FS
        fig, ax = plt.subplots(len(SIM_FILTERS), 1, figsize=(12, 1.7 * len(SIM_FILTERS)), sharex=True)
        for a, f in zip(ax, SIM_FILTERS):
            a.plot(tt, example[f][sl], color=COLORS[f], lw=0.8 if f == "F0" else 1.4)
            a.axhline(G["run_on"], color="r", ls="--", lw=0.8)
            a.set_ylim(0, 1.5); a.set_ylabel(NAMES[f], fontsize=8); a.grid(alpha=.3)
        ax[0].set_title("S2 biceps level with each filter, same 8 s of EMG (red line = run threshold)")
        ax[-1].set_xlabel("Time (s)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "FigG6_raw_vs_filtered_example.png"), dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--logs", default="GameLogs")
    ap.add_argument("--out", default="results_game")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    py_rows = load_python(a.data)
    lab, foot, smooth, rt = game_tables(a.logs, py_rows)
    steps = step_delays()
    resp, human = response_table(steps, rt)
    rt_raw = None
    if not rt.empty:
        rt_raw = pd.read_csv(os.path.join(a.logs, rt.iloc[0]["file"]), comment="#")
        rt_raw.attrs["filter"] = rt.iloc[0]["filter"]
    sim, example = offline_sim(a.data, sorted({r["subject"] for r in py_rows}))

    sm = pd.DataFrame([{k: v for k, v in s.items() if k not in ("t", "run")} for s in smooth])
    lab.drop(columns="start").to_csv(os.path.join(a.out, "labels_corrected.csv"), index=False)
    foot.drop(columns="start").to_csv(os.path.join(a.out, "table_football.csv"), index=False)
    sm.to_csv(os.path.join(a.out, "table_smoothness_game.csv"), index=False)
    resp.to_csv(os.path.join(a.out, "table_response_time.csv"), index=False)
    rt.to_csv(os.path.join(a.out, "table_reaction_test.csv"), index=False)
    sim.to_csv(os.path.join(a.out, "table_offline_simulation.csv"), index=False)
    figures(a.out, foot, smooth, resp, human, rt_raw, sim, example)

    pd.set_option("display.width", 200)
    print("\n== Labels (logged in the game vs filter Python was really running) ==")
    print(lab[["game_log", "logged_filter", "actual_filter", "corrected_name"]].to_string(index=False))
    print("\n== Football ==")
    print(foot[["subject", "order_played", "filter", "goals", "shots", "shooting_pct", "tackles",
                "kicks_without_ball", "avg_time_to_goal_s", "mean_kick_power"]].round(2).to_string(index=False))
    print("\n== Smoothness of the run command in the game ==")
    print(sm[["subject", "filter", "run_starts_per_min", "flickers_per_min", "running_pct",
              "run_ripple_pct"]].round(2).to_string(index=False))
    print("\n== Response time (filter delay from the step test; reaction = person + filter) ==")
    print(resp.round(0).to_string(index=False))
    if human is not None:
        print("Person's own reaction (from the %s reaction test): start %.0f ms, stop %.0f ms, kick %.0f ms"
              % (rt.iloc[0]["filter"], human["start"], human["stop"], human["kick"]))
    print("\n== Offline simulation, mean over subjects ==")
    print(sim.groupby("filter").mean(numeric_only=True).reindex(SIM_FILTERS).round(2).to_string())
    print("\nSaved in", a.out)


if __name__ == "__main__":
    main()
