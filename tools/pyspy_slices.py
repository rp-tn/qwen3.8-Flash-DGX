#!/usr/bin/env python3
"""Summarize tools/profile_boot.sh output (py-spy raw/collapsed stacks, one file per time slice).

  pyspy_slices.py DIR [--proc EngineCore|api] [--thread MainThread] [--from 3 --to 14]
                      [--self | --incl REGEX] [--top 15] [--per-slice] [--strip-idle]

--proc api   = process 1 only (the APIServer), EngineCore = frames under the EngineCore process.
--self       = rank leaf frames (default). --incl REGEX = rank frames matching REGEX by inclusive time.
--under REGEX= keep only stacks that pass through a frame matching REGEX.
"""
import argparse
import collections
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--proc", default="EngineCore")
ap.add_argument("--thread", default="MainThread")
ap.add_argument("--from", dest="lo", type=int, default=0)
ap.add_argument("--to", dest="hi", type=int, default=10**6)
ap.add_argument("--incl")
ap.add_argument("--under")
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--per-slice", action="store_true")
ap.add_argument("--depth", type=int, default=1, help="leaf frames joined for --self (1 = leaf only)")
a = ap.parse_args()

files = sorted(f for f in os.listdir(a.dir) if f.endswith(".txt"))
files = [f for f in files if a.lo <= int(f.split("-")[0]) <= a.hi]


def stacks(path):
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        stack, _, n = line.rpartition(" ")
        if not n.isdigit():
            continue
        fr = stack.split(";")
        procs = [i for i, x in enumerate(fr) if x.startswith("process ")]
        if a.proc == "api":
            if len(procs) != 1:
                continue
        elif not any(a.proc in fr[i] for i in procs):
            continue
        rest = fr[procs[-1] + 1:]
        if not rest or not rest[0].startswith("thread ") or a.thread not in rest[0]:
            continue
        yield rest[1:], int(n)


def short(f):
    return re.sub(r"\((?:/usr/local/lib/python3\.12/dist-packages/|/usr/lib/python3\.12/)", "(", f)


def summarize(fs):
    tot, cnt = 0, collections.Counter()
    for f in fs:
        for fr, n in stacks(os.path.join(a.dir, f)):
            if a.under and not any(re.search(a.under, x) for x in fr):
                tot += 0
                continue
            tot += n
            if a.incl:
                for x in set(fr):
                    if re.search(a.incl, x):
                        cnt[short(x)] += n
            else:
                cnt[" <- ".join(short(x) for x in reversed(fr[-a.depth:])) if fr else "(no python frame)"] += n
    return tot, cnt


groups = [[f] for f in files] if a.per_slice else [files]
for g in groups:
    tot, cnt = summarize(g)
    head = g[0] if len(g) == 1 else f"{g[0]} .. {g[-1]} ({len(g)} slices)"
    print(f"== {head}: {tot} samples")
    for k, n in cnt.most_common(a.top):
        print(f"  {100 * n / max(tot, 1):5.1f}%  {k}")
