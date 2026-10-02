"""Ad-hoc latency probe (not part of the test suite).

The build gate is "mean < 150 ms warm", so this reports a warm mean that means
something. Three corrections over the first version, which printed one blended
mean:

  * the first query is reported separately as *cold* -- it pays for opening the
    index and loading the vector matrix, and was running 300-500 ms against
    ~40 ms for everything after it. Averaging it in made the headline number
    both worse and less meaningful.
  * refusals are reported separately too. A stage-5 refusal returns before
    retrieval, so it costs ~0.2 ms; folding those in flattered the mean by
    making the assistant look faster at answering than it is.
  * the gate is evaluated explicitly rather than left for the reader to infer
    from whichever number happened to be printed last.

Run:  python bench.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):   # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from mf_faq.pipeline import FAQPipeline  # noqa: E402

WARM_GATE_MS = 150.0

ANSWER_QUERIES = [
    "What is the expense ratio of HDFC Large Cap Fund?",
    "minimum SIP for HDFC Small Cap Fund",
    "exit load HDFC ELSS Tax Saver Fund",
    "benchmark of HDFC Balanced Advantage Fund",
    "how do I download a capital gains statement?",
    "where can I download my account statement?",
    "Who manages HDFC Flexi Cap Fund?",
    "What is the NAV of HDFC Large Cap Fund?",
    "Is there a lock-in period in HDFC ELSS Tax Saver Fund?",
    "minimum lump sum for HDFC Small Cap Fund",
]

#: Refusals are measured, not averaged in: they short-circuit before retrieval.
REFUSAL_QUERIES = [
    "should I buy HDFC Small Cap Fund?",
    "What are the 3 year returns of HDFC Large Cap Fund?",
    "My PAN is ABCDE1234F, what is the exit load?",
]


def _stats(xs: list[float]) -> tuple[float, float]:
    return (sum(xs) / len(xs), max(xs)) if xs else (0.0, 0.0)


def main() -> int:
    t0 = time.perf_counter()
    pipe = FAQPipeline()
    open_ms = (time.perf_counter() - t0) * 1000

    cold: list[float] = []
    warm: list[float] = []
    refusals: list[float] = []

    print(f"index opened lazily in {open_ms:.1f} ms (constructor does no I/O)\n")

    for i, q in enumerate(ANSWER_QUERIES):
        a = pipe.ask(q)
        ms = a.timings_ms["total_ms"]
        (cold if i == 0 else warm).append(ms)
        tag = "cold" if i == 0 else "warm"
        print(f"  {ms:8.1f} ms  {tag:4s}  [{a.kind:10s}] {a.text[:56]}")

    for q in REFUSAL_QUERIES:
        a = pipe.ask(q)
        refusals.append(a.timings_ms["total_ms"])
        print(f"  {a.timings_ms['total_ms']:8.1f} ms  refr  [{a.kind:10s}] {a.text[:56]}")

    pipe.close()

    warm_mean, warm_max = _stats(warm)
    cold_mean, cold_max = _stats(cold)
    ref_mean, ref_max = _stats(refusals)
    all_mean, all_max = _stats(cold + warm + refusals)

    print()
    print(f"cold        n={len(cold):2d}  mean={cold_mean:7.1f} ms  max={cold_max:7.1f} ms")
    print(f"warm        n={len(warm):2d}  mean={warm_mean:7.1f} ms  max={warm_max:7.1f} ms")
    print(f"refusals    n={len(refusals):2d}  mean={ref_mean:7.1f} ms  max={ref_max:7.1f} ms")
    print(f"all queries n={len(cold) + len(warm) + len(refusals):2d}  "
          f"mean={all_mean:7.1f} ms  max={all_max:7.1f} ms   (not a gate figure)")

    ok = warm_mean < WARM_GATE_MS
    print()
    print(f"gate: warm mean {warm_mean:.1f} ms < {WARM_GATE_MS:.0f} ms  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  the build gate is not met; do not report the blended mean instead")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
