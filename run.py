#!/usr/bin/env python
"""Command line entry point for the MF facts assistant.

    python run.py build                 # stages 1-4: fetch, parse, chunk, index
    python run.py ask "expense ratio of HDFC Large Cap Fund?"
    python run.py serve                 # tiny web UI on http://127.0.0.1:8000
    python run.py eval                  # stage 9: golden + policy suite
    python run.py sources               # the citation allow-list
    python run.py stage5 "query"        # inspect the gate only
    python run.py stage6 "query"        # inspect hybrid retrieval only
    python run.py dump                  # every chunk + its embedding -> a .txt file
    python run.py retrieval             # score retrieval: recall@k, MRR, nDCG
    python run.py samples               # regenerate samples/sample_qa.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):   # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mf_faq import EXAMPLE_QUESTIONS, WELCOME                     # noqa: E402
from mf_faq.eval import (DEFAULT_KS, evaluate, evaluate_retrieval,   # noqa: E402
                         write_report)
from mf_faq.intent import classify, scrub_pii                     # noqa: E402
from mf_faq.pipeline import FAQPipeline                           # noqa: E402
from mf_faq.retrieval import Retriever, explain                   # noqa: E402
from mf_faq.sources import ALLOWED_URLS, BRIEF_URLS, SOURCES, SCHEMES  # noqa: E402

ROOT = Path(__file__).resolve().parent


def cmd_build(args) -> int:
    pipe = FAQPipeline(verbose=True)
    try:
        stats = pipe.build(force_fetch=args.refresh)
        print(json.dumps(stats, indent=2))
    finally:
        pipe.close()
    return 0


def cmd_ask(args) -> int:
    pipe = FAQPipeline(verbose=args.debug)
    try:
        if args.json:
            ans = pipe.ask(" ".join(args.query), debug=args.debug)
            print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
        else:
            # Echo the question back with any identifier redacted. Printing the
            # raw query would re-display the PAN the user just pasted, in the
            # one command a reviewer runs to check that FR-5 holds.
            shown, leaked = scrub_pii(" ".join(args.query))
            print(WELCOME)
            print("-" * 70)
            print("Q: " + shown)
            if leaked:
                print(f"   ({len(leaked)} identifier type(s) redacted from the echo above)")
            ans = pipe.ask(" ".join(args.query), debug=args.debug)
            print()
            print(ans.render())
            print("-" * 70)
            print(f"kind={ans.kind} stage={ans.stage} guards={ans.guards} "
                  f"timings={ans.timings_ms}")
    finally:
        pipe.close()
    return 0


def cmd_serve(args) -> int:
    from mf_faq.ui import serve
    pipe = FAQPipeline()
    serve(host=args.host, port=args.port, pipe=pipe)
    return 0


def cmd_eval(args) -> int:
    pipe = FAQPipeline(verbose=args.debug)
    rep = evaluate(pipe, verbose=True)
    out = ROOT / "data" / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(rep, out)
    print(f"report -> {out}")
    return 0 if rep.failed == 0 else 1


def cmd_sources(args) -> int:
    from sources_gen import generate as gen_sources
    print(f"Quotable corpus (the {len(BRIEF_URLS)} URLs from the brief):")
    for i, u in enumerate(BRIEF_URLS, 1):
        print(f"  {i}. {u}")
    print(f"\nFull allow-list ({len(ALLOWED_URLS)} URLs):")
    for s in SOURCES:
        flags = []
        if not s.quotable:
            flags.append("pointer-only")
        if not s.verified_at_build:
            flags.append("not-quotable-at-build")
        print(f"  [{s.source_id}] {s.url}")
        print(f"      {s.publisher} | {s.doc_type}"
              + (f" | {', '.join(flags)}" if flags else ""))
        if not s.quotable and s.note:
            print(f"      why: {s.note}")
    print("\nAMC: HDFC Mutual Fund")
    for k, v in SCHEMES.items():
        print(f"  {k:20s} {v['label']}  [{v['category']}]")
    if getattr(args, "write", False):
        c, m = gen_sources()
        print(f"\nwrote {c}\nwrote {m}")
    return 0


def cmd_stage5(args) -> int:
    d = classify(" ".join(args.query))
    print(json.dumps(d.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_stage6(args) -> int:
    pipe = FAQPipeline()
    try:
        r = pipe.retriever.retrieve(classify(" ".join(args.query)), top_k=args.k)
        print(explain(r))
    finally:
        pipe.close()
    return 0


def cmd_samples(args) -> int:
    from samples_gen import generate
    j, m = generate()
    print(f"wrote {j}\nwrote {m}")
    return 0


def cmd_dump(args) -> int:
    from mf_faq.indexing import write_txt_report
    out = Path(args.out) if args.out else ROOT / "data" / "chunks_and_embeddings.txt"
    info = write_txt_report(
        out, db_path=Path(args.db) if args.db else None,
        full=args.full, top_n=args.top, kind=args.kind, topic=args.topic,
        contains=args.contains, limit=args.limit, show_tokens=not args.no_tokens,
        neighbors=args.neighbors,
    )
    print(f"wrote {info['path']}  "
          f"({info['chunks_written']} of {info['chunks_matched']} chunks, "
          f"{info['lines']} lines, {info['missing_vectors']} missing vectors)")
    return 0


def cmd_retrieval(args) -> int:
    pipe = FAQPipeline()
    try:
        rep = evaluate_retrieval(pipe, ks=tuple(args.ks) if args.ks else DEFAULT_KS,
                                 verbose=not args.json)
    finally:
        pipe.close()
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        out = ROOT / "data" / "retrieval_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nreport -> {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mf-faq", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debug", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="run stages 1-4 (fetch, parse, chunk, index)")
    b.add_argument("--refresh", action="store_true", help="ignore the raw cache and refetch")
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("query", nargs="+")
    a.add_argument("--json", action="store_true")
    # `--debug` is also a global flag, so it is accepted on both sides of the
    # subcommand. default=SUPPRESS matters: a subparser default would otherwise
    # overwrite the global value, silently turning `run.py --debug ask "..."`
    # into a non-debug run.
    a.add_argument("--debug", action="store_true", default=argparse.SUPPRESS)
    a.set_defaults(func=cmd_ask)

    s = sub.add_parser("serve", help="run the tiny web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    e = sub.add_parser("eval", help="run the golden + policy suite")
    e.set_defaults(func=cmd_eval)

    sp = sub.add_parser("sources", help="print the source registry")
    sp.add_argument("--write", action="store_true",
                    help="also regenerate SOURCES.md and SOURCES.csv")
    sp.set_defaults(func=cmd_sources)

    s5 = sub.add_parser("stage5", help="show the intent gate decision")
    s5.add_argument("query", nargs="+")
    s5.set_defaults(func=cmd_stage5)

    s6 = sub.add_parser("stage6", help="show hybrid retrieval for a query")
    s6.add_argument("query", nargs="+")
    s6.add_argument("-k", type=int, default=5)
    s6.set_defaults(func=cmd_stage6)

    sm = sub.add_parser("samples", help="regenerate samples/sample_qa.{json,md}")
    sm.set_defaults(func=cmd_samples)

    dp = sub.add_parser("dump", help="write every chunk + its embedding to a .txt file")
    dp.add_argument("-o", "--out", help="output .txt path "
                                       "(default: data/chunks_and_embeddings.txt)")
    dp.add_argument("--db", help="index to read (default: data/index/mf_faq.sqlite)")
    dp.add_argument("--full", action="store_true",
                    help="print all 512 dims per chunk, not just the non-zero ones")
    dp.add_argument("--top", type=int, default=24,
                    help="how many strongest dims to summarise per chunk")
    dp.add_argument("--kind", choices=["fact", "section", "procedure"])
    dp.add_argument("--topic", help="only chunks with this topic")
    dp.add_argument("--contains", help="only chunks whose text/answer_hint/chunk_id contains this")
    dp.add_argument("--limit", type=int, help="write at most this many chunks")
    dp.add_argument("--no-tokens", action="store_true", help="skip the token list per chunk")
    dp.add_argument("-n", "--neighbors", type=int, default=0, metavar="N",
                    help="list each chunk's N nearest chunks by cosine similarity")
    dp.set_defaults(func=cmd_dump)

    rv = sub.add_parser("retrieval", help="score retrieval: recall@k, precision@k, MRR, nDCG")
    rv.add_argument("-k", type=int, nargs="+", dest="ks", metavar="K",
                    help=f"cut-offs to report (default: {list(DEFAULT_KS)})")
    rv.add_argument("--json", action="store_true", help="print the report instead of a table")
    rv.set_defaults(func=cmd_retrieval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
