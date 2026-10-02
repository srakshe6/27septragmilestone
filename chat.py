#!/usr/bin/env python
"""Interactive terminal REPL for trying the assistant before any UI exists.

    python chat.py                          # uses whatever .env/config says
    python chat.py --llm                    # force the LLM rewrite pass on
    python chat.py --llm --provider groq    # force a specific provider
    python chat.py --no-llm                 # force the offline extractive path
    python chat.py --model llama-3.1-8b-instant

In-session commands:
    /stage    show the synthesiser, provider, model and key state
    /hits     show the retrieved chunks for the last answer
    /raw      toggle verbose retrieval trace
    /json     toggle raw JSON output
    /quit     leave
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mf_faq import EXAMPLE_QUESTIONS, WELCOME          # noqa: E402
from mf_faq import llm                                  # noqa: E402
from mf_faq.config import CONFIG                        # noqa: E402
from mf_faq.intent import scrub_pii                     # noqa: E402
from mf_faq.pipeline import FAQPipeline                 # noqa: E402

RULE = "-" * 72
PROMPT = "you> "


def _banner(mode: str) -> None:
    print(RULE)
    print(WELCOME)
    print(RULE)
    key = os.environ.get("GROQ_API_KEY", "")
    prov = llm.provider()
    print(f"synthesis : {mode}")
    print(f"provider  : {prov}   model: {llm.model_for(prov) or 'n/a'}")
    if prov != "none" and key:
        print(f"api key   : present ({len(key)} chars, {key[:3]}...)")
    else:
        print(f"api key   : NONE FOUND for provider '{prov}' - "
              f"the offline extractive path will be used")
    print(f"examples  : {EXAMPLE_QUESTIONS[0]}")
    print("commands  : /stage /hits /raw /json /quit")
    print(RULE)


def _cmd_stage() -> None:
    print(f"  synthesis   : {'llm' if llm.available() else 'extractive'}")
    print(f"  provider    : {llm.provider()}  (auto -> groq when GROQ_API_KEY is set)")
    print(f"  groq model  : {CONFIG.groq_model}")
    print(f"  key set     : {bool(os.environ.get('GROQ_API_KEY'))}")
    print(f"  base url    : {os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1/chat/completions')}")
    print(f"  offline     : {CONFIG.offline}")


def main() -> int:
    p = argparse.ArgumentParser(prog="chat.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--llm", action="store_true", help="force the LLM rewrite pass on")
    g.add_argument("--no-llm", action="store_true", help="force the extractive path")
    p.add_argument("--provider", choices=["auto", "groq", "openai", "anthropic", "none"])
    p.add_argument("--model", help="override the provider model for this run")
    p.add_argument("--raw", action="store_true", help="start with retrieval tracing on")
    p.add_argument("--json", action="store_true", help="start with JSON output on")
    args = p.parse_args()

    target = args.provider or llm.provider()
    if args.model:
        if target == "groq":
            CONFIG.groq_model = args.model
        elif target == "anthropic":
            CONFIG.anthropic_model = args.model
        else:
            CONFIG.llm_model = args.model

    pipe = FAQPipeline(verbose=True)
    if args.no_llm:
        mode = pipe.set_llm(False, args.provider)
    elif args.llm:
        mode = pipe.set_llm(True, args.provider)
    else:
        mode = pipe.set_llm(CONFIG.llm_enabled, args.provider)

    verbose, as_json = args.raw, args.json
    last = None
    _banner(mode)

    try:
        while True:
            try:
                line = input(PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue

            if line.startswith("/"):
                cmd = line[1:].split()[0].lower()
                if cmd in ("q", "quit", "exit"):
                    return 0
                if cmd == "stage":
                    _cmd_stage()
                elif cmd == "hits":
                    if last is None:
                        print("  (no answer yet)")
                    elif not last.evidence:
                        print("  (no evidence on the last answer)")
                    else:
                        for i, e in enumerate(last.evidence, 1):
                            print(f"  [{i}] {e['score']:>7.4f}  {e['kind']}/{e['topic']}  {e['chunk_id']}")
                            print(f"      {e['url']}")
                elif cmd == "raw":
                    verbose = not verbose
                    print(f"  retrieval trace: {'on' if verbose else 'off'}")
                elif cmd == "json":
                    as_json = not as_json
                    print(f"  json output: {'on' if as_json else 'off'}")
                else:
                    print(f"  unknown command {line!r} - try /stage /hits /raw /json /quit")
                print()
                continue

            shown, leaked = scrub_pii(line)
            t0 = time.perf_counter()
            try:
                ans = pipe.ask(line, debug=verbose)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {type(exc).__name__}: {exc}\n")
                continue
            wall = (time.perf_counter() - t0) * 1000
            last = ans

            if as_json:
                print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
            else:
                print()
                print(ans.render())
                print(RULE)
                note = f"  [{leaked} identifier type(s) redacted from the echo]" if leaked else ""
                print(f"  you> {shown}{note}")
                print(f"  kind={ans.kind} stage={ans.stage} "
                      f"guards={','.join(ans.guards) or 'none'} "
                      f"conf={ans.confidence} {wall:.0f}ms")
            print()
    finally:
        pipe.close()


if __name__ == "__main__":
    raise SystemExit(main())
