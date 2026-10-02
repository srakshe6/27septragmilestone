#!/usr/bin/env python
"""Raw Groq REPL - ask a model anything, with no RAG, no guardrails, no index.

Answers the question "can Groq get this right?" directly, which the RAG path in
``chat.py`` cannot: that path only answers mutual-fund facts out of the built
corpus and refuses everything else by design.

    python groq_chat.py
    python groq_chat.py --temp 0.7            # --model defaults to the .env id
    python groq_chat.py --system "Answer only in one sentence."

    python groq_chat.py -q "What is 17*23?"   # one-shot, repeatable, no TTY
    python groq_chat.py -q "Q1?" -q "Q2?" --json

The API key and the model id come from ``.env`` (GROQ_API_KEY, GROQ_BASE_URL,
MF_FAQ_GROQ_MODEL) via ``mf_faq.config.load_env()``. ``.env.example`` is a
template and is never read. A value already exported in the real environment
wins over ``.env``, and ``--model`` wins over both for this run only.

Session commands:
    /model [id]   show or switch the model
    /models       list the ids this key can reach
    /temp [n]     show or set temperature (0 = deterministic)
    /effort [n]   show or set reasoning_effort (none/low/medium/high)
    /system [s]   show or replace the system prompt
    /history      print the conversation so far
    /clear        forget the conversation, keep the settings
    /json         toggle raw JSON output
    /quit         leave
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# A model answer can contain anything -- rupee signs, em dashes, narrow no-break
# spaces -- and a cp1252 console raises UnicodeEncodeError on the first one it
# cannot encode, killing the REPL mid-answer. Same guard as chat.py; it has to
# run before anything is printed.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from mf_faq import llm                                     # noqa: E402
from mf_faq.config import CONFIG, ENV_PATH                 # noqa: E402

RULE = "-" * 72
PROMPT = "you> "

#: `.env` is already in os.environ by this point -- the `CONFIG` import above runs
#: `load_env()` as a side effect, which is why these getenv calls see it.
BASE_URL = os.getenv("GROQ_BASE_URL",
                      "https://api.groq.com/openai/v1/chat/completions")
MODELS_URL = BASE_URL.rsplit("/chat/completions", 1)[0] + "/models"

DEFAULT_SYSTEM = ("You are a helpful assistant. Answer the question directly and "
                  "correctly. If you are unsure, say so instead of guessing.")

REASONING_MODELS = ("gpt-oss", "deepseek-r1", "qwen3", "llama4-scout")


def key() -> str:
    k = os.environ.get("GROQ_API_KEY", "").strip()
    if not k:
        sys.exit(f"GROQ_API_KEY is not set. Add it to {ENV_PATH} "
                 f"(that file is the only place this script reads a key from).")
    return k


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        BASE_URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key()}",
                 "User-Agent": llm._USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace").strip()[:300]
        except Exception:  # noqa: BLE001
            detail = ""
        if exc.code == 404 and "does not exist" in detail:
            detail = (f"{detail} -- the id is retired; run /models to see what "
                      f"this key can reach")
        sys.exit(f"HTTP {exc.code}: {detail}")


def list_models() -> list[str]:
    req = urllib.request.Request(
        MODELS_URL, method="GET",
        headers={"Authorization": f"Bearer {key()}",
                 "User-Agent": llm._USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        sys.exit(f"HTTP {exc.code}: could not list models")
    ids = sorted(m["id"] for m in data.get("data", []))
    return ids


def is_reasoning_model(model: str) -> bool:
    return any(tok in model.lower() for tok in REASONING_MODELS)


def banner(model: str, temp: float, effort: str, max_tokens: int) -> None:
    print(RULE)
    print(f"provider : groq   model: {model}")
    print(f"base url : {BASE_URL}")
    print(f"config   : {ENV_PATH}"
          + ("" if ENV_PATH.exists() else "  (MISSING - nothing is configured)"))
    print(f"api key  : present ({len(key())} chars, from {ENV_PATH.name})")
    print(f"temp     : {temp}   max_tokens: {max_tokens}   "
          f"reasoning_effort: {effort}")
    if is_reasoning_model(model):
        print(f"note     : {model} is a reasoning model - it spends tokens "
              f"before answering,\n           so an empty answer with "
              f"finish_reason=length means max_tokens was too small.")
    print(f"system   : {DEFAULT_SYSTEM[:60]}...")
    print("commands : /model /models /temp /effort /system /history /clear "
          "/json /quit")
    print(RULE)


def main() -> int:
    ap = argparse.ArgumentParser(prog="groq_chat.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=CONFIG.groq_model)
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--effort", choices=["none", "minimal", "low", "medium", "high"],
                    default="medium")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="raise this for long answers on a reasoning model")
    ap.add_argument("--no-history", action="store_true",
                    help="send each question with no prior turns")
    ap.add_argument("--json", action="store_true", help="start with JSON output on")
    ap.add_argument("-q", "--question", action="append", metavar="TEXT",
                    help="ask TEXT and exit; repeatable. Implies --no-history, "
                         "so it is also the scriptable form")
    args = ap.parse_args()

    if args.question:
        args.no_history = True

    model, system, temp = args.model, args.system, args.temp
    effort, max_tokens = args.effort, args.max_tokens
    history: list[dict] = []
    as_json = args.json
    # -q feeds the same code path as a typed line, so one-shot and interactive
    # cannot drift apart. The banner is suppressed in one-shot mode to keep the
    # output pipeable.
    one_shot = bool(args.question)
    pending = list(args.question or [])
    if not one_shot:
        banner(model, temp, effort, max_tokens)

    while True:
        if pending:
            line = pending.pop(0).strip()
            if not one_shot:
                print(f"{PROMPT}{line}")
        elif one_shot:
            return 0
        else:
            try:
                line = input(PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
        if not line:
            continue

        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
            if cmd in ("q", "quit", "exit"):
                return 0
            elif cmd == "model":
                model = arg.strip() or model
                print(f"  model: {model}"
                      + ("   (reasoning model)" if is_reasoning_model(model) else ""))
            elif cmd == "models":
                for mid in list_models():
                    print(f"  {mid}")
            elif cmd == "temp":
                try:
                    temp = float(arg) if arg.strip() else temp
                except ValueError:
                    print("  temp must be a number")
                else:
                    print(f"  temp: {temp}")
            elif cmd == "effort":
                if arg.strip():
                    if arg.strip() not in ("none", "minimal", "low", "medium", "high"):
                        print("  effort must be none|minimal|low|medium|high")
                    else:
                        effort = arg.strip()
                print(f"  reasoning_effort: {effort}")
            elif cmd == "system":
                system = arg.strip() or system
                print(f"  system: {system}")
            elif cmd == "history":
                for i, m in enumerate(history, 1):
                    body = m["content"].replace("\n", " ")[:90]
                    print(f"  [{i}] {m['role']:<9} {body}")
                if not history:
                    print("  (empty)")
            elif cmd == "clear":
                history = []
                print("  history cleared")
            elif cmd == "json":
                as_json = not as_json
                print(f"  json output: {'on' if as_json else 'off'}")
            else:
                print(f"  unknown command {line!r} - try /models /model /temp "
                      f"/effort /system /history /clear /json /quit")
            print()
            continue

        messages = [{"role": "system", "content": system}]
        if not args.no_history:
            messages += history
        messages.append({"role": "user", "content": line})

        payload = {"model": model, "messages": messages, "temperature": temp,
                   "max_tokens": max_tokens}
        if is_reasoning_model(model):
            payload["reasoning_effort"] = effort

        t0 = time.perf_counter()
        data = post(payload)
        wall = (time.perf_counter() - t0) * 1000

        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        finish = data["choices"][0].get("finish_reason", "?")
        usage = data.get("usage", {})

        if not args.no_history:
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": content})

        if as_json:
            print(json.dumps({"model": data.get("model"), "finish_reason": finish,
                              "content": content, "reasoning": reasoning,
                              "usage": usage, "latency_ms": round(wall)},
                             indent=2, ensure_ascii=False))
            print()
            continue

        print()
        if reasoning:
            print(f"  [reasoning] {reasoning[:400]}"
                  + ("..." if len(reasoning) > 400 else ""))
        print(content if content else "  <empty answer>")
        print(RULE)
        print(f"  kind={finish} model={data.get('model')} "
              f"tok={usage.get('total_tokens', '?')} "
              f"(reasoning {usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)}, "
              f"answer {usage.get('completion_tokens', '?') - usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)}) "
              f"{wall:.0f}ms")
        if finish == "length" and not content:
            print("  hint: raise the budget -- /quit then rerun with --max-tokens 4096")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
