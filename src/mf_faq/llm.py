"""Optional LLM client (standard library only).

The assistant is fully functional with no API key: `synthesis.py` falls back
to a deterministic extractive composer.  When a key *is* present, this module
adds a natural-language rewrite pass over the retrieved context, with the same
citation and sentence constraints applied afterwards by the guardrail stage.

Providers: **groq** (primary; OpenAI-compatible REST), plus openai and
anthropic. Groq is checked first in `auto` mode, so `GROQ_API_KEY` in the
environment is all it takes to route there.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .config import CONFIG


class LLMUnavailable(RuntimeError):
    pass


#: Groq sits behind Cloudflare, which rejects the stdlib default
#: (`Python-urllib/3.x`) with a bare `403 error code: 1010` and no JSON body --
#: indistinguishable from a bad key. Sending a browser UA is what gets past it.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

#: Reasoning models emit tokens into a separate `reasoning` field *before* the
#: answer. Those tokens are billed against `max_tokens`, so a budget sized for
#: the answer alone is exhausted before a word is emitted: Groq answers with
#: `finish_reason="length"` and an empty `content`. With the 300-token budget
#: this used to send, `openai/gpt-oss-120b` spent all 300 on reasoning and
#: returned nothing -- which looked identical to a working LLM path falling
#: back, so the rewrite was silently dead on the default model.
REASONING_MODELS = ("gpt-oss", "deepseek-r1", "qwen-qwq", "qwen3", "llama-4-scout")

#: Budget for a plain completion. The system prompt caps the answer at three
#: short sentences, so this is generous.
ANSWER_MAX_TOKENS = 300
#: Budget for a reasoning completion: reasoning + answer. Measured on
#: gpt-oss-120b at `effort=low`, a single JSON answer costs ~80 reasoning
#: tokens, so 2048 leaves an order of magnitude of headroom.
REASONING_MAX_TOKENS = int(os.getenv("MF_FAQ_LLM_MAX_TOKENS", "2048"))
#: The extraction task is narrow, so reasoning past `low` buys nothing here.
REASONING_EFFORT = os.getenv("MF_FAQ_LLM_REASONING_EFFORT", "low")


def is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return any(tok in m for tok in REASONING_MODELS)


def provider() -> str:
    p = CONFIG.llm_provider
    if p != "auto":
        return p
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


#: Which env var holds the credential for each provider. `available()` uses this
#: so a *named* provider with a *blank* key reads as unavailable. It used to
#: return True on `provider() != "none"` alone, which advertised the LLM path
#: with no credential and then failed at request time into a silent extractive
#: fallback -- a broken key was indistinguishable from a working system.
_KEY_VARS = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY",
             "anthropic": "ANTHROPIC_API_KEY"}


def available() -> bool:
    if not CONFIG.llm_enabled:
        return False
    p = provider()
    key = _KEY_VARS.get(p)
    return bool(os.environ.get(key, "").strip()) if key else False


def missing() -> str:
    """Human-readable reason `available()` is False, or "" when it is True."""
    if not CONFIG.llm_enabled:
        return "MF_FAQ_LLM is not 1 (LLM rewrite pass disabled)"
    p = provider()
    if p == "none":
        return "no provider key set (need GROQ_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY)"
    var = _KEY_VARS.get(p, "")
    if not os.environ.get(var, "").strip():
        return f"provider is '{p}' but {var} is empty"
    return ""


def model_for(prov: str) -> str:
    """The configured model id for `prov`, or "" when there is no provider.

    The three providers keep their ids in three differently named fields
    (`groq_model`, `anthropic_model`, and the OpenAI-shaped `llm_model`), so
    anything that *displays* the active model has to pick the right field or it
    reports `gpt-4o-mini` while talking to Anthropic.
    """
    return {"groq": CONFIG.groq_model,
            "openai": CONFIG.llm_model,
            "anthropic": CONFIG.anthropic_model}.get(prov, "")


# ------------------------------------------------------------------ requests

def _post_openai_compatible(prompt: str, system: str, *, base_url: str,
                            key: str, model: str) -> str:
    """One OpenAI-chat-completions-shaped POST.

    Groq, OpenAI and Ollama all speak this dialect (Groq's endpoint is literally
    the OpenAI schema), so there is one implementation and the surrounding
    functions just supply base URL, key header and model. The temperature and
    the max_tokens budget are pinned here; the budget is per-model because a
    reasoning model has to pay for its own thinking out of the same allowance.
    """
    reasoning = is_reasoning_model(model)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": REASONING_MAX_TOKENS if reasoning else ANSWER_MAX_TOKENS,
    }
    if reasoning:
        body["reasoning_effort"] = REASONING_EFFORT
    req = urllib.request.Request(
        base_url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read())
    choice = data["choices"][0]
    content = (choice["message"].get("content") or "").strip()
    if not content:
        # Surfaced as LLMUnavailable so the caller falls back *and names the
        # cause*, instead of reporting an unparseable reply and leaving the
        # real fault (an exhausted budget) to be guessed at.
        spent = data.get("usage", {}).get("completion_tokens_details", {}) \
                       .get("reasoning_tokens", "?")
        raise LLMUnavailable(
            f"{model} returned an empty answer (finish_reason="
            f"{choice.get('finish_reason', '?')}, reasoning_tokens={spent}); "
            f"the max_tokens budget is too small for this model")
    return content


def _post_groq(prompt: str, system: str) -> str:
    return _post_openai_compatible(
        prompt, system,
        base_url=os.getenv("GROQ_BASE_URL",
                           "https://api.groq.com/openai/v1/chat/completions"),
        key=os.environ["GROQ_API_KEY"], model=CONFIG.groq_model)


def _post_openai(prompt: str, system: str) -> str:
    return _post_openai_compatible(
        prompt, system,
        base_url=os.getenv("OPENAI_BASE_URL",
                           "https://api.openai.com/v1/chat/completions"),
        key=os.environ["OPENAI_API_KEY"], model=CONFIG.llm_model)


def _post_anthropic(prompt: str, system: str) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    body = json.dumps({
        "model": CONFIG.anthropic_model,
        "max_tokens": 400,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages"),
        data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        })
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read())
    return "".join(b.get("text", "") for b in data.get("content", []))


def complete(prompt: str, system: str) -> str:
    p = provider()
    try:
        if p == "groq":
            return _post_groq(prompt, system)
        if p == "openai":
            return _post_openai(prompt, system)
        if p == "anthropic":
            return _post_anthropic(prompt, system)
    except urllib.error.HTTPError as exc:
        # A Cloudflare 403 has a plain-text body, not the provider's JSON error,
        # so the useful detail has to be read off the raw bytes. Without this the
        # only clue is "HTTP 403", which reads exactly like a rejected key.
        try:
            detail = exc.read().decode("utf-8", "replace").strip()[:200]
        except Exception:  # noqa: BLE001
            detail = ""
        if exc.code == 403 and "1010" in detail:
            detail = ("blocked by Cloudflare at the edge; a User-Agent header is "
                      "required for this endpoint")
        raise LLMUnavailable(
            f"{provider()} HTTP {exc.code}" + (f": {detail}" if detail else "")
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
    raise LLMUnavailable("no LLM provider configured")


# ------------------------------------------------------------------ prompts

SYSTEM = (
    "You are a mutual fund facts assistant for HDFC Mutual Fund schemes. "
    "You answer factual questions ONLY, using ONLY the supplied CONTEXT. "
    "Hard rules: (1) never recommend, rate, rank or compare schemes, and never "
    "suggest buying/selling/holding; (2) never state, compute or compare returns "
    "or performance - if asked, say the factsheet is the source; (3) keep the "
    "answer to at most 3 short sentences; (4) never ask for or repeat PAN, "
    "Aadhaar, folio/account numbers, OTPs, email addresses, phone numbers, or "
    "UPI/IFSC bank details; "
    "(5) copy the single best citation URL verbatim from CONTEXT and output "
    "nothing else. Reply with JSON: {\"answer\": \"...\", \"citation\": \"...\"}."
)
