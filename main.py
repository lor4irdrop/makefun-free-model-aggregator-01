#!/usr/bin/env python3
"""Free-model aggregator for make.fun free-model-aggregator-01.

One long-running server (Python 3.13 standard library only) that reads the
trial brief at $MAKEFUN_PROMPT_FILE, probes the ten providers' catalog
routes, registers its port and catalog with the env at $MAKEFUN_ENV_URL,
and answers the env client's OpenAI Chat Completions and Anthropic Messages
requests (JSON and SSE) with the best free model that meets each request.

Routing is deterministic: candidates are ranked by attempt bucket first
(fewest upstream attempts to that provider win, so the load spreads
instead of hammering one provider to exhaustion) and then by the sealed
quality table's rating_milli (highest first, ties by provider then model
id). A provider call that refuses (402/429/5xx) fails over to the next
candidate; the refusal's message tells whether the provider's budget is
spent or it is out (never tried again) or it was one scheduled fail call
(the provider stays eligible). Every choice is a function of the brief,
the requests and the providers' answers: no wall clock, no unseeded
randomness, so a replay reproduces the live pass byte for byte.

  python3 main.py --list    print the catalog document and exit 0
  python3 main.py           run the trial server (needs the brief + env)
"""

import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL = 0.1
PROBE_TIMEOUT = 5.0
WORK_TIMEOUT = 2.0
PROVIDER_TIMEOUT = 8.0
MAX_BODY_BYTES = 32 * 1024 * 1024
SSE_DELTA_CHARS = 13

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


# --------------------------------------------------------------------------
# Brief parsing: the three JSON documents embedded in the prompt file.
# --------------------------------------------------------------------------

def object_end(text, start):
    """Index of the '}' closing the object opening at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in `text` that carries a string `schema`."""
    out = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def read_brief(path):
    """The brief's directory, ratings table and workload manifest."""
    brief = {"providers": [], "ratings": [], "best": {}, "manifest": []}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief["error"] = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory = next((d for d in docs
                      if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        if not is_record(raw):
            continue
        provider_id = raw.get("id")
        wire = raw.get("wire")
        if not isinstance(provider_id, str) or not provider_id:
            continue
        if wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = raw.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if isinstance(raw.get(key), str) and raw[key]:
                base = raw[key]
                break
        brief["providers"].append({
            "id": provider_id,
            "name": raw.get("name") if isinstance(raw.get("name"), str) else provider_id,
            "wire": wire,
            "auth": auth,
            "base": base,
            "key_env": raw.get("key_env") if isinstance(raw.get("key_env"), str) else "",
            "key": raw.get("key") if isinstance(raw.get("key"), str) else "",
        })
    ratings_doc = next((d for d in docs
                        if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            if not is_record(raw):
                continue
            model = raw.get("model")
            provider = raw.get("provider")
            if not isinstance(model, str) or not model:
                continue
            if not isinstance(provider, str) or not provider:
                continue
            brief["ratings"].append({
                "model": model,
                "provider": provider,
                "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                "context_length": raw.get("context_length") if isinstance(raw.get("context_length"), (int, float)) else None,
                "tools": raw.get("tools") is True,
                "vision": raw.get("vision") is True,
                "json": raw.get("json") is True,
                "free": raw.get("free") is True,
            })
        if isinstance(ratings_doc.get("best"), list):
            for raw in ratings_doc["best"]:
                if is_record(raw) and isinstance(raw.get("id"), str):
                    brief["best"][raw["id"]] = raw.get("rating_milli")
    workload_doc = next((d for d in docs
                         if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            if not is_record(raw):
                continue
            request_id = raw.get("id")
            if not isinstance(request_id, str) or not request_id:
                continue
            requires = raw.get("requires") if is_record(raw.get("requires")) else {}
            entry = {
                "id": request_id,
                "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                "stream": raw.get("stream") is True,
                "weight": 2 if raw.get("weight") == 2 else 1,
                "requires": {
                    "tools": requires.get("tools") is True,
                    "vision": requires.get("vision") is True,
                    "json": requires.get("json") is True,
                },
            }
            if isinstance(requires.get("min_context"), (int, float)):
                entry["requires"]["min_context"] = requires["min_context"]
            brief["manifest"].append(entry)
    brief["error"] = None
    return brief


# --------------------------------------------------------------------------
# Provider catalog probing.
# --------------------------------------------------------------------------

def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def join_base(env_url, base):
    text = base.strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text.rstrip("/")
    if not text:
        return env_url
    return env_url + ("" if text.startswith("/") else "/") + text.rstrip("/")


def http_json(method, url, headers, body=None, timeout=PROBE_TIMEOUT):
    """One HTTP call; returns (status, parsed JSON or None, raw bytes)."""
    data = None
    request_headers = dict(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers["content-type"] = "application/json"
    try:
        request = urllib.request.Request(url, data=data, method=method, headers=request_headers)
    except ValueError:
        return None, None, b""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as err:
        raw = err.read()
        status = err.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, None, b""
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        parsed = None
    return status, parsed, raw


def model_from_listing(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        name = raw.get("name")
        model_id = name[len("models/"):] if isinstance(name, str) and name.startswith("models/") else name
    else:
        model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    context = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        if isinstance(raw.get(key), (int, float)):
            context = raw[key]
            break
    display = raw.get("display_name") or raw.get("displayName") or model_id
    return {
        "id": model_id,
        "display_name": display if isinstance(display, str) else model_id,
        "free": raw.get("free") if isinstance(raw.get("free"), bool) else None,
        "context_length": context,
        "tools": raw.get("tools") if isinstance(raw.get("tools"), bool) else None,
        "vision": raw.get("vision") if isinstance(raw.get("vision"), bool) else None,
        "json": raw.get("json") if isinstance(raw.get("json"), bool) else None,
    }


def probe_provider(env_url, provider):
    """GET one provider's catalog route; returns (ok, [models])."""
    if not env_url:
        return False, []
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    status, parsed, _raw = http_json("GET", url, headers, timeout=PROBE_TIMEOUT)
    if status is None or status < 200 or status >= 300 or not is_record(parsed):
        return False, []
    items = parsed.get("models") if provider["wire"] == "gemini" else parsed.get("data")
    if not isinstance(items, list):
        return False, []
    models = []
    for raw in items:
        model = model_from_listing(provider["wire"], raw)
        if model is not None:
            models.append(model)
    return True, models


def build_catalog(brief, env_url):
    """The one catalog, plus a capability map the router uses.

    Returns (catalog, caps) where caps[(provider_id, model_id)] carries the
    listing's tools/vision/json flags (the catalog document itself holds
    only id/free/context_length, as the rules spell it).
    """
    ratings = {(r["provider"], r["model"]): r for r in brief["ratings"]}
    providers = []
    caps = {}
    for provider in brief["providers"]:
        ok, listed = probe_provider(env_url, provider)
        if not ok:
            log("catalog probe failed for %s; using the brief's table" % provider["id"])
        by_id = {}
        for model in listed:
            by_id[model["id"]] = model
        for row in brief["ratings"]:
            if row["provider"] != provider["id"]:
                continue
            if row["model"] not in by_id:
                by_id[row["model"]] = {
                    "id": row["model"], "display_name": row["model"],
                    "free": row["free"], "context_length": row["context_length"],
                    "tools": row["tools"], "vision": row["vision"], "json": row["json"],
                }
        models = []
        for model_id in sorted(by_id):
            model = by_id[model_id]
            row = ratings.get((provider["id"], model_id))
            free = row["free"] if row is not None else (model["free"] if model["free"] is not None else False)
            context = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model_id, "free": free, "context_length": context})
            caps[(provider["id"], model_id)] = {
                "tools": row["tools"] if row is not None else model["tools"],
                "vision": row["vision"] if row is not None else model["vision"],
                "json": row["json"] if row is not None else model["json"],
                "context_length": context,
                "free": free,
                "rating_milli": row["rating_milli"] if row is not None else 0,
            }
        providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": models,
        })
    return {"providers": providers}, caps


# --------------------------------------------------------------------------
# Routing: rank free candidates by rating, fail over on refusal.
# --------------------------------------------------------------------------

def capabilities_of(caps_map, provider_id, model_id):
    """What a model can do: the sealed table first, the listing second."""
    caps = caps_map.get((provider_id, model_id))
    if caps is None:
        return {
            "context_length": None, "tools": None, "vision": None,
            "json": None, "free": False, "rating_milli": 0,
        }
    return caps


def meets_requirements(requires, caps):
    if requires.get("tools") and caps["tools"] is not True:
        return False
    if requires.get("vision") and caps["vision"] is not True:
        return False
    if requires.get("json") and caps["json"] is not True:
        return False
    minimum = requires.get("min_context")
    if minimum is not None and (caps["context_length"] is None or caps["context_length"] < minimum):
        return False
    return True


class Router:
    """Rank candidates by rating; fail over to the next one on refusal.

    A provider that refuses twice in a row is most likely out of budget or
    in outage (a single refusal is usually its scheduled fail call, and
    the next call to it still answers), so it is demoted to the back of the
    ranking until it answers again; it is never dropped outright.

    Calls are also spread across providers: candidates are ranked by
    attempt bucket first (fewest upstream attempts to that provider win),
    so no provider is hammered to exhaustion while others sit idle.
    The first call to any provider always answers (a provider's failure
    schedule starts at call 2), so spreading the first wave over all ten
    providers costs no failed calls. A provider whose refusal says its
    budget is spent, or that it is out, is dead for the rest of the trial
    and never tried again; a plain refusal is one scheduled fail call and
    the provider stays eligible.
    """

    DEMOTE_AFTER = 2
    ATTEMPT_BUCKET = 3
    RATING_GATE = 45

    def __init__(self, brief, catalog, caps):
        self.brief = brief
        self.catalog = catalog
        self.caps = caps
        self.by_provider = {p["id"]: p for p in brief["providers"]}
        self.consecutive_failures = {}
        self.attempts = {}
        self.dead = set()

    def note_attempt(self, provider_id):
        self.attempts[provider_id] = self.attempts.get(provider_id, 0) + 1

    def mark_dead(self, provider_id):
        self.dead.add(provider_id)

    def note_success(self, provider_id):
        self.consecutive_failures[provider_id] = 0

    def note_failure(self, provider_id):
        self.consecutive_failures[provider_id] = self.consecutive_failures.get(provider_id, 0) + 1

    def _demoted(self, provider_id):
        return self.consecutive_failures.get(provider_id, 0) >= self.DEMOTE_AFTER

    def candidates(self, requires, client_api, json_mode):
        out = []
        for entry in self.catalog["providers"]:
            provider = self.by_provider.get(entry["id"])
            if provider is None:
                continue
            if entry["id"] in self.dead:
                continue
            for model in entry["models"]:
                caps = capabilities_of(self.caps, entry["id"], model["id"])
                if not caps["free"]:
                    continue
                if not meets_requirements(requires, caps):
                    continue
                # An Anthropic-wire provider cannot carry an OpenAI JSON-mode
                # request: it would echo json_mode absent and fail the check.
                if json_mode and client_api == "openai" and provider["wire"] == "anthropic":
                    continue
                out.append({
                    "provider": provider, "model_id": model["id"],
                    "rating": caps["rating_milli"],
                })
        if not out:
            # Every provider is dead (budgets spent / outages): fall back to
            # the full list so a request still gets an attempted answer
            # instead of an immediate 503.
            for entry in self.catalog["providers"]:
                provider = self.by_provider.get(entry["id"])
                if provider is None:
                    continue
                for model in entry["models"]:
                    caps = capabilities_of(self.caps, entry["id"], model["id"])
                    if not caps["free"]:
                        continue
                    if not meets_requirements(requires, caps):
                        continue
                    if json_mode and client_api == "openai" and provider["wire"] == "anthropic":
                        continue
                    out.append({
                        "provider": provider, "model_id": model["id"],
                        "rating": caps["rating_milli"],
                    })
        out.sort(key=lambda c: (1 if self._demoted(c["provider"]["id"]) else 0,
                                self.attempts.get(c["provider"]["id"], 0) // self.ATTEMPT_BUCKET,
                                -c["rating"], c["provider"]["id"], c["model_id"]))
        if out:
            # Do not trade more than RATING_GATE rating points to save a
            # call: the cheapest providers are far weaker, and a quality
            # of 98 instead of 99 leaves no margin at all in the final
            # trial. The best candidate always passes its own gate.
            best_rating = max(c["rating"] for c in out)
            near = [c for c in out if c["rating"] >= best_rating - self.RATING_GATE]
            if near:
                out = near
        return out


# --------------------------------------------------------------------------
# Request normalisation: one parsed form for all three upstream wires.
# --------------------------------------------------------------------------

def text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def parse_data_url(url):
    """A data: URL as (media_type, base64 data), else (None, None)."""
    if not isinstance(url, str):
        return None, None
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", url, re.DOTALL)
    if match is None:
        return None, None
    media = match.group(1) or "image/png"
    return media, match.group(3)


def collect_images(content):
    """Image parts of a message content: [{media_type, data} | {url}]."""
    images = []
    if not isinstance(content, list):
        return images
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "image_url" and is_record(block.get("image_url")):
            media, data = parse_data_url(block["image_url"].get("url"))
            if data is not None:
                images.append({"media_type": media, "data": data})
            elif isinstance(block["image_url"].get("url"), str):
                images.append({"url": block["image_url"]["url"]})
        elif block.get("type") == "image" and is_record(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                images.append({
                    "media_type": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png",
                    "data": source["data"],
                })
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                images.append({"url": source["url"]})
    return images


class Normalized:
    """A client request parsed into wire-neutral fields."""

    def __init__(self, api, body):
        self.api = api
        self.stream = body.get("stream") is True
        self.max_tokens = None
        for key in ("max_tokens", "max_completion_tokens"):
            if isinstance(body.get(key), (int, float)):
                self.max_tokens = body[key]
                break
        if self.max_tokens is None:
            self.max_tokens = 1024
        self.temperature = body.get("temperature") if isinstance(body.get("temperature"), (int, float)) else None
        self.top_p = body.get("top_p") if isinstance(body.get("top_p"), (int, float)) else None
        self.json_mode = False
        if api == "openai" and is_record(body.get("response_format")):
            self.json_mode = body["response_format"].get("type") in ("json_object", "json_schema")
        self.stops = []
        if api == "openai":
            stop = body.get("stop")
            if isinstance(stop, str):
                self.stops = [stop]
            elif isinstance(stop, list):
                self.stops = [s for s in stop if isinstance(s, str)]
        else:
            stops = body.get("stop_sequences")
            if isinstance(stops, list):
                self.stops = [s for s in stops if isinstance(s, str)]
        self.system = ""
        self.messages = []
        self.tools = []
        self.tool_choice = body.get("tool_choice")
        if api == "openai":
            for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
                if not is_record(message):
                    continue
                role = message.get("role")
                if role == "system":
                    text = text_of(message.get("content"))
                    if text:
                        self.system = (self.system + "\n" + text) if self.system else text
                else:
                    self.messages.append(message)
            for raw in body.get("tools") if isinstance(body.get("tools"), list) else []:
                if is_record(raw) and is_record(raw.get("function")) and isinstance(raw["function"].get("name"), str):
                    fn = raw["function"]
                    self.tools.append({
                        "name": fn["name"],
                        "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                        "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
                    })
        else:
            self.system = text_of(body.get("system"))
            for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
                if is_record(message):
                    self.messages.append(message)
            for raw in body.get("tools") if isinstance(body.get("tools"), list) else []:
                if is_record(raw) and isinstance(raw.get("name"), str):
                    self.tools.append({
                        "name": raw["name"],
                        "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                        "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                    })

    def has_vision(self):
        for message in self.messages:
            if collect_images(message.get("content")):
                return True
        return False


# --------------------------------------------------------------------------
# Upstream request builders: one per provider wire.
# --------------------------------------------------------------------------

def openai_messages(normalized):
    """The normalized request as OpenAI Chat Completions messages."""
    messages = []
    if normalized.system:
        messages.append({"role": "system", "content": normalized.system})
    for message in normalized.messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            text = text_of(content)
            entry = {"role": "assistant", "content": text if text else None}
            calls = []
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "tool_use":
                        calls.append({
                            "id": block.get("id") if isinstance(block.get("id"), str) else "",
                            "type": "function",
                            "function": {
                                "name": block.get("name") if isinstance(block.get("name"), str) else "",
                                "arguments": json.dumps(block.get("input") if is_record(block.get("input")) else {}),
                            },
                        })
            if calls:
                entry["tool_calls"] = calls
            messages.append(entry)
            continue
        if role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": message.get("tool_call_id", ""),
                "content": text_of(content),
            })
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        parts = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "tool_result":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                        "content": text_of(block.get("content")),
                    })
                elif block.get("type") in ("image", "image_url"):
                    for image in collect_images([block]):
                        if "data" in image:
                            parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (image["media_type"], image["data"])}})
                        else:
                            parts.append({"type": "image_url", "image_url": {"url": image["url"]}})
        if parts:
            messages.append({"role": "user", "content": parts})
    return messages


def build_openai_body(normalized, model_id):
    body = {"model": model_id, "messages": openai_messages(normalized)}
    if normalized.max_tokens is not None:
        body["max_tokens"] = normalized.max_tokens
    if normalized.temperature is not None:
        body["temperature"] = normalized.temperature
    if normalized.top_p is not None:
        body["top_p"] = normalized.top_p
    if normalized.stops:
        body["stop"] = normalized.stops
    if normalized.tools:
        body["tools"] = [{
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
        } for t in normalized.tools]
        choice = normalized.tool_choice
        if isinstance(choice, str):
            body["tool_choice"] = choice
        elif is_record(choice):
            if choice.get("type") == "auto":
                body["tool_choice"] = "auto"
            elif choice.get("type") == "any":
                body["tool_choice"] = "required"
            elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
                body["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if normalized.json_mode:
        body["response_format"] = {"type": "json_object"}
    if normalized.stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def build_anthropic_body(normalized, model_id):
    messages = []
    for message in normalized.messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text:
                blocks.append({"type": "text", "text": text})
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "tool_use":
                        blocks.append({
                            "type": "tool_use",
                            "id": block.get("id") if isinstance(block.get("id"), str) else "",
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "input": block.get("input") if is_record(block.get("input")) else {},
                        })
            messages.append({"role": "assistant", "content": blocks})
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        parts = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "tool_result":
                    parts.append({
                        "type": "tool_result",
                        "tool_use_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                        "content": text_of(block.get("content")),
                    })
                elif block.get("type") in ("image", "image_url"):
                    for image in collect_images([block]):
                        if "data" in image:
                            parts.append({"type": "image", "source": {"type": "base64", "media_type": image["media_type"], "data": image["data"]}})
                        else:
                            parts.append({"type": "image", "source": {"type": "url", "url": image["url"]}})
        if parts:
            messages.append({"role": "user", "content": parts})
    body = {"model": model_id, "messages": messages}
    if normalized.system:
        body["system"] = normalized.system
    if normalized.max_tokens is not None:
        body["max_tokens"] = normalized.max_tokens
    if normalized.temperature is not None:
        body["temperature"] = normalized.temperature
    if normalized.top_p is not None:
        body["top_p"] = normalized.top_p
    if normalized.stops:
        body["stop_sequences"] = normalized.stops
    if normalized.tools:
        body["tools"] = [{
            "name": t["name"], "description": t["description"], "input_schema": t["parameters"],
        } for t in normalized.tools]
        choice = normalized.tool_choice
        if isinstance(choice, str):
            if choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif choice == "none":
                body["tool_choice"] = {"type": "none"}
        elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
            body["tool_choice"] = {"type": "tool", "name": choice["function"].get("name", "")}
    if normalized.stream:
        body["stream"] = True
    return body


def build_gemini_body(normalized):
    contents = []
    for message in normalized.messages:
        role = "model" if message.get("role") == "assistant" else "user"
        parts = []
        content = message.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "tool_use":
                    parts.append({"text": ""})
                elif block.get("type") in ("image", "image_url"):
                    for image in collect_images([block]):
                        if "data" in image:
                            parts.append({"inlineData": {"mimeType": image["media_type"], "data": image["data"]}})
                        else:
                            parts.append({"fileData": {"fileUri": image["url"]}})
        if parts:
            contents.append({"role": role, "parts": parts})
    body = {"contents": contents}
    if normalized.system:
        body["systemInstruction"] = {"parts": [{"text": normalized.system}]}
    if normalized.tools:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"], "parameters": t["parameters"],
        } for t in normalized.tools]}]
    config = {}
    if normalized.max_tokens is not None:
        config["maxOutputTokens"] = normalized.max_tokens
    if normalized.stops:
        config["stopSequences"] = normalized.stops
    if normalized.temperature is not None:
        config["temperature"] = normalized.temperature
    if normalized.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# --------------------------------------------------------------------------
# Upstream answers: parse every provider wire into (text, usage, stop).
# --------------------------------------------------------------------------

def parse_sse_frames(text):
    """An SSE body as [(event, data)] frames, in order."""
    frames = []
    normalized_text = text.replace("\r\n", "\n")
    for frame in normalized_text.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def split_text(text):
    pieces = [text[i:i + SSE_DELTA_CHARS] for i in range(0, len(text), SSE_DELTA_CHARS)]
    return pieces if pieces else [""]


def parse_openai_answer(status, raw):
    """An OpenAI-wire provider answer: (text, usage, stop_family) or None."""
    if status < 200 or status >= 300:
        return None
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("data:") or "data:" in text[:200]:
        answer_text = ""
        usage = None
        stop = None
        for _event, data in parse_sse_frames(text):
            if data.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                return None
            if not is_record(chunk):
                return None
            if is_record(chunk.get("usage")):
                usage = {
                    "input": chunk["usage"].get("prompt_tokens"),
                    "output": chunk["usage"].get("completion_tokens"),
                }
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not is_record(choice):
                return None
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            if isinstance(delta.get("content"), str):
                answer_text += delta["content"]
            if isinstance(choice.get("finish_reason"), str):
                stop = choice["finish_reason"]
        if stop is None or usage is None:
            return None
        return answer_text, usage, STOP_FAMILIES.get(stop)
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not is_record(value):
        return None
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not is_record(choices[0]):
        return None
    choice = choices[0]
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    if isinstance(content, str):
        answer_text = content
    else:
        answer_text = text_of(content)
    usage_raw = value.get("usage") if is_record(value.get("usage")) else {}
    usage = {"input": usage_raw.get("prompt_tokens"), "output": usage_raw.get("completion_tokens")}
    stop = choice.get("finish_reason")
    if not isinstance(stop, str):
        return None
    return answer_text, usage, STOP_FAMILIES.get(stop)


def parse_anthropic_answer(status, raw):
    """An Anthropic-wire provider answer: (text, usage, stop_family) or None."""
    if status < 200 or status >= 300:
        return None
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("event:"):
        answer_text = ""
        input_tokens = None
        output_tokens = None
        stop = None
        for event, data in parse_sse_frames(text):
            try:
                payload = json.loads(data)
            except ValueError:
                return None
            if not is_record(payload):
                return None
            if event == "message_start" and is_record(payload.get("message")) and is_record(payload["message"].get("usage")):
                input_tokens = payload["message"]["usage"].get("input_tokens")
            elif event == "content_block_delta" and is_record(payload.get("delta")):
                delta = payload["delta"]
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    answer_text += delta["text"]
            elif event == "message_delta":
                if is_record(payload.get("delta")) and isinstance(payload["delta"].get("stop_reason"), str):
                    stop = payload["delta"]["stop_reason"]
                if is_record(payload.get("usage")):
                    output_tokens = payload["usage"].get("output_tokens")
        if stop is None or input_tokens is None or output_tokens is None:
            return None
        return answer_text, {"input": input_tokens, "output": output_tokens}, STOP_FAMILIES.get(stop)
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not is_record(value) or value.get("type") != "message":
        return None
    if not isinstance(value.get("content"), list):
        return None
    stop = value.get("stop_reason")
    usage_raw = value.get("usage") if is_record(value.get("usage")) else {}
    if not isinstance(stop, str):
        return None
    return (text_of(value["content"]),
            {"input": usage_raw.get("input_tokens"), "output": usage_raw.get("output_tokens")},
            STOP_FAMILIES.get(stop))


def parse_gemini_answer(status, raw):
    """A Gemini-wire provider answer: (text, usage, stop_family) or None."""
    if status < 200 or status >= 300:
        return None
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("data:") or "data:" in text[:200]:
        answer_text = ""
        usage = None
        stop = None
        for _event, data in parse_sse_frames(text):
            try:
                frame = json.loads(data)
            except ValueError:
                return None
            if not is_record(frame):
                return None
            candidates = frame.get("candidates")
            if isinstance(candidates, list) and candidates and is_record(candidates[0]):
                candidate = candidates[0]
                content = candidate.get("content") if is_record(candidate.get("content")) else {}
                parts = content.get("parts") if isinstance(content.get("parts"), list) else []
                for part in parts:
                    if is_record(part) and isinstance(part.get("text"), str):
                        answer_text += part["text"]
                if isinstance(candidate.get("finishReason"), str):
                    stop = candidate["finishReason"]
            meta = frame.get("usageMetadata") if is_record(frame.get("usageMetadata")) else None
            if meta is not None:
                usage = {"input": meta.get("promptTokenCount"), "output": meta.get("candidatesTokenCount")}
        if stop is None or usage is None:
            return None
        return answer_text, usage, STOP_FAMILIES.get(stop)
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not is_record(value):
        return None
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates or not is_record(candidates[0]):
        return None
    candidate = candidates[0]
    content = candidate.get("content") if is_record(candidate.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    answer_text = ""
    for part in parts:
        if is_record(part) and isinstance(part.get("text"), str):
            answer_text += part["text"]
    meta = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else {}
    stop = candidate.get("finishReason")
    if not isinstance(stop, str):
        return None
    return (answer_text,
            {"input": meta.get("promptTokenCount"), "output": meta.get("candidatesTokenCount")},
            STOP_FAMILIES.get(stop))


PARSERS = {
    "openai": parse_openai_answer,
    "anthropic": parse_anthropic_answer,
    "gemini": parse_gemini_answer,
}


def refusal_reason(parsed):
    """Classify a provider refusal from its wire error envelope.

    The env's refusal message names the cause: a spent free-tier budget
    or a begun outage is permanent for the trial, a scheduled fail call
    is transient, and a concurrency refusal spends no budget at all.
    Returns one of "spent", "outage", "fail", "concurrent", "other".
    """
    message = ""
    if is_record(parsed):
        error = parsed.get("error")
        if is_record(error) and isinstance(error.get("message"), str):
            message = error["message"]
        elif isinstance(error, str):
            message = error
        elif isinstance(parsed.get("message"), str):
            message = parsed["message"]
    if "budget is spent" in message:
        return "spent"
    if message.endswith(" is out"):
        return "outage"
    if "refused this call" in message:
        return "fail"
    if "at a time" in message:
        return "concurrent"
    return "other"


def call_provider(env_url, provider, model_id, normalized):
    """One upstream call; returns (parsed answer or None, refusal reason).

    The reason is None on a usable answer, else one of "spent", "outage",
    "fail", "concurrent", "other" (see refusal_reason).
    """
    base = join_base(env_url, provider["base"])
    wire = provider["wire"]
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if wire == "openai":
        url = base + "/v1/chat/completions"
        body = build_openai_body(normalized, model_id)
    elif wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        body = build_anthropic_body(normalized, model_id)
    else:
        if normalized.stream:
            url = base + "/v1beta/models/" + model_id + ":streamGenerateContent?alt=sse"
        else:
            url = base + "/v1beta/models/" + model_id + ":generateContent"
        body = build_gemini_body(normalized)
    if normalized.stream:
        headers["accept"] = "text/event-stream"
    _status, _parsed, raw = http_json("POST", url, headers, body=body, timeout=PROVIDER_TIMEOUT)
    if _status is None:
        return None, "other"
    if _status < 200 or _status >= 300:
        return None, refusal_reason(_parsed)
    answer = PARSERS[wire](_status, raw)
    if answer is None:
        return None, "other"
    return answer, None


# --------------------------------------------------------------------------
# Client answers: render the provider's answer in the client's wire.
# --------------------------------------------------------------------------

def stop_for_client(family, api):
    if api == "openai":
        return {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}.get(family, "stop")
    return {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}.get(family, "end_turn")


def render_openai_json(model_id, answer_text, usage, family):
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer_text},
            "finish_reason": stop_for_client(family, "openai"),
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": usage["input"],
            "completion_tokens": usage["output"],
            "total_tokens": (usage["input"] or 0) + (usage["output"] or 0),
        },
    }


def render_openai_sse(model_id, answer_text, usage, family):
    def chunk(delta, finish):
        return {"id": "chatcmpl-sim", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames = ["data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None))]
    for piece in split_text(answer_text):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, stop_for_client(family, "openai"))))
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk",
        "created": FIXED_CREATED, "model": model_id, "choices": [],
        "usage": {
            "prompt_tokens": usage["input"],
            "completion_tokens": usage["output"],
            "total_tokens": (usage["input"] or 0) + (usage["output"] or 0),
        },
    }))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(model_id, answer_text, usage, family):
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": [{"type": "text", "text": answer_text}],
        "stop_reason": stop_for_client(family, "anthropic"),
        "stop_sequence": None,
        "usage": {"input_tokens": usage["input"], "output_tokens": usage["output"]},
    }


def render_anthropic_sse(model_id, answer_text, usage, family):
    def event(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(data))
    frames = [event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_sim", "type": "message", "role": "assistant", "model": model_id,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": usage["input"], "output_tokens": 0},
        },
    })]
    frames.append(event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in split_text(answer_text):
        frames.append(event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_for_client(family, "anthropic"), "stop_sequence": None},
        "usage": {"output_tokens": usage["output"]},
    }))
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_error(message, error_type="invalid_request_error", code=None):
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


# --------------------------------------------------------------------------
# The server: routes, register, /work polling.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, catalog, caps, env_url):
        self.brief = brief
        self.catalog = catalog
        self.env_url = env_url
        self.router = Router(brief, catalog, caps)
        self.manifest_by_id = {m["id"]: m for m in brief["manifest"]}
        self.lock = threading.Lock()
        self.arrival_index = 0
        self.done_event = threading.Event()

    def free_model_ids(self):
        seen = set()
        ids = []
        for entry in self.catalog["providers"]:
            for model in entry["models"]:
                if model["free"] and model["id"] not in seen:
                    seen.add(model["id"])
                    ids.append(model["id"])
        return sorted(ids)

    def work_next_id(self):
        status, parsed, _raw = http_json("GET", self.env_url + "/work", {}, timeout=WORK_TIMEOUT)
        if status is None or status != 200 or not is_record(parsed):
            return None
        if parsed.get("done") is True:
            return None
        next_id = parsed.get("next")
        return next_id if isinstance(next_id, str) else None

    def requirements_for(self, body):
        """The manifest row of the request now arriving, else a body guess."""
        with self.lock:
            index = self.arrival_index
            self.arrival_index += 1
        manifest = self.brief["manifest"]
        next_id = self.work_next_id()
        if next_id is not None and next_id in self.manifest_by_id:
            if index < len(manifest) and manifest[index]["id"] != next_id:
                log("arrival %d is %s but /work names %s" % (index, manifest[index]["id"], next_id))
            return self.manifest_by_id[next_id]["requires"]
        if index < len(manifest):
            return manifest[index]["requires"]
        normalized = Normalized("openai", body)
        requires = {
            "tools": bool(normalized.tools),
            "vision": normalized.has_vision(),
            "json": normalized.json_mode,
        }
        return requires

    def serve(self, api, body):
        """Answer one client request: (status, content_type, payload)."""
        normalized = Normalized(api, body)
        requires = self.requirements_for(body)
        candidates = self.router.candidates(requires, api, normalized.json_mode)
        if not candidates:
            if api == "anthropic":
                return 503, "application/json", json.dumps(anthropic_error("overloaded_error", "no free model meets this request"))
            return 503, "application/json", json.dumps(openai_error("no free model meets this request", "service_unavailable_error"))
        for candidate in candidates:
            provider = candidate["provider"]
            answer, reason = call_provider(self.env_url, provider, candidate["model_id"], normalized)
            if answer is None:
                if reason in ("spent", "outage"):
                    # A spent budget or a begun outage is permanent: the
                    # provider can never answer again this trial.
                    self.router.note_attempt(provider["id"])
                    self.router.mark_dead(provider["id"])
                elif reason == "concurrent":
                    # Refused while another call was in flight: no budget
                    # was spent, so this is not an attempt to count.
                    pass
                else:
                    # A scheduled fail call (or an unreadable answer):
                    # the call number was spent, the provider stays alive.
                    self.router.note_attempt(provider["id"])
                    self.router.note_failure(provider["id"])
                continue
            answer_text, usage, family = answer
            self.router.note_attempt(provider["id"])
            if usage["input"] is None or usage["output"] is None or family is None:
                self.router.note_failure(provider["id"])
                continue
            self.router.note_success(provider["id"])
            if api == "openai":
                if normalized.stream:
                    return 200, "text/event-stream; charset=utf-8", render_openai_sse(candidate["model_id"], answer_text, usage, family)
                return 200, "application/json", json.dumps(render_openai_json(candidate["model_id"], answer_text, usage, family))
            if normalized.stream:
                return 200, "text/event-stream; charset=utf-8", render_anthropic_sse(candidate["model_id"], answer_text, usage, family)
            return 200, "application/json", json.dumps(render_anthropic_json(candidate["model_id"], answer_text, usage, family))
        if api == "anthropic":
            return 502, "application/json", json.dumps(anthropic_error("api_error", "every provider refused this request"))
        return 502, "application/json", json.dumps(openai_error("every provider refused this request", "api_error"))


class Handler(BaseHTTPRequestHandler):
    aggregator = None
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return value if is_record(value) else None

    def _send(self, status, content_type, payload):
        data = payload.encode("utf-8") if isinstance(payload, str) else payload
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            ids = self.aggregator.free_model_ids()
            if isinstance(self.headers.get("x-api-key"), str):
                self._send(200, "application/json", json.dumps({
                    "data": [{"type": "model", "id": model_id, "display_name": model_id,
                              "created_at": FIXED_ANTHROPIC_CREATED_AT} for model_id in ids],
                    "has_more": False,
                }))
            else:
                self._send(200, "application/json", json.dumps({
                    "object": "list",
                    "data": [{"id": model_id, "object": "model", "created": FIXED_CREATED,
                              "owned_by": "aggregator"} for model_id in ids],
                }))
            return
        self._send(404, "application/json", json.dumps(openai_error("no route GET %s" % self.path, "not_found_error")))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._send(404, "application/json", json.dumps(openai_error("no route POST %s" % path, "not_found_error")))
            return
        body = self._read_json()
        if body is None:
            if api == "anthropic":
                self._send(400, "application/json", json.dumps(anthropic_error("invalid_request_error", "the request body is not one JSON object")))
            else:
                self._send(400, "application/json", json.dumps(openai_error("the request body is not one JSON object")))
            return
        try:
            status, content_type, payload = self.aggregator.serve(api, body)
        except Exception as err:  # never leave the client hanging
            log("serve failed: %s" % err)
            if api == "anthropic":
                status, content_type, payload = 500, "application/json", json.dumps(anthropic_error("api_error", "the server failed to answer"))
            else:
                status, content_type, payload = 500, "application/json", json.dumps(openai_error("the server failed to answer", "api_error"))
        self._send(status, content_type, payload)


def register(env_url, port, catalog):
    status, parsed, _raw = http_json("POST", env_url + "/register", {},
                                     body={"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                                     timeout=PROBE_TIMEOUT)
    if status is None or status < 200 or status >= 300:
        raise RuntimeError("POST %s/register answered HTTP %s" % (env_url, status))
    if is_record(parsed) and isinstance(parsed.get("requests"), int):
        log("registered: the workload holds %d requests" % parsed["requests"])


def poll_work(aggregator, server):
    while not aggregator.done_event.is_set():
        status, parsed, _raw = http_json("GET", aggregator.env_url + "/work", {}, timeout=WORK_TIMEOUT)
        if status == 200 and is_record(parsed) and parsed.get("done") is True:
            aggregator.done_event.set()
            break
        aggregator.done_event.wait(POLL_INTERVAL)


def main(argv):
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    brief = read_brief(prompt_file)
    if brief.get("error"):
        log(brief["error"])
    if brief["providers"]:
        catalog, caps = build_catalog(brief, env_url)
    else:
        catalog, caps = {"providers": []}, {}
    if "--list" in argv:
        sys.stdout.write(json.dumps(catalog) + "\n")
        sys.stdout.flush()
        return 0
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    aggregator = Aggregator(brief, catalog, caps, env_url)
    Handler.aggregator = aggregator
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    address = server.server_address
    port = address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d providers" % (port, len(catalog["providers"])))
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    register(env_url, port, catalog)
    poller = threading.Thread(target=poll_work, args=(aggregator, server), daemon=True)
    poller.start()
    try:
        aggregator.done_event.wait()
    except KeyboardInterrupt:
        pass
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
