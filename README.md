# Free-model aggregator

A server (Python 3.13 standard library only) for the make.fun
`free-model-aggregator-01` bounty. It reads the trial brief, probes the ten
free-tier providers' catalog routes, registers its port and catalog with the
trial env, and answers the env client's OpenAI Chat Completions and
Anthropic Messages requests (JSON and SSE) with the best free model that
meets each request's requirements.

Routing is deterministic: candidates are ranked by the sealed quality
table's rating, a provider that refuses (402/429/5xx) fails over to the next
candidate, and a provider that refuses twice in a row is demoted until it
answers again. No wall clock, no unseeded randomness, so a replay
reproduces the live pass.

Practice trials (kit's env + scorer, run locally; docker is unavailable on
this machine):

| trial | score | quality | served | wire | failover | catalog | calls | working |
|-------|-------|---------|--------|------|----------|---------|-------|---------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 | 1 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 | 1 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 | 1 |

Run: `python3 main.py` with `MAKEFUN_PROMPT_FILE` and `MAKEFUN_ENV_URL`
set. `python3 main.py --list` prints the catalog document.
