"""RHA-RAG configuration.

API keys — set them in your shell, in a ``.env`` file (auto-loaded), or type
them when prompted.  For quick local dev, uncomment the lines below and paste
your keys directly (see the examples at the bottom of this docstring).
"""

import getpass
import os


def _prompt(key: str, *, env: dict[str, str] | None = None):
    """If *key* is already in ``os.environ``, do nothing.

    Otherwise look for a module-level variable with the same name (set by
    uncommenting the corresponding line in this file).  If found, copy it
    into ``os.environ``.  Otherwise prompt the user via ``getpass``
    (for API keys) or plain ``input`` (for PostgreSQL fields).  An empty
    input is treated as "skip".

    *env* lets the caller pass a non-``os.environ`` dict for testing.
    """
    store = os.environ if env is None else env

    if key in store:
        return

    # Check for a value set by uncommenting the corresponding line in this file.
    mod_val = globals().get(key)
    if mod_val is not None and mod_val not in ("your-key-here", ""):
        store[key] = str(mod_val)
        return

    prompt = getpass.getpass if key.startswith(("ZAI_", "QWEN_", "OPENAI_")) else input
    try:
        val = prompt(f"{key}: ").strip()
    except (EOFError, OSError):
        return
    if val:
        store[key] = val


# ── API keys ───────────────────────────────────────────────────
# Uncomment and fill in, or leave commented to be prompted.

# ZAI_API_KEY    = "your-key-here"        # https://www.z.ai/
# QWEN_API_KEY   = "your-key-here"        # https://dashscope.aliyun.com/
# OPENAI_API_KEY = "your-key-here"        # https://api.deepseek.com

for _k in ("ZAI_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"):
    _prompt(_k)


# ── Conversation memory ───────────────────────────────────────
MAX_HISTORY_TURNS = 6   # prior Q&A turns fed back each turn (0 = disable)

# PostgreSQL connection for persistent history.
# Uncomment and fill in, or leave unset (falls back to in-memory).

# PG_HOST     = "localhost"
# PG_PORT     = 5432
# PG_USER     = "postgres"
# PG_PASSWORD = "your-password"
# PG_DATABASE = "postgres"

_pg_parts = {}
for _k in ("PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD", "PG_DATABASE"):
    _prompt(_k, env=_pg_parts)

if all(_pg_parts.values()):
    DATABASE_URL = (
        f"postgresql+psycopg2://"
        f"{_pg_parts['PG_USER']}:{_pg_parts['PG_PASSWORD']}@"
        f"{_pg_parts['PG_HOST']}:{_pg_parts['PG_PORT']}/"
        f"{_pg_parts['PG_DATABASE']}"
    )
else:
    DATABASE_URL = ""
