"""RHA-RAG configuration.

API keys — set them in your shell, in a ``.env`` file (auto-loaded), or type
them when prompted.  For quick local dev, uncomment the lines below and paste
your keys directly.  In Docker, set them as environment variables instead;
interactive prompts are skipped when there is no terminal.
"""

import getpass
import os
import sys


def _resolve(key: str) -> str | None:
    """Return a value for *key* from the first available source, or None.

    1. ``os.environ`` (shell / ``.env`` / Docker).
    2. A module-level variable with the same name (uncommented in this file).
    """
    if key in os.environ:
        return os.environ[key]
    mod_val = globals().get(key)
    if mod_val is not None and mod_val not in ("your-key-here", ""):
        return str(mod_val)
    return None


def _prompt(key: str, *, secret: bool = False):
    """Ask the user for *key* interactively.  Skipped silently when stdin is
    not a terminal (Docker / redirect)."""
    if not sys.stdin.isatty():
        return
    prompt = getpass.getpass if secret else input
    try:
        val = prompt(f"{key}: ").strip()
    except (EOFError, OSError):
        return
    if val:
        os.environ[key] = val


# ── API keys ───────────────────────────────────────────────────
# Uncomment and fill in, or leave commented to be prompted / set via env.

# ZAI_API_KEY    = "your-key-here"        # https://www.z.ai/
# QWEN_API_KEY   = "your-key-here"        # https://dashscope.aliyun.com/
# OPENAI_API_KEY = "your-key-here"        # https://api.deepseek.com

for _k in ("ZAI_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"):
    _resolve(_k)  # picks up env / uncommented value
    _prompt(_k, secret=True)


# ── Model configuration ──────────────────────────────────────
LLM_MODEL     = os.environ.get("LLM_MODEL",     "deepseek-v4-pro")
LLM_BASE_URL  = os.environ.get("LLM_BASE_URL",  "https://api.deepseek.com")
EMBED_MODEL   = os.environ.get("EMBED_MODEL",   "text-embedding-v4")
EMBED_BASE_URL = os.environ.get(
    "EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
OCR_MODEL = os.environ.get("OCR_MODEL", "glm-ocr")

# DeepSeek thinking mode — sends {"thinking": {"type": "enabled"}} in the API
# call.  When enabled the model emits a chain-of-thought before the final answer
# (returned as `reasoning_content`).  Set to "false" / "0" to disable.
# https://api-docs.deepseek.com/guides/thinking_mode
LLM_THINKING = os.environ.get("LLM_THINKING", "true").lower() in (
    "1", "true", "yes"
)

# Effort level when thinking mode is on: low | medium | high | xhigh | max.
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "high")

# ── Server ───────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", "8000"))

# ── Conversation memory ───────────────────────────────────────
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "6"))

# PostgreSQL connection for persistent history.
# Uncomment and fill in, or set the PG_* env vars (Docker).
# Leave all blank for in-memory storage.

# PG_HOST     = "localhost"
# PG_PORT     = 5432
# PG_USER     = "postgres"
# PG_PASSWORD = "your-password"
# PG_DATABASE = "postgres"

_PG_FIELDS = ("PG_HOST", "PG_PORT", "PG_USER", "PG_PASSWORD", "PG_DATABASE")
for _k in _PG_FIELDS:
    _resolve(_k)
    _prompt(_k)

_pg = {k: os.environ.get(k, "") for k in _PG_FIELDS}
if all(_pg.values()):
    DATABASE_URL = (
        f"postgresql+psycopg2://"
        f"{_pg['PG_USER']}:{_pg['PG_PASSWORD']}@"
        f"{_pg['PG_HOST']}:{_pg['PG_PORT']}/"
        f"{_pg['PG_DATABASE']}"
    )
else:
    DATABASE_URL = ""
