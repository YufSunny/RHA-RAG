"""PostgreSQL-backed conversation history.

On first access, tries to connect and create the table. If PostgreSQL is
unreachable, every function is a no-op / returns empty — the server falls
back to its in-memory ``_conversations`` dict transparently.
"""

import logging
import sqlalchemy as sa
from sqlalchemy import text
from config import DATABASE_URL

log = logging.getLogger("rha_rag.db")

_engine: sa.Engine | None = None
_available: bool | None = None  # tri-state: None = not tried, True / False


def _ensure_db() -> bool:
    """Connect and create the table. Returns True on success (memoized)."""
    global _engine, _available

    if _available is not None:
        return _available

    try:
        _engine = sa.create_engine(DATABASE_URL, pool_size=2, pool_pre_ping=True)
        with _engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL,
                    seq INTEGER NOT NULL,
                    role VARCHAR(10) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_conv_session_seq "
                "ON conversation_messages (session_id, seq)"
            ))
        _available = True
        log.info("Connected to PostgreSQL; conversation_messages table ready")
        return True
    except Exception as e:
        log.warning("PostgreSQL unavailable — falling back to in-memory history: %s", e)
        _engine = None
        _available = False
        return False


def load_history(session_id: str, max_pairs: int) -> list[tuple[str, str]]:
    """Return the most recent (role, content) pairs for *session_id*.

    ``max_pairs`` caps the number of Q&A pairs returned (so at most
    ``max_pairs * 2`` rows).  Returns ``[]`` if the session is unknown.
    """
    if not _ensure_db() or max_pairs <= 0 or not session_id:
        return []

    limit = max_pairs * 2
    with _engine.begin() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT role, content FROM conversation_messages "
                    "WHERE session_id = :sid ORDER BY seq DESC LIMIT :n"
                ),
                {"sid": session_id, "n": limit},
            )
            .mappings()
            .all()
        )
    # rows are newest-first; reverse to chronological order.
    return [(r["role"], r["content"]) for r in reversed(rows)]


def save_turn(session_id: str, question: str, answer: str, max_pairs: int):
    """Persist one Q&A turn and trim excess rows to ``max_pairs`` pairs."""
    if not _ensure_db() or not session_id:
        return

    with _engine.begin() as conn:
        # Find the next seq for this session.
        row = conn.execute(
            text(
                "SELECT COALESCE(MAX(seq), -1) AS mx "
                "FROM conversation_messages WHERE session_id = :sid"
            ),
            {"sid": session_id},
        ).first()
        next_seq = row[0] + 1

        conn.execute(
            text(
                "INSERT INTO conversation_messages (session_id, seq, role, content) "
                "VALUES (:sid, :seq, 'human', :q)"
            ),
            {"sid": session_id, "seq": next_seq, "q": question},
        )
        conn.execute(
            text(
                "INSERT INTO conversation_messages (session_id, seq, role, content) "
                "VALUES (:sid, :seq, 'ai', :a)"
            ),
            {"sid": session_id, "seq": next_seq + 1, "a": answer},
        )

        # Trim excess rows beyond the cap.
        limit = max_pairs * 2
        conn.execute(
            text(
                "DELETE FROM conversation_messages WHERE id IN ("
                "  SELECT id FROM conversation_messages "
                "  WHERE session_id = :sid "
                "  ORDER BY seq DESC OFFSET :n"
                ")"
            ),
            {"sid": session_id, "n": limit},
        )


def clear_session(session_id: str):
    """Remove all rows for *session_id*."""
    if not _ensure_db() or not session_id:
        return
    with _engine.begin() as conn:
        conn.execute(
            text("DELETE FROM conversation_messages WHERE session_id = :sid"),
            {"sid": session_id},
        )


def list_sessions() -> list[dict]:
    """Return all sessions with preview info, newest first."""
    if not _ensure_db():
        return []
    with _engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT session_id,
                       MAX(created_at) AS updated_at,
                       COUNT(*) AS messages,
                       (SELECT LEFT(content, 80)
                        FROM conversation_messages m2
                        WHERE m2.session_id = m1.session_id
                        ORDER BY seq DESC LIMIT 1) AS last_preview
                FROM conversation_messages m1
                GROUP BY session_id
                ORDER BY updated_at DESC
            """)
        ).mappings().all()
    return [
        {
            "session_id": r["session_id"],
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            "messages": r["messages"],
            "last_preview": r["last_preview"],
        }
        for r in rows
    ]


def get_session_history(session_id: str) -> list[dict]:
    """Return all messages for a session, ordered by seq."""
    if not _ensure_db() or not session_id:
        return []
    with _engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT role, content, created_at FROM conversation_messages "
                "WHERE session_id = :sid ORDER BY seq"
            ),
            {"sid": session_id},
        ).mappings().all()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        }
        for r in rows
    ]
