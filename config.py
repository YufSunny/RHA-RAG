"""Admin-tunable settings for RHA-RAG.

Edit this file and restart the server (`python server.py`) to apply changes.
"""

# Number of prior Q&A turns fed back into each new turn of a conversation.
# Bounds token cost / latency as a chat grows. Set to 0 to disable memory.
MAX_HISTORY_TURNS = 6
