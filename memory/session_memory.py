"""
Session Memory module.
Provides a simple in-memory storage mechanism to maintain conversation history
and student metadata across multiple chat turns in a single session.
"""

from typing import Dict, List

class SessionMemory:
    """
    Keyed in-memory store for per-student chat history and metadata.
    Used by Streamlit's session state to persist the agent's context.
    """
    def __init__(self):
        """Initializes an empty internal store."""
        self._store: Dict[str, dict] = {}

    def save(self, student_id: str, data: dict):
        """
        Merges new data into the student's record.
        History is preserved while other fields are updated.
        """
        if student_id not in self._store:
            self._store[student_id] = {"history": []}
        self._store[student_id].update({k: v for k, v in data.items() if k != "history"})

    def load(self, student_id: str) -> dict:
        """Returns the full record for a specific student ID."""
        return self._store.get(student_id, {})

    def get_history(self, student_id: str) -> List[dict]:
        """Returns the list of conversation turns for a specific student."""
        return self._store.get(student_id, {}).get("history", [])

    def set_history(self, student_id: str, history: List[dict]) -> None:
        """Overwrites the conversation history for a specific student."""
        if student_id not in self._store:
            self._store[student_id] = {}
        self._store[student_id]["history"] = list(history)

    def add_turn(self, student_id: str, role: str, content: str):
        """Appends a single message (role and content) to the history."""
        if student_id not in self._store:
            self._store[student_id] = {"history": []}
        self._store[student_id]["history"].append({"role": role, "content": content})

    def clear(self, student_id: str) -> None:
        """Wipes all data associated with a specific student ID."""
        self._store.pop(student_id, None)

