"""Tests für nucleus/memoria/session_log.py.

Nutzt tmp_path (pytest built-in), kein Netzwerk, kein Qdrant.
"""

import os
import pytest

# session_log liest MEMORIA_SESSION_LOG_DIR beim Import — tmpdir vor Import setzen
# geht hier nicht sauber, daher instantiieren wir SessionLog mit monkey-patch.


def _make_log(tmp_path):
    from nucleus.memoria import session_log as sl_mod
    original = sl_mod.SESSION_LOG_DIR
    sl_mod.SESSION_LOG_DIR = tmp_path
    log = sl_mod.SessionLog()
    sl_mod.SESSION_LOG_DIR = original
    return log


def test_session_log_creates_file_on_first_append(tmp_path):
    log = _make_log(tmp_path)
    assert not log._path.exists()   # lazy — no empty file for sessions with no turns
    log.append("t1", "user", "Hallo")
    assert log._path.exists()


def test_session_log_appends_block(tmp_path):
    log = _make_log(tmp_path)
    log.append("turn-1", "user", "Hallo Welt")

    content = log._path.read_text(encoding="utf-8")
    assert "turn-1" in content
    assert "user" in content
    assert "Hallo Welt" in content
    assert content.endswith("\n\n")


def test_session_log_multiple_blocks(tmp_path):
    log = _make_log(tmp_path)
    log.append("t1", "user", "Erster Text")
    log.append("t2", "llm",  "Zweiter Text")

    content = log._path.read_text(encoding="utf-8")
    assert "t1" in content
    assert "t2" in content
    assert content.count("[") >= 2   # mindestens 2 Header-Blöcke
