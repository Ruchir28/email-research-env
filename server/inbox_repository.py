# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SQLite-backed inbox storage and retrieval helpers."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
BLUEPRINTS_PATH = DATA_DIR / "email_blueprints.json"
TASKS_PATH = DATA_DIR / "tasks.json"
DB_PATH = DATA_DIR / "inbox.db"


def _load_blueprints() -> dict[str, Any]:
    return json.loads(BLUEPRINTS_PATH.read_text())


def load_tasks() -> list[dict[str, Any]]:
    return json.loads(TASKS_PATH.read_text())


def ensure_inbox_db(force: bool = False) -> Path:
    """Create the inbox SQLite DB if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists() and not force:
        return DB_PATH

    blueprints = _load_blueprints()
    emails = blueprints["emails"]

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.executescript(
            """
            DROP TABLE IF EXISTS emails_fts;
            DROP TABLE IF EXISTS emails;

            CREATE TABLE emails (
                email_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipients_json TEXT NOT NULL,
                sent_at TEXT NOT NULL
            );

            CREATE INDEX idx_emails_sent_at ON emails(sent_at);
            CREATE INDEX idx_emails_sender ON emails(sender);

            CREATE VIRTUAL TABLE emails_fts USING fts5(
                subject,
                body,
                content='emails',
                content_rowid='rowid'
            );
            """
        )

        cursor.executemany(
            """
            INSERT INTO emails (email_id, subject, body, sender, recipients_json, sent_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    email["email_id"],
                    email["subject"],
                    email["body"],
                    email["sender"],
                    json.dumps(email["recipients"]),
                    email["sent_at"],
                )
                for email in emails
            ],
        )

        cursor.execute(
            """
            INSERT INTO emails_fts(rowid, subject, body)
            SELECT rowid, subject, body FROM emails
            """
        )
        conn.commit()
    finally:
        conn.close()
    return DB_PATH


def _connect() -> sqlite3.Connection:
    ensure_inbox_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fts_query(query: str | None) -> str:
    tokens = re.findall(r"[A-Za-z0-9@._-]+", query or "")
    # Quote each token so date-like or hyphenated terms remain literal FTS terms
    # instead of being parsed as FTS operators or column references.
    return " ".join(f'"{token}"' for token in tokens)


def search_emails(
    *,
    query: str | None,
    top_k: int,
    sent_after: str | None = None,
    sent_before: str | None = None,
    sender: str | None = None,
) -> list[dict[str, Any]]:
    """Search emails using FTS5 plus metadata filters."""
    fts_query = _fts_query(query)
    filters: list[str] = []
    params: list[Any] = []

    if sent_after:
        filters.append("e.sent_at >= ?")
        params.append(sent_after)
    if sent_before:
        filters.append("e.sent_at <= ?")
        params.append(sent_before)
    if sender:
        filters.append("LOWER(e.sender) = LOWER(?)")
        params.append(sender)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    conn = _connect()
    try:
        cursor = conn.cursor()
        if fts_query:
            sql = f"""
                SELECT
                    e.email_id,
                    e.subject,
                    e.sender,
                    e.sent_at,
                    snippet(emails_fts, 1, '[', ']', '...', 14) AS snippet
                FROM emails_fts
                JOIN emails e ON e.rowid = emails_fts.rowid
                WHERE emails_fts MATCH ?
                {'AND ' + ' AND '.join(filters) if filters else ''}
                ORDER BY bm25(emails_fts), e.sent_at DESC, e.email_id ASC
                LIMIT ?
            """
            rows = cursor.execute(sql, [fts_query, *params, top_k]).fetchall()
        else:
            sql = f"""
                SELECT
                    e.email_id,
                    e.subject,
                    e.sender,
                    e.sent_at,
                    substr(e.body, 1, 180) AS snippet
                FROM emails e
                {where_clause}
                ORDER BY e.sent_at DESC
                LIMIT ?
            """
            rows = cursor.execute(sql, [*params, top_k]).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()


def read_email(email_id: str) -> dict[str, Any] | None:
    """Read a single email by ID."""
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT email_id, subject, body, sender, recipients_json, sent_at
            FROM emails
            WHERE email_id = ?
            """,
            (email_id,),
        ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["recipients"] = json.loads(payload.pop("recipients_json"))
        return payload
    finally:
        conn.close()
