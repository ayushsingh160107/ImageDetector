"""TruthLens AI - Database & History Service
Manages local SQLite database for logging image detection analysis records.
Privacy-first: only metadata and model predictions are saved, never image files.
"""

from __future__ import annotations

import datetime
import os
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "truthlens.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates directory if needed and returns SQLite connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_size_kb REAL NOT NULL,
                    dimensions TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    ai_probability REAL NOT NULL,
                    real_probability REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    uncertainty_flag INTEGER NOT NULL DEFAULT 0,
                    notes TEXT
                )
                """
            )
    finally:
        conn.close()


def save_analysis(
    filename: str,
    file_size_kb: float,
    dimensions: str,
    predicted_label: str,
    ai_probability: float,
    real_probability: float,
    confidence_score: float,
    inference_time_ms: float,
    model_name: str,
    status: str = "Completed",
    uncertainty_flag: bool = False,
    notes: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Saves an analysis record to SQLite and returns the record ID."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_history (
                    timestamp, filename, file_size_kb, dimensions, predicted_label,
                    ai_probability, real_probability, confidence_score,
                    inference_time_ms, model_name, status, uncertainty_flag, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    filename,
                    round(file_size_kb, 2),
                    dimensions,
                    predicted_label,
                    round(ai_probability, 4),
                    round(real_probability, 4),
                    round(confidence_score, 4),
                    round(inference_time_ms, 2),
                    model_name,
                    status,
                    1 if uncertainty_flag else 0,
                    notes or "",
                ),
            )
            return cursor.lastrowid or 0
    finally:
        conn.close()


def get_history(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent analysis history records."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT * FROM analysis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_history_dataframe(limit: int = 100, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieves history as a pandas DataFrame."""
    records = get_history(limit, db_path)
    if not records:
        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "filename",
                "file_size_kb",
                "dimensions",
                "predicted_label",
                "ai_probability",
                "real_probability",
                "confidence_score",
                "inference_time_ms",
                "model_name",
                "status",
                "uncertainty_flag",
            ]
        )
    return pd.DataFrame(records)


def clear_history(db_path: str = DEFAULT_DB_PATH) -> None:
    """Clears all records from history."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM analysis_history")
    finally:
        conn.close()
