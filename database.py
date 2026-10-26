import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance")
DB_PATH = os.path.join(DB_DIR, "neuroscan.db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _connection(db_path: Optional[str] = None):
    connection = get_db_connection(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db(db_path: Optional[str] = None) -> None:
    with _connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                original_image TEXT NOT NULL,
                gradcam_image TEXT,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_used TEXT NOT NULL,
                all_probabilities TEXT,
                triage_level TEXT NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_class TEXT,
                review_note TEXT
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(scans)")
        }
        if "reviewed_class" not in columns:
            connection.execute("ALTER TABLE scans ADD COLUMN reviewed_class TEXT")
        if "review_note" not in columns:
            connection.execute("ALTER TABLE scans ADD COLUMN review_note TEXT")


def insert_scan(
    case_label: str,
    original_image: str,
    gradcam_image: str,
    predicted_class: str,
    confidence: float,
    model_used: str,
    probabilities: Dict[str, float],
    notes: str,
    db_path: Optional[str] = None,
) -> int:
    with _connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO scans (
                patient_id, patient_name, original_image, gradcam_image,
                predicted_class, confidence, model_used, all_probabilities,
                triage_level, notes
            ) VALUES (?, 'Anonymous case', ?, ?, ?, ?, ?, ?, 'Educational prototype', ?)
            """,
            (
                case_label,
                original_image,
                gradcam_image,
                predicted_class,
                round(float(confidence), 1),
                model_used,
                json.dumps(probabilities),
                notes,
            ),
        )
        return int(cursor.lastrowid)


def _public_scan(row: sqlite3.Row) -> Dict[str, Any]:
    scan = {
        key: row[key]
        for key in (
            "id",
            "patient_id",
            "original_image",
            "gradcam_image",
            "predicted_class",
            "confidence",
            "model_used",
            "all_probabilities",
            "notes",
            "created_at",
            "reviewed_class",
            "review_note",
        )
    }
    scan["case_label"] = scan.pop("patient_id")
    scan["probabilities"] = json.loads(scan.pop("all_probabilities") or "{}")
    return scan


def get_all_scans(
    search_query: Optional[str] = None,
    filter_class: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = """
        SELECT id, patient_id, original_image, gradcam_image, predicted_class,
               confidence, model_used, all_probabilities, notes, created_at,
               reviewed_class, review_note
        FROM scans WHERE 1=1
    """
    params: List[Any] = []
    if search_query:
        query += " AND patient_id LIKE ?"
        params.append(f"%{search_query.strip()}%")
    if filter_class and filter_class != "all":
        query += " AND LOWER(predicted_class) = LOWER(?)"
        params.append(filter_class.strip())
    query += " ORDER BY created_at DESC, id DESC LIMIT 100"

    with _connection(db_path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [_public_scan(row) for row in rows]


def get_scan_by_id(scan_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    with _connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT id, patient_id, original_image, gradcam_image, predicted_class,
                   confidence, model_used, all_probabilities, notes, created_at,
                   reviewed_class, review_note
            FROM scans WHERE id = ?
            """,
            (scan_id,),
        ).fetchone()
    return _public_scan(row) if row else None


def update_review(
    scan_id: int,
    reviewed_class: Optional[str],
    review_note: str,
    db_path: Optional[str] = None,
) -> bool:
    with _connection(db_path) as connection:
        cursor = connection.execute(
            "UPDATE scans SET reviewed_class = ?, review_note = ? WHERE id = ?",
            (reviewed_class, review_note, scan_id),
        )
        return cursor.rowcount > 0


def delete_scan(scan_id: int, db_path: Optional[str] = None) -> bool:
    with _connection(db_path) as connection:
        cursor = connection.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        return cursor.rowcount > 0


def get_stats(db_path: Optional[str] = None) -> Dict[str, int]:
    with _connection(db_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        reviewed = connection.execute(
            "SELECT COUNT(*) FROM scans WHERE reviewed_class IS NOT NULL"
        ).fetchone()[0]
    return {"total_scans": total, "reviewed_scans": reviewed}
