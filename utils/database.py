import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Database:

    def __init__(self, db_path: str = "disaster_history.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self._create_table()
        logger.info("Database connected: %s", db_path)

    def _create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            date      TEXT    NOT NULL,
            disaster  TEXT    NOT NULL,
            confidence REAL   NOT NULL,
            area      REAL    NOT NULL,
            severity  TEXT    NOT NULL
        )
        """
        self.connection.execute(query)
        self.connection.commit()

    def save_analysis(
        self,
        disaster: str,
        confidence: float,
        area: float,
        severity: str,
    ):
        query = """
        INSERT INTO history (date, disaster, confidence, area, severity)
        VALUES (?, ?, ?, ?, ?)
        """
        self.connection.execute(
            query,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                disaster,
                confidence,
                area,
                severity,
            ),
        )
        self.connection.commit()
        logger.info("Saved analysis: %s (%.1f%%)", disaster, confidence)

    def get_history(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM history ORDER BY id DESC")
        return cursor.fetchall()

    def close(self):
        """Explicitly close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed.")