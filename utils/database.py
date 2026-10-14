import sqlite3
from datetime import datetime



class Database:


    def __init__(self):

        self.connection = sqlite3.connect(
            "disaster_history.db"
        )


        self.create_table()



    def create_table(self):

        query = """

        CREATE TABLE IF NOT EXISTS history
        (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        date TEXT,

        disaster TEXT,

        confidence REAL,

        area REAL,

        severity TEXT

        )

        """


        self.connection.execute(
            query
        )


        self.connection.commit()



    def save_analysis(
            self,
            disaster,
            confidence,
            area,
            severity
    ):


        query = """

        INSERT INTO history
        (
        date,
        disaster,
        confidence,
        area,
        severity
        )

        VALUES (?,?,?,?,?)

        """


        self.connection.execute(
            query,
            (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),

                disaster,

                confidence,

                area,

                severity
            )
        )


        self.connection.commit()



    def get_history(self):

        cursor = self.connection.cursor()


        cursor.execute(
            "SELECT * FROM history"
        )


        return cursor.fetchall()