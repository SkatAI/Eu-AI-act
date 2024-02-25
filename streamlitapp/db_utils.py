import psycopg2
from sqlalchemy import create_engine
import os


class Database:
    def __init__(self):
        db_params = {
            "dbname": "AIARAGtimedb",
            "user": "doadmin",
            "password": os.environ.get("DOPGPASSWORD_AIARAGtime"),
            "host": os.environ.get("DOPGHOST"),
            "port": "25060",
            "sslmode": "require",
        }
        self.connection = psycopg2.connect(**db_params)
        self.engine = create_engine(
            f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        )

    def insert(self, insert_query):
        cursor = self.connection.cursor()

        cursor.execute(insert_query)
        self.connection.commit()
        cursor.close()

    def execute(self, query):
        cursor = self.connection.cursor()

        cursor.execute(query)
        self.connection.commit()
        cursor.close()

    def close(self):
        self.connection.close()
        self.engine.dispose()


if __name__ == "__main__":
    db = Database()
    query = "select count(*) from groundtruth;"
    cur = db.connection.cursor()
    # Execute a query
    cur.execute(query)
    # Retrieve query results
    n = cur.fetchone()
    print("n:", n)

    active_connections = db.engine.pool.status()
    print(f"-- active connections {active_connections}")
