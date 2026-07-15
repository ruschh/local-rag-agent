import json
import sqlite3
from typing import Any, Optional

from src.memory.database import SQLiteConnection
from src.observability.execution_context import ExecutionContext


class ConversationRepository:
    """
    Executa as operações SQL relacionadas ao histórico de conversas.
    """

    def __init__(
        self,
        database: Optional[SQLiteConnection] = None,
    ) -> None:
        self.database = database or SQLiteConnection()
        self.create_table()
        self.migrate_database()


#==============================================================================================================================
    def create_table(self) -> None:
        """
        Cria a tabela de histórico, caso ela ainda não exista.
        """
        query = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            execution_id TEXT NOT NULL UNIQUE,
            timestamp_start TEXT NOT NULL,
            timestamp_end TEXT,
            question TEXT NOT NULL,
            answer TEXT,
            model_name TEXT,
            temperature REAL,
            status TEXT NOT NULL,
            error TEXT,
            metrics_json TEXT NOT NULL,
            documents_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """

        with self.database.connect() as connection:
            connection.execute(query)
            connection.commit()


#==============================================================================================================================
    def migrate_database(self) -> None:
        """
        Adiciona colunas e índices necessários em bancos já existentes.
        """
        with self.database.connect() as connection:
            columns = connection.execute(
                "PRAGMA table_info(conversation_history);"
            ).fetchall()

            column_names = {
                column["name"]
                for column in columns
            }

            if "session_id" not in column_names:
                connection.execute(
                    """
                    ALTER TABLE conversation_history
                    ADD COLUMN session_id TEXT;
                    """
                )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_conversation_session
                ON conversation_history(session_id);
                """
            )

            connection.commit()
    

#==============================================================================================================================
    def save(
        self,
        execution: ExecutionContext,
    ) -> int:
        """
        Persiste uma execução completa no banco.
        """
        execution_data = execution.to_dict()

        query = """
        INSERT INTO conversation_history (
            execution_id,
            session_id,
            timestamp_start,
            timestamp_end,
            question,
            answer,
            model_name,
            temperature,
            status,
            error,
            metrics_json,
            documents_json,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        values = (
            execution.execution_id,
            execution.session_id,
            execution.timestamp_start,
            execution.timestamp_end,
            execution.question,
            execution.answer,
            execution.model_name,
            execution.temperature,
            execution.status,
            execution.error,
            json.dumps(
                execution_data["metrics"],
                ensure_ascii=False,
            ),
            json.dumps(
                execution_data["documents"],
                ensure_ascii=False,
            ),
            json.dumps(
                execution_data["metadata"],
                ensure_ascii=False,
            ),
        )

        try:
            with self.database.connect() as connection:
                cursor = connection.execute(
                    query,
                    values,
                )
                connection.commit()

                return cursor.lastrowid

        except sqlite3.IntegrityError as error:
            raise ValueError(
                "Esta execução já foi salva no banco."
            ) from error


#==============================================================================================================================
    def find_by_execution_id(
        self,
        execution_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Localiza uma execução pelo identificador único.
        """
        query = """
        SELECT *
        FROM conversation_history
        WHERE execution_id = ?;
        """

        with self.database.connect() as connection:
            row = connection.execute(
                query,
                (execution_id,),
            ).fetchone()

        if row is None:
            return None

        return self._deserialize_row(row)


#==============================================================================================================================
    def list_recent(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retorna as interações mais recentes.
        """
        query = """
        SELECT *
        FROM conversation_history
        ORDER BY id DESC
        LIMIT ?;
        """

        with self.database.connect() as connection:
            rows = connection.execute(
                query,
                (limit,),
            ).fetchall()

        return [
            self._deserialize_row(row)
            for row in rows
        ]


#==============================================================================================================================
    def search(
        self,
        term: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Pesquisa um termo nas perguntas e respostas.
        """
        query = """
        SELECT *
        FROM conversation_history
        WHERE question LIKE ?
           OR answer LIKE ?
        ORDER BY id DESC
        LIMIT ?;
        """

        search_term = f"%{term}%"

        with self.database.connect() as connection:
            rows = connection.execute(
                query,
                (
                    search_term,
                    search_term,
                    limit,
                ),
            ).fetchall()

        return [
            self._deserialize_row(row)
            for row in rows
        ]

#==============================================================================================================================
    def count(self) -> int:
        """
        Retorna o total de interações salvas.
        """
        query = """
        SELECT COUNT(*) AS total
        FROM conversation_history;
        """

        with self.database.connect() as connection:
            row = connection.execute(query).fetchone()

        return int(row["total"])


#==============================================================================================================================
    def delete_all(self) -> int:
        """
        Remove todo o histórico de conversas.
        """
        with self.database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_history;"
            )
            connection.commit()

            return cursor.rowcount


#==============================================================================================================================
    @staticmethod
    def _deserialize_row(
        row: sqlite3.Row,
    ) -> dict[str, Any]:
        """
        Converte uma linha SQLite em dicionário Python.
        """
        data = dict(row)

        data["metrics"] = json.loads(
            data.pop("metrics_json")
        )

        data["documents"] = json.loads(
            data.pop("documents_json")
        )

        data["metadata"] = json.loads(
            data.pop("metadata_json")
        )

        return data


#==============================================================================================================================
    def list_by_session(
    self,
    session_id: str,
    limit: int = 6,
) -> list[dict[str, Any]]:
        """
        Retorna as interações mais recentes de uma sessão.

        O resultado final é colocado em ordem cronológica.
        """
        query = """
            SELECT *
            FROM conversation_history
            WHERE session_id = ?
            AND status = 'success'
            AND answer IS NOT NULL
            ORDER BY id DESC
            LIMIT ?;
            """

        with self.database.connect() as connection:
            rows = connection.execute(
                query,
                (
                    session_id,
                    limit,
                ),
            ).fetchall()

        interactions = [
            self._deserialize_row(row)
            for row in rows
        ]

        interactions.reverse()

        return interactions