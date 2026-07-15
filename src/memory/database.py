import sqlite3
from pathlib import Path
from typing import Optional

from src.config import PASTA_MEMORY


class SQLiteConnection:
    """
    Gerencia conexões com o banco SQLite da aplicação.
    """

    def __init__(
        self,
        database_path: Optional[Path] = None,
    ) -> None:
        self.database_path = Path(
            database_path or PASTA_MEMORY
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def connect(self) -> sqlite3.Connection:
        """
        Cria e retorna uma conexão com o banco SQLite.
        """
        connection = sqlite3.connect(
            self.database_path
        )

        # Permite acessar as colunas pelo nome.
        connection.row_factory = sqlite3.Row

        # Habilita validação de chaves estrangeiras.
        connection.execute(
            "PRAGMA foreign_keys = ON;"
        )

        return connection