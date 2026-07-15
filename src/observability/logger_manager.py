import json
from datetime import datetime, timezone
from pathlib import Path
from src.config import PASTA_LOGS


class LoggerManager:
    """
    Gerencia logs estruturados em formato JSONL.
    """

    def __init__(self, log_file=None):

        if log_file is None:
            log_file = PASTA_LOGS / "rag_app.jsonl"

        self.log_path = Path(log_file)

        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log_event(
        self,
        event_type,
        payload,
    ):
        """
        Registra um evento genérico.
        """
        event = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "event_type": event_type,
            **payload,
        }

        self._write(event)

    def log_execution(
        self,
        execution_context,
    ):
        """
        Registra uma execução completa do RAG.
        """
        event = {
            "event_type": "rag_execution",
            **execution_context.to_dict(),
        }

        self._write(event)

    def _write(
        self,
        event: dict,
    ) -> None:
        """
        Grava uma linha JSON no arquivo de logs.
        """
        with open(
            self.log_path,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
            )
            file.write("\n")