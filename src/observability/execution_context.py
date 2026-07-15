from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from src.observability.metrics import PerformanceMetrics


@dataclass
class ExecutionContext:
    """
    Representa uma execução completa do pipeline RAG.

    O objeto reúne dados da requisição, resposta, documentos recuperados,
    métricas, configuração do modelo e informações de erro.

    Ele pode ser serializado para JSON e futuramente enviado para:
    - arquivos JSONL;
    - SQLite;
    - MLflow;
    - LangSmith;
    - OpenTelemetry.
    """

    question: str
    session_id: str

    execution_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    timestamp_start: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    timestamp_end: Optional[str] = None

    answer: Optional[str] = None

    documents: list[Any] = field(
        default_factory=list
    )

    metrics: PerformanceMetrics = field(
        default_factory=PerformanceMetrics
    )

    model_name: Optional[str] = None
    temperature: Optional[float] = None

    status: str = "started"
    error: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def set_result(
        self,
        answer: str,
        documents: list[Any],
    ) -> None:
        """
        Registra a resposta e os documentos recuperados.
        """
        self.answer = answer
        self.documents = documents
        self.status = "success"

        self.metrics.collect_text_metrics(
            question=self.question,
            answer=answer,
            documents=documents,
        )

    def set_error(
        self,
        error: Exception | str,
    ) -> None:
        """
        Registra uma falha na execução.
        """
        self.status = "error"
        self.error = str(error)

    def finish(self) -> None:
        """
        Finaliza a execução e encerra o temporizador total.
        """
        self.metrics.end_total_timer()

        self.timestamp_end = datetime.now(
            timezone.utc
        ).isoformat()

    def add_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Adiciona uma informação complementar à execução.
        """
        self.metadata[key] = value

    def documents_to_dict(self) -> list[dict]:
        """
        Converte os metadados dos documentos recuperados
        para uma estrutura serializável.
        """
        serialized_documents = []

        for index, document in enumerate(
            self.documents,
            start=1,
        ):
            metadata = getattr(
                document,
                "metadata",
                {},
            )

            page_content = getattr(
                document,
                "page_content",
                "",
            )

            serialized_documents.append(
                {
                    "rank": index,
                    "source": metadata.get(
                        "source",
                        "Fonte desconhecida",
                    ),
                    "page": metadata.get(
                        "page",
                        "Página desconhecida",
                    ),
                    "start_index": metadata.get(
                        "start_index",
                    ),
                    "content_chars": len(
                        page_content
                    ),
                    "content_preview": (
                        page_content[:300]
                        if page_content
                        else ""
                    ),
                }
            )

        return serialized_documents

    def to_dict(self) -> dict:
        """
        Retorna toda a execução em formato serializável para JSON.
        """
        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "status": self.status,
            "question": self.question,
            "answer": self.answer,
            "model": {
                "name": self.model_name,
                "temperature": self.temperature,
            },
            "documents": self.documents_to_dict(),
            "metrics": self.metrics.to_dict(),
            "error": self.error,
            "metadata": self.metadata,
        }