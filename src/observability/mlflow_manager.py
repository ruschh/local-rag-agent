import tempfile
from pathlib import Path
from typing import Any, Optional

import mlflow

from src.config import (
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELO_EMBEDDINGS,
)
from src.observability.execution_context import ExecutionContext


class MLflowManager:
    """
    Gerencia o registro das execuções RAG no MLflow.

    Para cada pergunta, registra:
    - parâmetros da aplicação;
    - métricas de desempenho;
    - tags de identificação;
    - ExecutionContext completo como artefato JSON.
    """

    def __init__(
        self,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        enabled: bool = MLFLOW_ENABLED,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = enabled

        if not self.enabled:
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    @staticmethod
    def _remove_none_values(
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Remove valores None antes de enviá-los ao MLflow.
        """
        return {
            key: value
            for key, value in data.items()
            if value is not None
        }

    @staticmethod
    def _extract_llm_monitor_metrics(
        execution: ExecutionContext,
    ) -> dict[str, float]:
        """
        Extrai métricas agregadas produzidas pelo LLMMonitor.
        """
        monitor_data = execution.metadata.get(
            "llm_monitor",
            {},
        )

        return {
            "llm_num_calls": float(
                monitor_data.get("num_calls", 0)
            ),
            "llm_successful_calls": float(
                monitor_data.get("successful_calls", 0)
            ),
            "llm_failed_calls": float(
                monitor_data.get("failed_calls", 0)
            ),
            "llm_monitor_total_duration_seconds": float(
                monitor_data.get(
                    "total_duration_seconds",
                    0,
                )
            ),
        }

    def _build_params(
        self,
        execution: ExecutionContext,
    ) -> dict[str, Any]:
        """
        Monta os parâmetros permanentes da execução.
        """
        return self._remove_none_values(
            {
                "model_name": execution.model_name,
                "temperature": execution.temperature,
                "embedding_model": MODELO_EMBEDDINGS,
                "application_type": "conversational_rag",
                "vector_store": "chroma",
                "llm_provider": "ollama",
            }
        )

    def _build_metrics(
        self,
        execution: ExecutionContext,
    ) -> dict[str, float]:
        """
        Monta as métricas numéricas da execução.
        """
        performance_metrics = execution.metrics.to_dict()

        metrics: dict[str, float] = {}

        for key, value in performance_metrics.items():
            if value is not None:
                metrics[key] = float(value)

        metrics.update(
            self._extract_llm_monitor_metrics(execution)
        )

        return metrics

    def _build_tags(
        self,
        execution: ExecutionContext,
        interface: str,
    ) -> dict[str, str]:
        """
        Monta tags usadas para filtrar as runs no MLflow.
        """
        return {
            "execution_id": execution.execution_id,
            "session_id": execution.session_id,
            "status": execution.status,
            "interface": interface,
            "has_error": str(
                execution.error is not None
            ).lower(),
        }

    def log_execution(
        self,
        execution: ExecutionContext,
        interface: str = "streamlit",
        additional_params: Optional[
            dict[str, Any]
        ] = None,
        additional_tags: Optional[
            dict[str, str]
        ] = None,
    ) -> Optional[str]:
        """
        Registra uma execução completa no MLflow.

        Retorna o run_id criado ou None quando o MLflow
        estiver desabilitado.
        """

        if not self.enabled:
            return None

        params = self._build_params(execution)
        metrics = self._build_metrics(execution)
        tags = self._build_tags(
            execution=execution,
            interface=interface,
        )

        if additional_params:
            params.update(
                self._remove_none_values(
                    additional_params
                )
            )

        if additional_tags:
            tags.update(additional_tags)

        run_name = (
            f"rag-{execution.execution_id[:8]}"
        )
        with mlflow.start_run(
            run_name=run_name,
            tags=tags,
        ) as run:

            mlflow.log_params(params)

            if metrics:
                mlflow.log_metrics(metrics)

            mlflow.log_text(
                execution.question,
                artifact_file="input/question.txt",
            )

            if execution.answer:
                mlflow.log_text(
                    execution.answer,
                    artifact_file="output/answer.txt",
                )

            mlflow.log_dict(
                execution.to_dict(),
                artifact_file=(
                    "execution/execution_context.json"
                ),
            )

            documents = execution.documents_to_dict()

            mlflow.log_dict(
                {
                    "documents": documents,
                },
                artifact_file=(
                    "retrieval/"
                    "retrieved_documents.json"
                ),
            )

            if execution.error:
                mlflow.log_text(
                    execution.error,
                    artifact_file="errors/error.txt",
                )

            return run.info.run_id