import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from src.observability.execution_context import ExecutionContext


class LLMMonitor(BaseCallbackHandler):
    """
    Monitora chamadas ao LLM realizadas internamente pelo LangChain.

    O monitor utiliza callbacks para observar:
    - mensagens enviadas ao modelo;
    - resposta produzida;
    - tempo de inferência;
    - metadados de uso;
    - erros da chamada.

    Os resultados são armazenados no ExecutionContext.
    """

    def __init__(
        self,
        execution_context: ExecutionContext,
        capture_prompt: bool = True,
        max_prompt_chars: int | None = 20_000,
        max_response_chars: int | None = 10_000,
    ) -> None:
        self.execution_context = execution_context
        self.capture_prompt = capture_prompt
        self.max_prompt_chars = max_prompt_chars
        self.max_response_chars = max_response_chars

        # Armazena o instante inicial de cada chamada ao LLM.
        self._start_times: dict[str, float] = {}

        # Armazena temporariamente os dados de cada chamada.
        self._calls: dict[str, dict[str, Any]] = {}

        # Lista final de chamadas monitoradas.
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _utc_now() -> str:
        """
        Retorna data e hora atual em UTC, no formato ISO 8601.
        """
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _run_id_to_string(run_id: UUID | str) -> str:
        """
        Normaliza o identificador da execução para string.
        """
        return str(run_id)

    @staticmethod
    def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
        """
        Converte uma mensagem do LangChain em estrutura serializável.
        """
        content = message.content

        if not isinstance(content, (str, int, float, bool, list, dict)):
            content = str(content)

        return {
            "type": getattr(message, "type", message.__class__.__name__),
            "content": content,
            "name": getattr(message, "name", None),
            "id": getattr(message, "id", None),
        }

    def _truncate(
        self,
        value: Any,
        max_chars: int | None,
    ) -> Any:
        """
        Limita o tamanho de textos gravados nos logs.

        Estruturas que não são strings são devolvidas sem alteração.
        """
        if not isinstance(value, str):
            return value

        if max_chars is None or len(value) <= max_chars:
            return value

        removed_chars = len(value) - max_chars

        return (
            value[:max_chars]
            + f"\n...[{removed_chars} caracteres omitidos]"
        )

    def _serialize_messages(
        self,
        messages: list[list[BaseMessage]],
    ) -> list[list[dict[str, Any]]] | None:
        """
        Serializa os lotes de mensagens enviados ao modelo.
        """
        if not self.capture_prompt:
            return None

        serialized_batches = []

        for batch in messages:
            serialized_batch = []

            for message in batch:
                message_data = self._message_to_dict(message)

                message_data["content"] = self._truncate(
                    message_data["content"],
                    self.max_prompt_chars,
                )

                serialized_batch.append(message_data)

            serialized_batches.append(serialized_batch)

        return serialized_batches

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Executado imediatamente antes da chamada ao modelo de chat.
        """
        run_key = self._run_id_to_string(run_id)
        start_time = time.perf_counter()

        self._start_times[run_key] = start_time

        # Preenche o temporizador do LLM na PerformanceMetrics.
        if self.execution_context.metrics.start_llm is None:
            self.execution_context.metrics.start_llm = start_time

        model_name = (
            (metadata or {}).get("ls_model_name")
            or serialized.get("name")
            or self.execution_context.model_name
        )

        self._calls[run_key] = {
            "run_id": run_key,
            "parent_run_id": (
                self._run_id_to_string(parent_run_id)
                if parent_run_id is not None
                else None
            ),
            "status": "started",
            "timestamp_start": self._utc_now(),
            "timestamp_end": None,
            "duration_seconds": None,
            "model_name": model_name,
            "tags": tags or [],
            "metadata": metadata or {},
            "messages": self._serialize_messages(messages),
            "response": None,
            "usage_metadata": None,
            "llm_output": None,
            "error": None,
        }

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Compatibilidade com modelos que utilizam prompts de texto,
        em vez da interface de mensagens.
        """
        run_key = self._run_id_to_string(run_id)
        start_time = time.perf_counter()

        # Evita duplicação caso on_chat_model_start já tenha sido chamado.
        if run_key in self._calls:
            return

        self._start_times[run_key] = start_time

        if self.execution_context.metrics.start_llm is None:
            self.execution_context.metrics.start_llm = start_time

        serialized_prompts = None

        if self.capture_prompt:
            serialized_prompts = [
                self._truncate(prompt, self.max_prompt_chars)
                for prompt in prompts
            ]

        self._calls[run_key] = {
            "run_id": run_key,
            "parent_run_id": (
                self._run_id_to_string(parent_run_id)
                if parent_run_id is not None
                else None
            ),
            "status": "started",
            "timestamp_start": self._utc_now(),
            "timestamp_end": None,
            "duration_seconds": None,
            "model_name": (
                (metadata or {}).get("ls_model_name")
                or serialized.get("name")
                or self.execution_context.model_name
            ),
            "tags": tags or [],
            "metadata": metadata or {},
            "prompts": serialized_prompts,
            "response": None,
            "usage_metadata": None,
            "llm_output": None,
            "error": None,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Executado quando o modelo termina de gerar a resposta.
        """
        run_key = self._run_id_to_string(run_id)
        end_time = time.perf_counter()

        call = self._calls.get(
            run_key,
            {
                "run_id": run_key,
                "parent_run_id": (
                    self._run_id_to_string(parent_run_id)
                    if parent_run_id is not None
                    else None
                ),
                "timestamp_start": None,
                "model_name": self.execution_context.model_name,
            },
        )

        start_time = self._start_times.get(run_key)

        duration = (
            end_time - start_time
            if start_time is not None
            else None
        )

        generated_text = None
        usage_metadata = None

        # Uma chamada pode conter mais de uma geração.
        if response.generations and response.generations[0]:
            generation = response.generations[0][0]

            message = getattr(generation, "message", None)

            if message is not None:
                generated_text = getattr(message, "content", None)
                usage_metadata = getattr(
                    message,
                    "usage_metadata",
                    None,
                )
            else:
                generated_text = getattr(
                    generation,
                    "text",
                    None,
                )

        call.update(
            {
                "status": "success",
                "timestamp_end": self._utc_now(),
                "duration_seconds": (
                    round(duration, 4)
                    if duration is not None
                    else None
                ),
                "response": self._truncate(
                    generated_text,
                    self.max_response_chars,
                ),
                "usage_metadata": usage_metadata,
                "llm_output": response.llm_output,
                "error": None,
            }
        )

        self.calls.append(call)

        # Considera o fim da última chamada observada.
        self.execution_context.metrics.end_llm = end_time

        self._synchronize_execution_context()

        self._calls.pop(run_key, None)
        self._start_times.pop(run_key, None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Executado quando ocorre uma falha na chamada ao modelo.
        """
        run_key = self._run_id_to_string(run_id)
        end_time = time.perf_counter()

        call = self._calls.get(
            run_key,
            {
                "run_id": run_key,
                "parent_run_id": (
                    self._run_id_to_string(parent_run_id)
                    if parent_run_id is not None
                    else None
                ),
                "timestamp_start": None,
                "model_name": self.execution_context.model_name,
            },
        )

        start_time = self._start_times.get(run_key)

        duration = (
            end_time - start_time
            if start_time is not None
            else None
        )

        call.update(
            {
                "status": "error",
                "timestamp_end": self._utc_now(),
                "duration_seconds": (
                    round(duration, 4)
                    if duration is not None
                    else None
                ),
                "response": None,
                "usage_metadata": None,
                "llm_output": None,
                "error": {
                    "type": error.__class__.__name__,
                    "message": str(error),
                },
            }
        )

        self.calls.append(call)

        self.execution_context.metrics.end_llm = end_time

        self._synchronize_execution_context()

        self._calls.pop(run_key, None)
        self._start_times.pop(run_key, None)

    def _synchronize_execution_context(self) -> None:
        """
        Transfere os dados monitorados para o ExecutionContext.
        """
        successful_calls = sum(
            call.get("status") == "success"
            for call in self.calls
        )

        failed_calls = sum(
            call.get("status") == "error"
            for call in self.calls
        )

        total_duration = sum(
            call.get("duration_seconds") or 0
            for call in self.calls
        )

        self.execution_context.metadata["llm_monitor"] = {
            "num_calls": len(self.calls),
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "total_duration_seconds": round(
                total_duration,
                4,
            ),
            "calls": self.calls,
        }