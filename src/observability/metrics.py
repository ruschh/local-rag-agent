import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class PerformanceMetrics:
    """
    Mede métricas de desempenho de uma execução RAG.

    Métricas coletadas:
    - tempo total da requisição;
    - tempo do retriever;
    - tempo do LLM;
    - número de documentos recuperados;
    - tamanho total do contexto;
    - tamanho da pergunta;
    - tamanho da resposta;
    - estimativa simples de tokens.
    """

    start_total: float = field(default_factory=time.perf_counter)
    end_total: Optional[float] = None

    start_retriever: Optional[float] = None
    end_retriever: Optional[float] = None

    start_llm: Optional[float] = None
    end_llm: Optional[float] = None

    num_documents: int = 0

    context_chars: int = 0
    question_chars: int = 0
    answer_chars: int = 0

    estimated_context_tokens: int = 0
    estimated_question_tokens: int = 0
    estimated_answer_tokens: int = 0
    estimated_total_tokens: int = 0

    def start_retrieval_timer(self) -> None:
        """Inicia a medição do tempo de recuperação."""
        self.start_retriever = time.perf_counter()
        self.end_retriever = None

    def end_retrieval_timer(self) -> None:
        """Encerra a medição do tempo de recuperação."""
        if self.start_retriever is None:
            raise RuntimeError(
                "O temporizador do retriever não foi iniciado."
            )

        self.end_retriever = time.perf_counter()

    def start_llm_timer(self) -> None:
        """Inicia a medição do tempo de geração do LLM."""
        self.start_llm = time.perf_counter()
        self.end_llm = None

    def end_llm_timer(self) -> None:
        """Encerra a medição do tempo de geração do LLM."""
        if self.start_llm is None:
            raise RuntimeError(
                "O temporizador do LLM não foi iniciado."
            )

        self.end_llm = time.perf_counter()

    def end_total_timer(self) -> None:
        """Encerra a medição do tempo total da execução."""
        self.end_total = time.perf_counter()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Calcula uma estimativa simples do número de tokens.

        Aproximação utilizada:
            1 token ≈ 4 caracteres.

        Esta métrica não substitui a tokenização real do modelo.
        """
        if not text:
            return 0

        return math.ceil(len(text) / 4)

    def collect_text_metrics(
        self,
        question: str,
        answer: str,
        documents: Sequence[Any],
    ) -> None:
        """
        Coleta métricas textuais da pergunta, da resposta e dos
        documentos recuperados.
        """
        context = "\n\n".join(
            getattr(document, "page_content", "")
            for document in documents
        )

        self.num_documents = len(documents)

        self.context_chars = len(context)
        self.question_chars = len(question)
        self.answer_chars = len(answer)

        self.estimated_context_tokens = self.estimate_tokens(context)
        self.estimated_question_tokens = self.estimate_tokens(question)
        self.estimated_answer_tokens = self.estimate_tokens(answer)

        self.estimated_total_tokens = (
            self.estimated_context_tokens
            + self.estimated_question_tokens
            + self.estimated_answer_tokens
        )

    @staticmethod
    def _calculate_duration(
        start: Optional[float],
        end: Optional[float],
    ) -> Optional[float]:
        """
        Calcula a duração entre dois instantes, quando ambos existem.
        """
        if start is None or end is None:
            return None

        return end - start

    def reset(self) -> None:
        """
        Reinicia todas as métricas para uma nova execução.
        """
        self.start_total = time.perf_counter()
        self.end_total = None

        self.start_retriever = None
        self.end_retriever = None

        self.start_llm = None
        self.end_llm = None

        self.num_documents = 0

        self.context_chars = 0
        self.question_chars = 0
        self.answer_chars = 0

        self.estimated_context_tokens = 0
        self.estimated_question_tokens = 0
        self.estimated_answer_tokens = 0
        self.estimated_total_tokens = 0

    def to_dict(self) -> dict:
        """
        Retorna as métricas em formato serializável para JSON.
        """
        total_time = self._calculate_duration(
            self.start_total,
            self.end_total,
        )

        retriever_time = self._calculate_duration(
            self.start_retriever,
            self.end_retriever,
        )

        llm_time = self._calculate_duration(
            self.start_llm,
            self.end_llm,
        )

        return {
            "total_time_seconds": (
                round(total_time, 4)
                if total_time is not None
                else None
            ),
            "retriever_time_seconds": (
                round(retriever_time, 4)
                if retriever_time is not None
                else None
            ),
            "llm_time_seconds": (
                round(llm_time, 4)
                if llm_time is not None
                else None
            ),
            "num_documents": self.num_documents,
            "context_chars": self.context_chars,
            "question_chars": self.question_chars,
            "answer_chars": self.answer_chars,
            "estimated_context_tokens": self.estimated_context_tokens,
            "estimated_question_tokens": self.estimated_question_tokens,
            "estimated_answer_tokens": self.estimated_answer_tokens,
            "estimated_total_tokens": self.estimated_total_tokens,
        }