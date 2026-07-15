from typing import Any

from src.config import (
    MODELO_LLM,
    TEMPERATURA_LLM,
    LIMITE_HISTORICO,
    CAPTURAR_PROMPT,
)

from src.core.vectorstore import carregar_banco_vetorial
from src.core.retriever import criar_retriever
from src.core.llm import criar_llm_local
from src.core.prompts import (
    criar_prompt,
    criar_prompt_contextualizacao,
)
from src.core.chains import (
    criar_retriever_com_historico,
    criar_document_chain,
    criar_chain_rag,
    responder_pergunta,
)

from src.memory.conversation_memory import ConversationMemory

from src.observability.execution_context import ExecutionContext
from src.observability.llm_monitor import LLMMonitor
from src.observability.logger_manager import LoggerManager
from src.observability.mlflow_manager import MLflowManager


class RAGApplication:
    """
    Orquestra toda a aplicação RAG.

    Responsabilidades:
    - carregar banco vetorial;
    - criar retriever e LLM;
    - montar as chains;
    - recuperar memória conversacional;
    - executar perguntas;
    - coletar métricas;
    - monitorar chamadas ao LLM;
    - registrar logs;
    - persistir conversas em SQLite.
    """

    def __init__(
        self,
        model_name: str = MODELO_LLM,
        temperature: float = TEMPERATURA_LLM,
        history_limit: int = LIMITE_HISTORICO,
        capture_prompt: bool = CAPTURAR_PROMPT,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.history_limit = history_limit
        self.capture_prompt = capture_prompt

        self.logger = LoggerManager()
        self.memory = ConversationMemory()

        self.retrieval_chain = self._build_rag_chain()

        self.mlflow_manager = MLflowManager()

    def _build_rag_chain(self):
        """
        Constrói o pipeline RAG conversacional.
        """
        db = carregar_banco_vetorial()
        retriever = criar_retriever(db)

        llm = criar_llm_local()

        contextualization_prompt = (
            criar_prompt_contextualizacao()
        )

        history_aware_retriever = (
            criar_retriever_com_historico(
                llm=llm,
                retriever=retriever,
                contextualization_prompt=(
                    contextualization_prompt
                ),
            )
        )

        response_prompt = criar_prompt()

        document_chain = criar_document_chain(
            llm=llm,
            prompt=response_prompt,
        )

        return criar_chain_rag(
            history_aware_retriever=(
                history_aware_retriever
            ),
            document_chain=document_chain,
        )

    @staticmethod
    def normalize_question(question: str) -> str:
        """
        Remove espaços desnecessários sem alterar
        maiúsculas, minúsculas ou acentos.
        """
        return " ".join(question.strip().split())

    def ask(
        self,
        question: str,
        session_id: str,
    ) -> dict[str, Any]:
        """
        Executa uma pergunta completa no RAG.

        Retorna um dicionário contendo:
        - resposta;
        - fontes;
        - dados completos da execução.
        """
        normalized_question = self.normalize_question(
            question
        )

        if not normalized_question:
            raise ValueError(
                "A pergunta não pode estar vazia."
            )

        if not session_id.strip():
            raise ValueError(
                "O identificador da sessão não pode estar vazio."
            )

        execution = ExecutionContext(
            question=normalized_question,
            session_id=session_id,
            model_name=self.model_name,
            temperature=self.temperature,
        )

        llm_monitor = LLMMonitor(
            execution_context=execution,
            capture_prompt=self.capture_prompt,
        )

        try:
            chat_history = self.memory.get_chat_history(
                session_id=session_id,
                limit=self.history_limit,
            )

            answer, documents = responder_pergunta(
                pergunta=normalized_question,
                retrieval_chain=self.retrieval_chain,
                chat_history=chat_history,
                callbacks=[llm_monitor],
            )

            execution.set_result(
                answer=answer,
                documents=documents,
            )

        except Exception as error:
            execution.set_error(error)
            raise

        finally:
            execution.finish()

            self.logger.log_execution(execution)

            try:
                self.memory.save(execution)
            except Exception as memory_error:
                self.logger.log_event(
                    event_type="memory_error",
                    payload={
                        "execution_id": execution.execution_id,
                        "error": str(memory_error),
                    },
                )

            try:
                self.mlflow_manager.log_execution(
                    execution=execution,
                    interface="streamlit",
                    additional_params={
                        "history_limit": self.history_limit,
                        "capture_prompt": self.capture_prompt,
                    },

                )

            except Exception as memory_error:
                self.logger.log_event(
                    event_type="mlflow_error",
                    payload={
                        "execution_id": execution.execution_id,
                        "session_id": session_id,
                        "error": str(mlflow_error),
                    },
                )

        return {
            "answer": execution.answer,
            "documents": execution.documents,
            "execution": execution.to_dict(),
        }

    def get_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retorna o histórico persistido de uma sessão.
        """
        return self.memory.repository.list_by_session(
            session_id=session_id,
            limit=limit,
        )

    def search_history(
        self,
        term: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Pesquisa perguntas e respostas persistidas.
        """
        return self.memory.search(
            term=term,
            limit=limit,
        )

    def clear_history(self) -> int:
        """
        Remove todo o histórico persistido.
        """
        return self.memory.clear()

    def get_statistics(self) -> dict[str, int]:
        """
        Retorna estatísticas básicas da memória.
        """
        return self.memory.statistics()