from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from src.memory.conversation_repository import (
    ConversationRepository,
)
from src.observability.execution_context import ExecutionContext


class ConversationMemory:
    """
    Interface de alto nível para a memória persistente.
    """

    def __init__(
        self,
        repository: Optional[
            ConversationRepository
        ] = None,
    ) -> None:
        self.repository = (
            repository
            or ConversationRepository()
        )

    def save(
        self,
        execution: ExecutionContext,
    ) -> int:
        return self.repository.save(execution)

    def get_chat_history(
        self,
        session_id: str,
        limit: int = 6,
    ) -> list[BaseMessage]:
        """
        Recupera interações anteriores e as converte
        em mensagens compreendidas pelo LangChain.
        """
        interactions = self.repository.list_by_session(
            session_id=session_id,
            limit=limit,
        )

        messages: list[BaseMessage] = []

        for interaction in interactions:
            messages.append(
                HumanMessage(
                    content=interaction["question"]
                )
            )

            messages.append(
                AIMessage(
                    content=interaction["answer"]
                )
            )

        return messages

    def last_messages(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self.repository.list_recent(limit)

    def search(
        self,
        term: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.repository.search(
            term=term,
            limit=limit,
        )

    def clear(self) -> int:
        return self.repository.delete_all()

    def statistics(self) -> dict[str, int]:
        return {
            "total_interactions": self.repository.count()
        }