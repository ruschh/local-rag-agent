from src.core.vectorstore import carregar_banco_vetorial
from src.core.retriever import criar_retriever
from src.core.llm import criar_llm_local
from uuid import uuid4

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

from src.observability.logger_manager import LoggerManager
from src.observability.execution_context import ExecutionContext 
from src.observability.llm_monitor import LLMMonitor
from src.memory.conversation_memory import ConversationMemory


def exibir_fontes(fontes):
    """
    Exibe os documentos recuperados pelo retriever.
    """
    print("\nFontes recuperadas:")

    for i, doc in enumerate(fontes, start=1):
        fonte = doc.metadata.get("source", "Fonte desconhecida")
        pagina = doc.metadata.get("page", "Página desconhecida")
        print(f"{i}. {fonte} — página {pagina}")


def main():
    """
    Executa o RAG V3 modularizado.
    """
    logger = LoggerManager()
    memory = ConversationMemory()

    db = carregar_banco_vetorial()
    retriever = criar_retriever(db)
    llm = criar_llm_local()
    prompt = criar_prompt()

    contextualizacao_prompt = (criar_prompt_contextualizacao())
    history_aware_retriever =   (
        criar_retriever_com_historico(
            llm=llm,
            retriever=retriever,
            contextualization_prompt=(contextualizacao_prompt),
        )
    )
    document_chain = criar_document_chain(llm, prompt)
    retrieval_chain = criar_chain_rag(
        history_aware_retriever=(history_aware_retriever),
        document_chain=document_chain,
    )

    print("RAG V3 iniciado.")
    print("Digite 'sair' para encerrar.\n")

    session_id = input("ID da sessão [terminal_default]: ").strip()
    if not session_id:
        session_id = "terminal_default"

    print(f"Sessão ativa: {session_id}\n")

    while True:
        pergunta = input("Pergunta: ")

        pergunta = pergunta.strip()
        pergunta = " ".join(pergunta.split())

        if pergunta.lower() in ["sair", "exit", "quit"]:
            break

        chat_history = memory.get_chat_history(
            session_id=session_id,
            limit=6,
        )
        
        execution = ExecutionContext(
            question=pergunta,
            session_id=session_id,
            model_name="qwen2.5:3b",
            temperature=0.0,
        )

        llm_monitor = LLMMonitor(
            execution_context=execution,
            capture_prompt=True,
        )

        try:
            resposta, fontes = responder_pergunta(
                pergunta=pergunta, 
                retrieval_chain=retrieval_chain,
                chat_history=chat_history,
                callbacks=[llm_monitor],
            )

            execution.set_result(
                answer=resposta,
                documents=fontes,
            )

            print("\nResposta:")
            print(resposta)

            exibir_fontes(fontes)

        except Exception as error:
            execution.set_error(error)

            print(f"\nErro ao responder: {error}")

        finally:
            execution.finish()
            logger.log_execution(execution)

            try:
                memory.save(execution)

            except Exception as memory_error:
                logger.log_event(
                    event_type="memory_error",
                    payload={
                        "execution_id": execution.execution_id,
                        "error": str(memory_error),
                    }
                )

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()