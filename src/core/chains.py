from typing import Any

try:
    from langchain_classic.chains import (
        create_history_aware_retriever,
        create_retrieval_chain,
    )
    from langchain_classic.chains.combine_documents import (
        create_stuff_documents_chain,
    )
except ModuleNotFoundError:
    from langchain.chains import (
        create_history_aware_retriever,
        create_retrieval_chain,
    )
    from langchain.chains.combine_documents import (
        create_stuff_documents_chain,
    )


def criar_retriever_com_historico(
    llm,
    retriever,
    contextualization_prompt,
):
    """
    Cria um retriever que reescreve perguntas dependentes
    do histórico antes de realizar a busca vetorial.
    """
    return create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualization_prompt,
    )


def criar_document_chain(llm, prompt):
    """
    Cria a chain que combina documentos e prompt.
    """
    return create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )


def criar_chain_rag(
    history_aware_retriever,
    document_chain,
):
    """
    Cria a cadeia RAG conversacional.
    """
    return create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=document_chain,
    )


def responder_pergunta(
    pergunta: str,
    retrieval_chain,
    chat_history: list,
    callbacks: list[Any] | None = None,
):
    """
    Executa uma pergunta utilizando o histórico da sessão.
    """
    config = {}

    if callbacks:
        config["callbacks"] = callbacks

    response = retrieval_chain.invoke(
        {
            "input": pergunta,
            "chat_history": chat_history,
        },
        config=config,
    )

    return (
        response["answer"],
        response["context"],
    )