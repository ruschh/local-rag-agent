from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


BASE_DIR = Path(__file__).resolve().parent
PASTA_DB = BASE_DIR / "db"


def criar_embeddings():
    """
    Mesmo modelo usado na criação do banco vetorial.
    O modelo de consulta precisa ser o mesmo usado na indexação.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def carregar_banco_vetorial():
    """
    Carrega o banco Chroma já criado na pasta db/.
    """
    embeddings = criar_embeddings()

    db = Chroma(
        persist_directory=str(PASTA_DB),
        embedding_function=embeddings,
    )

    return db


def criar_retriever(db):
    """
    Cria o mecanismo de busca semântica.
    k=4 significa que o sistema buscará os 4 chunks mais relevantes.
    """
    return db.as_retriever(
        search_kwargs={"k": 4}
    )


def criar_llm_local():
    """
    Cria um modelo local utilizando o Ollama.

    O modelo já deve estar baixado com:

        ollama pull qwen2.5:3b
    """

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
    )

    return llm


def formatar_documentos(documentos):
    """
    Junta os chunks recuperados em um único texto de contexto.
    """
    return "\n\n".join(doc.page_content for doc in documentos)


def responder_pergunta(pergunta, retriever, llm):
    """
    Executa o fluxo RAG:

    pergunta → busca no banco vetorial → contexto → modelo gerador → resposta.
    """
    documentos_relevantes = retriever.invoke(pergunta)
    contexto = formatar_documentos(documentos_relevantes)

    prompt = ChatPromptTemplate.from_template(
        """
        Você é um assistente especializado.

        Responda utilizando SOMENTE as informações do contexto.

        Se a resposta não estiver presente no contexto, responda exatamente:

        "Não encontrei essa informação na base de documentos."

        Nunca invente informações.

        Nunca explique sua resposta.

        Nunca escreva comentários.

        Responda apenas ao que foi perguntado.

        ------------------------

        Contexto:

        {contexto}

        ------------------------

        Pergunta:

        {pergunta}

        ------------------------

        Resposta:
        """
    )

    chain = prompt | llm

    resposta = chain.invoke(
        {
            "contexto": contexto,
            "pergunta": pergunta,
        }
    )

    return resposta, documentos_relevantes


def main():
    """
    Loop simples de perguntas pelo terminal.
    A interface gráfica ficará para a próxima etapa.
    """
    db = carregar_banco_vetorial()
    retriever = criar_retriever(db)
    llm = criar_llm_local()

    print("RAG local iniciado.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Pergunta: ")

        if pergunta.lower() in ["sair", "exit", "quit"]:
            break

        resposta, fontes = responder_pergunta(pergunta, retriever, llm)

        print("\nResposta:")
        print(resposta.content)

        print("\nFontes recuperadas:")
        for i, doc in enumerate(fontes, start=1):
            fonte = doc.metadata.get("source", "Fonte desconhecida")
            pagina = doc.metadata.get("page", "Página desconhecida")
            print(f"{i}. {fonte} — página {pagina}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()