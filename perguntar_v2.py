from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from langchain_classic.chains import create_retrieval_chain  
from langchain_classic.chains.combine_documents import create_stuff_documents_chain 



BASE_DIR = Path(__file__).resolve().parent
PASTA_DB = BASE_DIR / "db"


def criar_embeddings():
    """
    Cria o mesmo modelo de embeddings usado na indexação dos documentos.

    Importante:
    o modelo de embeddings usado para consultar o banco vetorial deve ser
    o mesmo usado para criar o banco em criar_db.py.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def carregar_banco_vetorial():
    """
    Carrega o banco vetorial Chroma já criado na pasta db/.
    """
    embeddings = criar_embeddings()

    db = Chroma(
        persist_directory=str(PASTA_DB),
        embedding_function=embeddings,
    )

    return db


def criar_retriever(db):
    """
    Cria o retriever responsável por buscar os documentos mais relevantes.

    Nesta versão usamos MMR para reduzir redundância entre os chunks
    recuperados e aumentar a diversidade do contexto.
    """
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
        },
    )

    return retriever


def criar_llm_local():
    """
    Cria o modelo gerador local usando Ollama.

    Antes de executar este arquivo, o modelo precisa estar baixado:

        ollama pull qwen2.5:3b
    """
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
    )

    return llm

def criar_prompt():
    """
    Cria o prompt usado pela chain de documentos.

    Atenção:
    create_stuff_documents_chain espera a variável {context}.
    create_retrieval_chain envia a pergunta na variável {input}.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Você é um assistente especializado em responder perguntas com base em documentos.

                Use SOMENTE as informações presentes no contexto.

                Se a resposta não estiver presente no contexto, responda exatamente:

                "Não encontrei essa informação na base de documentos."

                Regras:
                    - Não invente informações.
                    - Não use conhecimento externo.
                    - Não explique o processo.
                    - Não escreva comentários sobre a resposta.
                    - Responda de forma direta e objetiva.
                """,
            ),
            (
                "human",
                """
                Contexto:

                {context}

                Pergunta:

                {input}
                """,
            ),
        ]
    )

    return prompt


def criar_document_chain(llm, prompt):
    """
    Cria a chain que combina os documentos recuperados em um único contexto.

    O método "stuff" simplesmente insere todos os documentos recuperados
    dentro do prompt, na variável {context}.
    """
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    return document_chain


def criar_chain_rag(retriever, document_chain):
    """
    Cria a Retrieval Chain de alto nível.

    Fluxo:
    pergunta do usuário → retriever → documentos relevantes → document_chain → resposta.
    """
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )

    return retrieval_chain


def responder_pergunta(pergunta, retrieval_chain):
    """
    Executa a pergunta usando a Retrieval Chain.

    A resposta retorna um dicionário com:
    - "answer": resposta final do modelo;
    - "context": documentos recuperados pelo retriever.
    """
    response = retrieval_chain.invoke(
        {
            "input": pergunta,
        }
    )

    resposta = response["answer"]
    fontes = response["context"]

    return resposta, fontes


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
    Executa o RAG V2 no terminal.

    Esta versão usa:
    ChromaDB + HuggingFaceEmbeddings + Ollama + create_retrieval_chain.
    """
    db = carregar_banco_vetorial()
    retriever = criar_retriever(db)
    llm = criar_llm_local()
    prompt = criar_prompt()

    document_chain = criar_document_chain(
        llm=llm,
        prompt=prompt,
    )

    retrieval_chain = criar_chain_rag(
        retriever=retriever,
        document_chain=document_chain,
    )

    print("RAG V2 iniciado.")
    print("Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Pergunta: ")

        pergunta = pergunta.strip()
        pergunta = " ".join(pergunta.split())

        if pergunta.lower() in ["sair", "exit", "quit"]:
            break

        resposta, fontes = responder_pergunta(
            pergunta=pergunta,
            retrieval_chain=retrieval_chain,
        )

        print("\nResposta:")
        print(resposta)

        exibir_fontes(fontes)

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()