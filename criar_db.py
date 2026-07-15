import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent
PASTA_BASE = BASE_DIR / "base"
PASTA_DB = BASE_DIR / "db"


def carregar_documentos():
    """
    Carrega todos os arquivos PDF presentes na pasta base/.
    Cada página carregada vira um documento do LangChain.
    """
    carregador = PyPDFDirectoryLoader(str(PASTA_BASE), glob="*.pdf")
    documentos = carregador.load()

    print(f"{len(documentos)} páginas/documentos carregados.")
    return documentos


def dividir_chunks(documentos):
    """
    Divide os documentos em pedaços menores.
    Isso é necessário porque o modelo não deve receber PDFs inteiros de uma vez.
    """
    separador = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )

    chunks = separador.split_documents(documentos)

    print(f"{len(chunks)} chunks criados.")
    return chunks


def criar_embeddings():
    """
    Cria o modelo local de embeddings.
    Esse modelo roda no computador, sem usar API paga.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def vetorizar_chunks(chunks):
    """
    Converte os chunks em vetores numéricos e salva no ChromaDB.
    """

    embeddings = criar_embeddings()

    if PASTA_DB.exists(): 
        shutil.rmtree(PASTA_DB)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PASTA_DB),
    )

    print("Banco vetorial criado com sucesso!")


def criar_db():
    """
    Pipeline completo de indexação:
    PDF → documentos → chunks → embeddings → ChromaDB.
    """
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)


if __name__ == "__main__":
    criar_db()