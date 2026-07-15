from langchain_chroma import Chroma
from src.config import PASTA_DB
from src.core.embeddings import criar_embeddings


def carregar_banco_vetorial():
    """
    Carrega o banco vetorial Chroma persistido em db/.
    """
    embeddings = criar_embeddings()

    return Chroma(
        persist_directory=str(PASTA_DB),
        embedding_function=embeddings,
    )