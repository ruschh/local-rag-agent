from langchain_huggingface import HuggingFaceEmbeddings
from src.config import MODELO_EMBEDDINGS


def criar_embeddings():
    """
    Cria o modelo de embeddings usado tanto na indexação quanto na consulta.
    """
    return HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)