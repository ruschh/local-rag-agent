from langchain_ollama import ChatOllama
from src.config import MODELO_LLM


def criar_llm_local():
    """
    Cria o modelo local via Ollama.

    Antes, execute:
        ollama pull qwen2.5:3b
    """
    return ChatOllama(
        model=MODELO_LLM,
        temperature=0,
    )