from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"
PASTA_BASE = BASE_DIR / "base"
PASTA_DB = BASE_DIR / "db"
PASTA_LOGS = BASE_DIR / "logs"
PASTA_DATA = BASE_DIR / "data"
PASTA_MEMORY = PASTA_DATA / "rag_memory.db"

MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
MODELO_LLM = "qwen2.5:3b"

MLFLOW_EXPERIMENT_NAME = "rag_local_experiment"
MLFLOW_DATABASE = PASTA_DATA / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DATABASE}"
MLFLOW_ENABLED = True

TEMPERATURA_LLM = 0.0
LIMITE_HISTORICO = 6
CAPTURAR_PROMPT = True