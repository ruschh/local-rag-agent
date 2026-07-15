def criar_retriever(db):
    """
    Cria o retriever com MMR para reduzir redundância entre chunks.
    """
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 10,
        },
    )