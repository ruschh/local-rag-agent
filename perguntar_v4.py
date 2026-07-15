from src.application.rag_application import RAGApplication


def exibir_fontes(documentos):
    """
    Exibe as fontes recuperadas pelo RAG.
    """
    print("\nFontes recuperadas:")

    for indice, documento in enumerate(
        documentos,
        start=1,
    ):
        fonte = documento.metadata.get(
            "source",
            "Fonte desconhecida",
        )

        pagina = documento.metadata.get(
            "page",
            "Página desconhecida",
        )

        print(
            f"{indice}. {fonte} — página {pagina}"
        )


def main():
    """
    Interface de terminal da versão 3 do RAG.
    """
    app = RAGApplication()

    print("RAG V3 iniciado.")
    print("Digite 'sair' para encerrar.\n")

    session_id = input(
        "ID da sessão [terminal_default]: "
    ).strip()

    if not session_id:
        session_id = "terminal_default"

    print(f"Sessão ativa: {session_id}\n")

    while True:
        pergunta = input("Pergunta: ")

        if pergunta.strip().casefold() in {
            "sair",
            "exit",
            "quit",
        }:
            break

        try:
            resultado = app.ask(
                question=pergunta,
                session_id=session_id,
            )

            print("\nResposta:")
            print(resultado["answer"])

            exibir_fontes(
                resultado["documents"]
            )

        except ValueError as error:
            print(f"\nEntrada inválida: {error}")

        except Exception as error:
            print(f"\nErro ao responder: {error}")

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()