from uuid import uuid4

import streamlit as st

from src.application.rag_application import RAGApplication


st.set_page_config(
    page_title="Agente RAG",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def carregar_aplicacao() -> RAGApplication:
    """
    Carrega uma única instância da aplicação RAG.

    O cache evita reconstruir o banco vetorial, o retriever,
    o modelo e as chains a cada interação do Streamlit.
    """
    return RAGApplication()


def inicializar_estado() -> None:
    """
    Inicializa as variáveis mantidas durante a sessão do Streamlit.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_execution" not in st.session_state:
        st.session_state.last_execution = None


def carregar_historico_persistido(
    app: RAGApplication,
) -> None:
    """
    Carrega do SQLite as mensagens da sessão atual.

    O carregamento ocorre apenas quando a lista da interface
    ainda está vazia.
    """
    if st.session_state.messages:
        return

    history = app.get_history(
        session_id=st.session_state.session_id,
        limit=20,
    )

    for interaction in history:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": interaction["question"],
            }
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": interaction["answer"],
                "documents": interaction.get(
                    "documents",
                    [],
                ),
                "execution": {
                    "metrics": interaction.get(
                        "metrics",
                        {},
                    )
                },
            }
        )


def iniciar_nova_conversa() -> None:
    """
    Cria uma nova sessão sem apagar o histórico anterior do SQLite.
    """
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []
    st.session_state.last_execution = None


def formatar_fonte(documento) -> str:
    """
    Extrai uma representação legível da fonte recuperada.
    """
    if isinstance(documento, dict):
        source = documento.get(
            "source",
            "Fonte desconhecida",
        )
        page = documento.get(
            "page",
            "Página desconhecida",
        )
    else:
        source = documento.metadata.get(
            "source",
            "Fonte desconhecida",
        )
        page = documento.metadata.get(
            "page",
            "Página desconhecida",
        )

    return f"{source} — página {page}"


def exibir_fontes(documentos) -> None:
    """
    Exibe as fontes recuperadas em uma área expansível.
    """
    if not documentos:
        return

    with st.expander("Fontes recuperadas"):
        for index, documento in enumerate(
            documentos,
            start=1,
        ):
            st.write(
                f"{index}. {formatar_fonte(documento)}"
            )


def exibir_metricas(execution: dict | None) -> None:
    """
    Exibe as métricas da execução atual.
    """
    if not execution:
        return

    metrics = execution.get("metrics", {})

    if not metrics:
        return

    with st.expander("Métricas de desempenho"):
        column_1, column_2, column_3 = st.columns(3)

        column_1.metric(
            "Tempo total",
            f"{metrics.get('total_time_seconds') or 0:.2f} s",
        )

        column_2.metric(
            "Tempo do LLM",
            f"{metrics.get('llm_time_seconds') or 0:.2f} s",
        )

        column_3.metric(
            "Documentos",
            metrics.get("num_documents", 0),
        )

        st.write(
            {
                "Caracteres do contexto": metrics.get(
                    "context_chars",
                    0,
                ),
                "Tokens estimados do contexto": metrics.get(
                    "estimated_context_tokens",
                    0,
                ),
                "Tokens estimados da resposta": metrics.get(
                    "estimated_answer_tokens",
                    0,
                ),
                "Tokens estimados totais": metrics.get(
                    "estimated_total_tokens",
                    0,
                ),
            }
        )


def exibir_historico() -> None:
    """
    Renderiza todas as mensagens da conversa atual.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                exibir_fontes(
                    message.get("documents", [])
                )

                exibir_metricas(
                    message.get("execution")
                )


def configurar_sidebar(app: RAGApplication) -> None:
    """
    Constrói a barra lateral da interface.
    """
    with st.sidebar:
        st.title("Configurações")

        if st.button(
            "Nova conversa",
            use_container_width=True,
        ):
            iniciar_nova_conversa()
            st.rerun()

        st.divider()

        st.caption("Sessão atual")

        st.code(
            st.session_state.session_id,
            language=None,
        )

        statistics = app.get_statistics()

        st.metric(
            "Interações armazenadas",
            statistics.get(
                "total_interactions",
                0,
            ),
        )

        st.divider()

        st.caption(
            "Modelo local: qwen2.5:3b"
        )


def main() -> None:
    """
    Executa a interface Streamlit do agente RAG.
    """
    inicializar_estado()

    try:
        app = carregar_aplicacao()

    except Exception as error:
        st.error(
            f"Não foi possível inicializar o RAG: {error}"
        )
        st.stop()

    carregar_historico_persistido(app)
    configurar_sidebar(app)

    st.title("Agente RAG")
    st.caption(
        "Faça perguntas sobre Retrieval-Augmented Generation."
    )

    exibir_historico()

    question = st.chat_input(
        "Digite sua pergunta",
        max_chars=2000,
    )

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Respondendo aguarde..."
        ):
            try:
                result = app.ask(
                    question=question,
                    session_id=(
                        st.session_state.session_id
                    ),
                )

                answer = result["answer"]
                documents = result["documents"]
                execution = result["execution"]

                st.markdown(answer)
                exibir_fontes(documents)
                exibir_metricas(execution)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "documents": documents,
                        "execution": execution,
                    }
                )

                st.session_state.last_execution = (
                    execution
                )

            except ValueError as error:
                st.warning(str(error))

            except Exception as error:
                st.error(
                    f"Erro ao processar a pergunta: {error}"
                )


if __name__ == "__main__":
    main()