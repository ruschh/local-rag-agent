from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)


def criar_prompt_contextualizacao():
    """
    Reescreve perguntas dependentes do histórico para que
    possam ser compreendidas isoladamente pelo retriever.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Dado o histórico da conversa e a pergunta mais recente,
reescreva a pergunta como uma pergunta independente.

Não responda à pergunta.

Apenas produza a pergunta reescrita.

Se a pergunta já for independente, devolva-a sem alterações.
""",
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )


def criar_prompt():
    """
    Prompt usado para gerar a resposta final do RAG.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Você é um assistente especializado em responder
perguntas com base em documentos.

Use SOMENTE as informações presentes no contexto.

O histórico pode ser usado para compreender a conversa,
mas informações factuais devem estar fundamentadas no contexto.

Se a resposta não estiver no contexto, responda exatamente:

"Não encontrei essa informação na base de documentos."

Não invente informações.
Responda de forma direta e objetiva.
""",
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
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