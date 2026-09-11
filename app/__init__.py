"""CKP01 — Chatbot Profissional (FIAP · 2º Semestre · Módulo 1).

Pacote do chatbot de domínio "Treino de academia e prescrição de exercícios".
Módulos:
    config          -> leitura do .env e configuração central
    prompts         -> system prompts com XML tagging (Aula 04)
    schemas         -> modelos Pydantic v2 que validam as saídas
    memory_manager  -> as 3 estratégias de memória gerenciada (Aula 02)
    chain           -> arquitetura de 2 chains da Aula 03 (conversa + LCEL)
    context_rot     -> demonstração empírica da degradação por contexto
    meta_prompting  -> diferencial: o modelo reescrevendo o próprio prompt
    main            -> interface Gradio + entry point (python -m app.main)
"""

import warnings


def silenciar_avisos_de_legado() -> None:
    """Silencia os LangChainDeprecationWarning das classes exigidas pela Aula 03.

    O checkpoint pede explicitamente a arquitetura de 2 chains com
    `ConversationChain` + memórias do `langchain.memory`. Desde o LangChain
    0.2.7 essas classes são marcadas como deprecated (a substituta é a
    `RunnableWithMessageHistory`): elas funcionam normalmente, mas o aviso
    polui a saída do terminal e da interface.

    Precisa ser chamada DEPOIS de importar o langchain: o próprio pacote roda
    `surface_langchain_deprecation_warnings()` na importação, o que reinstala
    um filtro "default" à frente de qualquer filtro nosso.

    O silenciamento é restrito por mensagem — nenhum outro aviso do projeto é
    escondido.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*(ConversationChain|ConversationBufferMemory|"
        r"ConversationSummaryMemory|ConversationTokenBufferMemory|"
        r"migration guide|migrating_memory).*",
    )


__version__ = "1.0.0"
__all__ = [
    "config",
    "prompts",
    "schemas",
    "memory_manager",
    "chain",
    "context_rot",
    "meta_prompting",
]
