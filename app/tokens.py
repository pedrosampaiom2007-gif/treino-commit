"""Contagem de tokens compartilhada pelo projeto.

Por que este módulo existe: a `ConversationTokenBufferMemory` precisa contar
tokens para saber o que descartar, e o `ChatOllama` não implementa contagem
própria. Sem isto, o LangChain cai no tokenizer GPT-2 do pacote `transformers`
— que não está nas dependências e ainda baixaria o tokenizer da internet na
primeira chamada. Injetamos então um contador baseado em tiktoken via o campo
`custom_get_token_ids` do modelo.

Ressalva honesta: o gemma4 usa um tokenizer SentencePiece próprio, então o
`cl100k_base` é uma APROXIMAÇÃO (erro típico de 10 a 20% em português). Como o
mesmo contador é usado no teto da memória e na medição de context rot, a
comparação relativa permanece válida — e o teto de tokens fica conservador.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

CODIFICACAO = "cl100k_base"


@lru_cache(maxsize=1)
def _codificador():
    """Carrega o tiktoken uma única vez. Devolve None se não estiver instalado."""
    try:
        import tiktoken

        return tiktoken.get_encoding(CODIFICACAO)
    except Exception:
        return None


def token_ids(texto: str) -> List[int]:
    """IDs dos tokens do texto — assinatura exigida por `custom_get_token_ids`.

    Sem tiktoken, cai numa heurística de ~4 caracteres por token, devolvendo
    IDs fictícios apenas para que a CONTAGEM (que é o que a memória usa) seja
    plausível.
    """
    codificador = _codificador()
    if codificador is not None:
        return codificador.encode(texto)
    return [0] * max(1, len(texto) // 4)


def contar_tokens(texto: str) -> int:
    """Número de tokens do texto."""
    return len(token_ids(texto))


def usando_tiktoken() -> bool:
    """True quando a contagem é a do tiktoken, e não a heurística."""
    return _codificador() is not None
