"""Contagem de tokens usada pela memória gerenciada.

Por que este módulo existe: a `ConversationTokenBufferMemory` (memória
gerenciada exigida pelo checkpoint) precisa contar tokens para saber o que
descartar, e o `ChatOllama` não implementa contagem própria. Sem isto, o
LangChain cairia no tokenizer GPT-2 do pacote `transformers` — que não é
dependência deste projeto e ainda tentaria BAIXAR o tokenizer da internet na
primeira chamada, o que quebraria o programa sem rede.

A contagem aqui é uma heurística simples (~4 caracteres por token), sem
depender de nenhuma biblioteca externa. Não é uma contagem exata do
tokenizer do gemma4, mas é estável, sempre disponível e suficiente para o
que a memória precisa: uma estimativa para decidir quando cortar o
histórico mais antigo.
"""

from __future__ import annotations

from typing import List

CARACTERES_POR_TOKEN = 4


def token_ids(texto: str) -> List[int]:
    """IDs de token — assinatura exigida por `custom_get_token_ids` do LangChain.

    Os IDs são fictícios (zeros); só a CONTAGEM deles importa para a memória.
    """
    return [0] * max(1, len(texto) // CARACTERES_POR_TOKEN)


def contar_tokens(texto: str) -> int:
    """Número estimado de tokens do texto."""
    return len(token_ids(texto))
