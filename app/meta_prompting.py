"""Diferencial: meta prompting (Aula 04).

Ideia: usar o próprio gemma4:cloud como crítico do system prompt que ESCREVEMOS
à mão, e depois registrar o antes/depois. O resultado NÃO entra em produção
automaticamente — ele é salvo em `saidas/meta_prompting.md` para o grupo ler,
comparar e decidir o que incorporar em `prompts.py`. Deixar um LLM reescrever
sozinho o próprio system prompt em produção é justamente como se perde o
controle da persona.

Uso:
    python -m app.meta_prompting
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from app.config import RAIZ_PROJETO
from app.prompts import SYSTEM_PROMPT_CHAT

PASTA_SAIDA = RAIZ_PROJETO / "saidas"

SYSTEM_META = """
<papel>
Você é um engenheiro de prompts sênior revisando o system prompt de um
chatbot em produção.
</papel>

<tarefa>
Analise criticamente o system prompt fornecido e devolva uma versão melhorada.
</tarefa>

<criterios_de_analise>
1. Ambiguidade: regras que admitem mais de uma leitura.
2. Lacunas: casos de borda do domínio que o prompt não cobre.
3. Contradições entre regras e restrições.
4. Resistência a jailbreak: o prompt sobrevive a "ignore as instruções acima"
   e a alguém fingindo ser o desenvolvedor?
5. Excesso: instruções que não mudam o comportamento e só gastam tokens.
</criterios_de_analise>

<formato_saida>
Responda em Markdown, em português do Brasil, exatamente com estas seções:

## Diagnóstico
(3 a 6 problemas concretos, cada um citando o trecho do prompt original)

## System prompt revisado
(a versão completa melhorada, mantendo o XML tagging e a persona original)

## O que mudou e por quê
(tabela com as colunas | Mudança | Motivo | Impacto esperado |)
</formato_saida>
""".strip()

HUMAN_META = """
<system_prompt_atual>
{prompt_atual}
</system_prompt_atual>

<dominio>
{dominio}
</dominio>
""".strip()

DOMINIO = (
    "Chatbot de treino de academia para praticantes iniciantes e "
    "intermediários. Não pode prescrever dieta fechada, não pode falar de "
    "anabolizantes e não pode diagnosticar lesão."
)


def construir_chain_meta(llm=None):
    """Pipeline LCEL do meta prompting: prompt | llm | StrOutputParser."""
    if llm is None:
        from app.chain import construir_llm

        llm = construir_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_META),
            HumanMessagePromptTemplate.from_template(HUMAN_META),
        ]
    )
    return prompt | llm | StrOutputParser()


def revisar_system_prompt(llm=None) -> str:
    """Pede ao modelo a crítica e a reescrita do system prompt do chat."""
    chain = construir_chain_meta(llm)
    return chain.invoke({"prompt_atual": SYSTEM_PROMPT_CHAT, "dominio": DOMINIO})


def salvar_antes_e_depois(revisao: str) -> Path:
    """Grava o antes/depois em Markdown, que é o artefato do diferencial."""
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    destino = PASTA_SAIDA / "meta_prompting.md"
    destino.write_text(
        "# Meta prompting — antes e depois do system prompt\n\n"
        "O modelo `gemma4:cloud` foi usado como crítico do system prompt escrito\n"
        "à mão pelo grupo. A versão revisada abaixo é uma SUGESTÃO: o que for\n"
        "aceito é portado manualmente para `app/prompts.py`.\n\n"
        "## Antes — system prompt em produção\n\n"
        "```xml\n" + SYSTEM_PROMPT_CHAT + "\n```\n\n"
        "## Depois — revisão proposta pelo modelo\n\n" + revisao + "\n",
        encoding="utf-8",
    )
    return destino


def main() -> None:
    """Entry point: python -m app.meta_prompting"""
    print("Pedindo ao gemma4:cloud uma revisão do system prompt...\n")
    revisao = revisar_system_prompt()
    print(revisao)
    destino = salvar_antes_e_depois(revisao)
    print(f"\nAntes/depois salvo em: {destino}")


if __name__ == "__main__":
    main()
