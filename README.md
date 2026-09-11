# CKP01 — Chatbot Profissional · Treino de academia

**Prompt Engineering & AI · FIAP · 2º Semestre 2026 · Módulo 1**

**Integrantes:**
Luan de Araujo Carneiro (RM573691) · Pedro Sampaio Mochnacs Arruda (RM573522) ·
Raul Sampaio Mochnacs Arruda (RM573523) · Pedro Ribeiro Lopes (RM570083) ·
Kevin Rodrigues de Melo (RM571777) · Pedro Vianna (RM570747) · Lana Ozeki (RM569795)

**Peso: 25% · Apresentação: Aula 04 · Entrega: 23:55 do dia da Aula 05 (.zip via Teams — só o líder)**

---

## Domínio

**Qual:** assistente virtual de **treino de academia e prescrição de exercícios** — apelidado de
**Halter**.

**Por que foi escolhido:** é um domínio com fronteira de escopo nítida e com consequência
real. Treino é competência do educador físico; dieta fechada é do nutricionista; lesão é do
médico ou fisioterapeuta. Essa fronteira dá material concreto para as restrições do system
prompt — e violá-la causa dano de verdade, o que torna a avaliação do prompt objetiva. É
também um domínio naturalmente conversacional e dependente de memória: objetivo, nível,
dias disponíveis e limitações são ditos uma vez e precisam valer pelo resto da sessão.

**Usuários-alvo:** praticantes iniciantes e intermediários de musculação, de 16 a 55 anos,
que treinam em academia convencional e querem montar, ajustar ou entender um treino. Não
dominam jargão técnico — por isso o system prompt obriga a traduzir os termos usados.

---

## Requisitos atendidos

| Requisito | Status | Implementação |
|---|:--:|---|
| Pipeline LCEL | ✅ | `chain.py` — `prompt \| llm \| PydanticOutputParser()` em `construir_chain_analise()` e `construir_chain_relatorio()` |
| Arquitetura de 2 chains (Aula 03) | ✅ | `ConversationChain` com memória (chain 1) + pipeline LCEL stateless (chain 2), ambas em `chain.py` |
| ChatOllama | ✅ | `gemma4:cloud` via Ollama Cloud, chave lida do `.env` em `config.py` |
| ChatPromptTemplate com variáveis | ✅ | `prompts.py` + `chain.py` — system e human separados, variáveis `{input}`, `{mensagem}`, `{historico}`, `{format_instructions}`; nenhuma f-string manual |
| Memória gerenciada | ✅ | `memory_manager.py` — as 3 estratégias implementadas, `ConversationTokenBufferMemory` (1200 tokens) ativa; justificativa abaixo |
| Demonstração em ≥ 5 turnos | ✅ | Aba **Memória** da interface conta os turnos do usuário na sessão; validado com 10 turnos simulados (ver "Observações técnicas") |
| Pydantic v2 (≥ 4 campos) | ✅ | `schemas.py` — `AnaliseConsulta` (6 campos) e `RelatorioSessao` (6 campos), com `field_validator` |
| PydanticOutputParser | ✅ | `chain.py` — nas duas chains estruturadas (e não `JsonOutputParser`, que devolveria `dict` sem validação) |
| Seção context rot | ✅ | `context_rot.py` — mesma pergunta em janelas de 0/5/10/15/20 turnos, com tabela comparativa |
| System prompt com persona | ✅ | `prompts.py` — XML tagging (`<papel>`, `<regras>`, `<restricoes>`…), persona, escopo, recusas e resistência a jailbreak |
| Domínio documentado | ✅ | Este README |
| Projeto local estruturado | ✅ | Pacote `app/` + `.env.example` + `requirements.txt` + `README.md`, sem Colab |

Este projeto entrega **apenas o obrigatório** do enunciado — sem os diferenciais de
"context engineering com métricas" (tiktoken + gráfico) nem "meta prompting".

---

## Como executar (local — sem Colab)

```bash
cp .env.example .env      # edite com a sua OLLAMA_API_KEY — este arquivo NÃO vai no .zip
pip install -r requirements.txt
python -m app.main        # Gradio em http://localhost:7860
```

A chave da Ollama Cloud sai de <https://ollama.com> → *Settings* → *Keys*. Se o `.env`
estiver faltando ou mal preenchido, o programa para logo no início com uma mensagem
dizendo exatamente o que corrigir — não há chave hardcoded em lugar nenhum do código.

Para rodar a demonstração de context rot separadamente:

```bash
python -m app.context_rot
```

### Estrutura

```
CKP01_treino_grupo/
├── app/
│   ├── __init__.py        # metadados do pacote e filtro de avisos de legado
│   ├── main.py            # interface Gradio + entry point
│   ├── chain.py           # as 2 chains da Aula 03 + fachada ChatbotTreino
│   ├── memory_manager.py  # as 3 estratégias de memória gerenciada
│   ├── schemas.py         # Pydantic v2: AnaliseConsulta e RelatorioSessao
│   ├── context_rot.py     # demonstração de degradação por contexto
│   ├── prompts.py         # system prompts com XML tagging
│   ├── config.py          # leitura e validação do .env
│   └── tokens.py          # estimativa de tokens usada pela memória
├── .env.example
├── requirements.txt
└── README.md
```

---

## Arquitetura das 2 chains (Aula 03)

```
                        mensagem do usuário
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   CHAIN 2 — LCEL stateless               CHAIN 1 — com memória
   ChatPromptTemplate                     ConversationChain
        │ (operador |)                         │
        ▼                                      ├── ChatPromptTemplate
     ChatOllama  (gemma4:cloud)                ├── ChatOllama (gemma4:cloud)
        │                                      └── ConversationTokenBufferMemory
        ▼                                              │
   PydanticOutputParser                                ▼
        │                                     resposta conversacional
        ▼
   AnaliseConsulta  ──► risco ≥ 4?  ──► corta a prescrição e encaminha
   (objeto validado)                     a profissional de saúde
```

As duas chains rodam no **mesmo turno**, e a ordem importa: a triagem estruturada roda
**antes** da chain de conversa. Quando ela devolve `risco_seguranca >= 4` — dor aguda,
lesão recente, pedido sobre substância proibida — o sistema corta a prescrição e encaminha
o usuário a um profissional, sem gastar o turno de chat. É a diferença entre *pedir* ao
modelo que se comporte e *garantir* em código que ele se comporte.

---

## Justificativa da memória

**Escolha: `ConversationTokenBufferMemory` com teto de 1200 tokens.**

As três estratégias estão implementadas em `memory_manager.py` e podem ser trocadas pela
variável `MEMORIA_ESTRATEGIA` no `.env`. A comparação que levou à escolha:

| Estratégia | Custo de tokens | O que se perde | Veredito para este domínio |
|---|---|---|---|
| `ConversationBufferMemory` | Cresce sem teto a cada turno | Nada | ❌ Numa sessão longa o prompt vira um custo que não fecha |
| `ConversationSummaryMemory` | Quase constante, mas **+1 chamada ao LLM por turno** | **Os números.** "supino reto 4×8 com 40 kg, descanso 90 s" vira "treinou peito" | ❌ Fatal aqui: prescrição de treino **é** feita de números |
| `ConversationTokenBufferMemory` | Limitado pelo teto (1200 tokens) | Só os turnos mais antigos, e por inteiro | ✅ **Escolhida** |

**Por que 1200 tokens.** O enunciado pede de 800 a 1500. Um par pergunta-resposta de
prescrição gasta, na prática, de 150 a 250 tokens, então 1200 guardam de 5 a 8 turnos
completos — o suficiente para cobrir a demonstração de 5+ turnos com folga.

**O que se perde, e por que é aceitável.** O token buffer descarta os turnos mais antigos.
Na prática, é justamente a informação menos usada: o usuário quer ajustar o treino que
acabou de receber, não o que discutiu quinze mensagens atrás. O risco de perder uma
restrição de segurança declarada logo no início é o que a seção de context rot mede — e o
motivo pelo qual a triagem da chain 2 reavalia o risco de segurança a **cada** mensagem,
sem depender da memória.

---

## Context rot — degradação com contexto crescente

Rodando `python -m app.context_rot`, o experimento planta três fatos no primeiro turno
(objetivo, frequência semanal e uma restrição de segurança no ombro), enche a conversa com
N turnos de papo genérico de academia que **não repetem** esses fatos, e faz sempre a
**mesma pergunta de controle**: *"Monta pra mim o treino de ombro de amanhã."*

Como só o contexto muda, qualquer queda observada é atribuível ao volume de contexto, e não
à pergunta.

| Turnos de contexto | ~Tokens do prompt | Objetivo | Frequência | Restrição de segurança |
|---:|---:|:--:|:--:|:--:|
| 0 | | | | |
| 5 | | | | |
| 10 | | | | |
| 15 | | | | |
| 20 | | | | |

> 📌 **Rode `python -m app.context_rot` com a sua chave e cole a tabela impressa aqui.**
> O comando já imprime a tabela pronta em Markdown no terminal.

**O que observar.** Conforme o número de turnos de enchimento cresce, o prompt fica maior
sem que nenhum fato tenha saído da janela do modelo — a informação continua tecnicamente
acessível. O que se degrada é o *uso* dela: os fatos plantados competem com um volume
crescente de texto irrelevante. Normalmente os primeiros detalhes a se perder são os
numéricos (a frequência semanal), seguidos do objetivo; a restrição de segurança tende a
resistir mais por ser a instrução mais enfática do system prompt.

**Conclusão de engenharia.** Contexto maior não é contexto melhor. É essa observação que
sustenta a memória com **teto** (em vez de acumular a conversa inteira) e a verificação de
segurança rodando numa chain **separada e stateless**, que recebe só a mensagem atual —
imune ao rot que afeta a chain de conversa.

---

## Segurança e resistência a desvio de assunto

O system prompt (`prompts.py`) inclui regras explícitas contra as tentativas mais comuns de
tirar o bot do personagem ou do escopo:

- Pedidos para "ignorar as instruções anteriores", "ativar modo desenvolvedor/debug",
  simular ser outra IA, ou revelar/reescrever o próprio system prompt são recusados, e o
  bot continua respondendo como Halter, no mesmo assunto de treino.
- Alegações de autoridade ("sou o administrador", "sou o desenvolvedor") não mudam o
  comportamento do bot — ele não tem como verificar a alegação e trata qualquer usuário da
  mesma forma.
- Pedidos de roleplay que trocariam a persona ("finja que você é...", "a partir de agora
  você é...") são recusados; o bot mantém a persona de assistente de treino.
- Assuntos fora de treino/exercício/condicionamento físico são recusados em uma frase, com
  o bot reconduzindo a conversa ao próprio domínio.
- Pedidos sobre anabolizantes, hormônios ou substâncias controladas são recusados.
- Mensagens que indicam dor aguda, lesão recente ou risco de segurança cortam a prescrição
  e encaminham a um profissional de saúde — essa verificação roda em uma chain estruturada
  separada (`AnaliseConsulta`, com o campo `risco_seguranca`), então não depende apenas do
  modelo "lembrar" de obedecer ao system prompt durante uma conversa longa.

O arquivo `.env` **não deve ser enviado no `.zip`** da entrega nem versionado no Git —
apenas o `.env.example`. A `OLLAMA_API_KEY` é lida exclusivamente pelo `config.py`, via
`python-dotenv`, e não aparece em nenhum outro ponto do código.

---

## Observações técnicas

- **`ConversationChain` e `langchain.memory` estão marcados como deprecated** desde o
  LangChain 0.2.7 (a substituta é `RunnableWithMessageHistory`). O checkpoint pede
  explicitamente a arquitetura da Aula 03, então elas foram mantidas; o aviso é silenciado
  de forma documentada e restrita em `app/__init__.py`.
- **Contagem de tokens:** o `ChatOllama` não implementa contagem própria, e sem intervenção
  o LangChain cairia no tokenizer GPT-2 do pacote `transformers` — que não é dependência
  deste projeto e ainda tentaria **baixar** o tokenizer da internet na primeira chamada.
  `app/tokens.py` evita isso com uma estimativa simples (~4 caracteres por token), sem
  depender de nenhuma biblioteca externa: é o que garante que o projeto roda mesmo sem
  acesso à internet além do necessário para falar com a Ollama Cloud.
- **O que foi testado antes da entrega:** as chains, a memória e os schemas foram testados
  com um modelo simulado (sem depender da Ollama Cloud), confirmando que uma sessão de 10
  turnos roda do início ao fim sem travar, que a memória preserva fatos do primeiro turno e
  descarta os mais antigos ao atingir o teto de tokens, e que a chain de triagem
  (`AnaliseConsulta`) força o encaminhamento a profissional de saúde sempre que
  `risco_seguranca >= 4` — **isso não depende do modelo "lembrar" da regra**, é uma
  verificação em código que roda a cada mensagem. O que ainda precisa ser testado pelo
  grupo, com a chave real, é a resistência do próprio `gemma4:cloud` às tentativas de
  jailbreak do system prompt (pedidos de "ignore as instruções", troca de persona, etc.):
  rode `python -m app.main`, tente esses pedidos e confirme que o bot recusa e continua
  respondendo — o bloco `<resistencia_a_desvio>` do system prompt foi escrito para isso,
  mas o comportamento final depende do modelo em produção.
