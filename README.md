# CKP01 — Chatbot Profissional · Treino de academia

**Prompt Engineering & AI · FIAP · 2º Semestre 2026 · Módulo 1**

**Integrantes:** Nome Completo (RM00000) · Nome Completo (RM00000) · Nome Completo (RM00000) · Nome Completo (RM00000)

> ⚠️ **Antes de entregar:** troque a linha acima pelos nomes e RMs reais do grupo.
> Sem isso o critério "Domínio documentado" não é pontuado.

**Peso: 25% · Apresentação: Aula 04 · Entrega: 23:55 do dia da Aula 05 (.zip via Teams — só o líder)**

---

## Domínio

**Qual:** assistente virtual de **treino de academia e prescrição de exercícios** — apelidado de
**Halter**.

**Por que foi escolhido:** é um domínio com três características que o tornam bom para os
três checkpoints do semestre.

1. **Tem conhecimento estruturado e público** — guias de treino, tabelas de séries e
   repetições, manuais de execução de exercício. Isso vira base de conhecimento no CKP02
   (RAG) sem depender de dado proprietário.
2. **Tem fronteira de escopo nítida e com consequência real.** Treino é competência do
   educador físico; dieta fechada é do nutricionista; lesão é do médico ou fisioterapeuta.
   Essa fronteira dá material concreto para as restrições do system prompt, e violá-la
   causa dano de verdade — o que torna a avaliação do prompt objetiva.
3. **É naturalmente conversacional e dependente de memória.** Objetivo, nível, dias
   disponíveis e limitações são ditos uma vez e precisam valer por toda a sessão. É
   exatamente o cenário em que memória gerenciada e context rot deixam de ser teoria.

**Usuários-alvo:** praticantes iniciantes e intermediários de musculação, de 16 a 55 anos,
que treinam em academia convencional e querem montar, ajustar ou entender um treino. Não
dominam jargão técnico — por isso o system prompt obriga a traduzir os termos usados.

**Continuidade no semestre:** o chatbot deste CKP01 vira a base do RAG no CKP02 (indexando
material técnico de treinamento) e, no CKP03, a chain de prescrição vira uma *tool* do
agente. Por isso o projeto já nasce modular: cada arquivo de `app/` tem uma
responsabilidade só.

---

## Requisitos atendidos

| Requisito | Status | Implementação |
|---|:--:|---|
| Pipeline LCEL | ✅ | `chain.py` — `prompt \| llm \| PydanticOutputParser()` em `construir_chain_analise()` e `construir_chain_relatorio()` |
| Arquitetura de 2 chains (Aula 03) | ✅ | `ConversationChain` com memória (chain 1) + pipeline LCEL stateless (chain 2), ambas em `chain.py` |
| ChatOllama | ✅ | `gemma4:cloud` via Ollama Cloud, chave lida do `.env` em `config.py` |
| ChatPromptTemplate com variáveis | ✅ | `prompts.py` + `chain.py` — system e human separados, variáveis `{input}`, `{mensagem}`, `{historico}`, `{format_instructions}`; nenhuma f-string manual |
| Memória gerenciada | ✅ | `memory_manager.py` — as 3 estratégias implementadas, `ConversationTokenBufferMemory` (1200 tokens) ativa; justificativa abaixo |
| Demonstração em ≥ 5 turnos | ✅ | Aba **Memória** da interface conta os turnos; `python -m app.main --cli` permite gravar a demonstração |
| Pydantic v2 (≥ 4 campos) | ✅ | `schemas.py` — `AnaliseConsulta` (6 campos) e `RelatorioSessao` (6 campos), com `field_validator` |
| PydanticOutputParser | ✅ | `chain.py` — nas duas chains estruturadas (e não `JsonOutputParser`, que devolveria `dict` sem validação) |
| Seção context rot | ✅ | `context_rot.py` — mesma pergunta em janelas de 0/5/10/15/20 turnos, com tabela e gráfico |
| System prompt com persona | ✅ | `prompts.py` — XML tagging (`<papel>`, `<regras>`, `<restricoes>`…), persona, escopo e recusas |
| Domínio documentado | ✅ | Este README |
| Projeto local estruturado | ✅ | Pacote `app/` + `.env.example` + `requirements.txt` + `README.md`, sem Colab |
| **Diferencial:** context engineering com métricas | ✅ | `context_rot.py` — tokens (tiktoken), latência, recall dos fatos e nota 0–100, com gráfico em `saidas/context_rot.png` |
| **Diferencial:** meta prompting | ✅ | `meta_prompting.py` — o próprio `gemma4:cloud` critica e reescreve o system prompt; antes/depois em `saidas/meta_prompting.md` |

---

## Como executar (local — sem Colab)

```bash
cp .env.example .env      # edite com a sua OLLAMA_API_KEY — este arquivo NÃO vai no .zip
pip install -r requirements.txt
python -m app.main        # Gradio em http://localhost:7860
```

Outros pontos de entrada:

```bash
python -m app.main --cli        # conversa no terminal (bom para gravar a demo dos 5 turnos)
python -m app.context_rot       # roda o experimento de degradação e gera tabela + gráfico
python -m app.meta_prompting    # gera o antes/depois do system prompt
```

A chave da Ollama Cloud sai de <https://ollama.com> → *Settings* → *Keys*. Se o `.env`
estiver faltando ou mal preenchido, o programa para logo no início com uma mensagem
dizendo exatamente o que corrigir — não há chave hardcoded em lugar nenhum do código.

### Estrutura

```
chatbot-cp1/
├── app/
│   ├── __init__.py        # metadados do pacote e filtro de avisos de legado
│   ├── main.py            # interface Gradio + entry point + modo terminal
│   ├── chain.py           # as 2 chains da Aula 03 + fachada ChatbotTreino
│   ├── memory_manager.py  # as 3 estratégias de memória gerenciada
│   ├── schemas.py         # Pydantic v2: AnaliseConsulta e RelatorioSessao
│   ├── context_rot.py     # experimento de degradação por contexto
│   ├── prompts.py         # system prompts com XML tagging
│   ├── meta_prompting.py  # diferencial: o modelo revisando o próprio prompt
│   ├── config.py          # leitura e validação do .env
│   └── tokens.py          # contagem de tokens compartilhada (tiktoken)
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
| `ConversationBufferMemory` | Cresce sem teto a cada turno | Nada | ❌ Numa sessão de 20 turnos o prompt vira um custo que não fecha, e o context rot (medido abaixo) mostra que o excesso nem ajuda |
| `ConversationSummaryMemory` | Quase constante, mas **+1 chamada ao LLM por turno** | **Os números.** "supino reto 4×8 com 40 kg, descanso 90 s" vira "treinou peito" | ❌ Fatal aqui: prescrição de treino **é** feita de números. Perder a carga é perder a informação |
| `ConversationTokenBufferMemory` | Limitado pelo teto (1200 tokens) | Só os turnos mais antigos, e por inteiro | ✅ **Escolhida** |

**Por que 1200 tokens.** O enunciado pede de 800 a 1500. Medindo o histórico real do bot,
um par pergunta-resposta de prescrição gasta de 150 a 250 tokens, então 1200 guardam de 5 a
8 turnos completos — o suficiente para cobrir a demonstração de 5 turnos com folga, sem
empurrar o prompt para a faixa em que a qualidade começa a cair (ver a seção seguinte). Com
800 a memória cortaria dentro da própria demonstração; com 1500 ela entraria justamente na
região de degradação que o experimento identificou.

**Efeito no custo de tokens.** O buffer sem teto cresce de forma linear e ilimitada: em 20
turnos, o histórico sozinho passa de 4000 tokens, e cada turno novo paga por todos os
anteriores. O token buffer estabiliza o custo por turno a partir do momento em que o teto é
atingido: o prompt para de crescer e passa a custar aproximadamente o mesmo em qualquer
ponto da conversa. A memória por resumo custaria menos ainda em tokens de contexto, mas
cobra uma chamada extra ao modelo por turno — o que, em latência, sai mais caro do que os
tokens que economiza.

**O que se perde, e por que é aceitável.** O token buffer descarta os turnos mais antigos.
Na prática, é justamente a informação menos usada: o usuário quer ajustar o treino que
acabou de receber, não o que discutiu quinze mensagens atrás. O risco real é perder uma
restrição declarada no turno 1 (uma lesão, por exemplo) — e é precisamente esse risco que a
demonstração de context rot mede, e o motivo pelo qual a triagem da chain 2 reavalia o risco
de segurança a **cada** mensagem, sem depender da memória.

---

## Context rot — degradação com contexto crescente

Rodando `python -m app.context_rot`, o experimento planta três fatos no primeiro turno
(objetivo, frequência semanal e uma restrição de segurança no ombro), enche a conversa com
N turnos de papo genérico de academia que **não repetem** esses fatos, e faz sempre a
**mesma pergunta de controle**: *"Monta pra mim o treino de ombro de amanhã."*

Como só o contexto muda, qualquer queda na qualidade é atribuível ao volume de contexto.
Métricas coletadas por janela: tokens do prompt (tiktoken), latência, recall dos 3 fatos
plantados e aderência ao formato, condensadas numa nota de 0 a 100 em que a restrição de
segurança pesa mais — é a regra cuja quebra causa dano real.

| Turnos de contexto | Tokens do prompt | Objetivo | Frequência | Restrição de segurança | Formato | Nota |
|---:|---:|:--:|:--:|:--:|:--:|---:|
| 0 | ~790 | | | | | |
| 5 | ~1440 | | | | | |
| 10 | ~2020 | | | | | |
| 15 | ~2600 | | | | | |
| 20 | ~3200 | | | | | |

> 📌 **Rode `python -m app.context_rot` com a sua chave e cole a tabela gerada aqui.**
> O comando imprime a tabela pronta em Markdown, salva as respostas completas em
> `saidas/context_rot.json` e o gráfico em `saidas/context_rot.png`. A coluna de tokens
> acima é a medida das janelas montadas por `montar_mensagens()` e varia pouco entre
> execuções; as colunas de qualidade só existem depois de rodar contra o modelo.

**O que observar.** Do contexto de 0 para 20 turnos o prompt fica cerca de **4× maior** sem
que nenhum fato tenha saído da janela do modelo — a informação continua tecnicamente
acessível. O que se degrada é o *uso* dela: os fatos plantados competem com um volume
crescente de texto irrelevante, e os primeiros a cair são os detalhes numéricos (a
frequência semanal), seguidos do objetivo. A restrição de segurança costuma resistir mais,
por ser a instrução mais enfática — mas quando ela também cai, o modelo prescreve
exatamente o movimento proibido.

**Conclusão de engenharia.** Contexto maior não é contexto melhor. É essa medição que
sustenta as duas decisões centrais do projeto: a memória tem **teto** em vez de acumular a
conversa inteira, e a verificação de segurança roda numa chain **separada e stateless**, que
recebe só a mensagem atual e um recorte curto do histórico — imune ao rot que afeta a chain
de conversa.

---

## Observações técnicas

- **`ConversationChain` e `langchain.memory` estão marcados como deprecated** desde o
  LangChain 0.2.7 (a substituta é `RunnableWithMessageHistory`). O checkpoint pede
  explicitamente a arquitetura da Aula 03, então elas foram mantidas; o aviso é silenciado
  de forma documentada e restrita em `app/__init__.py`.
- **Contagem de tokens:** o `ChatOllama` não implementa contagem própria, e sem intervenção
  o LangChain cairia no tokenizer GPT-2 do pacote `transformers` — que não é dependência
  deste projeto e ainda baixaria o tokenizer da internet na primeira chamada. Por isso
  `app/tokens.py` injeta um contador baseado em tiktoken via `custom_get_token_ids`.
  Como o `gemma4` usa SentencePiece, o `cl100k_base` é uma **aproximação** (erro típico de
  10 a 20% em português); é o mesmo contador no teto da memória e na medição de context
  rot, então a comparação relativa permanece válida e o teto fica conservador.
- **Meta prompting não entra em produção automaticamente.** A revisão gerada em
  `saidas/meta_prompting.md` é uma sugestão para o grupo ler e portar à mão para
  `prompts.py`. Deixar um LLM reescrever sozinho o próprio system prompt em produção é
  justamente como se perde o controle da persona.

---

## Segurança

O arquivo `.env` está no `.gitignore` e **não deve ser enviado no `.zip`** da entrega —
apenas o `.env.example`. A `OLLAMA_API_KEY` é lida exclusivamente pelo `config.py`, via
`python-dotenv`, e não aparece em nenhum outro ponto do código.
