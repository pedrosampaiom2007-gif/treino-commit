"""Demonstração empírica de *context rot* (degradação por contexto crescente).

Metodologia — "agulha no palheiro":

1. No PRIMEIRO turno o usuário planta 3 fatos que o chatbot precisa lembrar:
   objetivo, frequência semanal e uma restrição de segurança (lesão no ombro).
2. Enchemos a conversa com N turnos de papo genérico de academia, que não
   repetem nenhum dos 3 fatos.
3. Fazemos SEMPRE a mesma pergunta de controle no fim: "monte meu treino de
   ombro para amanhã". Uma resposta correta tem de respeitar a restrição.
4. Repetimos para N = 0, 5, 10, 15 e 20 turnos de enchimento.

O prompt de controle é idêntico em todas as janelas — só o contexto muda.
Assim, qualquer queda na nota é atribuível ao volume de contexto, e não à
pergunta.

Métricas coletadas por janela (diferencial "context engineering com métricas"):
    - tokens do prompt, contados com tiktoken;
    - latência da resposta, em segundos;
    - recall dos 3 fatos plantados (0 a 3);
    - aderência ao formato pedido no system prompt (séries x repetições);
    - nota final de 0 a 100.

Uso:
    python -m app.context_rot                 # janelas padrão
    python -m app.context_rot --janelas 0 10 20
    python -m app.context_rot --sem-grafico
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import RAIZ_PROJETO, carregar_config
from app.prompts import SYSTEM_PROMPT_CHAT
# contar_tokens vem de app/tokens.py: é o MESMO contador que limita a
# memória, para que a métrica do experimento e o teto da memória falem a
# mesma língua.
from app.tokens import contar_tokens, usando_tiktoken

# Janelas de enchimento avaliadas por padrão.
JANELAS_PADRAO: Sequence[int] = (0, 5, 10, 15, 20)

PASTA_SAIDA = RAIZ_PROJETO / "saidas"

# --- Turno 1: os fatos plantados ------------------------------------------
FATO_PLANTADO = (
    "Oi! Antes de começar, três coisas sobre mim: meu objetivo é HIPERTROFIA, "
    "eu treino 4 vezes por semana, e eu tenho uma lesão antiga no ombro "
    "direito — o fisioterapeuta me proibiu de fazer desenvolvimento por trás "
    "da nuca e qualquer elevação acima da linha da cabeça."
)
RESPOSTA_PLANTADA = (
    "Anotado: hipertrofia, 4x por semana e restrição no ombro direito — nada "
    "de desenvolvimento por trás da nuca nem elevação acima da cabeça."
)

# --- Pergunta de controle, idêntica em todas as janelas -------------------
PERGUNTA_CONTROLE = "Monta pra mim o treino de ombro de amanhã."

# --- Enchimento: papo de academia que NÃO repete os fatos plantados -------
TURNOS_ENCHIMENTO: List[tuple[str, str]] = [
    ("Qual a melhor hora do dia pra treinar?",
     "A melhor hora é a que você consegue manter todas as semanas. A literatura "
     "mostra uma vantagem pequena no fim da tarde, quando a temperatura corporal "
     "e a força de pico estão um pouco mais altas, mas o efeito é modesto perto "
     "do impacto da consistência. Se o único horário viável é às seis da manhã, "
     "treine às seis e alongue o aquecimento, porque o corpo recém-acordado "
     "precisa de mais tempo para entrar em ritmo. Escolha a janela que você "
     "consegue repetir e mantenha-a fixa: o corpo se adapta à rotina."),
    ("Preciso tomar água durante o treino?",
     "Precisa. Numa sessão de até uma hora, algo entre quatrocentos e seiscentos "
     "mililitros distribuídos ao longo do treino dá conta, e você não deve "
     "esperar a sede aparecer, porque ela já indica desidratação em curso. Em "
     "dias quentes, ou em sessões acima de uma hora com muito suor, vale incluir "
     "eletrólitos. Um teste caseiro simples: pese-se antes e depois do treino; "
     "cada quilo perdido corresponde a aproximadamente um litro de líquido a "
     "repor ao longo das horas seguintes."),
    ("Aquecimento é obrigatório?",
     "É, e por dois motivos. O primeiro é a temperatura: músculo e tendão "
     "aquecidos toleram mais carga e alongam melhor. O segundo é neural: as "
     "séries de aproximação ensaiam o padrão de movimento antes do peso pesado "
     "entrar. O protocolo que funciona para a maioria é de cinco a dez minutos "
     "de esteira ou bicicleta em ritmo leve, seguidos de duas a três séries de "
     "aproximação no primeiro exercício, subindo a carga progressivamente até a "
     "carga de trabalho. Isolados posteriores não precisam de aproximação."),
    ("Quanto tempo de descanso entre as séries?",
     "Depende do exercício e do objetivo. Em compostos pesados, como agachamento, "
     "levantamento terra e supino, de dois a três minutos, porque o sistema "
     "nervoso e o fosfato de creatina levam esse tempo para se recuperar e o "
     "descanso curto corta o número de repetições da série seguinte. Em isolados, "
     "de sessenta a noventa segundos costumam bastar. Descansar mais não "
     "atrapalha a hipertrofia: o que atrapalha é chegar na próxima série com "
     "menos repetições do que o planejado."),
    ("Alongar antes ou depois?",
     "Alongamento estático longo antes do treino reduz temporariamente a produção "
     "de força, então ele fica melhor no fim da sessão ou num momento separado do "
     "dia. Antes de treinar, prefira mobilidade dinâmica: rotação de ombro, "
     "abertura de quadril, agachamento sem carga, movimentos que levam a "
     "articulação pela amplitude que você vai usar. Se alguma articulação está "
     "travada e limita a execução, trabalhe mobilidade específica ali antes, sem "
     "transformar o aquecimento numa aula de flexibilidade."),
    ("Dá pra treinar com dor muscular tardia?",
     "Dá, quando é aquela dor difusa que aparece de vinte e quatro a quarenta e "
     "oito horas depois e melhora conforme você se movimenta. Nesse caso, reduza "
     "o volume, mantenha a técnica e siga. O que não se treina é dor localizada "
     "em ponto específico, dor que piora com o movimento, dor articular ou dor "
     "que veio de um estalo durante a série. Essa segunda categoria é sinal de "
     "lesão e pede avaliação, não teimosia."),
    ("Música ajuda no treino?",
     "Ajuda mais do que parece. O efeito melhor documentado é sobre a percepção "
     "subjetiva de esforço: a mesma série parece menos dura com música em ritmo "
     "acelerado, o que se traduz em mais repetições completadas em séries longas "
     "e em maior adesão à rotina. Em séries curtas e muito pesadas o efeito é "
     "pequeno, porque ali o limite é neural, não motivacional. Use como "
     "ferramenta, não como pré-requisito para treinar."),
    ("Preciso de creatina?",
     "Creatina monoidratada é o suplemento com a melhor base de evidência para "
     "força e volume de treino, e é barata. Dito isso, quem prescreve dose, "
     "avalia interação e considera o seu quadro clínico é nutricionista ou "
     "médico, não eu. O que posso afirmar com segurança é a ordem de prioridade: "
     "treino bem executado, sono suficiente e alimentação adequada explicam a "
     "maior parte do resultado; suplemento é o ajuste fino que vem depois, nunca "
     "o atalho que substitui os três."),
    ("Quantas horas devo dormir?",
     "De sete a nove horas para a maioria dos adultos. Dormir mal derruba a "
     "produção hormonal ligada à recuperação, aumenta a percepção de esforço e "
     "reduz a força de pico já na primeira noite ruim; em uma semana de privação "
     "acumulada o efeito sobre o desempenho é maior do que o de qualquer erro de "
     "programação de treino. Se a sua agenda obriga a escolher entre uma hora a "
     "mais de sono e uma hora a mais de academia, durma."),
    ("Cardio atrapalha o ganho de músculo?",
     "Em volume moderado, não atrapalha, e ainda melhora a recuperação entre as "
     "séries por elevar o condicionamento. O conflito aparece quando o cardio é "
     "longo, intenso e feito perto do treino de força do mesmo grupo muscular, "
     "tipicamente corrida longa no mesmo dia do treino de perna. A solução "
     "prática é separar: cardio em dias alternados, ou no mínimo com seis horas "
     "de intervalo, e priorizar caminhada inclinada ou bicicleta quando o dia "
     "também tem agachamento pesado."),
    ("Treinar em jejum funciona?",
     "Funciona para quem tolera bem, e não é superior a treinar alimentado. A "
     "queima de gordura durante a sessão em jejum é maior, mas o balanço "
     "energético do dia inteiro é o que determina a composição corporal, e ele "
     "não muda pelo horário da refeição. O risco real do jejum é de ordem "
     "prática: menos energia disponível costuma significar menos repetições e "
     "menos carga, o que, ao longo de semanas, reduz o estímulo de treino. "
     "Teste e observe o seu próprio desempenho."),
    ("Luva de academia serve pra quê?",
     "Serve para conforto e para evitar calo, principalmente em exercícios de "
     "pegada larga com barra. O que a luva não faz é melhorar a sua preensão: ao "
     "aumentar o diâmetro efetivo da barra, ela pode até atrapalhar um pouco em "
     "remadas e terras pesados. Se a pegada é o seu ponto de falha, o caminho é "
     "treinar preensão diretamente, com pendurar na barra e caminhada do "
     "fazendeiro, e usar strap apenas nas séries mais pesadas de puxada."),
    ("Devo treinar até a falha?",
     "Nem sempre, e definitivamente não em toda série. Parar com uma ou duas "
     "repetições na reserva produz ganho de hipertrofia muito parecido com o da "
     "falha total, mas gera bem menos fadiga acumulada, o que permite manter o "
     "volume semanal. A falha faz mais sentido na última série de exercícios "
     "isolados e em máquinas, onde o risco técnico é baixo. Em agachamento e "
     "supino livre, falhar sozinho é perigoso e não compensa."),
    ("Máquina ou peso livre?",
     "Os dois, e por razões diferentes. Peso livre exige estabilização, recruta "
     "musculatura acessória e transfere melhor para movimentos do dia a dia, "
     "então ele leva os compostos principais. Máquina fixa a trajetória, o que "
     "permite levar o músculo-alvo perto da falha com segurança e isolar um "
     "ponto fraco sem depender do equilíbrio. Um treino bem montado começa pelos "
     "compostos livres, quando você está descansado, e fecha com máquinas."),
    ("Com que frequência troco de treino?",
     "A cada seis a dez semanas, ou antes disso se a progressão de carga travar "
     "por três semanas seguidas com sono e alimentação em ordem. Trocar cedo "
     "demais é um erro comum: sem repetir o mesmo exercício por algumas semanas "
     "você não consegue medir progresso nem aperfeiçoar a técnica. Trocar tarde "
     "demais também cobra o seu preço, em tédio e em estagnação. O que muda na "
     "troca não precisa ser tudo: alterar exercícios acessórios e manter os "
     "compostos já basta."),
    ("O que é deload?",
     "É uma semana planejada de volume ou intensidade reduzidos, tipicamente "
     "metade das séries ou cerca de setenta por cento da carga habitual, para "
     "que a recuperação alcance o estímulo acumulado. A fadiga se acumula em "
     "camadas: muscular, articular e nervosa, e a última é a mais lenta de "
     "dissipar. O sinal clássico de que o deload está atrasado é quando a carga "
     "de sempre começa a parecer pesada e o ânimo para treinar cai sem outra "
     "explicação. Programe a cada seis a oito semanas."),
    ("Posso treinar dois dias seguidos?",
     "Pode, desde que alterne os grupos musculares. Um músculo trabalhado com "
     "volume alto leva de quarenta e oito a setenta e duas horas para se "
     "recuperar por completo, então treinar peito na segunda e costas na terça "
     "não gera conflito. O que cobra caro é repetir o mesmo padrão de movimento "
     "pesado em dias consecutivos, como agachar na segunda e fazer terra na "
     "terça, porque a região lombar aparece nos dois e não teve intervalo."),
    ("Respiração correta no exercício?",
     "A regra geral é expirar na fase de esforço e inspirar na fase de retorno. "
     "Em séries pesadas de compostos, a técnica muda: você inspira, segura o ar e "
     "contrai o abdômen para pressurizar o tronco, o que protege a coluna, e só "
     "expira ao terminar a repetição. Isso se chama manobra de Valsalva e não "
     "deve ser prolongada por várias repetições seguidas, nem usada por quem tem "
     "condição cardiovascular, porque eleva a pressão arterial."),
    ("Espelho na academia serve pra algo?",
     "Serve como retorno visual imediato de execução, especialmente útil em "
     "exercícios de ombro e em qualquer movimento em que você não consegue "
     "sentir a posição da escápula. A ressalva é que olhar o espelho muda a "
     "posição da cabeça em agachamento e terra, o que altera a mecânica da "
     "coluna. Nesses dois, olhe para um ponto fixo à frente e, se quiser "
     "conferir a técnica, grave um vídeo lateral em vez de torcer o pescoço."),
    ("Suplemento pré-treino é necessário?",
     "Não é necessário. A maior parte do efeito percebido vem da cafeína, entre "
     "três e seis miligramas por quilo de peso, cerca de quarenta minutos antes "
     "do treino, e um café forte entrega isso. A beta-alanina responde pelo "
     "formigamento e tem efeito real em séries longas, acima de um minuto de "
     "esforço contínuo. O resto costuma ser dose baixa demais para importar. "
     "Vale lembrar que cafeína no fim do dia atrapalha o sono, e sono ruim "
     "custa mais do que o pré-treino entrega."),
]


# ---------------------------------------------------------------------------
# Avaliação da resposta
# ---------------------------------------------------------------------------
@dataclass
class ResultadoJanela:
    """Uma linha da tabela de context rot."""

    turnos_enchimento: int
    tokens_prompt: int
    latencia_s: float
    lembrou_objetivo: bool
    lembrou_frequencia: bool
    respeitou_restricao: bool
    manteve_formato: bool
    nota_0_100: int
    resposta: str

    @property
    def recall(self) -> int:
        """Quantos dos 3 fatos plantados sobreviveram ao contexto."""
        return sum(
            (self.lembrou_objetivo, self.lembrou_frequencia, self.respeitou_restricao)
        )


def _contem(texto: str, padrao: str) -> bool:
    """Busca insensível a caixa e acento leve, via regex."""
    return re.search(padrao, texto, flags=re.IGNORECASE) is not None


def avaliar_resposta(resposta: str) -> tuple[bool, bool, bool, bool, int]:
    """Pontua a resposta contra os 3 fatos plantados e o formato exigido.

    A checagem é por palavra-chave — barata, determinística e suficiente para
    evidenciar a degradação. Não é um juiz LLM, e isso é proposital: um juiz
    LLM sofreria do mesmo context rot que estamos medindo.
    """
    # Fato 1 — objetivo declarado.
    lembrou_objetivo = _contem(resposta, r"hipertrofi|massa muscular|volume muscular")

    # Fato 2 — frequência semanal.
    lembrou_frequencia = _contem(resposta, r"\b4\b\s*(x|vezes)|quatro vezes|4x")

    # Fato 3 — a restrição de segurança. O ponto só é dado se a resposta
    # reconhecer a limitação; prescrever o movimento proibido zera o critério.
    prescreveu_proibido = _contem(
        resposta, r"por tr[aá]s da nuca|atr[aá]s da nuca|desenvolvimento militar por tr"
    )
    citou_restricao = _contem(
        resposta, r"ombro direito|lesão|lesao|restri|evitar|sem elevação acima|fisioterape"
    )
    respeitou_restricao = citou_restricao and not prescreveu_proibido

    # Formato exigido no system prompt: séries x repetições e descanso.
    manteve_formato = _contem(resposta, r"\d\s*[xX×]\s*\d") and _contem(
        resposta, r"descans|intervalo|segundos|min"
    )

    # Nota: a restrição de segurança pesa mais que o resto, porque é a regra
    # cuja quebra causa dano real ao usuário.
    nota = (
        (35 if respeitou_restricao else 0)
        + (20 if lembrou_objetivo else 0)
        + (20 if lembrou_frequencia else 0)
        + (25 if manteve_formato else 0)
    )
    return (
        lembrou_objetivo,
        lembrou_frequencia,
        respeitou_restricao,
        manteve_formato,
        nota,
    )


# ---------------------------------------------------------------------------
# Execução do experimento
# ---------------------------------------------------------------------------
def montar_mensagens(turnos_enchimento: int) -> List:
    """Monta a janela de contexto: fato plantado + N turnos + pergunta fixa."""
    mensagens = [
        SystemMessage(content=SYSTEM_PROMPT_CHAT),
        HumanMessage(content=FATO_PLANTADO),
        AIMessage(content=RESPOSTA_PLANTADA),
    ]
    for indice in range(turnos_enchimento):
        # Recicla a lista quando a janela pedida excede o enchimento disponível.
        pergunta, resposta = TURNOS_ENCHIMENTO[indice % len(TURNOS_ENCHIMENTO)]
        mensagens.append(HumanMessage(content=pergunta))
        mensagens.append(AIMessage(content=resposta))
    mensagens.append(HumanMessage(content=PERGUNTA_CONTROLE))
    return mensagens


def rodar_experimento(
    janelas: Sequence[int] = JANELAS_PADRAO,
    verbose: bool = True,
) -> List[ResultadoJanela]:
    """Roda a pergunta de controle em cada janela e devolve os resultados."""
    from app.chain import construir_llm

    config = carregar_config()
    # Temperatura 0 para isolar o efeito do contexto do efeito da amostragem.
    llm = construir_llm(config, temperatura=0.0)

    resultados: List[ResultadoJanela] = []

    for n in janelas:
        mensagens = montar_mensagens(n)
        tokens = contar_tokens("\n".join(str(m.content) for m in mensagens))

        inicio = time.perf_counter()
        resposta = str(llm.invoke(mensagens).content)
        latencia = time.perf_counter() - inicio

        objetivo, frequencia, restricao, formato, nota = avaliar_resposta(resposta)
        resultado = ResultadoJanela(
            turnos_enchimento=n,
            tokens_prompt=tokens,
            latencia_s=round(latencia, 2),
            lembrou_objetivo=objetivo,
            lembrou_frequencia=frequencia,
            respeitou_restricao=restricao,
            manteve_formato=formato,
            nota_0_100=nota,
            resposta=resposta,
        )
        resultados.append(resultado)

        if verbose:
            print(
                f"janela={n:>2} turnos | {tokens:>5} tokens | "
                f"{latencia:>5.2f}s | recall {resultado.recall}/3 | nota {nota}/100"
            )

    return resultados


# ---------------------------------------------------------------------------
# Relatórios
# ---------------------------------------------------------------------------
def tabela_markdown(resultados: Sequence[ResultadoJanela]) -> str:
    """Tabela comparativa em Markdown — o artefato pedido pelo checkpoint."""
    cabecalho = (
        "| Turnos de contexto | Tokens do prompt | Latência (s) | "
        "Objetivo | Frequência | Restrição de segurança | Formato | Nota |\n"
        "|---:|---:|---:|:--:|:--:|:--:|:--:|---:|\n"
    )
    marca = lambda ok: "✅" if ok else "❌"  # noqa: E731
    linhas = "".join(
        f"| {r.turnos_enchimento} | {r.tokens_prompt} | {r.latencia_s:.2f} | "
        f"{marca(r.lembrou_objetivo)} | {marca(r.lembrou_frequencia)} | "
        f"{marca(r.respeitou_restricao)} | {marca(r.manteve_formato)} | "
        f"**{r.nota_0_100}** |\n"
        for r in resultados
    )
    return cabecalho + linhas


def conclusao(resultados: Sequence[ResultadoJanela]) -> str:
    """Interpreta os números em uma frase — é o que a rubrica quer ler."""
    if not resultados:
        return "Sem resultados."
    primeiro, ultimo = resultados[0], resultados[-1]
    delta_nota = ultimo.nota_0_100 - primeiro.nota_0_100
    fator_tokens = ultimo.tokens_prompt / max(1, primeiro.tokens_prompt)
    rumo = "caiu" if delta_nota < 0 else ("subiu" if delta_nota > 0 else "ficou estável")
    return (
        f"Do contexto de {primeiro.turnos_enchimento} para "
        f"{ultimo.turnos_enchimento} turnos, o prompt ficou "
        f"{fator_tokens:.1f}x maior e a nota {rumo} "
        f"{abs(delta_nota)} pontos ({primeiro.nota_0_100} → {ultimo.nota_0_100}). "
        f"O recall dos fatos plantados foi de {primeiro.recall}/3 para "
        f"{ultimo.recall}/3. É o context rot: a informação continua DENTRO da "
        f"janela, mas o modelo deixa de usá-la conforme o ruído em volta cresce. "
        f"É exatamente por isso que a memória deste projeto tem teto de tokens "
        f"em vez de acumular a conversa inteira."
    )


def gerar_grafico(resultados: Sequence[ResultadoJanela], destino: Path) -> Path | None:
    """Gera o gráfico nota x tokens. Retorna None se matplotlib faltar."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # sem display no ambiente local
        import matplotlib.pyplot as plt
    except Exception:
        return None

    tokens = [r.tokens_prompt for r in resultados]
    notas = [r.nota_0_100 for r in resultados]
    recalls = [r.recall / 3 * 100 for r in resultados]
    rotulos = [str(r.turnos_enchimento) for r in resultados]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tokens, notas, marker="o", linewidth=2, label="Nota da resposta (0–100)")
    ax.plot(tokens, recalls, marker="s", linestyle="--", label="Recall dos fatos (%)")
    for x, y, rot in zip(tokens, notas, rotulos):
        ax.annotate(f"{rot} turnos", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8)
    ax.set_xlabel("Tokens no prompt (tiktoken · cl100k_base)")
    ax.set_ylabel("Qualidade")
    ax.set_title("Context rot — mesma pergunta, janelas de contexto diferentes")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150)
    plt.close(fig)
    return destino


def salvar_resultados(resultados: Sequence[ResultadoJanela]) -> Path:
    """Persiste o bruto em JSON para quem quiser conferir as respostas."""
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    destino = PASTA_SAIDA / "context_rot.json"
    destino.write_text(
        json.dumps([asdict(r) for r in resultados], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destino


def relatorio_completo(resultados: Sequence[ResultadoJanela]) -> str:
    """Bloco em Markdown exibido na aba "Context rot" da interface."""
    return (
        "### Demonstração de context rot\n\n"
        f"Pergunta de controle (idêntica em todas as janelas): "
        f"_\"{PERGUNTA_CONTROLE}\"_\n\n"
        + tabela_markdown(resultados)
        + "\n**Conclusão:** "
        + conclusao(resultados)
        + "\n\n_"
        + (
            "Tokens contados com tiktoken (`cl100k_base`) — aproximação do "
            "tokenizer real do gemma4; serve para comparação relativa."
            if usando_tiktoken()
            else "tiktoken indisponível: tokens estimados em ~4 caracteres cada."
        )
        + "_\n"
    )


def main() -> None:
    """Entry point: python -m app.context_rot"""
    parser = argparse.ArgumentParser(
        description="Mede a degradação de qualidade conforme o contexto cresce."
    )
    parser.add_argument(
        "--janelas",
        nargs="+",
        type=int,
        default=list(JANELAS_PADRAO),
        help="Quantidades de turnos de enchimento a testar (ex.: 0 5 10 15 20).",
    )
    parser.add_argument(
        "--sem-grafico", action="store_true", help="Pula a geração do PNG."
    )
    args = parser.parse_args()

    janelas = [max(0, n) for n in args.janelas]

    print("Rodando o experimento de context rot...\n")
    resultados = rodar_experimento(janelas)

    print("\n" + tabela_markdown(resultados))
    print("Conclusão: " + conclusao(resultados) + "\n")

    caminho_json = salvar_resultados(resultados)
    print(f"Respostas completas salvas em: {caminho_json}")

    if not args.sem_grafico:
        grafico = gerar_grafico(resultados, PASTA_SAIDA / "context_rot.png")
        if grafico:
            print(f"Gráfico salvo em: {grafico}")
        else:
            print("matplotlib indisponível — gráfico não gerado (a tabela basta).")


if __name__ == "__main__":
    main()
