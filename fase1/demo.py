import correlacao as cr
import regressao as rg
import numpy as np
import matplotlib.pyplot as plt
import re

def ler_datasets(caminho: str) -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()

    padrao = r"(\w+)\s*=\s*\[([^\]]+)\]"
    variaveis = {}
    for nome, valores in re.findall(padrao, conteudo):
        nums = [float(v.strip()) for v in valores.split(";") if v.strip()]
        variaveis[nome] = np.array(nums)

    datasets = {}
    indices = sorted(set(re.sub(r"[xy]", "", k) for k in variaveis))
    for i, idx in enumerate(indices, start=1):
        x_key = f"x{idx}"
        y_key = f"y{idx}"
        if x_key in variaveis and y_key in variaveis:
            datasets[f"Grupo {i}"] = {"x": variaveis[x_key], "y": variaveis[y_key]}

    return datasets


datasets = ler_datasets("datasetFase1.txt")

for nome, dados in datasets.items():
    x = dados["x"]
    y = dados["y"]

    r = cr.correlacao(x, y)

    beta0, beta1 = rg.regressao(x, y)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(x, y, color="steelblue", edgecolors="white", s=80, zorder=3, label="Dados")

    x_linha = np.linspace(x.min(), x.max(), 200)
    y_linha = beta0 + beta1 * x_linha
    ax.plot(x_linha, y_linha, color="tomato", linewidth=2, label="Regressão")

    ax.set_title(
        f"{nome}  —  r = {r:.4f}\n"
        f"ŷ = {beta0:.4f} + {beta1:.4f}·x",
        fontsize=11,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    arquivo = f"fase1_{nome.replace(' ', '_').lower()}.png"
    plt.savefig(arquivo, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    
    # O segundo DataSet não é o mais apropriado para o metódo de regressão linear
    # porque os dados possuem uma correlação alta, porém a estrutura dos
    # dados é mais complexa do que uma simples linha reta, o que pode levar a um ajuste inadequado
    # e a uma interpretação errada dos resultados.