import sys                                   # Permite manipular o caminho de busca de módulos
import os                                    # Funções para manipulação de caminhos de arquivos

# Insere a pasta fase1 no início do caminho de busca para que correlacao.py e regressao.py sejam encontrados
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'fase1'))

import numpy as np                           # Operações matemáticas e matriciais
import pandas as pd                          # Leitura e análise de dados tabulares
import matplotlib                            # Biblioteca base de visualização
matplotlib.use('Agg')                        # Usa backend sem interface gráfica (salva em arquivo em vez de exibir janela)
import matplotlib.pyplot as plt              # Interface de criação de gráficos
from mpl_toolkits.mplot3d import Axes3D      # Suporte a gráficos em 3 dimensões
from sklearn.linear_model import LinearRegression  # Regressão linear da biblioteca scikit-learn (para comparação)

from correlacao import correlacao            # Função de correlação implementada na fase1
from regressao import regressao              # Função de regressão simples implementada na fase1
from regmultipla import regmultipla          # Função de regressão múltipla implementada nesta fase

# ─────────────────────────────────────────────
# a) Carregamento dos dados
# ─────────────────────────────────────────────
caminho_csv = os.path.join(os.path.dirname(__file__), 'data.csv')  # Monta o caminho absoluto até data.csv
data = pd.read_csv(caminho_csv, header=None)                        # Lê o CSV sem cabeçalho (colunas sem nome)
data.columns = ['tamanho', 'quartos', 'preco']                      # Nomeia as 3 colunas manualmente

print("=" * 55)
print("  FASE 2 — Regressão Linear Múltipla")
print("=" * 55)

# b) Montagem de X e y
# ─────────────────────────────────────────────
X = data[['tamanho', 'quartos']].values.astype(float)  # Matriz de variáveis independentes (m x 2)
y = data['preco'].values.astype(float)                  # Vetor da variável dependente (m x 1)

# ─────────────────────────────────────────────
# c) Correlação e regressão simples (2 gráficos)
# ─────────────────────────────────────────────
pares = [                                               # Lista com os dois pares de variáveis a analisar
    ('tamanho', 'preco', 'Tamanho da Casa',  'Preço'),  # Par 1: tamanho × preço
    ('quartos', 'preco', 'Número de Quartos', 'Preço'), # Par 2: quartos × preço
]

fig_d, axes_d = plt.subplots(1, 2, figsize=(13, 5))    # Cria figura com 2 subgráficos lado a lado

for ax, (col_x, col_y, label_x, label_y) in zip(axes_d, pares):  # Itera sobre cada subgráfico e seu par de variáveis
    vx = data[col_x].values.astype(float)              # Extrai os valores da variável independente como array
    vy = data[col_y].values.astype(float)              # Extrai os valores da variável dependente como array

    r            = correlacao(vx, vy)                  # Calcula o coeficiente de correlação r entre as variáveis
    beta0, beta1 = regressao(vx, vy)                   # Calcula os coeficientes β₀ e β₁ da reta de regressão simples

    x_linha = np.linspace(vx.min(), vx.max(), 200)     # Gera 200 pontos igualmente espaçados para desenhar a reta
    y_linha = beta0 + beta1 * x_linha                  # Calcula os valores previstos pela reta nesses 200 pontos

    ax.scatter(vx, vy, color='steelblue', edgecolors='white', s=70, zorder=3, label='Dados')  # Plota os pontos reais
    ax.plot(x_linha, y_linha, color='tomato', linewidth=2, label='Regressão')                 # Plota a reta de regressão
    ax.set_title(                                      # Define o título do subgráfico com r e a equação da reta
        f"{label_x} × {label_y}\n"
        f"r = {r:.4f}   ŷ = {beta0:.2f} + {beta1:.2f}·x",
        fontsize=10,
    )
    ax.set_xlabel(label_x)                             # Rótulo do eixo X
    ax.set_ylabel(label_y)                             # Rótulo do eixo Y
    ax.legend(fontsize=8)                              # Exibe a legenda no subgráfico
    ax.grid(True, linestyle='--', alpha=0.4)           # Adiciona grade tracejada com transparência

    print(f"\nd) {label_x} × {label_y}")
    print(f"   r      = {r:.6f}")
    print(f"   β₀     = {beta0:.4f}")
    print(f"   β₁     = {beta1:.4f}")

fig_d.suptitle("d) Correlação e Regressão Simples", fontsize=13, fontweight='bold')  # Título geral da figura
plt.tight_layout()                                     # Ajusta espaçamento automático entre os subgráficos
saida_d = os.path.join(os.path.dirname(__file__), 'fase2_d_regressao_simples.png')  # Caminho do arquivo de saída
fig_d.savefig(saida_d, dpi=150, bbox_inches='tight')   # Salva a figura em PNG com resolução 150 dpi
plt.close(fig_d)                                       # Fecha a figura para liberar memória
print(f"\n  [Salvo] {saida_d}")

# ─────────────────────────────────────────────
# d) Regressão Múltipla + Gráfico 3D
# ─────────────────────────────────────────────
beta = regmultipla(X, y)                               # Calcula os 3 coeficientes β via fórmula matricial
print(f"\ne) Coeficientes da Regressão Múltipla:")
print(f"   β₀ (intercepto) = {beta[0]:.4f}")
print(f"   β₁ (tamanho)    = {beta[1]:.4f}")
print(f"   β₂ (quartos)    = {beta[2]:.4f}")
print(f"   Modelo: ŷ = {beta[0]:.2f} + {beta[1]:.4f}·tamanho + {beta[2]:.4f}·quartos")

# ─────────────────────────────────────────────
# e) Gráfico 3D com plano de regressão
# ─────────────────────────────────────────────
fig_3d = plt.figure(figsize=(10, 7))                   # Cria nova figura para o gráfico 3D
ax3d = fig_3d.add_subplot(111, projection='3d')        # Adiciona um único subgráfico com projeção tridimensional

tamanho_arr = X[:, 0]                                  # Extrai a coluna de tamanhos (todos os índices, coluna 0)
quartos_arr = X[:, 1]                                  # Extrai a coluna de quartos (todos os índices, coluna 1)

ax3d.scatter(tamanho_arr, quartos_arr, y,              # Plota os dados reais no espaço 3D
             color='steelblue', edgecolors='white', s=60, zorder=5, label='Dados reais')

t_grid = np.linspace(tamanho_arr.min(), tamanho_arr.max(), 30)  # 30 pontos no intervalo de tamanho para a grade
q_grid = np.linspace(quartos_arr.min(), quartos_arr.max(), 30)  # 30 pontos no intervalo de quartos para a grade
TT, QQ = np.meshgrid(t_grid, q_grid)                  # Cria grade 2D combinando todos os pontos de tamanho e quartos
PP = beta[0] + beta[1] * TT + beta[2] * QQ            # Calcula o preço previsto para cada ponto da grade (plano de regressão)

ax3d.plot_surface(TT, QQ, PP, alpha=0.35, color='tomato', rstride=1, cstride=1)  # Renderiza o plano semitransparente

# ─────────────────────────────────────────────
# f) Correlações no título do gráfico
# ─────────────────────────────────────────────
r_tam_preco = correlacao(tamanho_arr, y)               # Correlação entre tamanho e preço
r_qua_preco = correlacao(quartos_arr, y)               # Correlação entre quartos e preço

ax3d.set_xlabel('Tamanho')                             # Rótulo do eixo X
ax3d.set_ylabel('Quartos')                             # Rótulo do eixo Y
ax3d.set_zlabel('Preço')                               # Rótulo do eixo Z
ax3d.set_title(                                        # Título com os dois coeficientes de correlação
    f"Regressão Linear Múltipla\n"
    f"r(tamanho, preço) = {r_tam_preco:.4f}   |   r(quartos, preço) = {r_qua_preco:.4f}",
    fontsize=10,
)
ax3d.legend(fontsize=9)                                # Exibe a legenda no gráfico 3D
ax3d.view_init(elev=20, azim=225)                      # Define o ângulo de visão inicial (elevação 20°, azimute 225°)

plt.tight_layout()                                     # Ajusta margens da figura
saida_3d = os.path.join(os.path.dirname(__file__), 'fase2_ef_grafico3d.png')  # Caminho do arquivo de saída
fig_3d.savefig(saida_3d, dpi=150, bbox_inches='tight') # Salva o gráfico 3D em PNG
plt.close(fig_3d)                                      # Fecha a figura para liberar memória
print(f"\n  [Salvo] {saida_3d}")

print(f"\ng) Coeficientes de correlação:")
print(f"   r(tamanho, preço) = {r_tam_preco:.6f}")
print(f"   r(quartos, preço) = {r_qua_preco:.6f}")

# ─────────────────────────────────────────────
# g) Previsão de preço para casa 1650 ft² / 3 quartos
# ─────────────────────────────────────────────
casa_teste = np.array([[1650, 3]])                        # Array com os dados da casa a ser prevista
preco_previsto = beta[0] + beta[1] * 1650 + beta[2] * 3  # Aplica manualmente o modelo: ŷ = β₀ + β₁·1650 + β₂·3
print(f"\nh) Previsão para casa de 1650 ft² e 3 quartos: {preco_previsto:,.2f}")
print(f"   (Valor esperado pelo enunciado: 293.081)")

print("\n   Variação de quartos (tamanho fixo = 1650):")
for q in range(1, 7):                                  # Itera de 1 a 6 quartos mantendo o tamanho fixo em 1650
    p = beta[0] + beta[1] * 1650 + beta[2] * q        # Recalcula o preço previsto para cada quantidade de quartos
    print(f"     {q} quartos → preço = {p:,.2f}")

print("""
   Explicação: cada quarto adicional aumenta o preço em β₂ unidades.
   Como β₂ é positivo, mais quartos → preço mais alto.
   Isso ocorre porque quartos adicionais representam mais espaço/valor
   percebido, mas o aumento é proporcional ao coeficiente aprendido.
""")

# ─────────────────────────────────────────────
# h) Comparação com scikit-learn
# ─────────────────────────────────────────────
modelo_sklearn = LinearRegression()                    # Instancia o modelo de regressão linear do scikit-learn
modelo_sklearn.fit(X, y)                               # Treina o modelo com os mesmos dados X e y

print("i) Comparação com scikit-learn:")
print(f"   sklearn  — β₀={modelo_sklearn.intercept_:.4f}  β₁={modelo_sklearn.coef_[0]:.4f}  β₂={modelo_sklearn.coef_[1]:.4f}")  # Coeficientes do sklearn
print(f"   Próprio  — β₀={beta[0]:.4f}  β₁={beta[1]:.4f}  β₂={beta[2]:.4f}")                                                    # Coeficientes da nossa implementação

preco_sklearn = modelo_sklearn.predict([[1650, 3]])[0] # Previsão do sklearn para a mesma casa de teste
print(f"\n   Previsão sklearn para 1650 ft² / 3 quartos: {preco_sklearn:,.2f}")
print(f"   Previsão própria para 1650 ft² / 3 quartos: {preco_previsto:,.2f}")
print(f"   Diferença: {abs(preco_previsto - preco_sklearn):.6f}  (deve ser ≈ 0)")  # Diferença deve ser zero, validando nossa implementação

print("\n  Todos os gráficos foram salvos na pasta fase2/")
print("=" * 55)