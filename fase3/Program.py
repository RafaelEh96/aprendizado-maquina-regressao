import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# a) Leitura dos dados
# ─────────────────────────────────────────────
caminho_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_preg.csv')
data = pd.read_csv(caminho_csv, header=None)
data.columns = ['valor_x', 'valor_y']

x = data['valor_x'].values
y = data['valor_y'].values

# ─────────────────────────────────────────────
# Função manual para calcular y a partir dos
# coeficientes do polyfit (ordem decrescente)
# y = β0 + β1*x + β2*x² + ... + βN*xN
# polyfit retorna [βN, βN-1, ..., β1, β0]
# ─────────────────────────────────────────────
def polinomio_manual(x_vals, coefs):
    """
    Avalia o polinômio manualmente.
    coefs: saída do np.polyfit (ordem decrescente).
    Converte para ordem crescente: β0, β1, ..., βN
    """
    grau = len(coefs) - 1
    y_pred = np.zeros_like(x_vals, dtype=float)
    for i, beta in enumerate(reversed(coefs)):   # reversed → ordem crescente
        y_pred += beta * (x_vals ** i)
    return y_pred

# EQM
def eqm(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)

# R²
def r_quadrado(y_real, y_pred):
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - ss_res / ss_tot

# ─────────────────────────────────────────────
# b) Gráfico de dispersão completo (todos os dados)
# ─────────────────────────────────────────────
x_linha = np.linspace(x.min(), x.max(), 500)   # eixo x suavizado para os plots

graus   = [1, 2, 3, 8]
cores   = ['red', 'green', 'black', 'yellow']
rotulos = ['N=1', 'N=2', 'N=3', 'N=8']

# ── FIGURA 1: TODOS OS DADOS ──────────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(x, y, color='steelblue', alpha=0.6, edgecolors='white',
            linewidths=0.5, label='Dados', zorder=3)

print("=" * 55)
print("  REGRESSÃO COM TODOS OS DADOS")
print("=" * 55)
print(f"{'Grau':>6} | {'EQM':>14} | {'R²':>10}")
print("-" * 55)

for N, cor, rot in zip(graus, cores, rotulos):
    coefs = np.polyfit(x, y, N)                 # c) / d) / e) / f)
    y_pred_linha = polinomio_manual(x_linha, coefs)
    y_pred_dados = polinomio_manual(x, coefs)

    ax1.plot(x_linha, y_pred_linha, color=cor, linewidth=2, label=rot)

    e = eqm(y, y_pred_dados)
    r2 = r_quadrado(y, y_pred_dados)
    print(f"  {rot:>4} | {e:>14.4f} | {r2:>10.6f}")

print("=" * 55)

ax1.set_title('Regressão Polinomial — Todos os dados', fontsize=14, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# ─────────────────────────────────────────────
# h) Divisão treino / teste  (90% / 10%)
# ─────────────────────────────────────────────
np.random.seed(42)
indices   = np.random.permutation(len(x))
n_teste   = max(1, int(len(x) * 0.10))
idx_teste = indices[:n_teste]
idx_tren  = indices[n_teste:]

x_tren, y_tren = x[idx_tren], y[idx_tren]
x_test, y_test = x[idx_teste], y[idx_teste]

print(f"\nTotal: {len(x)} amostras | Treino: {len(x_tren)} | Teste: {len(x_test)}")

# ── FIGURA 2: TREINO / TESTE ──────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(x_tren, y_tren, color='steelblue', alpha=0.6, edgecolors='white',
            linewidths=0.5, label='Treino', zorder=3)
ax2.scatter(x_test, y_test, color='orange', alpha=0.9, edgecolors='black',
            linewidths=0.8, s=80, label='Teste', zorder=4)

print("\n" + "=" * 65)
print("  REGRESSÃO COM TREINO — EQM E R² CALCULADOS NO TESTE")
print("=" * 65)
print(f"{'Grau':>6} | {'EQM (teste)':>16} | {'R² (teste)':>12}")
print("-" * 65)

for N, cor, rot in zip(graus, cores, rotulos):
    # i) Ajuste usando apenas dados de TREINO
    coefs = np.polyfit(x_tren, y_tren, N)
    y_pred_linha = polinomio_manual(x_linha, coefs)

    # j) EQM calculado nos dados de TESTE
    y_pred_test = polinomio_manual(x_test, coefs)
    e_test  = eqm(y_test, y_pred_test)
    r2_test = r_quadrado(y_test, y_pred_test)

    ax2.plot(x_linha, y_pred_linha, color=cor, linewidth=2, label=rot)

    print(f"  {rot:>4} | {e_test:>16.4f} | {r2_test:>12.6f}")

print("=" * 65)

ax2.set_title('Regressão Polinomial — Treino/Teste (90%/10%)', fontsize=14, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.show()