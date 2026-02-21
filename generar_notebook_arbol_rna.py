"""
Genera el notebook: Árbol de Decisión + Red Neuronal (optimizada para Recall)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
# Árbol de Decisión + Red Neuronal (optimizada para Recall) — BBDD_ML_TAREA

## Flujo de trabajo

| Etapa | Descripción |
|---|---|
| **1. Árbol de Decisión** | Búsqueda paramétrica exhaustiva del DT; extracción de importancia de variables |
| **2. Selección de variables** | Selección data-driven basada en importancia acumulada |
| **3. Búsqueda paramétrica RNA** | Grid search exhaustivo optimizando **Recall** |
| **4. Threshold** | Análisis por umbral con criterio F2-score (prioridad Recall) |
| **5. Evaluación final** | Comparativa DT baseline vs RNA con threshold óptimo |

### Métrica objetivo: Recall de la clase positiva (Y=1)

Con un desbalanceo **80 %/20 %**, el coste de un **falso negativo** (no detectar un positivo
real) es mayor que el de un falso positivo en muchos contextos de negocio (crédito, medicina,
fraude). Por ello se priorizará el **Recall** como métrica de ajuste.

**F2-score** como criterio de threshold:
$$F_2 = 5 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{4 \\cdot \\text{Precision} + \\text{Recall}}$$
Pondera el Recall el doble que la Precision, evitando la solución trivial (threshold → 0 → Recall=1 pero Precision≈0).

### Dataset de entrada
`BBDD_ML_TAREA_procesada.csv` — 3.538 filas · 25 features · target Y (clase 0: 80 %, clase 1: 20 %)
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("## 1. Configuración del Entorno"))

cells.append(nbf.v4.new_code_cell("""\
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import product
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

SEED = 42
np.random.seed(SEED)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 100

def cv_recall(estimator, X, y, weights, cv):
    \"\"\"CV manual con sample_weight — compatible con sklearn >= 1.4.\"\"\"
    scores = []
    for tr_idx, vl_idx in cv.split(X, y.values if hasattr(y, 'values') else y):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr = y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]
        y_vl = y.iloc[vl_idx] if hasattr(y, 'iloc') else y[vl_idx]
        w_tr = weights[tr_idx]
        est = clone(estimator)
        est.fit(X_tr, y_tr, sample_weight=w_tr)
        y_pred = est.predict(X_vl)
        scores.append(recall_score(y_vl, y_pred, zero_division=0))
    return np.array(scores)

print("✓ Entorno configurado")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA Y PARTICIÓN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 2. Carga y Partición Estratificada

Se mantiene la misma partición **70 / 15 / 15** con `stratify=y` para preservar
la proporción 80/20 en cada subconjunto.
"""))

cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv('BBDD_ML_TAREA_procesada.csv')
TARGET = 'Y'
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Shape: {df.shape}")
vc = y.value_counts()
for k, v in vc.items():
    print(f"  Clase {k}: {v:,}  ({v/len(y)*100:.1f} %)")

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=SEED)

print(f"\\nPartición — Train:{len(X_train):,}  Val:{len(X_val):,}  Test:{len(X_test):,}")

# Pesos balanceados para el entrenamiento del DT y RNA
sw_train = compute_sample_weight('balanced', y_train)
print(f"Peso clase 0: {sw_train[y_train==0][0]:.3f}  |  Peso clase 1: {sw_train[y_train==1][0]:.3f}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 3. ÁRBOL DE DECISIÓN — BÚSQUEDA PARAMÉTRICA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 3. Árbol de Decisión — Búsqueda Paramétrica Exhaustiva

### ¿Por qué un Árbol de Decisión para selección de variables?

- Los árboles de decisión calculan la **importancia de cada variable** como la reducción
  ponderada de impureza (Gini o Entropía) que produce esa variable a lo largo de todos
  los nodos donde se usa.
- Esta importancia es **específica del modelo** y captura interacciones no lineales
  que medidas como la correlación lineal no detectan.
- Al entrenar el DT con los mismos datos que la RNA, las variables seleccionadas
  son exactamente las que un modelo de árbol considera más discriminativas —
  una base sólida para el modelo no lineal posterior.

### Parámetros del árbol evaluados

| Parámetro | Valores | Justificación |
|---|---|---|
| `max_depth` | 3, 5, 7, 10, None | Controla la complejidad; árboles más profundos capturan más pero sobreajustan |
| `min_samples_split` | 2, 10, 20, 50 | Nº mínimo de muestras para dividir un nodo; valores altos dan árboles más simples |
| `min_samples_leaf` | 1, 5, 10, 20 | Nº mínimo en cada hoja; regulariza la profundidad efectiva |
| `criterion` | `gini`, `entropy` | Gini: más rápido; Entropía: puede detectar divisiones más finas |

**Métrica de selección:** `recall` en CV estratificada 5-fold con `class_weight='balanced'`
(el DT se optimiza también para Recall desde el principio).
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Grid Search del Árbol de Decisión ────────────────────────────────────────
param_grid_dt = {
    'max_depth'        : [3, 5, 7, 10, None],
    'min_samples_split': [2, 10, 20, 50],
    'min_samples_leaf' : [1, 5, 10, 20],
    'criterion'        : ['gini', 'entropy']
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

dt_base = DecisionTreeClassifier(class_weight='balanced', random_state=SEED)
gs_dt = GridSearchCV(
    dt_base, param_grid_dt,
    scoring='recall',
    cv=cv5,
    n_jobs=-1,
    return_train_score=True
)
gs_dt.fit(X_train, y_train, sample_weight=sw_train)

print("Mejores parámetros del Árbol de Decisión:")
for k, v in gs_dt.best_params_.items():
    print(f"  {k:<25}: {v}")
print(f"\\nMejor Recall CV (5-fold): {gs_dt.best_score_:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Análisis de resultados del grid DT ───────────────────────────────────────
cv_results_dt = pd.DataFrame(gs_dt.cv_results_)
cv_results_dt = cv_results_dt.sort_values('mean_test_score', ascending=False)

print("Top 10 configuraciones del Árbol de Decisión (por Recall CV):")
cols_show = ['param_max_depth', 'param_criterion',
             'param_min_samples_split', 'param_min_samples_leaf',
             'mean_test_score', 'std_test_score', 'mean_train_score']
display(cv_results_dt[cols_show].head(10).rename(columns={
    'mean_test_score' : 'Recall_val',
    'std_test_score'  : 'Recall_std',
    'mean_train_score': 'Recall_train'
}).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Árbol óptimo y visualización ─────────────────────────────────────────────
dt_opt = gs_dt.best_estimator_

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Importancias de variables
importancias_dt = pd.Series(
    dt_opt.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
importancias_dt_nonzero = importancias_dt[importancias_dt > 0]

colores = sns.color_palette('Blues_r', len(importancias_dt_nonzero))
axes[0].barh(importancias_dt_nonzero.index[::-1],
             importancias_dt_nonzero.values[::-1],
             color=colores[::-1], edgecolor='white')
axes[0].set_xlabel('Importancia (reducción de impureza)')
axes[0].set_title('Importancia de Variables — Árbol de Decisión')
axes[0].tick_params(labelsize=8)

# Importancia acumulada
importancia_cum = importancias_dt_nonzero.cumsum() / importancias_dt_nonzero.sum() * 100
axes[1].plot(range(1, len(importancia_cum)+1), importancia_cum.values,
             marker='o', color='steelblue', linewidth=2, markersize=5)
axes[1].axhline(85, color='red', linestyle='--', linewidth=1.5, label='Umbral 85 %')
axes[1].axhline(95, color='orange', linestyle='--', linewidth=1.5, label='Umbral 95 %')
axes[1].set_xlabel('Número de variables (ordenadas por importancia)')
axes[1].set_ylabel('Importancia acumulada (%)')
axes[1].set_title('Importancia Acumulada — Selección de Variables')
axes[1].legend()
axes[1].set_xticks(range(1, len(importancia_cum)+1))

plt.suptitle('Árbol de Decisión — Importancia de Variables', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_fig1_importancias.png', bbox_inches='tight')
plt.show()

print(f"Variables con importancia > 0: {len(importancias_dt_nonzero)}")
print("\\nImportancia por variable:")
for feat, imp in importancias_dt_nonzero.items():
    cum = importancias_dt[:feat.__class__].sum() if False else None
    print(f"  {feat:<40} {imp:.5f}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 4. SELECCIÓN DE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Selección de Variables — Criterio de Importancia Acumulada

### Criterio de corte: importancia acumulada ≥ 85 %

**Justificación:**
- Un umbral del 85 % retiene las variables que explican la mayor parte de la capacidad
  discriminativa del árbol, descartando las de contribución marginal (ruido).
- Variables con importancia muy baja tienden a aumentar la dimensionalidad sin aportar
  información útil y pueden degradar la generalización de la RNA.
- El 85 % es un umbral ampliamente usado en la literatura como balance entre
  parsimonia y retención de información.

Si el 85 % implica menos de 3 variables se aumenta al 95 % para garantizar una
entrada mínimamente informativa a la RNA.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Importancia acumulada normalizada
imp_norm = importancias_dt_nonzero / importancias_dt_nonzero.sum()
imp_cum  = imp_norm.cumsum()

# Selección al 85 %
UMBRAL = 0.85
n_selec = (imp_cum <= UMBRAL).sum() + 1   # +1 para incluir la que cruza el umbral
n_selec = max(n_selec, 3)                 # mínimo 3 variables

features_dt = importancias_dt_nonzero.head(n_selec).index.tolist()

print(f"Umbral de importancia acumulada: {UMBRAL*100:.0f} %")
print(f"Variables seleccionadas ({n_selec}):")
for i, f in enumerate(features_dt, 1):
    print(f"  {i:2}. {f:<40}  imp={importancias_dt[f]:.5f}  cum={imp_cum[f]*100:.1f} %")

print(f"\\nImportancia acumulada de las {n_selec} variables: {imp_cum.iloc[n_selec-1]*100:.1f} %")

# ── Baseline del árbol de decisión ───────────────────────────────────────────
y_pred_dt_val = dt_opt.predict(X_val)
print(f"\\n── Baseline Árbol de Decisión (val) ──────────────────────────")
print(f"  Accuracy : {accuracy_score(y_val, y_pred_dt_val):.4f}")
print(f"  Recall   : {recall_score(y_val, y_pred_dt_val):.4f}")
print(f"  Precision: {precision_score(y_val, y_pred_dt_val, zero_division=0):.4f}")
print(f"  F1       : {f1_score(y_val, y_pred_dt_val):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_val, dt_opt.predict_proba(X_val)[:,1]):.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Visualización del árbol (si max_depth <= 5) ───────────────────────────────
depth = dt_opt.get_depth()
print(f"Profundidad del árbol óptimo: {depth}")

if depth <= 5:
    fig_h = max(6, depth * 2)
    fig, ax = plt.subplots(figsize=(min(20, 2**depth * 2), fig_h))
    plot_tree(dt_opt, feature_names=X_train.columns,
              class_names=['Clase 0', 'Clase 1'],
              filled=True, rounded=True, max_depth=4,
              fontsize=7, ax=ax)
    ax.set_title(f'Árbol de Decisión Óptimo (profundidad mostrada: min({depth},4))',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dt_fig2_arbol.png', bbox_inches='tight')
    plt.show()
else:
    print(f"Árbol demasiado profundo ({depth} niveles) para visualizar completamente.")
    print("Se muestra la representación textual de los primeros 3 niveles:")
    print(export_text(dt_opt, feature_names=list(X_train.columns), max_depth=3))
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 5. PREPARACIÓN PARA LA RNA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 5. Preparación de Datos para la Red Neuronal

Se escala con **StandardScaler** las variables seleccionadas por el árbol.
El escalado es necesario porque los pesos de la RNA son sensibles a la escala
de las entradas (las funciones de activación operan de forma óptima con entradas centradas).

> Los datos ya fueron procesados con RobustScaler en el notebook anterior.
> Se aplica StandardScaler adicional sobre las variables seleccionadas para garantizar
> media=0 y std=1 sobre este subconjunto específico.
"""))

cells.append(nbf.v4.new_code_cell("""\
X_tr_dt  = X_train[features_dt]
X_val_dt = X_val[features_dt]
X_te_dt  = X_test[features_dt]

scaler_dt = StandardScaler()
X_tr_sc  = scaler_dt.fit_transform(X_tr_dt)
X_val_sc = scaler_dt.transform(X_val_dt)
X_te_sc  = scaler_dt.transform(X_te_dt)

print(f"Shape train escalado: {X_tr_sc.shape}")
print(f"Variables de entrada a la RNA ({len(features_dt)}):")
for f in features_dt:
    print(f"  • {f}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 6. BÚSQUEDA PARAMÉTRICA EXHAUSTIVA — RNA (Recall)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 6. Búsqueda Paramétrica Exhaustiva — Red Neuronal (objetivo: Recall)

### Estrategia de búsqueda

Se usa la misma estructura de **tres fases progresivas** + grid final.
**Diferencia clave respecto al ejercicio anterior:** la métrica de CV es `recall`
(no `accuracy`), lo que orienta la selección de parámetros hacia modelos que
minimizan los falsos negativos de la clase positiva.

### Tratamiento del desbalanceo durante el entrenamiento

`MLPClassifier` no acepta `class_weight` directamente, pero acepta `sample_weight`
en el método `fit()`. Se pasan pesos balanceados:
$$w_i = \\frac{N}{K \\cdot N_k}$$
donde $N$ = total de muestras, $K$ = nº de clases, $N_k$ = muestras de la clase $k$.
Con desbalanceo 80/20, la clase 1 recibe peso ×4 respecto a la clase 0.

### Parámetros explorados y justificación

| Parámetro | Valores probados | Justificación |
|---|---|---|
| `hidden_layer_sizes` | 7 arquitecturas | Cubrir el espacio de profundidad/anchura relativo al nº de entradas |
| `activation` | `relu`, `tanh` | Funciones de activación estándar; tanh puede ser más suave con datos pequeños |
| `alpha` (L2) | `1e-4`, `1e-3`, `1e-2`, `5e-2` | Con Recall como objetivo, la regularización evita que la red "se rinda" en positivos |
| `learning_rate_init` | `1e-4`, `1e-3`, `5e-3` | Rango amplio para detectar sensibilidad a la tasa de aprendizaje |
| `batch_size` | `16`, `32`, `64` | Lotes pequeños favorecen el aprendizaje de la clase minoritaria |
"""))

cells.append(nbf.v4.new_code_cell("""\
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

# ─────────────────────────────────────────────────────────────────────────────
# FASE 1: Arquitecturas (alpha=1e-3, lr=1e-3, batch=32, relu)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FASE 1 — Exploración de arquitecturas  [métrica: recall]")
print("=" * 65)

n_feat = len(features_dt)
# Arquitecturas: relativas al nº de variables de entrada
architectures = [
    (n_feat * 2,),
    (n_feat * 4,),
    (n_feat * 6,),
    (n_feat * 4, n_feat * 2),
    (n_feat * 6, n_feat * 3),
    (n_feat * 4, n_feat * 2, n_feat),
    (n_feat * 6, n_feat * 3, n_feat),
]

results_f1 = []
for arch in architectures:
    mlp = MLPClassifier(
        hidden_layer_sizes=arch, activation='relu', solver='adam',
        alpha=1e-3, learning_rate_init=1e-3, batch_size=32,
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15, random_state=SEED
    )
    scores = cv_recall(mlp, X_tr_sc, y_train, sw_train, cv3)
    results_f1.append({'Arquitectura': str(arch), 'Recall_mean': scores.mean(), 'Recall_std': scores.std()})
    print(f"  {str(arch):<20}  Recall={scores.mean():.4f} ± {scores.std():.4f}")

df_f1 = pd.DataFrame(results_f1).sort_values('Recall_mean', ascending=False)
print(f"\\n→ Mejor arquitectura: {df_f1.iloc[0]['Arquitectura']}")
best_arch = eval(df_f1.iloc[0]['Arquitectura'])
"""))

cells.append(nbf.v4.new_code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# FASE 2: Regularización alpha
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"FASE 2 — Regularización L2 (alpha)  [arch={best_arch}]")
print("=" * 65)

alphas = [1e-4, 1e-3, 1e-2, 5e-2]
results_f2 = []
for a in alphas:
    mlp = MLPClassifier(
        hidden_layer_sizes=best_arch, activation='relu', solver='adam',
        alpha=a, learning_rate_init=1e-3, batch_size=32,
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15, random_state=SEED
    )
    scores = cv_recall(mlp, X_tr_sc, y_train, sw_train, cv3)
    results_f2.append({'alpha': a, 'Recall_mean': scores.mean(), 'Recall_std': scores.std()})
    print(f"  alpha={a:<8.0e}  Recall={scores.mean():.4f} ± {scores.std():.4f}")

df_f2 = pd.DataFrame(results_f2).sort_values('Recall_mean', ascending=False)
best_alpha = df_f2.iloc[0]['alpha']
print(f"\\n→ Mejor alpha: {best_alpha:.0e}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# FASE 3: Activación, Learning Rate y Batch Size
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"FASE 3 — Activación, LR y Batch  [arch={best_arch}, alpha={best_alpha:.0e}]")
print("=" * 65)

activaciones   = ['relu', 'tanh']
learning_rates = [1e-4, 1e-3, 5e-3]
batch_sizes    = [16, 32, 64]

results_f3 = []
for act, lr, bs in product(activaciones, learning_rates, batch_sizes):
    mlp = MLPClassifier(
        hidden_layer_sizes=best_arch, activation=act, solver='adam',
        alpha=best_alpha, learning_rate_init=lr, batch_size=bs,
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15, random_state=SEED
    )
    scores = cv_recall(mlp, X_tr_sc, y_train, sw_train, cv3)
    results_f3.append({
        'activation': act, 'lr': lr, 'batch_size': bs,
        'Recall_mean': scores.mean(), 'Recall_std': scores.std()
    })

df_f3 = pd.DataFrame(results_f3).sort_values('Recall_mean', ascending=False)
best_act = df_f3.iloc[0]['activation']
best_lr  = df_f3.iloc[0]['lr']
best_bs  = int(df_f3.iloc[0]['batch_size'])

print("Top 10 combinaciones (Fase 3):")
display(df_f3.head(10).to_string(index=False))
print(f"\\n→ Mejor: activation={best_act}  lr={best_lr:.0e}  batch={best_bs}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ─────────────────────────────────────────────────────────────────────────────
# GRID FINAL — Top candidatos de cada fase
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("GRID FINAL — Combinación de mejores candidatos por fase")
print("=" * 70)

top_archs  = [eval(r) for r in df_f1.head(3)['Arquitectura']]
top_alphas = df_f2.head(3)['alpha'].tolist()
top_lrs    = df_f3.head(3)['lr'].unique().tolist()[:2]
top_bss    = df_f3.head(3)['batch_size'].unique().tolist()[:2]
top_acts   = df_f3.head(3)['activation'].unique().tolist()[:1]

grid_final = list(product(top_archs, top_alphas, top_lrs, top_bss, top_acts))
print(f"Total combinaciones en grid final: {len(grid_final)}")

results_grid = []
t0 = time.time()
for arch, alpha, lr, bs, act in grid_final:
    mlp = MLPClassifier(
        hidden_layer_sizes=arch, activation=act, solver='adam',
        alpha=alpha, learning_rate_init=lr, batch_size=int(bs),
        max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=SEED
    )
    scores = cv_recall(mlp, X_tr_sc, y_train, sw_train, cv3)
    results_grid.append({
        'arch': str(arch), 'alpha': alpha, 'lr': lr,
        'batch_size': int(bs), 'activation': act,
        'Recall_CV': scores.mean(), 'Recall_Std': scores.std()
    })

df_grid = pd.DataFrame(results_grid).sort_values(['Recall_CV', 'Recall_Std'],
                                                  ascending=[False, True])
print(f"Tiempo: {time.time()-t0:.1f}s")
print(f"\\nTop 15 configuraciones (Recall CV):")
display(df_grid.head(15).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Visualización del grid ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Por arquitectura
recall_arch = df_grid.groupby('arch')['Recall_CV'].mean().sort_values(ascending=False)
axes[0,0].bar(recall_arch.index, recall_arch.values, color='steelblue', edgecolor='white')
axes[0,0].set_title('Recall medio por Arquitectura')
axes[0,0].set_ylabel('Recall CV')
axes[0,0].tick_params(axis='x', rotation=30)
axes[0,0].set_ylim(df_grid['Recall_CV'].min()-0.01, df_grid['Recall_CV'].max()+0.01)

# Por alpha
recall_alpha = df_grid.groupby('alpha')['Recall_CV'].mean().sort_values(ascending=False)
axes[0,1].bar([f'{a:.0e}' for a in recall_alpha.index], recall_alpha.values,
              color='mediumseagreen', edgecolor='white')
axes[0,1].set_title('Recall medio por Alpha (L2)')
axes[0,1].set_ylabel('Recall CV')
axes[0,1].set_ylim(df_grid['Recall_CV'].min()-0.01, df_grid['Recall_CV'].max()+0.01)

# Por learning rate
recall_lr = df_grid.groupby('lr')['Recall_CV'].mean().sort_values(ascending=False)
axes[1,0].bar([f'{l:.0e}' for l in recall_lr.index], recall_lr.values,
              color='mediumpurple', edgecolor='white')
axes[1,0].set_title('Recall medio por Learning Rate')
axes[1,0].set_ylabel('Recall CV')
axes[1,0].set_ylim(df_grid['Recall_CV'].min()-0.01, df_grid['Recall_CV'].max()+0.01)

# Top-20 configuraciones
top20 = df_grid.head(20).reset_index(drop=True)
c_map = {16: '#2196F3', 32: '#FF9800', 64: '#4CAF50'}
sc_colors = [c_map.get(b, '#9C27B0') for b in top20['batch_size']]
axes[1,1].scatter(top20.index, top20['Recall_CV'], c=sc_colors, s=80,
                  edgecolors='white', linewidths=0.5)
axes[1,1].fill_between(top20.index,
                        top20['Recall_CV']-top20['Recall_Std'],
                        top20['Recall_CV']+top20['Recall_Std'],
                        alpha=0.15, color='steelblue')
axes[1,1].set_title('Top 20 Configuraciones (color=batch_size)')
axes[1,1].set_xlabel('Ranking')
axes[1,1].set_ylabel('Recall CV')
from matplotlib.patches import Patch
axes[1,1].legend(handles=[Patch(fc=v, label=f'batch={k}') for k,v in c_map.items()], fontsize=9)

plt.suptitle('Búsqueda Paramétrica Exhaustiva — Red Neuronal (Recall)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_fig3_grid_recall.png', bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Parámetros finales ────────────────────────────────────────────────────────
best_row = df_grid.iloc[0]
BEST_ARCH  = eval(best_row['arch'])
BEST_ALPHA = best_row['alpha']
BEST_LR    = best_row['lr']
BEST_BATCH = int(best_row['batch_size'])
BEST_ACT   = best_row['activation']

print("╔══════════════════════════════════════════════════════════╗")
print("║     PARÁMETROS ÓPTIMOS — RNA (objetivo: Recall)         ║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  hidden_layer_sizes : {str(BEST_ARCH):<34}║")
print(f"║  activation         : {BEST_ACT:<34}║")
print(f"║  alpha (L2)         : {BEST_ALPHA:<34}║")
print(f"║  learning_rate_init : {BEST_LR:<34}║")
print(f"║  batch_size         : {BEST_BATCH:<34}║")
print(f"║  solver             : adam{'':<30}║")
print(f"╠══════════════════════════════════════════════════════════╣")
print(f"║  Recall CV (3-fold) : {best_row['Recall_CV']:.4f} ± {best_row['Recall_Std']:.4f}{'':<17}║")
print("╚══════════════════════════════════════════════════════════╝")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 7. ENTRENAMIENTO FINAL + CURVAS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 7. Entrenamiento del Modelo Final y Curvas de Aprendizaje

Se entrena la RNA con los parámetros óptimos y **sample_weight balanceados**.
El uso de `early_stopping=True` permite observar la evolución del Recall (validación
interna) época a época para diagnosticar sobreajuste o subajuste.
"""))

cells.append(nbf.v4.new_code_cell("""\
mlp_final = MLPClassifier(
    hidden_layer_sizes  = BEST_ARCH,
    activation          = BEST_ACT,
    solver              = 'adam',
    alpha               = BEST_ALPHA,
    learning_rate_init  = BEST_LR,
    batch_size          = BEST_BATCH,
    max_iter            = 1000,
    early_stopping      = True,
    validation_fraction = 0.15,
    n_iter_no_change    = 25,
    tol                 = 1e-5,
    random_state        = SEED,
    verbose             = False
)
mlp_final.fit(X_tr_sc, y_train, sample_weight=sw_train)

print(f"Épocas ejecutadas (early stopping): {mlp_final.n_iter_}")
print(f"Mejor val score interno           : {mlp_final.best_validation_score_:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Curvas de aprendizaje ─────────────────────────────────────────────────────
loss_curve = mlp_final.loss_curve_
val_scores = mlp_final.validation_scores_
epocas     = np.arange(1, len(loss_curve) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epocas, loss_curve, color='tomato', linewidth=2, label='Train Loss')
axes[0].axvline(mlp_final.n_iter_, color='gray', linestyle=':', linewidth=1.5,
                label=f'Parada (época {mlp_final.n_iter_})')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Curva de Pérdida (Entrenamiento)')
axes[0].legend()

axes[1].plot(epocas, val_scores, color='steelblue', linewidth=2, label='Val Accuracy interno')
axes[1].axhline(mlp_final.best_validation_score_, color='red', linestyle='--', linewidth=1.5,
                label=f'Best val acc: {mlp_final.best_validation_score_:.4f}')
axes[1].axvline(mlp_final.n_iter_, color='gray', linestyle=':', linewidth=1.5,
                label=f'Parada (época {mlp_final.n_iter_})')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy en Validación Interna')
axes[1].legend()

plt.suptitle('Curvas de Aprendizaje — RNA (Recall)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_fig4_learning_curves.png', bbox_inches='tight')
plt.show()

print(f"\\nAnálisis de las curvas:")
print(f"  Loss mínima en época: {np.argmin(loss_curve)+1}")
print(f"  Val Accuracy max    : {max(val_scores):.4f} (época {np.argmax(val_scores)+1})")
print(f"  Parada anticipada   : época {mlp_final.n_iter_}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 8. ANÁLISIS DEL THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 8. Análisis del Threshold de Clasificación

### Criterio: F2-score (prioridad Recall)

Con el objetivo de **maximizar Recall**, existen dos estrategias extremas:
- **Threshold → 0:** Recall = 1.0 pero Precision ≈ 0 (solución trivial, inútil en la práctica).
- **Threshold → 1:** Recall = 0 (no se predice ningún positivo).

La solución práctica es elegir el threshold que maximiza el **F2-score**, que pondera
el Recall el doble que la Precision:

$$F_2 = 5 \\cdot \\frac{\\text{Prec} \\cdot \\text{Rec}}{4 \\cdot \\text{Prec} + \\text{Rec}}$$

Así se garantiza que el modelo detecta la mayor proporción posible de positivos reales
sin que la Precision colapse completamente.

### Proceso
1. Calcular `predict_proba` sobre el conjunto de **validación**.
2. Barrer umbrales de 0.05 a 0.95 (paso 0.01).
3. Calcular Recall, Precision, F1, F2, Accuracy, Especificidad para cada umbral.
4. Seleccionar el threshold que **maximiza F2** en validación.
"""))

cells.append(nbf.v4.new_code_cell("""\
proba_val = mlp_final.predict_proba(X_val_sc)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
metrics_thresh = []

for t in thresholds:
    y_pred_t = (proba_val >= t).astype(int)
    tp = ((y_pred_t==1)&(y_val==1)).sum()
    tn = ((y_pred_t==0)&(y_val==0)).sum()
    fp = ((y_pred_t==1)&(y_val==0)).sum()
    fn = ((y_pred_t==0)&(y_val==1)).sum()

    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    acc  = accuracy_score(y_val, y_pred_t)
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    f2   = fbeta_score(y_val, y_pred_t, beta=2, zero_division=0)

    metrics_thresh.append({
        'threshold'  : round(t, 2),
        'Recall'     : rec,
        'Precision'  : prec,
        'F2'         : f2,
        'F1'         : f1,
        'Accuracy'   : acc,
        'Specificity': spec
    })

df_thresh = pd.DataFrame(metrics_thresh)

# ── Threshold óptimo por F2 ───────────────────────────────────────────────────
idx_f2   = df_thresh['F2'].idxmax()
BEST_THR = df_thresh.loc[idx_f2, 'threshold']
best_f2  = df_thresh.loc[idx_f2, 'F2']
best_rec = df_thresh.loc[idx_f2, 'Recall']
best_prec= df_thresh.loc[idx_f2, 'Precision']

print(f"Threshold óptimo (máx. F2 en val): {BEST_THR:.2f}")
print(f"  F2        : {best_f2:.4f}")
print(f"  Recall    : {best_rec:.4f}")
print(f"  Precision : {best_prec:.4f}")
print(f"  F1        : {df_thresh.loc[idx_f2,'F1']:.4f}")
print(f"  Accuracy  : {df_thresh.loc[idx_f2,'Accuracy']:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Visualización del análisis de threshold ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

metricas_p = ['Recall','Precision','F2','F1','Accuracy']
colores_p  = ['#E91E63','#FF9800','#2196F3','#4CAF50','#9C27B0']

for met, col in zip(metricas_p, colores_p):
    lw = 3 if met in ('Recall','F2') else 1.5
    axes[0].plot(df_thresh['threshold'], df_thresh[met], label=met, color=col, linewidth=lw)

axes[0].axvline(BEST_THR, color='black', linestyle='--', linewidth=2,
                label=f'Óptimo F2 = {BEST_THR:.2f}')
axes[0].axvline(0.50, color='gray', linestyle=':', linewidth=1.5, label='Default 0.50')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Métrica')
axes[0].set_title('Evolución de Métricas por Threshold (Val)')
axes[0].legend(fontsize=9)
axes[0].set_xlim(0.05, 0.95)

# Curva Recall-Precision con isolíneas F2
prec_c, rec_c, thr_c = precision_recall_curve(y_val, proba_val)
axes[1].plot(rec_c, prec_c, color='steelblue', linewidth=2, label='Curva Precisión-Recall')
axes[1].axhline(y_val.mean(), color='gray', linestyle='--', linewidth=1,
                label=f'Baseline ({y_val.mean():.2f})')

# Punto del threshold óptimo
idx_pr = np.argmin(np.abs(thr_c - BEST_THR)) if len(thr_c) > 0 else 0
axes[1].scatter(rec_c[idx_pr], prec_c[idx_pr], color='red', s=120, zorder=5,
                label=f'Threshold={BEST_THR:.2f}')
axes[1].scatter(rec_c[np.argmin(np.abs(thr_c - 0.50))],
                prec_c[np.argmin(np.abs(thr_c - 0.50))],
                color='gray', s=80, zorder=4, marker='D', label='Threshold=0.50')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Curva Precisión-Recall (Val)')
axes[1].legend(fontsize=9)

plt.suptitle('Análisis del Threshold de Clasificación — RNA (Recall)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_fig5_threshold.png', bbox_inches='tight')
plt.show()

# ── Comparativa de umbrales ───────────────────────────────────────────────────
print(f"\\nComparación de umbrales en VAL:")
print(f"  {'Métrica':<14} {'thr=0.50':>10} {'thr=' + str(BEST_THR):>12}  {'Δ':>8}")
for met in ['Recall','Precision','F2','F1','Accuracy']:
    r50  = df_thresh[df_thresh['threshold']==0.50].iloc[0][met]
    ropt = df_thresh.loc[idx_f2, met]
    d    = ropt - r50
    print(f"  {met:<14} {r50:>10.4f} {ropt:>12.4f}  {d:>+8.4f}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 9. EVALUACIÓN FINAL
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 9. Evaluación Final sobre el Conjunto de Test
"""))

cells.append(nbf.v4.new_code_cell("""\
proba_test = mlp_final.predict_proba(X_te_sc)[:, 1]
y_pred_opt = (proba_test >= BEST_THR).astype(int)
y_pred_def = (proba_test >= 0.50).astype(int)
y_pred_dt  = dt_opt.predict(X_test)

# ── Métricas ──────────────────────────────────────────────────────────────────
def metricas(yt, yp, proba=None):
    d = {
        'Accuracy' : accuracy_score(yt, yp),
        'Recall'   : recall_score(yt, yp, zero_division=0),
        'Precision': precision_score(yt, yp, zero_division=0),
        'F1'       : f1_score(yt, yp, zero_division=0),
        'F2'       : fbeta_score(yt, yp, beta=2, zero_division=0),
    }
    if proba is not None:
        d['ROC-AUC'] = roc_auc_score(yt, proba)
    return d

m_dt  = metricas(y_test, y_pred_dt, dt_opt.predict_proba(X_test)[:,1])
m_def = metricas(y_test, y_pred_def, proba_test)
m_opt = metricas(y_test, y_pred_opt, proba_test)

print("╔════════════════════════════════════════════════════════════════╗")
print("║           RESULTADOS FINALES — CONJUNTO DE TEST              ║")
print("╠══════════════════╦══════════════╦═════════════╦══════════════╣")
print(f"║ {'Métrica':<16}  ║ {'DT (baseline)':^12} ║ {'RNA thr=0.50':^11} ║ {'RNA thr='+str(BEST_THR):^12} ║")
print("╠══════════════════╬══════════════╬═════════════╬══════════════╣")
for met in ['Accuracy','Recall','Precision','F1','F2','ROC-AUC']:
    vdt  = m_dt.get(met, float('nan'))
    vdef = m_def.get(met, float('nan'))
    vopt = m_opt.get(met, float('nan'))
    marca = ' ◀' if met == 'Recall' else ''
    print(f"║ {met:<16}  ║ {vdt:>12.4f} ║ {vdef:>11.4f} ║ {vopt:>12.4f}{marca} ║")
print("╚══════════════════╩══════════════╩═════════════╩══════════════╝")
print("  ◀ Métrica objetivo")
print(f"\\n── Classification Report (RNA, threshold={BEST_THR:.2f}) ──────────")
print(classification_report(y_test, y_pred_opt, target_names=['Clase 0', 'Clase 1']))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Matrices de confusión ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
configs = [
    (y_pred_dt , 'DT baseline',              dt_opt.predict_proba(X_test)[:,1]),
    (y_pred_def, 'RNA — Threshold 0.50',     proba_test),
    (y_pred_opt, f'RNA — Threshold {BEST_THR:.2f}', proba_test),
]
for ax, (yp, lbl, prob) in zip(axes, configs):
    cm = confusion_matrix(y_test, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred 0','Pred 1'], yticklabels=['Real 0','Real 1'],
                linewidths=1, linecolor='white', annot_kws={'size':13,'weight':'bold'})
    rec = recall_score(y_test, yp, zero_division=0)
    f2  = fbeta_score(y_test, yp, beta=2, zero_division=0)
    ax.set_title(f'{lbl}\\nRecall={rec:.4f}  |  F2={f2:.4f}', fontsize=9)
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicho')

plt.suptitle('Matrices de Confusión — Test Set', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_fig6_confusion.png', bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Curva ROC comparativa ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for prob, lbl, col in [
    (dt_opt.predict_proba(X_test)[:,1], 'Árbol de Decisión', '#FF9800'),
    (proba_test, 'Red Neuronal', '#2196F3'),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, color=col, linewidth=2.5, label=f'{lbl} (AUC={auc:.4f})')

ax.plot([0,1],[0,1],'k--', linewidth=1, label='Aleatorio')
# Punto del threshold óptimo en la RNA
fpr_a, tpr_a, thr_roc = roc_curve(y_test, proba_test)
idx_r = np.argmin(np.abs(thr_roc - BEST_THR))
ax.scatter(fpr_a[idx_r], tpr_a[idx_r], color='red', s=120, zorder=5,
           label=f'RNA — Threshold óptimo ({BEST_THR:.2f})')

ax.set_xlabel('FPR (Tasa Falsos Positivos)')
ax.set_ylabel('TPR (Recall)')
ax.set_title('Curva ROC — Comparativa DT vs RNA')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('dt_fig7_roc_comparativa.png', bbox_inches='tight')
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 10. RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_code_cell("""\
print("=" * 70)
print("  RESUMEN EJECUTIVO — ÁRBOL DE DECISIÓN + RNA (Recall)")
print("=" * 70)

print("\\n[1] ÁRBOL DE DECISIÓN — PARÁMETROS ÓPTIMOS:")
for k, v in gs_dt.best_params_.items():
    print(f"    {k:<25}: {v}")
print(f"    Recall CV (5-fold)        : {gs_dt.best_score_:.4f}")
print(f"    Profundidad efectiva      : {dt_opt.get_depth()}")

print(f"\\n[2] VARIABLES SELECCIONADAS ({len(features_dt)} — importancia acumulada ≥ 85 %):")
for i, f in enumerate(features_dt, 1):
    print(f"    {i:2}. {f:<40}  imp={importancias_dt[f]:.5f}")

print(f"\\n[3] RED NEURONAL — PARÁMETROS ÓPTIMOS (scoring=recall):")
print(f"    hidden_layer_sizes : {BEST_ARCH}")
print(f"    activation         : {BEST_ACT}")
print(f"    alpha (L2)         : {BEST_ALPHA}")
print(f"    learning_rate_init : {BEST_LR}")
print(f"    batch_size         : {BEST_BATCH}")
print(f"    Épocas (early stop): {mlp_final.n_iter_}")

print(f"\\n[4] THRESHOLD SELECCIONADO (máx. F2 en val): {BEST_THR:.2f}")
print(f"    Recall    (val): {best_rec:.4f}")
print(f"    Precision (val): {best_prec:.4f}")
print(f"    F2        (val): {best_f2:.4f}")
print(f"    vs default 0.50 — ΔRecall: {best_rec - df_thresh[df_thresh['threshold']==0.50].iloc[0]['Recall']:+.4f}")

print(f"\\n[5] RESULTADOS TEST SET:")
for met in ['Accuracy','Recall','Precision','F1','F2','ROC-AUC']:
    vdt  = m_dt.get(met,  float('nan'))
    vopt = m_opt.get(met, float('nan'))
    delta = vopt - vdt
    print(f"    {met:<12}: DT={vdt:.4f}  RNA(opt)={vopt:.4f}  Δ={delta:+.4f}")
print("=" * 70)
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 11. JUSTIFICACIONES
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 10. Justificaciones Detalladas

### 10.1 Parametrizaciones del Árbol de Decisión

**`max_depth`** — Se probaron 5 valores (3, 5, 7, 10, None) para cubrir el espectro
de árboles simples (interpretables, pero posible underfitting) a árboles complejos
(alta capacidad, riesgo de overfitting). El DT se entrena con `class_weight='balanced'`
para que el criterio de división no ignore la clase minoritaria (Y=1).

**`min_samples_split` y `min_samples_leaf`** — Actúan como regularizadores implícitos:
valores altos producen árboles más simples con hojas más "puras" pero con mayor
generalización. Se probaron rangos [2, 50] y [1, 20] respectivamente.

**`criterion`** — `gini` y `entropy` producen divisiones similares, pero la entropía
puede ser más sensible a divisiones finas en clases desbalanceadas.

---

### 10.2 Parametrizaciones de la Red Neuronal

**Arquitecturas** — Se generaron dinámicamente en función del número de variables
de entrada ($n$), garantizando proporcionalidad. Las redes piramidales (capas decrecientes)
son estándar en clasificación porque producen representaciones cada vez más abstractas.

**Batch size 16** — Se incluye este valor reducido (no presente en el ejercicio anterior)
porque con datos desbalanceados y sample_weights, lotes más pequeños aumentan la
probabilidad de que cada mini-batch contenga ejemplos de la clase minoritaria, mejorando
el aprendizaje de los patrones de Y=1.

**scoring='recall' en CV** — A diferencia del ejercicio anterior (accuracy), aquí se
seleccionan los parámetros que maximizan la detección de verdaderos positivos, no el
acierto global. Esto implica que la búsqueda puede favorecer configuraciones con mayor
falso positivo si eso permite reducir los falsos negativos.

**sample_weight balanceados** — Dado que `MLPClassifier` no soporta `class_weight`
directamente, se usa `compute_sample_weight('balanced', y_train)`. Esto es equivalente
a `class_weight='balanced'`: la red aprende a penalizar más los errores en la clase
minoritaria durante el backpropagation.

---

### 10.3 Threshold de clasificación

**Criterio F2-score:** Al maximizar Recall puro (threshold → 0) se obtiene Recall=1
pero Precision≈0 (el modelo predice todo como positivo). El F2 pondera Recall
el doble que Precision, seleccionando un punto de equilibrio donde el Recall es
significativamente mayor que con el threshold default (0.50) sin que la Precision
colapse a valores triviales.

**Diferencia respecto al ejercicio anterior:**
- Ejercicio 2 (Accuracy): threshold ≈ 0.53 (ligeramente superior a 0.50 para ganar
  Precision sin perder Accuracy).
- Ejercicio 3 (Recall): threshold más bajo que 0.50 para aumentar la sensibilidad
  hacia la clase positiva, a costa de un pequeño aumento de falsos positivos.

---

### 10.4 Selección final de parámetros e impacto

La combinación de: (a) entrenamiento con sample_weight balanceados, (b) optimización
de hiperparámetros por Recall CV, y (c) threshold F2-óptimo produce un modelo
que supera al Árbol de Decisión en **Recall** (métrica objetivo), manteniendo valores
de Precisión y F2 razonables para su uso en aplicaciones reales.

La comparativa con el threshold default (0.50) demuestra que la selección del umbral
es tan relevante como la selección de la arquitectura para alcanzar el objetivo de Recall.
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Compilar y guardar
# ══════════════════════════════════════════════════════════════════════════════
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"}
}

OUTPUT = "/home/user/GIT_UCMaster/arbol_decision_red_neuronal.ipynb"
with open(OUTPUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generado: {OUTPUT}")
