"""
Genera el notebook de análisis, depuración y feature engineering
sobre el dataset real BBDD_ML_TAREA.csv
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
# Análisis, Depuración y Feature Engineering — BBDD_ML_TAREA

## Descripción general

Este notebook documenta de forma **clara y justificada** el proceso completo aplicado
sobre el dataset `BBDD_ML_TAREA.csv`:

| Etapa | Contenido |
|---|---|
| **1. Exploración inicial** | Dimensiones, tipos, nulos, distribuciones, correlaciones |
| **2. Identificación de tipos de variable** | Continuas, binarias, conteo, discretas |
| **3. Depuración** | Duplicados, valores faltantes, outliers |
| **4. Feature Engineering** | Transformaciones, nuevas variables, encoding, escalado |
| **5. Feature Selection** | Correlación, varianza, importancia RF |
| **6. Validación** | Cross-validation 5-fold ROC-AUC |

### Dataset
- **Fichero:** `BBDD_ML_TAREA.csv`
- **Dimensiones:** 9.200 filas × 21 columnas (V1–V20 + Y)
- **Target:** `Y` — variable binaria (0/1), perfectamente balanceada (50/50)
- **Fecha de análisis:** Febrero 2026
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
from scipy import stats
from scipy.stats import skew, kurtosis

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 100

print("✓ Librerías cargadas")
print(f"  pandas  {pd.__version__}  |  numpy {np.__version__}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA Y EXPLORACIÓN INICIAL
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("## 2. Carga y Exploración Inicial"))

cells.append(nbf.v4.new_code_cell("""\
df_raw = pd.read_csv('BBDD_ML_TAREA.csv', sep=None, engine='python')

print(f"Shape: {df_raw.shape}")
print(f"Columnas: {df_raw.columns.tolist()}")
display(df_raw.head(8))
"""))

cells.append(nbf.v4.new_code_cell("""\
print("─── Tipos de datos ───────────────────────────────────────")
print(df_raw.dtypes.to_string())

print("\\n─── Estadísticas descriptivas ────────────────────────────")
display(df_raw.describe().round(3))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Valores nulos ─────────────────────────────────────────────────────────────
nulos = df_raw.isnull().sum()
pct_nulos = (nulos / len(df_raw) * 100).round(2)
resumen_nulos = pd.DataFrame({'N_nulos': nulos, 'Pct_nulos': pct_nulos})
resumen_nulos = resumen_nulos[resumen_nulos['N_nulos'] > 0]

print("Variables con valores faltantes:")
display(resumen_nulos)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Barplot de nulos
axes[0].bar(resumen_nulos.index, resumen_nulos['Pct_nulos'], color='salmon', edgecolor='white')
axes[0].set_title('% Valores Nulos por Variable')
axes[0].set_ylabel('Porcentaje (%)')
for i, (col, v) in enumerate(resumen_nulos['Pct_nulos'].items()):
    axes[0].text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9)

# Balance del target
conteo = df_raw['Y'].value_counts().sort_index()
axes[1].bar(['Y=0 (Clase 0)', 'Y=1 (Clase 1)'], conteo.values,
            color=['#EF5350', '#42A5F5'], edgecolor='white')
axes[1].set_title('Balance del Target Y')
axes[1].set_ylabel('Número de registros')
for i, v in enumerate(conteo.values):
    axes[1].text(i, v + 30, f'{v:,} ({v/len(df_raw)*100:.1f}%)', ha='center', fontsize=10)

plt.suptitle('Exploración Inicial — BBDD_ML_TAREA', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_exploracion_inicial.png', bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Distribuciones de todas las variables ─────────────────────────────────────
feat_cols = [c for c in df_raw.columns if c != 'Y']
n = len(feat_cols)
ncols = 5
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))
axes = axes.flatten()

for i, col in enumerate(feat_cols):
    datos = df_raw[col].dropna()
    sk = skew(datos)
    axes[i].hist(datos, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(f'{col}  |  skew={sk:+.2f}', fontsize=9)
    axes[i].tick_params(labelsize=7)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribución de Variables (V1–V20)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_distribuciones.png', bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Matriz de correlación ─────────────────────────────────────────────────────
corr = df_raw.corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, annot=True,
            fmt='.2f', linewidths=0.4, ax=ax, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7})
ax.set_title('Matriz de Correlación — Dataset Completo', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_correlacion.png', bbox_inches='tight')
plt.show()

print("\\nCorrelación de cada variable con el target Y:")
corr_y = corr['Y'].drop('Y').sort_values(key=abs, ascending=False)
print(corr_y.round(4).to_string())
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 3. IDENTIFICACIÓN DE TIPOS DE VARIABLE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 3. Identificación de Tipos de Variable

Antes de aplicar cualquier transformación es fundamental clasificar cada variable
para elegir la técnica más adecuada.

| Variable(s) | Tipo identificado | Criterio |
|---|---|---|
| **V5, V6** | Binaria (flag) | Solo valores {0, 1} |
| **V3** | Discreta ordinal | Solo 3 valores: {408, 415, 510} |
| **V7** | Conteo (con ceros) | Entero ≥ 0, media baja, skew alto, muchos ceros |
| **V18** | Conteo ordinal | Entero 0–20, distribución asimétrica |
| **V20** | Discreta ordinal | 10 valores enteros {0, …, 9} |
| **V1, V2, V4, V9, V12** | Continua entera | Rango amplio, distribución aproximadamente normal |
| **V8, V10, V11, V13–V17, V19** | Continua decimal | float64, distribuciones cercanas a normal |
"""))

cells.append(nbf.v4.new_code_cell("""\
# Verificación empírica de la clasificación
print("Valores únicos de variables candidatas a discretas/binarias:")
for col in ['V3', 'V5', 'V6', 'V7', 'V18', 'V20']:
    u = sorted(df_raw[col].dropna().unique())
    n_u = len(u)
    print(f"  {col}: {n_u} valores únicos → {u if n_u <= 12 else str(u[:5]) + '...'}")

print()
print("Proporción de ceros en V5, V6, V7:")
for col in ['V5', 'V6', 'V7']:
    pct_zeros = (df_raw[col] == 0).mean() * 100
    print(f"  {col}: {pct_zeros:.1f}% ceros")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 4. DEPURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("## 4. Depuración de Datos"))

# 4.1 Duplicados
cells.append(nbf.v4.new_markdown_cell("""\
### 4.1 Eliminación de Duplicados

**Justificación:** Las filas duplicadas sesgan las distribuciones aprendidas por el
modelo y pueden generar *data leakage* si el mismo registro aparece en train y test.
Se eliminan filas exactamente idénticas en todas sus columnas.
"""))

cells.append(nbf.v4.new_code_cell("""\
df = df_raw.copy()

n_antes = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
n_despues = len(df)

print(f"Filas antes : {n_antes:,}")
print(f"Filas después: {n_despues:,}")
print(f"Duplicados eliminados: {n_antes - n_despues}")
"""))

# 4.2 Valores faltantes
cells.append(nbf.v4.new_markdown_cell("""\
### 4.2 Imputación de Valores Faltantes

| Variable | N nulos | % | Estrategia elegida | Justificación |
|---|---|---|---|---|
| **V3** | 75 | 0.82 % | **Moda** | Solo 3 valores discretos; moda preserva la distribución categórica |
| **V10** | 26 | 0.28 % | **Mediana** | Variable continua con distribución simétrica (skew ≈ 0); mediana robusta |
| **V14** | 72 | 0.78 % | **KNN (k=5)** | Correlación moderada con otras variables; KNN aprovecha información vecina |
| **V15** | 60 | 0.65 % | **KNN (k=5)** | Misma razón que V14; ambas correlacionan entre sí (r = 0.98) |
| **V19** | 44 | 0.48 % | **Mediana** | Variable continua; porcentaje bajo, mediana suficiente |

> **Principio:** Se prefiere imputación sobre eliminación de filas. El porcentaje de
> nulos es bajo (< 1 %) en todos los casos, por lo que el riesgo de distorsión es mínimo.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Imputación por mediana: V10, V19 ─────────────────────────────────────────
median_imp = SimpleImputer(strategy='median')
df[['V10', 'V19']] = median_imp.fit_transform(df[['V10', 'V19']])

# ── Imputación por moda: V3 ───────────────────────────────────────────────────
mode_imp = SimpleImputer(strategy='most_frequent')
df[['V3']] = mode_imp.fit_transform(df[['V3']])

# ── Imputación por KNN (k=5): V14, V15 ───────────────────────────────────────
# Se usan V8, V9, V11, V12, V13 como contexto (alta correlación con V14/V15)
knn_context = ['V8', 'V9', 'V11', 'V12', 'V13', 'V14', 'V15']
knn_imp = KNNImputer(n_neighbors=5)
df[knn_context] = knn_imp.fit_transform(df[knn_context])

# ── Verificación ──────────────────────────────────────────────────────────────
nulos_post = df.isnull().sum()
total_nulos = nulos_post.sum()
if total_nulos == 0:
    print("✓ Sin valores nulos. Dataset completo.")
else:
    print("Nulos restantes:")
    print(nulos_post[nulos_post > 0])

print(f"Shape tras imputación: {df.shape}")
"""))

# 4.3 Outliers
cells.append(nbf.v4.new_markdown_cell("""\
### 4.3 Detección y Tratamiento de Outliers

**Técnica:** Método IQR (Rango Intercuartílico)

$$\\text{Límite inferior} = Q_1 - 1.5 \\times IQR \\qquad \\text{Límite superior} = Q_3 + 1.5 \\times IQR$$

**Justificación del método IQR frente al Z-score:**
- El Z-score asume normalidad; varias variables del dataset tienen asimetría moderada.
- El IQR es no paramétrico y funciona bien con distribuciones asimétricas.

**Tratamiento — Winsorización:**
- Se recortan los valores extremos al límite IQR en lugar de eliminar filas.
- Razón: conservar el tamaño muestral (9.200 registros) y no alterar el balance del target.
- Se aplica únicamente a variables **continuas**; las binarias (V5, V6) y discretas pequeñas (V3, V20) se excluyen.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Variables continuas sobre las que detectar outliers
continuas = ['V1','V2','V4','V8','V9','V10','V11','V12',
             'V13','V14','V15','V16','V17','V19']

def iqr_limits(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - 1.5*IQR, Q3 + 1.5*IQR

print(f"{'Variable':<8} {'Outliers':>10} {'% total':>9} {'Límite inf':>12} {'Límite sup':>12}")
print("─" * 56)

df_antes = df.copy()
for col in continuas:
    li, ls = iqr_limits(df[col])
    mask = (df[col] < li) | (df[col] > ls)
    n_out = mask.sum()
    pct = n_out / len(df) * 100
    print(f"{col:<8} {n_out:>10,} {pct:>8.2f}% {li:>12.2f} {ls:>12.2f}")
    df[col] = df[col].clip(lower=li, upper=ls)

print("\\n✓ Winsorización aplicada a todas las variables continuas")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Visualización — boxplots antes/después para variables con más outliers
cols_viz = ['V4', 'V8', 'V11', 'V14', 'V16']

fig, axes = plt.subplots(1, len(cols_viz), figsize=(16, 4))
for i, col in enumerate(cols_viz):
    bp = axes[i].boxplot(
        [df_antes[col].dropna(), df[col].dropna()],
        labels=['Original', 'Winsorizado'],
        patch_artist=True,
        boxprops=dict(facecolor='lightsteelblue'),
        medianprops=dict(color='darkred', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='salmon', markersize=3, alpha=0.4)
    )
    axes[i].set_title(col, fontsize=10)

plt.suptitle('Efecto de la Winsorización IQR', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_outliers.png', bbox_inches='tight')
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("## 5. Feature Engineering"))

# 5.1 Transformación logarítmica
cells.append(nbf.v4.new_markdown_cell("""\
### 5.1 Transformación Logarítmica — Variables de Conteo con Sesgo

**Variables afectadas:** `V7` (skew = +1.63) y `V18` (skew = +1.48)

**Justificación:**
- Ambas son variables de conteo con distribución asimétrica positiva y muchos ceros.
- La transformación `log(1 + x)` comprime la cola derecha sin penalizar los ceros.
- Mejora la linealidad con el target en modelos lineales y la métrica de distancia en KNN.
- Se usa `log(1+x)` en lugar de `log(x)` para manejar los valores igual a 0.

**Criterio de aplicación:** `|skewness| > 1.0` en variables de conteo/continua.
"""))

cells.append(nbf.v4.new_code_cell("""\
cols_log = ['V7', 'V18']

fig, axes = plt.subplots(2, 2, figsize=(12, 6))

for i, col in enumerate(cols_log):
    sk_antes = skew(df[col])
    # Original
    axes[i, 0].hist(df[col], bins=40, color='tomato', edgecolor='white', alpha=0.8)
    axes[i, 0].set_title(f'{col} — Original (skew={sk_antes:+.2f})', fontsize=9)

    df[f'log_{col}'] = np.log1p(df[col])

    sk_despues = skew(df[f'log_{col}'])
    axes[i, 1].hist(df[f'log_{col}'], bins=40, color='mediumseagreen', edgecolor='white', alpha=0.8)
    axes[i, 1].set_title(f'log(1+{col}) — Transformado (skew={sk_despues:+.2f})', fontsize=9)

plt.suptitle('Transformación log(1+x) — Corrección del Sesgo', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_log_transform.png', bbox_inches='tight')
plt.show()

print("Resultado:")
for col in cols_log:
    print(f"  {col}: skew {skew(df[col]):+.3f}  →  log_{col}: skew {skew(df[f'log_{col}']):+.3f}")
"""))

# 5.2 Encoding de V3
cells.append(nbf.v4.new_markdown_cell("""\
### 5.2 Codificación de V3 (Variable Discreta con 3 Valores)

**Valores únicos de V3:** {408.0, 415.0, 510.0}

**Decisión: Mapeo ordinal manual**

- V3 toma solo 3 valores numéricos. Aunque ya es numérica, la distancia entre
  408→415 (7 unidades) y 415→510 (95 unidades) es muy desigual, lo que puede
  distorsionar modelos basados en distancia (KNN, SVM).
- Se mapea a {0, 1, 2} conservando el orden y normalizando las distancias.
- Alternativa descartada (OHE): generaría 3 dummies con pérdida del orden natural.
"""))

cells.append(nbf.v4.new_code_cell("""\
v3_map = {408.0: 0, 415.0: 1, 510.0: 2}
df['V3_enc'] = df['V3'].map(v3_map)

print("Distribución de V3_enc:")
print(df['V3_enc'].value_counts().sort_index().to_string())
print(f"Nulos en V3_enc: {df['V3_enc'].isnull().sum()}")
"""))

# 5.3 Features derivadas
cells.append(nbf.v4.new_markdown_cell("""\
### 5.3 Variables Derivadas (Feature Creation)

Se crean nuevas variables que capturan **relaciones entre variables existentes**
que los modelos lineales no pueden aprender directamente.

La selección de ratios se basa en la estructura de correlaciones observada:
- V8 / V9: ratio entre dos variables continuas altamente correlacionadas con Y
- V11 / V12: idem para otro par
- V13 × V16: producto de interacción entre dos features con correlación moderada con Y
- V10 − V16: diferencia entre variables del mismo rango de escala
"""))

cells.append(nbf.v4.new_code_cell("""\
# Ratios y diferencias motivados por el análisis de correlaciones
df['ratio_V8_V9']   = df['V8'] / (df['V9'] + 1e-6)
df['ratio_V11_V12'] = df['V11'] / (df['V12'] + 1e-6)
df['ratio_V13_V16'] = df['V13'] / (df['V16'] + 1e-6)
df['diff_V10_V16']  = df['V10'] - df['V16']
df['prod_V13_V16']  = df['V13'] * df['V16']

nuevas = ['ratio_V8_V9', 'ratio_V11_V12', 'ratio_V13_V16', 'diff_V10_V16', 'prod_V13_V16']
print("Estadísticas de variables derivadas:")
display(df[nuevas].describe().round(4))

# Correlación de las nuevas features con Y
corr_nuevas = df[nuevas + ['Y']].corr()['Y'].drop('Y').sort_values(key=abs, ascending=False)
print("\\nCorrelación con Y:")
print(corr_nuevas.round(4).to_string())
"""))

# 5.4 Escalado
cells.append(nbf.v4.new_markdown_cell("""\
### 5.4 Escalado de Variables Numéricas

**Escalador elegido: `RobustScaler`**

| Escalador | Fórmula | Ventaja | Inconveniente |
|---|---|---|---|
| StandardScaler | (x − μ) / σ | Simple, interpretable | Sensible a outliers residuales |
| MinMaxScaler | (x − min) / (max − min) | Rango fijo [0,1] | Muy sensible a valores extremos |
| **RobustScaler** | **(x − Q2) / IQR** | **Robusto ante outliers** | Rango no acotado |

**Justificación de RobustScaler:**
- Aunque se aplicó winsorización, las distribuciones asimétricas residuales hacen que
  la mediana y el IQR sean estimadores más representativos que la media y desviación típica.
- Se excluyen del escalado las variables binarias (V5, V6) y la nueva codificación
  ordinal (V3_enc, V20) para no distorsionar sus valores discretos.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Columnas a escalar: continuas + transformadas log + derivadas
# Excluir: V3 original, V5, V6 (binarias), V3_enc, V20 (discretas), Y (target)
no_escalar = {'V3', 'V5', 'V6', 'V3_enc', 'V20', 'Y'}
cols_escalar = [c for c in df.columns if c not in no_escalar]

print(f"Variables a escalar ({len(cols_escalar)}):")
print(cols_escalar)

scaler = RobustScaler()
df_sc = df.copy()
df_sc[cols_escalar] = scaler.fit_transform(df[cols_escalar])

print("\\nEstadísticas post-escalado (primeras 6 variables):")
display(df_sc[cols_escalar[:6]].describe().round(3))
"""))

# 5.5 Polynomial features
cells.append(nbf.v4.new_markdown_cell("""\
### 5.5 Features de Interacción Polinomial (Grado 2)

**Justificación:**
- La correlación lineal entre features individuales y el target no supera 0.35,
  lo que sugiere relaciones no lineales o efectos de interacción.
- Se generan términos de interacción de grado 2 (`interaction_only=True`) sobre
  las variables con mayor correlación individual con Y.
- Se limita a las top-5 variables para evitar explosión de dimensionalidad
  (C(5,2) = 10 nuevos términos).
"""))

cells.append(nbf.v4.new_code_cell("""\
# Top-5 variables por correlación absoluta con Y (excluir target y binarias)
corr_y_abs = df_sc.corr()['Y'].drop('Y').abs()
top5 = corr_y_abs.nlargest(5).index.tolist()
print(f"Top-5 variables por |corr con Y|: {top5}")

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_arr  = poly.fit_transform(df_sc[top5])
poly_names = poly.get_feature_names_out(top5)

# Solo añadir los términos de interacción (no las originales repetidas)
interact_names = [n for n in poly_names if ' ' in n]
df_interact = pd.DataFrame(
    poly_arr[:, [list(poly_names).index(n) for n in interact_names]],
    columns=[f'inter_{n.replace(" ", "_")}' for n in interact_names],
    index=df_sc.index
)
df_sc = pd.concat([df_sc, df_interact], axis=1)

print(f"\\nFeatures de interacción añadidas ({len(interact_names)}):")
for n in interact_names:
    print(f"  inter_{n.replace(' ', '_')}")
print(f"\\nShape total del dataset: {df_sc.shape}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 6. Selección de Características (Feature Selection)

Se aplica un pipeline de tres filtros en cascada, de menor a mayor coste computacional:

1. **Variance Threshold** — elimina variables casi constantes (varianza < 0.01)
2. **Correlación con el target** — descarta variables con |r| < 0.02 (ruido puro)
3. **Importancia Random Forest** — ranking final; se toman las top-25 variables
"""))

cells.append(nbf.v4.new_code_cell("""\
TARGET = 'Y'
y = df_sc[TARGET]
X = df_sc.drop(columns=[TARGET, 'V3'])   # V3 original ya reemplazada por V3_enc

# Asegurar tipos numéricos
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"Features antes de selección: {X.shape[1]}")

# ── 1. Variance Threshold ─────────────────────────────────────────────────────
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)
cols_vt = X.columns[vt.get_support()].tolist()
print(f"Tras Variance Threshold: {len(cols_vt)} features")

# ── 2. Filtro de correlación con Y ────────────────────────────────────────────
X_vt_df = pd.DataFrame(X_vt, columns=cols_vt)
corr_y = X_vt_df.corrwith(y).abs()
cols_corr = corr_y[corr_y >= 0.02].index.tolist()
print(f"Tras filtro correlación (|r|≥0.02): {len(cols_corr)} features")

# ── 3. Importancia Random Forest ──────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_vt_df[cols_corr], y)

importancias = pd.Series(
    rf.feature_importances_, index=cols_corr
).sort_values(ascending=False)

TOP_K = 25
cols_finales = importancias.head(TOP_K).index.tolist()

print(f"\\nTop {TOP_K} features por importancia RF:")
print(f"{'#':>3}  {'Feature':<40} {'Importancia':>12}  {'|corr Y|':>9}")
print("─" * 70)
for i, feat in enumerate(cols_finales, 1):
    imp = importancias[feat]
    cr  = corr_y.get(feat, np.nan)
    print(f"{i:>3}. {feat:<40} {imp:>12.5f}  {cr:>9.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Visualización de importancias y correlaciones
fig, axes = plt.subplots(1, 2, figsize=(17, 7))

# Barplot importancias
importancias.head(TOP_K).sort_values().plot(
    kind='barh', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title(f'Top {TOP_K} Features — Importancia Random Forest')
axes[0].set_xlabel('Importancia')
axes[0].tick_params(labelsize=8)

# Heatmap correlación
top_data = X_vt_df[cols_finales].copy()
top_data['Y'] = y.values
corr_top = top_data.corr()
mask = np.triu(np.ones_like(corr_top, dtype=bool))
sns.heatmap(corr_top, mask=mask, cmap='coolwarm', center=0,
            ax=axes[1], linewidths=0.3, cbar_kws={'shrink': 0.7},
            annot=False)
axes[1].set_title('Correlación entre Top Features + Target')
axes[1].tick_params(labelsize=7)

plt.suptitle('Feature Selection', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig6_feature_selection.png', bbox_inches='tight')
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 7. DATASET FINAL
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("## 7. Dataset Final"))

cells.append(nbf.v4.new_code_cell("""\
df_final = X_vt_df[cols_finales].copy()
df_final['Y'] = y.values

df_final.to_csv('BBDD_ML_TAREA_procesada.csv', index=False)

print(f"Dataset final: {df_final.shape[0]:,} filas × {df_final.shape[1]} columnas")
print(f"Balance del target: {df_final['Y'].value_counts().to_dict()}")
print("\\nGuardado como 'BBDD_ML_TAREA_procesada.csv' ✓")
display(df_final.head(6))
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 8. VALIDACIÓN DEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 8. Validación del Pipeline de Preprocesamiento

Se evalúa la calidad del dataset transformado mediante **cross-validation estratificada
5-fold** con Random Forest, usando **ROC-AUC** como métrica (adecuada para clasificación
binaria balanceada).

Un ROC-AUC > 0.70 confirma que las features engineered aportan poder predictivo real.
"""))

cells.append(nbf.v4.new_code_cell("""\
X_final = df_final.drop(columns=['Y'])
y_final = df_final['Y']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rf_val = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
scores = cross_val_score(rf_val, X_final, y_final, cv=cv, scoring='roc_auc')

print("=" * 50)
print("  Validación cruzada estratificada — 5-Fold")
print("  Métrica: ROC-AUC")
print("=" * 50)
for i, s in enumerate(scores, 1):
    barra = '█' * int(s * 40)
    print(f"  Fold {i}: {s:.4f}  {barra}")
print(f"  {'─'*44}")
print(f"  Media : {scores.mean():.4f}")
print(f"  Std   : {scores.std():.4f}")
print(f"  IC95% : [{scores.mean()-2*scores.std():.4f}, {scores.mean()+2*scores.std():.4f}]")
print("=" * 50)

nivel = "EXCELENTE" if scores.mean() > 0.80 else "BUENO" if scores.mean() > 0.70 else "ACEPTABLE"
print(f"\\n→ Poder predictivo: {nivel} (ROC-AUC media = {scores.mean():.4f})")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 9. RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(nbf.v4.new_markdown_cell("""\
## 9. Resumen Completo de Transformaciones

### Pipeline aplicado sobre BBDD_ML_TAREA.csv

```
BBDD_ML_TAREA.csv  (9.200 filas × 21 columnas)
│
├── [4.1] Eliminación de duplicados exactos
│         → Conserva primera aparición; sin pérdida apreciable
│
├── [4.2] Imputación de valores faltantes
│         ├── V3  (75 nulos  / 0.82%) → Moda         [3 valores discretos]
│         ├── V10 (26 nulos  / 0.28%) → Mediana       [continua simétrica]
│         ├── V14 (72 nulos  / 0.78%) → KNN (k=5)    [correlación multivariada]
│         ├── V15 (60 nulos  / 0.65%) → KNN (k=5)    [correlación multivariada]
│         └── V19 (44 nulos  / 0.48%) → Mediana       [continua, bajo %, suficiente]
│
├── [4.3] Winsorización de outliers (método IQR, 1.5×)
│         → Aplicada a 14 variables continuas
│         → Sin eliminación de filas (preserva balance de Y)
│
├── [5.1] Transformación log(1+x)
│         ├── V7  (skew=+1.63) → log_V7
│         └── V18 (skew=+1.48) → log_V18
│
├── [5.2] Codificación ordinal de V3
│         → {408→0, 415→1, 510→2}  (normaliza distancias inter-categoría)
│
├── [5.3] Variables derivadas (ratios e interacciones de dominio)
│         ├── ratio_V8_V9, ratio_V11_V12, ratio_V13_V16
│         ├── diff_V10_V16
│         └── prod_V13_V16
│
├── [5.4] Escalado RobustScaler
│         → Sobre todas las variables continuas/derivadas
│         → Excluye binarias (V5, V6) y discretas pequeñas (V3_enc, V20)
│
├── [5.5] Features de interacción polinomial (grado 2, interaction_only)
│         → Sobre Top-5 variables por |corr con Y|
│         → Genera 10 términos de interacción adicionales
│
└── [6.]  Feature Selection en cascada
          ├── Variance Threshold (umbral = 0.01)
          ├── Filtro correlación con Y (|r| ≥ 0.02)
          └── Top-25 por importancia RandomForest (200 árboles)

BBDD_ML_TAREA_procesada.csv  (~9.200 filas × 26 columnas)
```

### Decisiones de diseño justificadas

| Decisión | Alternativa considerada | Razón de elección |
|---|---|---|
| KNN imputer para V14/V15 | Mediana simple | Alta correlación entre variables; KNN aprovecha estructura multivariada |
| Winsorización vs. eliminación | Eliminar filas | Preserva n=9.200 y balance 50/50 del target |
| RobustScaler | StandardScaler | Más robusto ante asimetría residual post-winsorización |
| log(1+x) para V7/V18 | Sin transformar | skew > 1.0; mejora linealidad y distancias |
| Mapeo ordinal V3 | One-Hot Encoding | Preserva orden; evita 2 columnas extra por solo 3 valores |
| interaction_only PolyFeatures | Grado completo | Evita explosión de dimensionalidad; C(5,2)=10 términos |
| Top-25 RF features | Todas las features | Reduce dimensionalidad manteniendo poder predictivo |
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Compilar y guardar notebook
# ══════════════════════════════════════════════════════════════════════════════
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"}
}

OUTPUT = "/home/user/GIT_UCMaster/analisis_feature_engineering.ipynb"
import nbformat as nbf2
with open(OUTPUT, "w", encoding="utf-8") as f:
    nbf2.write(nb, f)

print(f"Notebook generado: {OUTPUT}")
