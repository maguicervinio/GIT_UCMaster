"""
Script para generar el notebook de análisis y feature engineering.
Ejecutar con: python3 generar_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 0: Título principal
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# Análisis, Depuración y Feature Engineering de Base de Datos

## Descripción General

Este notebook documenta de forma clara y justificada el proceso completo de:

1. **Exploración inicial** de la base de datos
2. **Depuración y limpieza** de los datos (valores faltantes, outliers, duplicados, inconsistencias)
3. **Transformaciones** de variables existentes
4. **Feature Engineering**: creación de nuevas variables con valor predictivo
5. **Selección de características** (Feature Selection)

### Dataset utilizado
Se trabaja con un **dataset sintético de solicitudes de crédito bancario** (1.000 registros), diseñado para reflejar
problemas comunes en datos reales: valores nulos, outliers, variables sesgadas, categorías mal codificadas y
mezcla de tipos de datos.

| Variable | Tipo | Descripción |
|---|---|---|
| `edad` | Numérica | Edad del solicitante |
| `ingreso_anual` | Numérica | Ingreso anual en USD |
| `monto_prestamo` | Numérica | Monto solicitado en USD |
| `puntaje_crediticio` | Numérica | Score crediticio (300–850) |
| `anios_empleo` | Numérica | Años en el empleo actual |
| `num_dependientes` | Numérica | Número de dependientes |
| `nivel_educacion` | Categórica | Nivel de educación (ordinal) |
| `estado_civil` | Categórica | Estado civil |
| `tipo_propiedad` | Categórica | Tipo de propiedad que posee |
| `proposito_prestamo` | Categórica | Propósito del préstamo |
| `fecha_solicitud` | Fecha | Fecha de la solicitud |
| `prestamo_aprobado` | Binaria | Variable objetivo (0/1) |

---
**Fecha:** Febrero 2026
**Objetivo:** Preparar los datos para un modelo de clasificación binaria (aprobación de crédito)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 1: Imports y configuración
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Configuración del Entorno"))

cells.append(nbf.v4.new_code_cell("""\
# ─── Librerías estándar ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ─── Visualización ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ─── Machine Learning / Feature Engineering ──────────────────────────────────
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder, PolynomialFeatures
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.stats import skew, kurtosis

# ─── Reproducibilidad ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── Estilo de gráficas ───────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12, 5)

print("✓ Entorno configurado correctamente")
print(f"  pandas  {pd.__version__}")
print(f"  numpy   {np.__version__}")
print(f"  sklearn disponible")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 2: Generación del dataset sintético
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 2. Generación del Dataset Sintético

> **Justificación:** Se genera un dataset que imita defectos comunes de datos reales:
> valores nulos (~8–12 % por variable), outliers graves (3 σ), distribuciones sesgadas,
> y categorías con errores de codificación. Esto permite demostrar cada técnica de
> limpieza y feature engineering de forma motivada.
"""))

cells.append(nbf.v4.new_code_cell("""\
N = 1_000  # tamaño del dataset

# ── Variables numéricas ───────────────────────────────────────────────────────
edad              = np.random.normal(40, 12, N).clip(18, 80).astype(int)
ingreso_anual     = np.random.lognormal(10.8, 0.6, N)   # distribución log-normal (sesgada)
monto_prestamo    = np.random.lognormal(10.2, 0.8, N)
puntaje_crediticio= np.random.normal(650, 80, N).clip(300, 850).astype(int)
anios_empleo      = np.random.exponential(5, N).clip(0, 40)
num_dependientes  = np.random.poisson(1.5, N).clip(0, 6)

# ── Variables categóricas ─────────────────────────────────────────────────────
nivel_educacion   = np.random.choice(
    ['Primaria', 'Secundaria', 'Tecnico', 'Universitario', 'Posgrado'],
    N, p=[0.05, 0.25, 0.20, 0.35, 0.15])
estado_civil      = np.random.choice(
    ['Soltero', 'Casado', 'Divorciado', 'Viudo'],
    N, p=[0.30, 0.50, 0.15, 0.05])
tipo_propiedad    = np.random.choice(
    ['Ninguna', 'Alquilada', 'Propia', 'Hipotecada'],
    N, p=[0.20, 0.30, 0.35, 0.15])
proposito_prestamo= np.random.choice(
    ['Vivienda', 'Educacion', 'Vehiculo', 'Negocios', 'Personal'],
    N, p=[0.30, 0.15, 0.20, 0.15, 0.20])

# ── Variable de fecha ─────────────────────────────────────────────────────────
fecha_base = datetime(2020, 1, 1)
fechas     = [fecha_base + timedelta(days=int(d))
              for d in np.random.uniform(0, 365*4, N)]
fecha_solicitud = pd.to_datetime(fechas)

# ── Variable objetivo (lógica de aprobación) ──────────────────────────────────
prob_aprobacion = (
    0.3
    + 0.25 * (puntaje_crediticio > 650).astype(float)
    + 0.20 * (ingreso_anual > 50_000).astype(float)
    - 0.15 * (monto_prestamo / ingreso_anual > 3).astype(float)
    - 0.10 * (num_dependientes > 3).astype(float)
).clip(0.05, 0.95)
prestamo_aprobado = np.random.binomial(1, prob_aprobacion, N)

# ── Ensamblado del DataFrame ──────────────────────────────────────────────────
df_raw = pd.DataFrame({
    'edad'              : edad,
    'ingreso_anual'     : ingreso_anual.round(2),
    'monto_prestamo'    : monto_prestamo.round(2),
    'puntaje_crediticio': puntaje_crediticio,
    'anios_empleo'      : anios_empleo.round(1),
    'num_dependientes'  : num_dependientes,
    'nivel_educacion'   : nivel_educacion,
    'estado_civil'      : estado_civil,
    'tipo_propiedad'    : tipo_propiedad,
    'proposito_prestamo': proposito_prestamo,
    'fecha_solicitud'   : fecha_solicitud,
    'prestamo_aprobado' : prestamo_aprobado
})

# ── Inyección de defectos realistas ──────────────────────────────────────────
rng = np.random.default_rng(SEED)

# Valores nulos (8–12 % por variable)
for col, pct in [('ingreso_anual', 0.10), ('puntaje_crediticio', 0.08),
                 ('anios_empleo', 0.12), ('tipo_propiedad', 0.09),
                 ('edad', 0.06)]:
    mask = rng.choice([True, False], N, p=[pct, 1-pct])
    df_raw.loc[mask, col] = np.nan

# Outliers graves
idx_out = rng.choice(N, 15, replace=False)
df_raw.loc[idx_out[:5],  'ingreso_anual']  = rng.uniform(800_000, 2_000_000, 5)
df_raw.loc[idx_out[5:10],'monto_prestamo'] = rng.uniform(1_500_000, 3_000_000, 5)
df_raw.loc[idx_out[10:], 'puntaje_crediticio'] = rng.choice([150, 170, 180], 5)

# Duplicados
df_raw = pd.concat([df_raw, df_raw.iloc[:12]], ignore_index=True)

# Ruido en categorías
erroneos = {'Soltero': 'soltero', 'Casado': 'CASADO', 'Universitario': 'universitario'}
for i in rng.choice(len(df_raw), 30, replace=False):
    col = rng.choice(['estado_civil', 'nivel_educacion'])
    if df_raw.at[i, col] in erroneos:
        df_raw.at[i, col] = erroneos[df_raw.at[i, col]]

print(f"Dataset creado: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")
print(df_raw.dtypes)
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 3: Exploración Inicial
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 3. Exploración Inicial (EDA)

Antes de cualquier transformación se estudia la estructura del dataset para tomar
decisiones informadas sobre el tratamiento de cada variable.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Vista general ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SHAPE:", df_raw.shape)
print("=" * 60)
display(df_raw.head())

print("\\n─── Tipos de datos ───────────────────────────────────────")
print(df_raw.dtypes)

print("\\n─── Estadísticas descriptivas (numéricas) ────────────────")
display(df_raw.describe().round(2))

print("\\n─── Estadísticas descriptivas (categóricas) ─────────────")
display(df_raw.describe(include='object'))
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Mapa de calor de valores nulos ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Porcentaje de nulos
nulos = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=False)
nulos = nulos[nulos > 0]
axes[0].bar(nulos.index, nulos.values, color='salmon')
axes[0].set_title('% Valores Nulos por Variable')
axes[0].set_ylabel('Porcentaje (%)')
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(nulos.values):
    axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)

# Balance de la variable objetivo
conteo = df_raw['prestamo_aprobado'].value_counts()
axes[1].pie(conteo.values, labels=['Aprobado (1)', 'Rechazado (0)'],
            autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
axes[1].set_title('Balance de la Variable Objetivo')

plt.suptitle('Exploración Inicial', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_inicial.png', bbox_inches='tight')
plt.show()

print("\\nResumen de nulos:")
print(nulos.to_string())
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Distribuciones de variables numéricas ─────────────────────────────────────
num_cols = ['edad', 'ingreso_anual', 'monto_prestamo',
            'puntaje_crediticio', 'anios_empleo', 'num_dependientes']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    datos = df_raw[col].dropna()
    axes[i].hist(datos, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].axvline(datos.mean(), color='red', linestyle='--', label=f'Media: {datos.mean():.0f}')
    axes[i].axvline(datos.median(), color='orange', linestyle='--', label=f'Mediana: {datos.median():.0f}')
    sk = skew(datos)
    axes[i].set_title(f'{col}\\nSkewness: {sk:.2f}')
    axes[i].legend(fontsize=8)

plt.suptitle('Distribución de Variables Numéricas', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distribuciones.png', bbox_inches='tight')
plt.show()
print("Skewness por variable:")
for col in num_cols:
    sk = skew(df_raw[col].dropna())
    print(f"  {col:<25} {sk:+.3f}{'  ← SESGADA' if abs(sk) > 1 else ''}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 4: Depuración – Duplicados
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Depuración de Datos

### 4.1 Eliminación de Duplicados

**Problema:** Filas completamente duplicadas sesgan las distribuciones y pueden producir
*data leakage* si el duplicado cae en test.

**Decisión:** Se eliminan duplicados exactos conservando la primera aparición.
"""))

cells.append(nbf.v4.new_code_cell("""\
df = df_raw.copy()

n_antes = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
n_despues = len(df)

print(f"Filas antes : {n_antes}")
print(f"Filas después: {n_despues}")
print(f"Duplicados eliminados: {n_antes - n_despues}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 5: Depuración – Ruido en categorías
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 4.2 Normalización de Variables Categóricas

**Problema:** Errores de capitalización producen categorías duplicadas lógicamente
idénticas (`Soltero` ≠ `soltero` ≠ `SOLTERO`), lo que infla la cardinalidad y degrada
cualquier codificación posterior.

**Decisión:** Convertir a `Title Case` (primera letra mayúscula) para unificar.
"""))

cells.append(nbf.v4.new_code_cell("""\
cat_cols = ['nivel_educacion', 'estado_civil', 'tipo_propiedad', 'proposito_prestamo']

print("Valores únicos ANTES de normalizar:")
for col in cat_cols:
    print(f"  {col}: {sorted(df[col].dropna().unique())}")

for col in cat_cols:
    df[col] = df[col].str.strip().str.title()

print("\\nValores únicos DESPUÉS de normalizar:")
for col in cat_cols:
    print(f"  {col}: {sorted(df[col].dropna().unique())}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 6: Depuración – Outliers
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 4.3 Detección y Tratamiento de Outliers

**Técnica utilizada: Método IQR (Rango Intercuartílico)**

$$\\text{Límite inferior} = Q_1 - 1.5 \\times IQR \\quad ; \\quad \\text{Límite superior} = Q_3 + 1.5 \\times IQR$$

**Justificación:** El método IQR es robusto ante distribuciones asimétricas (como
`ingreso_anual` y `monto_prestamo`), a diferencia del método Z-score que asume normalidad.

**Tratamiento:** Winsorización (recorte al límite del IQR) en lugar de eliminación,
para conservar el tamaño muestral y no distorsionar la distribución del target.
"""))

cells.append(nbf.v4.new_code_cell("""\
def detectar_outliers_iqr(serie):
    \"\"\"Devuelve máscara booleana de outliers según el método IQR.\"\"\"
    Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
    IQR = Q3 - Q1
    li, ls = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return (serie < li) | (serie > ls), li, ls

def winsorizacion(df, col):
    \"\"\"Winsoriza la columna al rango [límite_inf, límite_sup] IQR.\"\"\"
    _, li, ls = detectar_outliers_iqr(df[col].dropna())
    return df[col].clip(lower=li, upper=ls)

cols_outlier = ['ingreso_anual', 'monto_prestamo', 'puntaje_crediticio']

print(f"{'Variable':<25} {'Outliers detectados':>20} {'% del total':>12}")
print("-" * 60)
for col in cols_outlier:
    mask, li, ls = detectar_outliers_iqr(df[col].dropna())
    n_out = mask.sum()
    print(f"{col:<25} {n_out:>20} {n_out/len(df)*100:>11.1f}%")
    df[col] = winsorizacion(df, col)

print("\\n✓ Winsorización aplicada")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Visualización: Boxplots antes y después
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(cols_outlier):
    datos_raw = df_raw[col].dropna()
    datos_clean = df[col].dropna()
    axes[i].boxplot([datos_raw, datos_clean],
                    labels=['Original', 'Winsorizado'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col)
    axes[i].set_ylabel('Valor')

plt.suptitle('Efecto de la Winsorización sobre Outliers', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outliers_boxplot.png', bbox_inches='tight')
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 7: Imputación de valores faltantes
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 4.4 Imputación de Valores Faltantes

| Variable | Estrategia | Justificación |
|---|---|---|
| `edad` | Mediana | Distribución asimétrica; la mediana es robusta ante outliers |
| `ingreso_anual` | KNN (k=5) | Correlación con otras variables; KNN aprovecha información multivariada |
| `puntaje_crediticio` | Mediana | Score ordenado; mediana preserva la distribución sin suponer linealidad |
| `anios_empleo` | Mediana | Distribución exponencial; mediana más representativa que la media |
| `tipo_propiedad` | Moda | Variable categórica; se rellena con la categoría más frecuente |

> **Principio:** Se prefiere la **imputación** sobre la eliminación de filas para no
> perder el ~10 % de registros y mantener representatividad de todas las subpoblaciones.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Imputación numérica: mediana simple ───────────────────────────────────────
mediana_imputer = SimpleImputer(strategy='median')
cols_mediana = ['edad', 'puntaje_crediticio', 'anios_empleo']
df[cols_mediana] = mediana_imputer.fit_transform(df[cols_mediana])

# ── Imputación por KNN para ingreso_anual ─────────────────────────────────────
# Se incluyen features numéricas completas como contexto
knn_features = ['puntaje_crediticio', 'anios_empleo', 'edad', 'ingreso_anual']
knn_imputer = KNNImputer(n_neighbors=5)
df[knn_features] = knn_imputer.fit_transform(df[knn_features])

# ── Imputación categórica: moda ───────────────────────────────────────────────
moda_imputer = SimpleImputer(strategy='most_frequent')
df[['tipo_propiedad']] = moda_imputer.fit_transform(df[['tipo_propiedad']])

# ── Verificación ──────────────────────────────────────────────────────────────
nulos_post = df.isnull().sum()
print("Valores nulos tras imputación:")
print(nulos_post[nulos_post > 0] if nulos_post.sum() > 0 else "  → Ninguno. Dataset limpio ✓")
print(f"\\nShape final del dataset limpio: {df.shape}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 8: Feature Engineering – Variables de fecha
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 5. Feature Engineering

### 5.1 Extracción de Características desde Fechas

**Justificación:** Las fechas en bruto no son interpretables por los modelos de ML.
Se extraen componentes temporales que pueden capturar estacionalidad (trimestre, mes)
y la antigüedad relativa de la solicitud.
"""))

cells.append(nbf.v4.new_code_cell("""\
df['solicitud_anio']     = df['fecha_solicitud'].dt.year
df['solicitud_mes']      = df['fecha_solicitud'].dt.month
df['solicitud_trimestre']= df['fecha_solicitud'].dt.quarter
df['solicitud_dia_sem']  = df['fecha_solicitud'].dt.dayofweek   # 0=lunes, 6=domingo
df['dias_desde_inicio']  = (df['fecha_solicitud'] - df['fecha_solicitud'].min()).dt.days

# Se elimina la columna original (no es consumible por modelos directamente)
df.drop(columns=['fecha_solicitud'], inplace=True)

print("Variables de fecha creadas:")
print(df[['solicitud_anio','solicitud_mes','solicitud_trimestre',
          'solicitud_dia_sem','dias_desde_inicio']].describe().round(1))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 9: Features numéricas derivadas
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 5.2 Creación de Variables Numéricas Derivadas

Estas variables **capturan relaciones de dominio** que un modelo lineal no puede
aprender automáticamente sin ellas:

| Nueva variable | Fórmula | Relevancia |
|---|---|---|
| `ratio_deuda_ingreso` | monto / ingreso | Indicador clave de capacidad de pago |
| `ingreso_por_dependiente` | ingreso / (dependientes+1) | Ingreso disponible real |
| `cuota_mensual_estimada` | monto / (años_empleo·12 + 1) | Proxy de carga mensual |
| `carga_familiar` | dependientes / (edad - 18 + 1) | Responsabilidad relativa a la vida laboral |
"""))

cells.append(nbf.v4.new_code_cell("""\
df['ratio_deuda_ingreso']    = df['monto_prestamo'] / (df['ingreso_anual'] + 1)
df['ingreso_por_dependiente']= df['ingreso_anual'] / (df['num_dependientes'] + 1)
df['cuota_mensual_estimada'] = df['monto_prestamo'] / (df['anios_empleo'] * 12 + 1)
df['carga_familiar']         = df['num_dependientes'] / (df['edad'] - 18 + 1)

nuevas = ['ratio_deuda_ingreso','ingreso_por_dependiente',
          'cuota_mensual_estimada','carga_familiar']
print("Estadísticas de variables derivadas:")
display(df[nuevas].describe().round(3))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 10: Binning
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 5.3 Discretización (Binning)

**Justificación:** Algunas variables numéricas tienen relaciones **no lineales** con el
target (por ejemplo, el riesgo crediticio no baja de forma proporcional al aumentar el
puntaje). El binning captura estos umbrales categóricos conocidos en la industria.
"""))

cells.append(nbf.v4.new_code_cell("""\
# ── Grupos de edad ─────────────────────────────────────────────────────────────
df['grupo_edad'] = pd.cut(
    df['edad'],
    bins=[17, 25, 35, 50, 65, 81],
    labels=['18-25', '26-35', '36-50', '51-65', '66+'])

# ── Categoría de puntaje crediticio (estándar industria FICO) ─────────────────
df['categoria_score'] = pd.cut(
    df['puntaje_crediticio'],
    bins=[299, 580, 670, 740, 800, 851],
    labels=['Muy Malo', 'Regular', 'Bueno', 'Muy Bueno', 'Excelente'])

# ── Nivel de ingresos ──────────────────────────────────────────────────────────
df['nivel_ingresos'] = pd.qcut(
    df['ingreso_anual'], q=4,
    labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])

print("Distribución grupo_edad:")
print(df['grupo_edad'].value_counts().sort_index())
print("\\nDistribución categoria_score (FICO):")
print(df['categoria_score'].value_counts().sort_index())
print("\\nDistribución nivel_ingresos (cuartiles):")
print(df['nivel_ingresos'].value_counts().sort_index())
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 11: Transformaciones de escala
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 5.4 Transformación Logarítmica para Variables Sesgadas

**Problema:** `ingreso_anual`, `monto_prestamo` e `ingreso_por_dependiente` presentan
asimetría positiva (skewness > 1). Esto perjudica modelos lineales y KNN.

**Solución:** Transformación `log(1 + x)` para comprimir la cola derecha y aproximar
la distribución a la normal, sin eliminar registros con valor 0.

**Criterio de aplicación:** Se aplica cuando `|skewness| > 1.0`.
"""))

cells.append(nbf.v4.new_code_cell("""\
cols_log = ['ingreso_anual', 'monto_prestamo', 'ingreso_por_dependiente',
            'cuota_mensual_estimada', 'ratio_deuda_ingreso']

fig, axes = plt.subplots(2, len(cols_log), figsize=(18, 7))

for i, col in enumerate(cols_log):
    sk_antes = skew(df[col].dropna())
    axes[0, i].hist(df[col], bins=40, color='tomato', alpha=0.7, edgecolor='white')
    axes[0, i].set_title(f'Original\\nSkew: {sk_antes:.2f}', fontsize=9)
    axes[0, i].set_xlabel(col, fontsize=8)

    df[f'log_{col}'] = np.log1p(df[col])

    sk_despues = skew(df[f'log_{col}'].dropna())
    axes[1, i].hist(df[f'log_{col}'], bins=40, color='mediumseagreen', alpha=0.7, edgecolor='white')
    axes[1, i].set_title(f'log(1+x)\\nSkew: {sk_despues:.2f}', fontsize=9)
    axes[1, i].set_xlabel(f'log_{col}', fontsize=8)

axes[0, 0].set_ylabel('Original', fontsize=10)
axes[1, 0].set_ylabel('Transformado', fontsize=10)
plt.suptitle('Transformación Logarítmica: Corrección del Sesgo', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('log_transformacion.png', bbox_inches='tight')
plt.show()

print("Skewness antes → después:")
for col in cols_log:
    s_b = skew(df[col].dropna())
    s_d = skew(df[f'log_{col}'].dropna())
    print(f"  {col:<35} {s_b:+.3f}  →  {s_d:+.3f}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 12: Escalado
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 5.5 Escalado de Variables Numéricas

| Escalador | Fórmula | Cuándo usarlo |
|---|---|---|
| **StandardScaler** | `(x − μ) / σ` | Variables aproximadamente normales; requerido por SVM, regresión logística |
| **MinMaxScaler** | `(x − min) / (max − min)` | Redes neuronales; cuando los límites son conocidos |
| **RobustScaler** | `(x − Q2) / IQR` | Distribuciones con outliers residuales |

**Decisión:** Se utiliza **RobustScaler** como escalador principal dado que el dataset
presenta distribuciones asimétricas incluso después de la transformación logarítmica.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Selección de columnas numéricas para escalar
cols_num_escalar = [
    'edad', 'puntaje_crediticio', 'anios_empleo', 'num_dependientes',
    'dias_desde_inicio', 'solicitud_mes', 'solicitud_dia_sem',
    'log_ingreso_anual', 'log_monto_prestamo',
    'log_ingreso_por_dependiente', 'log_cuota_mensual_estimada',
    'log_ratio_deuda_ingreso', 'carga_familiar'
]

# Asegurar que todas existen
cols_num_escalar = [c for c in cols_num_escalar if c in df.columns]

scaler = RobustScaler()
df_escalado = df.copy()
df_escalado[cols_num_escalar] = scaler.fit_transform(df[cols_num_escalar])

print("Estadísticas DESPUÉS del RobustScaler:")
display(df_escalado[cols_num_escalar[:6]].describe().round(3))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 13: Codificación de categóricas
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 5.6 Codificación de Variables Categóricas

| Variable | Técnica | Justificación |
|---|---|---|
| `nivel_educacion` | **Ordinal Encoding** | Existe orden natural: Primaria < Secundaria < Técnico < Universitario < Posgrado |
| `estado_civil` | **One-Hot Encoding** | Sin orden inherente; evita asumir relación ordinal falsa |
| `tipo_propiedad` | **One-Hot Encoding** | Sin orden inherente; 4 categorías — cardinalidad manejable |
| `proposito_prestamo` | **One-Hot Encoding** | Sin orden; 5 categorías — cardinalidad manejable |
| `grupo_edad` | **Ordinal Encoding** | Orden natural definido por los rangos etarios |
| `categoria_score` | **Ordinal Encoding** | Orden natural definido por el sistema FICO |
| `nivel_ingresos` | **Ordinal Encoding** | Orden natural: Bajo < Medio-Bajo < Medio-Alto < Alto |
"""))

cells.append(nbf.v4.new_code_cell("""\
df_enc = df_escalado.copy()

# ── Ordinal Encoding ──────────────────────────────────────────────────────────
ordinal_mappings = {
    'nivel_educacion' : [['Primaria', 'Secundaria', 'Tecnico', 'Universitario', 'Posgrado']],
    'grupo_edad'      : [['18-25', '26-35', '36-50', '51-65', '66+']],
    'categoria_score' : [['Muy Malo', 'Regular', 'Bueno', 'Muy Bueno', 'Excelente']],
    'nivel_ingresos'  : [['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto']]
}

for col, cats in ordinal_mappings.items():
    if col in df_enc.columns:
        oe = OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1)
        df_enc[[col]] = oe.fit_transform(df_enc[[col]].astype(str))

# ── One-Hot Encoding ──────────────────────────────────────────────────────────
ohe_cols = ['estado_civil', 'tipo_propiedad', 'proposito_prestamo']
ohe_cols_existentes = [c for c in ohe_cols if c in df_enc.columns]
df_enc = pd.get_dummies(df_enc, columns=ohe_cols_existentes, drop_first=True, dtype=int)

print(f"Shape tras codificación: {df_enc.shape}")
print("\\nColumnas one-hot generadas:")
for c in ohe_cols_existentes:
    cols_ohe = [col for col in df_enc.columns if col.startswith(c)]
    print(f"  {c}: {cols_ohe}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 14: Features de interacción
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
### 5.7 Features de Interacción (Polynomial Features)

**Justificación:** El riesgo crediticio depende de **combinaciones** de variables:
un ingreso alto con una deuda alta es diferente a un ingreso alto con deuda baja.
Las features polinomiales de grado 2 capturan estas interacciones sin necesidad de
especificarlas manualmente.

**Aplicación selectiva:** Se aplica solo sobre las variables más relevantes (puntaje,
ingreso, deuda) para evitar explosión de dimensionalidad.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Variables seleccionadas para interacciones
poly_features = ['puntaje_crediticio', 'log_ingreso_anual', 'log_ratio_deuda_ingreso']
poly_features_ok = [c for c in poly_features if c in df_enc.columns]

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_arr = poly.fit_transform(df_enc[poly_features_ok])
poly_names = poly.get_feature_names_out(poly_features_ok)

df_poly = pd.DataFrame(poly_arr, columns=poly_names, index=df_enc.index)
# Solo agregar las columnas de interacción (no las originales repetidas)
interact_cols = [c for c in poly_names if ' ' in c]  # contienen espacio → interacción
df_enc[interact_cols] = df_poly[interact_cols]

print(f"Features de interacción añadidas: {len(interact_cols)}")
print(interact_cols)
print(f"\\nShape total tras interacciones: {df_enc.shape}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 15: Selección de features
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 6. Selección de Características (Feature Selection)

Se aplican tres métodos complementarios:

1. **Variance Threshold:** Elimina variables con varianza casi nula (aportan poca información)
2. **Correlación con el target:** Filtra variables con correlación de Pearson ≥ 0.05 en valor absoluto
3. **Importancia con Random Forest:** Selecciona las top-K variables más discriminativas
"""))

cells.append(nbf.v4.new_code_cell("""\
TARGET = 'prestamo_aprobado'

# ── Preparar X, y ─────────────────────────────────────────────────────────────
y = df_enc[TARGET]
X = df_enc.drop(columns=[TARGET])

# Eliminar columnas originales (sin transformar) si coexisten con log_
cols_originales = ['ingreso_anual', 'monto_prestamo', 'ingreso_por_dependiente',
                   'cuota_mensual_estimada', 'ratio_deuda_ingreso']
X.drop(columns=[c for c in cols_originales if c in X.columns], inplace=True)

# Tipos corruptos → float
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"Features antes de selección: {X.shape[1]}")

# ── 1. Variance Threshold ─────────────────────────────────────────────────────
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)
cols_vt = X.columns[vt.get_support()]
print(f"Tras Variance Threshold (umbral=0.01): {len(cols_vt)} features")

# ── 2. Correlación con el target ──────────────────────────────────────────────
X_vt_df = pd.DataFrame(X_vt, columns=cols_vt)
corr_target = X_vt_df.corrwith(y).abs().sort_values(ascending=False)
cols_corr = corr_target[corr_target >= 0.03].index.tolist()
print(f"Tras filtro de correlación (≥0.03): {len(cols_corr)} features")

# ── 3. Importancia con Random Forest ─────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_vt_df[cols_corr], y)
importancias = pd.Series(rf.feature_importances_, index=cols_corr).sort_values(ascending=False)

# Top-20 features
TOP_K = 20
cols_finales = importancias.head(TOP_K).index.tolist()
print(f"Top {TOP_K} features seleccionadas por Random Forest:")
for i, (feat, imp) in enumerate(importancias.head(TOP_K).items(), 1):
    print(f"  {i:2}. {feat:<45} {imp:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Visualización de importancias
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Barplot importancias top 20
importancias.head(TOP_K).sort_values().plot(
    kind='barh', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title(f'Top {TOP_K} Variables por Importancia (Random Forest)')
axes[0].set_xlabel('Importancia')

# Heatmap de correlación entre top features y target
top_data = X_vt_df[cols_finales].copy()
top_data['prestamo_aprobado'] = y.values
corr_matrix = top_data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
            ax=axes[1], square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.7}, fmt='.2f', annot=False)
axes[1].set_title('Matriz de Correlación (Top Features + Target)')

plt.suptitle('Feature Selection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_selection.png', bbox_inches='tight')
plt.show()
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 16: Dataset final
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 7. Dataset Final

Se construye el dataset listo para modelado con las features seleccionadas y el target.
"""))

cells.append(nbf.v4.new_code_cell("""\
df_final = X_vt_df[cols_finales].copy()
df_final[TARGET] = y.values

# Guardar a CSV
df_final.to_csv('dataset_final.csv', index=False)

print(f"Dataset final: {df_final.shape[0]} filas × {df_final.shape[1]} columnas")
print(f"Balance del target: {df_final[TARGET].value_counts().to_dict()}")
print("\\nGuardado como 'dataset_final.csv' ✓")
display(df_final.head())
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 17: Validación rápida del pipeline
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 8. Validación del Pipeline de Preprocesamiento

Se ejecuta una validación cruzada (5-fold) con Random Forest para confirmar que
las transformaciones aplicadas producen un dataset útil para el modelado.
"""))

cells.append(nbf.v4.new_code_cell("""\
X_final = df_final.drop(columns=[TARGET])
y_final = df_final[TARGET]

rf_val = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
scores = cross_val_score(rf_val, X_final, y_final, cv=5, scoring='roc_auc')

print("=" * 50)
print("Validación cruzada (5-fold) – ROC-AUC")
print("=" * 50)
for i, s in enumerate(scores, 1):
    print(f"  Fold {i}: {s:.4f}")
print(f"  Media : {scores.mean():.4f} ± {scores.std():.4f}")
print("=" * 50)

if scores.mean() > 0.65:
    print("✓ El pipeline produce features con poder predictivo aceptable.")
else:
    print("⚠ ROC-AUC bajo. Revisar transformaciones o añadir más features.")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# CELDA 18: Resumen final
# ─────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
## 9. Resumen de Transformaciones Aplicadas

### Pipeline completo

```
Dataset raw (1012 filas × 12 columnas)
│
├── 4.1  Eliminación de duplicados               → −12 filas
├── 4.2  Normalización de categorías (Title Case) → 0 pérdidas, sin ambigüedades
├── 4.3  Detección y winsorización de outliers    → sin pérdida de filas
├── 4.4  Imputación de valores faltantes
│        ├── Mediana → edad, puntaje_crediticio, anios_empleo
│        ├── KNN (k=5) → ingreso_anual
│        └── Moda → tipo_propiedad
│
├── 5.1  Extracción de features temporales        → +5 columnas
├── 5.2  Variables numéricas derivadas            → +4 columnas
├── 5.3  Discretización (binning)                 → +3 columnas
├── 5.4  Transformación log(1+x) para sesgo       → +5 columnas log_
├── 5.5  RobustScaler sobre variables numéricas   → in-place
├── 5.6  Ordinal Encoding (4 variables)           → in-place
│        One-Hot Encoding (3 variables)            → +7 columnas dummy
├── 5.7  Polynomial features de interacción       → +3 columnas
│
└── 6.   Selección de características
         ├── Variance Threshold (0.01)
         ├── Filtro de correlación con target (≥0.03)
         └── Top-20 por importancia Random Forest

Dataset final: ~1000 filas × 21 columnas (20 features + 1 target)
```

### Decisiones de diseño clave

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| **KNN imputer** para ingreso | Media | Aprovecha correlaciones multivariadas |
| **Winsorización** vs eliminación | Drop de filas | Preserva tamaño muestral |
| **RobustScaler** | StandardScaler | Más robusto ante skewness residual |
| **Ordinal Encoding** para nivel educativo | OHE | Orden real: más info, menos columnas |
| **log(1+x)** | Raíz cuadrada / Box-Cox | Maneja ceros; interpretable; simétrica |
| **Interaction_only PolyFeatures** | Grado completo | Evita explosión de dimensionalidad |
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Compilar y guardar notebook
# ─────────────────────────────────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

OUTPUT = "/home/user/GIT_UCMaster/analisis_feature_engineering.ipynb"
with open(OUTPUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook generado: {OUTPUT}")
