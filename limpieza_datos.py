from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(__file__).parent / 'Insighted_2_2025.xlsx'
PLOT_DIR = DATA_FILE.parent / 'grafica'
TARGET = 'probabilidad_abandono'
MAX_NULL_RATIO = 0.50
KNN_NEIGHBORS = 5


df = pd.read_excel(DATA_FILE)
df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()


def find_column(text):
    """Full name of the column that contains `text`.

    Several questions carry the whole statement as the column name.
    """
    return next(c for c in df.columns if text in c)


df = df[df[TARGET].notna()].reset_index(drop=True).copy()
print(f'Filas de trabajo: {len(df)}')


drop_reasons = {
    c: f'{df[c].isna().mean()*100:.0f}% de nulos, mas del {int(MAX_NULL_RATIO*100)}%'
    for c in df.columns if df[c].isna().mean() > MAX_NULL_RATIO
}

drop_reasons.update({
    'Fecha': 'columna de fecha',
    'verbatim_nps': 'pregunta abierta',
    'verbatim_ces': 'pregunta abierta',
    'verbatim_retroalimentacion': 'pregunta abierta',
    'id': 'identificador unico, no predice nada',
    'nombre_universidad': 'ya esta en las columnas dummy de universidad',
    'nombre_corto': 'igual que nombre_universidad',
    'id_universidad': 'codigo nominal, ya esta en las dummy',
    'cuatrimestre': 'repetida en numero_cuatrimestre',
    'id_provincia': 'codigo nominal, se conserva provincia',
    'sede_universidad': f'demasiadas categorias ({df["sede_universidad"].nunique()} sedes)',
    'nombre_carrera': f'demasiadas categorias ({df["nombre_carrera"].nunique()}), se conserva area_conocimiento',
})

drop_reasons.update({
    c: 'un solo valor en toda la columna'
    for c in df.columns if c not in drop_reasons and df[c].nunique() <= 1
})

print(f'\nColumnas eliminadas: {len(drop_reasons)}')
for c, reason in drop_reasons.items():
    print(f'  {c[:55]:55s} | {reason}')

df = df.drop(columns=list(drop_reasons))


ORDINAL_COLS = {
    'grado_academico': {'Curso Libre': 1, 'Diplomado': 2, 'Técnico': 3, 'Bachillerato': 4,
                        'Licenciatura': 5, 'Maestría': 6, 'Doctorado': 7},
    'etapa_carrera': {'Etpa inicial': 1, 'Etapa intermedia': 2, 'Etapa avanzada': 3},
    'nivel_satisfaccion': {'Insatisfecho': 1, 'Indiferente': 2, 'Satisfecho': 3,
                           'Muy satisfecho': 4},
    'nivel_ces': {'Muy difícil': 1, 'Moderado': 2, 'Muy fácil': 3},
    'tipo_estudiante': {'Detractores': 1, 'Pasivos': 2, 'Promotores': 3},
    find_column('valor o beneficio'): {
        'Me da menor valor/beneficio del que espero por su precio': 1,
        'Me da justo el valor/beneficio que espero por su precio': 2,
        'Me da mayor valor/beneficio del que espero por su precio': 3},
    find_column('precio ideal'): {
        'Menos de ₡60 000': 1, '₡60 000 – ₡100 000': 2, '₡100 001 – ₡150 000': 3,
        '₡150 001 – ₡200 000': 4, '₡200 001 – ₡300 000': 5, '₡300 001 – ₡400 000': 6,
        'Más de ₡400 000': 7},
    find_column('Has considerado'): {
        'No, no he pensado en cambiarme': 1,
        'Sí, pero todavía no he buscado alternativas específicas': 2,
        'Sí, ya he investigado opciones concretas': 3},
}

for column, mapping in ORDINAL_COLS.items():
    unmapped = set(df[column].dropna().unique()) - set(mapping)
    assert not unmapped, f'{column}: categorias sin mapear {unmapped}'
    df[column] = df[column].map(mapping)

print(f'\nOrdinales convertidas a numero: {len(ORDINAL_COLS)}')


NOMINAL_COLS = ['sexo', 'modalidad_estudio', 'area_conocimiento', 'situacion_laboral',
                'provincia', find_column('de *financiamiento')]

before = df.shape[1]
df = pd.get_dummies(df, columns=NOMINAL_COLS, dtype=float)
print(f'Nominales a one-hot: {len(NOMINAL_COLS)} columnas -> {df.shape[1] - before + len(NOMINAL_COLS)}')

non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
print(f'Columnas de texto restantes: {non_numeric}')


scaler = StandardScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(df), columns=df.columns, index=df.index
)

print(f'\nNulos antes de imputar: {int(scaled_data.isna().sum().sum())}')

data = pd.DataFrame(
    KNNImputer(n_neighbors=KNN_NEIGHBORS).fit_transform(scaled_data),
    columns=scaled_data.columns,
    index=scaled_data.index,
)

print(f'Nulos despues de imputar: {int(data.isna().sum().sum())}')
print(f'\nDataset limpio: {data.shape[0]} filas x {data.shape[1]} columnas '
      f'(estandarizado, sin nulos)')


features = data.drop(columns=[TARGET])
pca = PCA()
coords = pca.fit_transform(features)
var_pct = pca.explained_variance_ratio_ * 100
cum_var = var_pct.cumsum()
print(f'\nPCA: {len(var_pct)} componentes, varianza explicada: {cum_var[-1]:.1f}%')
n_90 = int((cum_var < 90).sum() + 1)
print(f'\nVarianza explicada: PC1 {var_pct[0]:.1f}% + PC2 {var_pct[1]:.1f}% = '
      f'{cum_var[1]:.1f}% de la varianza')
print(f'Componentes para el 90% de la varianza: {n_90} de {len(var_pct)}')




pca_2d = PCA(n_components=2)
coords_2d = pca_2d.fit_transform(features)
print(f'\nPCA 2D para graficar: {coords_2d.shape}')

PLOT_DIR.mkdir(exist_ok=True)

var_pct_2d = pca_2d.explained_variance_ratio_ * 100
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=10, alpha=0.5, color='#2c7fb8')
ax.set_xlabel(f'PC1 ({var_pct_2d[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var_pct_2d[1]:.1f}%)')
ax.set_title('PCA con 2 componentes')
fig.savefig(PLOT_DIR / 'pca_2componentes.png', dpi=110, bbox_inches='tight')

high_risk = (df[TARGET] >= 7).to_numpy()
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(coords_2d[~high_risk, 0], coords_2d[~high_risk, 1],
           s=10, alpha=0.5, color='#2c7fb8')
ax.scatter(coords_2d[high_risk, 0], coords_2d[high_risk, 1],
           s=14, alpha=0.8, color='#c0392b')
ax.set_xlabel(f'PC1 ({var_pct[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var_pct[1]:.1f}%)')
ax.set_title('Proyeccion PCA de los estudiantes')
fig.savefig(PLOT_DIR / 'pca_proyeccion.png', dpi=110, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, len(cum_var) + 1), cum_var, color='#2c7fb8')
ax.axhline(90, ls='--', color='#c0392b', label='90% de varianza')
ax.axvline(n_90, ls='--', color='#c0392b')
ax.set_xlabel('numero de componentes')
ax.set_ylabel('% de varianza acumulada')
ax.set_title('Varianza explicada acumulada (PCA)')
ax.legend()
fig.savefig(PLOT_DIR / 'pca_varianza_acumulada.png', dpi=110, bbox_inches='tight')


raw_data = pd.DataFrame(
    scaler.inverse_transform(data), columns=data.columns, index=data.index
)
raw_features = raw_data.drop(columns=[TARGET])

pca_raw = PCA()
coords_raw = pca_raw.fit_transform(raw_features)
var_raw = pca_raw.explained_variance_ratio_ * 100
cum_var_raw = var_raw.cumsum()
n_90_raw = int((cum_var_raw < 90).sum() + 1)
print(f'\nPCA sin estandarizar: PC1 {var_raw[0]:.1f}% + PC2 {var_raw[1]:.1f}% = '
      f'{cum_var_raw[1]:.1f}% de la varianza')
print(f'Componentes para el 90% de la varianza: {n_90_raw} de {len(var_raw)}')
top_pc1 = abs(pca_raw.components_[0]).argsort()[::-1][:3]
print('Variables con mas peso en PC1: ' +
      ', '.join(c[:40] for c in raw_features.columns[top_pc1]))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(coords_raw[:, 0], coords_raw[:, 1], s=10, alpha=0.5, color='#7b3294')
ax.set_xlabel(f'PC1 ({var_raw[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({var_raw[1]:.1f}%)')
ax.set_title('PCA con 2 componentes (datos sin estandarizar)')
fig.savefig(PLOT_DIR / 'pca_sin_estandarizar.png', dpi=110, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, len(cum_var_raw) + 1), cum_var_raw, color='#7b3294')
ax.axhline(90, ls='--', color='#c0392b', label='90% de varianza')
ax.axvline(n_90_raw, ls='--', color='#c0392b')
ax.set_xlabel('numero de componentes')
ax.set_ylabel('% de varianza acumulada')
ax.set_title('Varianza explicada acumulada (datos sin estandarizar)')
ax.legend()
fig.savefig(PLOT_DIR / 'pca_varianza_acumulada_sin_estandarizar.png',
            dpi=110, bbox_inches='tight')

print(f'Graficas guardadas en: {PLOT_DIR}')
