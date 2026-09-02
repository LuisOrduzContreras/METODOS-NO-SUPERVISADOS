from pathlib import Path
import pandas as pd
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)


KEY_INDICATORS = ['nps', 'ces', 'satisfaccion', 'probabilidad_abandono', 'edad']
CATEGORICAL_COLUMNS = [
    'sexo', 'grado_academico', 'modalidad_estudio', 'area_conocimiento',
    'tipo_estudiante', 'etapa_carrera', 'nivel_satisfaccion', 'nivel_ces',
    'situacion_laboral', 'rango_ingresos', 'provincia',
]


DROPOUT_REASONS = [
    'dificultad_pago', 'sin_beca', 'carga_dificil', 'duda_carrera',
    'trabajo_estudio', 'problemas_familiares', 'sin_acompanamiento',
    'inflexibilidad_horaria', 'sin_empleo_area', 'trabajo_sin_grado',
    'bajas_expectativas', 'materias_reprobadas',
]


PERCEPTION_COLUMNS = [
    'calidad_academica', 'prestigio_laboral', 'proyeccion_internacional',
    'infraestructura', 'ubicación', 'plataforma_administrativa',
    'plataforma_virtual', 'satisfaccion_modalidad', 'satisfaccion_financiamiento',
]


AGE_BINS = [16, 20, 25, 30, 40, 60]
AGE_LABELS = ['17-20', '21-25', '26-30', '31-40', '41-59']

file_path = Path(__file__).parent / 'Insighted_2_2025.xlsx'
df = pd.read_excel(file_path)


def separator(title):
    print('\n' + ' ' * 30)
    print(title)
    print(' ' * 30)


separator('1. DIMENSIONES Y TIPOS DE DATOS')

print(f'Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}')

print(df.dtypes.value_counts())


separator('2. DUPLICADOS')

print('Filas totalmente duplicadas:', df.duplicated().sum())

print('IDs duplicados (excluyendo nulos):', df['id'].dropna().duplicated().sum())


separator('3. VALORES NULOS (top 15 columnas)')
null_counts = df.isnull().sum().sort_values(ascending=False)
null_pct = (null_counts / len(df) * 100).round(1)

print(pd.concat([null_counts.head(15), null_pct.head(15)], axis=1, keys=['cantidad', 'porcentaje']))


missing_id = df['id'].isna()

print(f'\nFilas sin "id" (ola de encuesta reducida): {missing_id.sum()}')

separator('4. ESTADÍSTICOS DESCRIPTIVOS - INDICADORES CLAVE')

print(df[KEY_INDICATORS].describe().round(2))

separator('5. DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS')

for col in CATEGORICAL_COLUMNS:
    print(f'\n--- {col} ---')
    counts = df[col].value_counts(dropna=False)
    pct = df[col].value_counts(dropna=False, normalize=True).mul(100).round(1)
    print(pd.concat([counts, pct.astype(str) + '%'], axis=1, keys=['cantidad', 'porcentaje']))


separator('6. TOP 10 UNIVERSIDADES POR NÚMERO DE RESPUESTAS')
print(df['nombre_universidad'].value_counts().head(10))


separator('7. MOTIVOS DE POSIBLE ABANDONO')
print(df[DROPOUT_REASONS].mean().mul(100).round(1).sort_values(ascending=False))


separator('8. CORRELACIÓN ENTRE INDICADORES CLAVE')
print(df[KEY_INDICATORS].corr().round(2))


separator('9. DISTRIBUCIÓN POR RANGO DE EDAD')
age_range = pd.cut(df['edad'], bins=AGE_BINS, labels=AGE_LABELS)
age_pct = age_range.value_counts(normalize=True).sort_index().mul(100).round(1)

print(age_pct.astype(str) + '%')


separator('10. PERCEPCIÓN INSTITUCIONAL - ESTADÍSTICOS DESCRIPTIVOS')
print(df[PERCEPTION_COLUMNS].describe().round(2).T)


separator('11. CANTIDAD DE MOTIVOS DE ABANDONO POR ESTUDIANTE')
reasons_per_student = df[DROPOUT_REASONS].sum(axis=1)
reasons_pct = reasons_per_student.value_counts(normalize=True).sort_index().mul(100).round(1)

print(reasons_pct.astype(str) + '%')
print(f'\nPromedio de motivos marcados: {reasons_per_student.mean().round(2)}')
print(f'Estudiantes sin ningún motivo marcado: {(reasons_per_student == 0).mean() * 100:.1f}%')


separator('12. CORRELACIÓN ENTRE PERCEPCIÓN INSTITUCIONAL E INDICADORES CLAVE')
print(df[PERCEPTION_COLUMNS + KEY_INDICATORS].corr().loc[PERCEPTION_COLUMNS, KEY_INDICATORS].round(2))


# ===========================================================================
#   LIMPIEZA GENERAL DE LOS DATOS  ->  DataFrame `datos`
#
#   Resultado: numérico, sin nulos y sin redundancia. Toda columna eliminada
#   queda registrada en el dict `motivos` con su justificación; ninguna se
#   descarta "porque sí".
# ===========================================================================

datos = df.copy()
datos.columns = datos.columns.str.replace(r'\s+', ' ', regex=True).str.strip()


def col(texto):
    """Nombre completo de la columna que contiene `texto` (varias preguntas
    traen el enunciado entero como nombre de columna)."""
    return next(c for c in datos.columns if texto in c)


def nps_score(escala):
    """NPS = %promotores (9-10) - %detractores (0-6), en puntos."""
    escala = escala.dropna()
    return ((escala >= 9).mean() - (escala <= 6).mean()) * 100


def mapa_por_patron(serie, patrones):
    """categoria -> 1..n segun el orden de `patrones`, buscados como subcadena."""
    cats = list(serie.dropna().unique())
    mapa = {}
    for i, p in enumerate(patrones, 1):
        for c in cats:
            if c not in mapa and p in c:
                mapa[c] = i
    faltan = set(cats) - set(mapa)
    assert not faltan, f'categorias sin mapear -> {faltan}'
    return mapa


separator('13. LIMPIEZA: NPS COMO CARACTERÍSTICA')
# El NPS es una métrica de grupo, no de individuo: se calcula por universidad
# y se le asigna a cada estudiante el de su institución. Se hace ahora porque
# más abajo se elimina la columna nombre_universidad.

datos['nps_universidad'] = datos.groupby('nombre_universidad')['nps'].transform(nps_score)
print(f'NPS global: {nps_score(datos["nps"]):.1f}')
print(f'nps_universidad -> {datos["nps_universidad"].notna().sum()} filas, '
      f'rango [{datos["nps_universidad"].min():.1f}, {datos["nps_universidad"].max():.1f}]')


separator('14. LIMPIEZA: ELIMINACIÓN JUSTIFICADA DE COLUMNAS')

motivos = {}

# Regla: la columna de fecha se elimina.
motivos['Fecha'] = 'columna de fecha; ademas constante (un solo valor)'

# Regla: las preguntas abiertas (texto libre) no se consideran.
for c in datos.columns:
    if c.startswith('verbatim_'):
        motivos[c] = 'pregunta abierta / texto libre'

# Regla: columnas con mas de 50% de nulos se eliminan de una.
for c in datos.columns:
    p = datos[c].isna().mean()
    if p > 0.50:
        motivos.setdefault(c, f'nulos {p*100:.1f}% (> 50%)')

# Identificadores y columnas cuyo contenido ya esta en otra: ordenarlas o
# dejarlas seria ruido, no informacion.
motivos['id'] = f'identificador unico ({datos["id"].nunique()} valores), no predice nada'
motivos['id_universidad'] = 'codigo nominal; la universidad ya esta en las columnas dummy'
motivos['nombre_universidad'] = 'ya codificada en las columnas dummy de universidad'
motivos['nombre_corto'] = 'mismo contenido que nombre_universidad'
motivos['sede_universidad'] = f'alta cardinalidad ({datos["sede_universidad"].nunique()} sedes); se conserva la universidad'
motivos['nombre_carrera'] = f'alta cardinalidad ({datos["nombre_carrera"].nunique()} carreras); se conserva area_conocimiento'
motivos['cuatrimestre'] = 'texto redundante con numero_cuatrimestre (ya numerica)'
motivos['id_provincia'] = 'codigo nominal redundante con provincia'

# Columnas constantes: no aportan varianza a ningun analisis.
for c in datos.columns:
    if c not in motivos and datos[c].nunique(dropna=True) <= 1:
        motivos[c] = 'varianza cero: un solo valor en toda la columna'

print(f'Columnas eliminadas: {len(motivos)} de {datos.shape[1]}')
for c, m in motivos.items():
    print(f'  - {c[:52]:52s} | {m}')

datos = datos.drop(columns=list(motivos))
print(f'Columnas restantes: {datos.shape[1]}')


separator('15. LIMPIEZA: CONVERSIÓN A NUMÉRICO Y ENCODING')
# Las escalas 0-10 ya vienen numéricas y pasan directo. Las ordinales en texto
# se mapean a número conservando el orden de los niveles. Las nominales sin
# orden natural van a one-hot: asignarles un número inventaría una jerarquía.

ORDINALES = {
    'grado_academico': ['Curso Libre', 'Diplomado', 'Técnico', 'Bachillerato',
                        'Licenciatura', 'Maestría', 'Doctorado'],
    'etapa_carrera': ['Etpa inicial', 'Etapa intermedia', 'Etapa avanzada'],
    'nivel_satisfaccion': ['Insatisfecho', 'Indiferente', 'Satisfecho', 'Muy satisfecho'],
    'nivel_ces': ['Muy difícil', 'Moderado', 'Muy fácil'],
    'tipo_estudiante': ['Detractores', 'Pasivos', 'Promotores'],
}
for c, orden in ORDINALES.items():
    mapa = {v: i for i, v in enumerate(orden, 1)}
    faltan = set(datos[c].dropna().unique()) - set(mapa)
    assert not faltan, f'{c}: categorias sin mapear -> {faltan}'
    datos[c] = datos[c].map(mapa)
    print(f'  ordinal   {c:22s} {len(mapa)} niveles')

# Ordinales con etiquetas sucias (simbolo colon, guion largo): el orden se fija
# por subcadena en vez de escribir la etiqueta completa.
ORDINALES_PATRON = {
    col('valor o beneficio'): ['menor', 'justo', 'mayor'],
    col('Has considerado'): ['No, no he pensado', 'todavía no he buscado', 'ya he investigado'],
    col('precio ideal'): ['Menos de', '₡60 000', '₡100 001', '₡150 001',
                          '₡200 001', '₡300 001', 'Más de'],
}
for c, patrones in ORDINALES_PATRON.items():
    datos[c] = datos[c].map(mapa_por_patron(datos[c], patrones))
    print(f'  ordinal   {c[:45]:45s} {len(patrones)} niveles')

NOMINALES = ['sexo', 'modalidad_estudio', 'area_conocimiento', 'situacion_laboral',
             'provincia', col('financiamiento estás utilizando')]
antes = datos.shape[1]
datos = pd.get_dummies(datos, columns=NOMINALES, dtype=float)
print(f'  one-hot   {len(NOMINALES)} nominales -> +{datos.shape[1] - antes + len(NOMINALES)} columnas')

resto = [c for c in datos.columns if not pd.api.types.is_numeric_dtype(datos[c])]
assert not resto, f'quedan columnas no numericas: {resto}'
datos = datos.astype(float)
print(f'  dataframe totalmente numérico: {datos.shape[1]} columnas')


separator('16. LIMPIEZA: IMPUTACIÓN KNN (UNA SOLA LLAMADA)')
# KNNImputer se aplica UNA vez sobre todo el dataframe ya numérico: cada
# faltante se estima con los 5 estudiantes más parecidos en el resto de
# variables. Hacerlo columna por columna perdería esa vecindad.

print(f'Nulos antes: {int(datos.isna().sum().sum())} en '
      f'{int((datos.isna().sum() > 0).sum())} columnas')
datos = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(datos),
                     columns=datos.columns, index=datos.index)
print(f'Nulos despues: {int(datos.isna().sum().sum())}')


separator('17. LIMPIEZA: REDUNDANCIA POR CORRELACIÓN (|r| > 0.90)')
# Solo por encima de 0.90 se considera que dos variables dicen lo mismo. De
# cada par se conserva la primera y se elimina la segunda.

UMBRAL_CORR = 0.90
corr = datos.corr()
nombres = list(corr.columns)
redundantes = set()
pares = []
for i in range(len(nombres)):
    for j in range(i + 1, len(nombres)):
        r = corr.iloc[i, j]
        if abs(r) > UMBRAL_CORR:
            pares.append((nombres[i], nombres[j], r))
            if nombres[i] not in redundantes:
                redundantes.add(nombres[j])

print(f'Pares con |r| > {UMBRAL_CORR}: {len(pares)}')
for a, b, r in sorted(pares, key=lambda t: -abs(t[2])):
    quita = b if b in redundantes else a
    print(f'  r={r:+.3f} | {a[:32]:32s} <-> {b[:32]:32s} | elimina: {quita[:28]}')

datos = datos.drop(columns=list(redundantes))
print(f'\nEliminadas por redundancia: {len(redundantes)}')
print(f'DATASET LIMPIO -> {datos.shape[0]} filas x {datos.shape[1]} columnas | '
      f'nulos: {int(datos.isna().sum().sum())}')


separator('18. LIMPIEZA: EXPORTAR DATASET LIMPIO')
# `datos` se guarda junto al original con el sufijo _cleaned.

salida = file_path.with_name(file_path.stem + '_cleaned' + file_path.suffix)
datos.to_excel(salida, index=False)
print(f'Guardado: {salida.name}  ({datos.shape[0]} filas x {datos.shape[1]} columnas)')
