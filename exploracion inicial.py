from pathlib import Path
import pandas as pd

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
