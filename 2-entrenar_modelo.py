import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
warnings.filterwarnings("ignore")

# ── Crear carpeta para modelos ──────────────────────────────────────────────
CARPETA_MODELOS = "models"
if not os.path.exists(CARPETA_MODELOS):
    os.makedirs(CARPETA_MODELOS)
    print(f"📁 Carpeta '{CARPETA_MODELOS}' creada")

# ── 1. Cargar dataset ────────────────────────────────────────────────────────
try:
    data = pd.read_csv("dataset/gestos.csv", header=None, encoding='utf-8')
except UnicodeDecodeError:
    try:
        print("⚠️  UTF-8 falló, intentando con Latin-1...")
        data = pd.read_csv("dataset/gestos.csv", header=None, encoding='latin-1')
    except UnicodeDecodeError:
        print("⚠️  Latin-1 falló, intentando con cp1252...")
        data = pd.read_csv("dataset/gestos.csv", header=None, encoding='cp1252')

print(f"Dataset original: {len(data)} muestras, {data.iloc[:, -1].nunique()} gestos distintos")
print("Distribución por clase (original):")
print(data.iloc[:, -1].value_counts())

# ── 2. FILTRAR clases con pocas muestras ──────────────────────────────────
MIN_MUESTRAS = 10

conteos = data.iloc[:, -1].value_counts()
clases_validas = conteos[conteos >= MIN_MUESTRAS].index.tolist()
print(f"\n📊 Clases con al menos {MIN_MUESTRAS} muestras: {clases_validas}")

data_filtrado = data[data.iloc[:, -1].isin(clases_validas)]
print(f"Dataset filtrado: {len(data_filtrado)} muestras")

print("\nDistribución por clase (filtrado):")
print(data_filtrado.iloc[:, -1].value_counts())

# ── 3. Verificar balance ─────────────────────────────────────────────────────
X = data_filtrado.iloc[:, :-1].values
y = data_filtrado.iloc[:, -1].values

conteos = pd.Series(y).value_counts()
minimo = conteos.min()
maximo = conteos.max()
if maximo / minimo > 2:
    print(f"\n⚠  Dataset desbalanceado (ratio {maximo/minimo:.1f}x). "
          "Considera capturar más muestras del gesto minoritario.")

# ── 4. Dividir datos ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 División de datos:")
print(f"  - Entrenamiento: {len(X_train)} muestras")
print(f"  - Prueba: {len(X_test)} muestras")

# ── 5. Definir modelos a comparar ────────────────────────────────────────────
candidatos = {
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                    probability=True, random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                           max_depth=4, random_state=42))
    ]),
}

# ── 6. Validación cruzada para elegir el mejor modelo ───────────────────────
print("\n── Validación cruzada (5-fold) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

for nombre, pipeline in candidatos.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    resultados[nombre] = scores
    print(f"  {nombre:<20} acc: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

mejor_nombre = max(resultados, key=lambda k: resultados[k].mean())
print(f"\n✓ Mejor modelo: {mejor_nombre}")
modelo_final = candidatos[mejor_nombre]

# ── 7. Ajuste fino con GridSearch del mejor modelo ───────────────────────────
if mejor_nombre == "RandomForest":
    param_grid = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_split": [2, 5],
    }
elif mejor_nombre == "SVM_RBF":
    param_grid = {
        "clf__C": [1, 10, 50],
        "clf__gamma": ["scale", "auto"],
    }
else:
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
    }

print("\n── GridSearchCV (puede tardar) ──")
gs = GridSearchCV(modelo_final, param_grid, cv=cv,
                  scoring="accuracy", n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
print(f"  Mejores parámetros: {gs.best_params_}")
print(f"  Mejor CV acc:       {gs.best_score_*100:.2f}%")

modelo_optimo = gs.best_estimator_

# ── 8. Evaluación final ──────────────────────────────────────────────────────
y_pred = modelo_optimo.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"\n── Resultado en test ──")
print(f"  Accuracy: {acc*100:.2f}%")
print("\nConfianza por gesto:")
print(classification_report(y_test, y_pred))

# ── 9. Guardar modelo en carpeta 'models' ──────────────────────────────────
print("\n── Guardando modelo ──")

ruta_modelo = os.path.join(CARPETA_MODELOS, "modelo_gestos.pkl")
joblib.dump(modelo_optimo, ruta_modelo)
print(f"  ✓ Modelo guardado en: {ruta_modelo}")

scaler = modelo_optimo.named_steps['scaler']
ruta_scaler = os.path.join(CARPETA_MODELOS, "scaler_gestos.pkl")
joblib.dump(scaler, ruta_scaler)
print(f"  ✓ Escalador guardado en: {ruta_scaler}")

info_modelo = {
    'nombre': 'modelo_gestos.pkl',
    'fecha_creacion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'mejor_modelo': mejor_nombre,
    'mejores_parametros': gs.best_params_,
    'clases': modelo_optimo.classes_.tolist() if hasattr(modelo_optimo, 'classes_') else None,
    'caracteristicas': X.shape[1],
    'accuracy_train': gs.best_score_,
    'accuracy_test': acc,
    'num_muestras': len(data_filtrado),
    'num_gestos': data_filtrado.iloc[:, -1].nunique(),
    'distribucion_gestos': data_filtrado.iloc[:, -1].value_counts().to_dict(),
    'muestras_eliminadas': len(data) - len(data_filtrado),
    'clases_eliminadas': [c for c in conteos.index if c not in clases_validas]
}

ruta_info = os.path.join(CARPETA_MODELOS, "info_modelo.txt")
with open(ruta_info, 'w', encoding='utf-8') as f:
    f.write("=" * 50 + "\n")
    f.write("INFORMACIÓN DEL MODELO DE GESTOS\n")
    f.write("=" * 50 + "\n\n")
    
    f.write(f"📅 Fecha creación: {info_modelo['fecha_creacion']}\n")
    f.write(f"🏆 Mejor modelo: {info_modelo['mejor_modelo']}\n")
    f.write(f"⚙️  Mejores parámetros: {info_modelo['mejores_parametros']}\n\n")
    
    f.write("📊 Rendimiento:\n")
    f.write(f"  - Accuracy (CV): {info_modelo['accuracy_train']*100:.2f}%\n")
    f.write(f"  - Accuracy (Test): {info_modelo['accuracy_test']*100:.2f}%\n\n")
    
    f.write("📋 Dataset:\n")
    f.write(f"  - Total muestras: {info_modelo['num_muestras']}\n")
    f.write(f"  - Gestos únicos: {info_modelo['num_gestos']}\n")
    f.write(f"  - Características: {info_modelo['caracteristicas']}\n")
    f.write(f"  - Clases: {info_modelo['clases']}\n")
    f.write(f"  - Muestras eliminadas: {info_modelo['muestras_eliminadas']}\n")
    if info_modelo['clases_eliminadas']:
        f.write(f"  - Clases eliminadas: {info_modelo['clases_eliminadas']}\n\n")
    
    f.write("📈 Distribución por gesto:\n")
    for gesto, count in info_modelo['distribucion_gestos'].items():
        f.write(f"  - {gesto}: {count} muestras\n")
    
    f.write("\n" + "=" * 50 + "\n")
    f.write("ESTRUCTURA DE ARCHIVOS\n")
    f.write("=" * 50 + "\n")
    f.write(f"{CARPETA_MODELOS}/\n")
    f.write(f"├── modelo_gestos.pkl     # Modelo entrenado\n")
    f.write(f"├── scaler_gestos.pkl     # Escalador para preprocesar\n")
    f.write(f"└── info_modelo.txt       # Esta información\n")

print(f"  ✓ Información guardada en: {ruta_info}")

print(f"\n📂 Contenido de '{CARPETA_MODELOS}/':")
for archivo in sorted(os.listdir(CARPETA_MODELOS)):
    ruta_archivo = os.path.join(CARPETA_MODELOS, archivo)
    tamaño = os.path.getsize(ruta_archivo)
    print(f"  - {archivo} ({tamaño/1024:.2f} KB)")

# ── 10. Gráficas diagnóstico ─────────────────────────────────────────────────
print("\n── Generando gráficas ──")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

nombres = list(resultados.keys())
medias = [resultados[n].mean() * 100 for n in nombres]
stds = [resultados[n].std() * 100 for n in nombres]
bars = axes[0].bar(nombres, medias, yerr=stds, capsize=6,
                   color=["#4CAF50" if n == mejor_nombre else "#90A4AE" for n in nombres])
axes[0].set_ylim(max(0, min(medias) - 5), 101)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Comparación de modelos (5-fold CV)")
axes[0].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)

clases = modelo_optimo.classes_
cm = confusion_matrix(y_test, y_pred, labels=clases)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Matriz de confusión (test)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
ruta_grafica = os.path.join(CARPETA_MODELOS, "diagnostico_modelo.png")
plt.savefig(ruta_grafica, dpi=120, bbox_inches="tight")
print(f"  ✓ Gráficas guardadas en: {ruta_grafica}")
plt.show()