# Detección de Estrés Académico mediante Fine-Tuning de RoBERTuito

Clasificación binaria de estrés académico en comentarios estudiantiles universitarios, mediante ajuste fino (fine-tuning) de un modelo transformer preentrenado en español (RoBERTuito), siguiendo la metodología CRISP-DM.

## Resumen

Este proyecto ajusta el modelo `pysentimiento/robertuito-base-cased` para identificar señales de estrés académico en comentarios de estudiantes sobre sus profesores y cursos, expresados en español informal con jerga universitaria limeña. El modelo final alcanza un **F1-score de 0.97** sobre un conjunto de prueba no visto durante el entrenamiento.

## Metodología

El proyecto sigue las fases de CRISP-DM:

1. **Comprensión del negocio** — definición del problema como clasificación binaria (0: sin estrés, 1: con estrés), y diseño de un esquema de etiquetado basado en cuatro dimensiones teóricas: rasgos afectivos, sintácticos, sociales y temáticos.
2. **Comprensión y preparación de los datos** — construcción de un corpus de 800 comentarios sintéticos, auditado mediante análisis de similitud de coseno sobre representaciones TF-IDF para detectar y eliminar redundancia estructural y mitigar riesgo de fuga de datos (data leakage) entre particiones. El corpus depurado quedó en 720 observaciones. Se validó la dificultad resultante frente a un modelo de referencia (TF-IDF + Regresión Logística, F1 = 0.868) antes de proceder al ajuste fino.
3. **Modelado** — tokenización con Byte-Pair Encoding (longitud máxima de 96 tokens), split estratificado 70/15/15 (train/validation/test), y fine-tuning con la librería `transformers` de Hugging Face.
4. **Evaluación** — medición sobre el conjunto de prueba aislado, con matriz de confusión y reporte de clasificación completo.
5. **Despliegue** — modelo publicado en Hugging Face Hub en formato `safetensors`, con notebook de inferencia funcional.

## Resultados

| Clase | Precisión | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Estrés (0) | 0.966 | 0.983 | 0.974 | 58 |
| Estrés (1) | 0.980 | 0.960 | 0.970 | 50 |
| **Accuracy** | | | **0.972** | 108 |

## Estructura del repositorio

```
├── README.md
├── data_estudiantes.csv          # Corpus final (720 filas, auditado)
├── 1_Expo_IA_PNL.ipynb           # Pipeline completo: carga, preparación, fine-tuning, evaluación
├── Prueba_robertitu.ipynb        # Inferencia con el modelo ya entrenado
└── matriz_confusion_final.png    # Matriz de confusión sobre el conjunto de prueba
```

## Modelo

El modelo ajustado está publicado públicamente en Hugging Face Hub:
**[huggingface.co/tu-usuario/robertuito-estres-academico](https://huggingface.co/tu-usuario/robertuito-estres-academico)**

## Cómo ejecutar

Ambos notebooks están diseñados para ejecutarse en Google Colab sin dependencias locales ni acceso a almacenamiento personal.

**Pipeline completo** (`1_Expo_IA_PNL.ipynb`): reproduce el proceso desde la carga de datos hasta la evaluación final. Requiere entorno de ejecución con GPU (Entorno de ejecución → Cambiar tipo de entorno → GPU T4).

**Solo inferencia** (`Prueba_robertitu.ipynb`): carga el modelo ya entrenado desde Hugging Face Hub y permite clasificar texto nuevo directamente.

En ambos casos, basta con abrir el notebook en Colab y ejecutar las celdas en orden; los datos y el modelo se cargan automáticamente desde este repositorio y desde Hugging Face Hub, respectivamente.

## Stack técnico

- Python, PyTorch
- Hugging Face `transformers`, `datasets`, `evaluate`
- `scikit-learn` (métricas, baseline, split estratificado)
- `pandas`, `matplotlib`, `seaborn`
- Modelo base: [pysentimiento/robertuito-base-cased](https://huggingface.co/pysentimiento/robertuito-base-cased)

## Limitaciones

El corpus utilizado es de origen sintético, generado mediante un modelo de lenguaje y sometido a un proceso de auditoría y depuración descrito en la sección de metodología. Las métricas reportadas corresponden al desempeño del modelo sobre este corpus específico; su validación sobre comentarios reales de estudiantes en un entorno de producción constituye trabajo futuro pendiente.

## Autor

Leo — Ingeniería de Sistemas, Universidad de Lima
