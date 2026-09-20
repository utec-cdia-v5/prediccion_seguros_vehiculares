# Taller - Predicción de Seguros Vehiculares

## Objetivo
Comparar dos problemas de aprendizaje automático (Machine Learning) supervisado y publicar ambos modelos en una sola API REST FastAPI `prediccion_seguros_vehiculares`.

- **Clasificación:** predecir `abandona` (Sí/No).
- **Regresión:** estimar `costo_reparacion` en PEN. Un valor como **S/ 9,350** es un ejemplo; la predicción real depende de los datos y del modelo.

## Arquitectura
```text
        Predicción de Seguros Vehiculares
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   CLASIFICACIÓN              REGRESIÓN
clientes_abandono.csv    siniestros_costos.csv
    1,000 filas               1,000 filas
          │                       │
          ▼                       ▼
 RandomForestClassifier    RandomForestRegressor
          │                       │
          ▼                       ▼
modelo_abandono.joblib    modelo_costo_reparacion.joblib
          │                       │
          └──────────┬────────────┘
                     ▼
       prediccion_seguros_vehiculares
                  FastAPI
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 POST /abandono       POST /costo_reparacion
          │                     │
      Sí / No             Monto en PEN
                     │
                     ▼
                   Docker
```

## Base teórica
**Aprendizaje supervisado** aprende una relación entre variables X y un objetivo usando ejemplos etiquetados: 
- **Clasificación**: predice una clase discreta; aquí 0=permanece y 1=abandona. 
- **Regresión**: predice un valor continuo; aquí el costo de reparación.

### Métricas de Clasificación
- **Accuracy:** proporción total correcta.
- **Precision:** de los abandonos predichos, cuántos eran abandono.
- **Recall:** de los abandonos reales, cuántos detectó.
- **F1:** balance entre Precision y Recall.
- **Matriz de confusión:** TP, FP, TN y FN.

### Métricas de Regresión
- **MAE:** error absoluto promedio, interpretable en soles.
- **RMSE:** penaliza más los errores grandes.
- **R²:** proporción de variabilidad explicada; se interpreta junto con MAE/RMSE y un conjunto de prueba.

| Aspecto | Clasificación | Regresión |
|---|---|---|
| Pregunta | ¿Abandonará? | ¿Cuánto costará? |
| Target | 0/1 | monto continuo |
| Modelo | RandomForestClassifier | RandomForestRegressor |
| Métricas | Accuracy, Precision, Recall, F1 | MAE, RMSE, R² |

## Datos
Ambos CSV contienen **exactamente 1,000 registros ficticios**:
- `clientes_abandono.csv` usa edad, antigüedad, prima, siniestros, reclamos y renovaciones. 
- `siniestros_costos.csv` usa antigüedad/valor del vehículo, severidad, tipo de siniestro, piezas, grúa y combustible.

## Preparación
Ingresa a MV Desarrollo en su cuenta de AWS Academy.

```bash (MV Linux)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenamiento y Persistencia
```bash
python training/train_abandono.py
python training/train_costo_reparacion.py
```
Cada script hace `train_test_split` 80/20. El clasificador usa un Pipeline con escalado y Random Forest. El regresor usa `ColumnTransformer`: numéricas con `StandardScaler` y categóricas con `OneHotEncoder(handle_unknown="ignore")`, seguido de Random Forest. Se guarda el **Pipeline completo**, evitando discrepancias entre entrenamiento e inferencia.

Se generan los modelos ya entrenados: `models/modelo_abandono.joblib`, `models/modelo_costo_reparacion.joblib` y JSON con métricas.

## API (Docker)
```bash
docker build -t prediccion_seguros_vehiculares:1.0 .
docker run -d --rm --name prediccion_seguros_vehiculares_c -p 8000:8000 prediccion_seguros_vehiculares:1.0
```

## Predicción
Pruebe ambos endpoints con la documentación de FastApi o Postman:
- Documentación FastApi en Swagger: `http://IP_Pública_MV:8000/docs` (Abra el puerto 8000 en MV de ser necesario)
- Postman: Importe la colección postman: `prediccion_seguros_vehiculares.postman_collection.json` 

## Conclusión
Ambos casos comparten el ciclo **Datos -> Preparación -> Entrenamiento -> Persistencia -> API (Docker) -> Predicción**, pero responden preguntas distintas: una clase probable frente a un valor numérico estimado.