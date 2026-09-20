# LAB1b — Predicción de Seguros Vehiculares

## Objetivo
Comparar dos problemas de aprendizaje supervisado y publicar ambos modelos en una sola API REST FastAPI `prediccion_seguros_vehiculares`.

- **Clasificación:** predecir `abandona` (Sí/No).
- **Regresión:** estimar `costo_reparacion` en PEN. Un valor como **S/ 9,350** es solo un ejemplo pedagógico; la predicción real depende de los datos y del modelo.

## Arquitectura
```text
                    LAB1b
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   CLASIFICACIÓN              REGRESIÓN
clientes_abandono.csv      siniestros_costos.csv
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
**Aprendizaje supervisado** aprende una relación entre variables X y un objetivo y usando ejemplos etiquetados. **Clasificación** predice una clase discreta; aquí 0=permanece y 1=abandona. **Regresión** predice un valor continuo; aquí el costo de reparación.

### Métricas de clasificación
- **Accuracy:** proporción total correcta.
- **Precision:** de los abandonos predichos, cuántos eran abandono.
- **Recall:** de los abandonos reales, cuántos detectó.
- **F1:** balance entre Precision y Recall.
- **Matriz de confusión:** TP, FP, TN y FN.

### Métricas de regresión
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
Ambos CSV contienen **exactamente 1,000 registros ficticios** y son solo educativos. `clientes_abandono.csv` usa edad, antigüedad, prima, siniestros, reclamos y renovaciones. `siniestros_costos.csv` usa antigüedad/valor del vehículo, severidad, tipo de siniestro, piezas, grúa y combustible.

## Preparación
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
En Windows: `.venv\Scripts\Activate.ps1`.

## Entrenamiento
```bash
python training/train_abandono.py
python training/train_costo_reparacion.py
```
Cada script hace `train_test_split` 80/20. El clasificador usa un Pipeline con escalado y Random Forest. El regresor usa `ColumnTransformer`: numéricas con `StandardScaler` y categóricas con `OneHotEncoder(handle_unknown="ignore")`, seguido de Random Forest. Se guarda el **Pipeline completo**, evitando discrepancias entre entrenamiento e inferencia.

Se generan `models/modelo_abandono.joblib`, `models/modelo_costo_reparacion.joblib` y JSON con métricas.

## API local
```bash
uvicorn api.main:app --reload
```
Swagger: `http://localhost:8000/docs`

### POST /abandono
```bash
curl -X POST http://localhost:8000/abandono -H "Content-Type: application/json" -d '{"edad":31,"antiguedad":2,"prima_mensual":260,"numero_siniestros":1,"numero_reclamos":4,"renovaciones":2}'
```
Devuelve clase y probabilidad estimada.

### POST /costo_reparacion
Categorías: severidad=`Leve|Media|Grave`; tipo=`Colision|Robo_Parcial|Choque_Multiple|Volcadura|Inundacion`; combustible=`Gasolina|Diesel|Hibrido|Electrico`.
```bash
curl -X POST http://localhost:8000/costo_reparacion -H "Content-Type: application/json" -d '{"antiguedad_vehiculo":6,"valor_vehiculo":70000,"severidad":"Media","tipo_siniestro":"Colision","piezas_afectadas":5,"requiere_grua":1,"combustible":"Gasolina"}'
```
Devuelve un costo estimado en PEN.

## Docker
Primero entrene ambos modelos. Luego:
```bash
docker build -t prediccion_seguros_vehiculares:1.0 .
docker run --rm -p 8000:8000 prediccion_seguros_vehiculares:1.0
```
Pruebe nuevamente ambos endpoints en Swagger, curl o Postman.

## Secuencia pedagógica sugerida
1. Revisar aprendizaje supervisado y diferenciar clasificación/regresión.
2. Explorar ambos CSV con Pandas.
3. Identificar X e y.
4. Separar train/test y discutir fuga de datos.
5. Entrenar y comparar métricas.
6. Explicar Pipeline, serialización y `joblib`.
7. Probar inferencia sin reentrenar.
8. Exponer ambos modelos en FastAPI.
9. Contenerizar y repetir las pruebas.

## Preguntas de análisis
1. ¿Por qué abandono es clasificación y costo es regresión?
2. ¿Por qué no evaluar con los mismos datos de entrenamiento?
3. ¿Cuándo Recall sería más importante que Accuracy?
4. ¿Qué significa un MAE de S/ 1,200?
5. ¿Por qué RMSE penaliza más errores grandes?
6. ¿Por qué usamos One-Hot Encoding?
7. ¿Qué ventaja ofrece guardar el Pipeline completo?
8. ¿Por qué la API no debe reentrenar en cada request?
9. ¿Qué diferencia existe entre `predict()` y `predict_proba()`?
10. ¿Qué controles adicionales exigiría un caso productivo?

## Ejercicios
Cambiar `test_size`; variar `n_estimators`; agregar variables; comparar Logistic Regression para abandono; comparar Linear/Gradient Boosting para costo; agregar validación estricta de categorías; medir latencia; versionar endpoints `/v1/...`.

## Conclusión
Ambos casos comparten el ciclo **Datos -> Preparación -> Entrenamiento -> Evaluación -> Pipeline -> Persistencia -> API -> Docker -> Inferencia**, pero responden preguntas distintas: una clase probable frente a un valor numérico estimado.
