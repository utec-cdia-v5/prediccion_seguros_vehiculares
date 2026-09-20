from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

R = Path(__file__).resolve().parents[1]
a = joblib.load(R / 'models/modelo_abandono.joblib')
c = joblib.load(R / 'models/modelo_costo_reparacion.joblib')
app = FastAPI(
	title='prediccion_seguros_vehiculares',
	version='1.0',
)


class A(BaseModel):
	edad: int = Field(ge=18, le=100)
	antiguedad: int = Field(ge=0)
	prima_mensual: float = Field(gt=0)
	numero_siniestros: int = Field(ge=0)
	numero_reclamos: int = Field(ge=0)
	renovaciones: int = Field(ge=0)


class C(BaseModel):
	antiguedad_vehiculo: int = Field(ge=0)
	valor_vehiculo: float = Field(gt=0)
	severidad: str
	tipo_siniestro: str
	piezas_afectadas: int = Field(ge=1)
	requiere_grua: int = Field(ge=0, le=1)
	combustible: str


@app.get('/health')
def health():
	return {'status': 'ok', 'modelos': ['abandono', 'costo_reparacion']}


@app.post('/abandono')
def abandono(x: A):
	d = pd.DataFrame([x.model_dump()])
	y = int(a.predict(d)[0])
	p = float(a.predict_proba(d)[0][1])
	return {
		'abandona': bool(y),
		'clase': y,
		'probabilidad_abandono': round(p, 4),
	}


@app.post('/costo_reparacion')
def costo(x: C):
	d = pd.DataFrame([x.model_dump()])
	y = float(c.predict(d)[0])
	return {
		'costo_reparacion_estimado': round(y, 2),
		'moneda': 'PEN',
	}
