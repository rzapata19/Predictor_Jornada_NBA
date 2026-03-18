# Predictor del Bracket de Playoff NBA

Proyecto en Python para:

- construir un dataset de partidos NBA con features temporales
- entrenar un modelo de prediccion de ganador local
- predecir los partidos de las proximas 24 horas

## Estructura

```text
Predictor-del-Bracket-de-Playoff/
|-- data/
|   |-- dataset_ml.csv
|   `-- remaining_schedule_template.csv
|-- models/
|   `-- nba_model.pkl
|-- outputs/
|   `-- next_24h_predictions.csv
|-- scripts/
|   |-- build_dataset.py
|   |-- daily_update.py
|   |-- train_model.py
|   `-- predict_next_24h.py
|-- .gitignore
`-- README.md
```

## Requisitos

- Python 3.9+
- Dependencias:

```bash
pip install pandas scikit-learn xgboost joblib nba_api requests
```

## 1. Construir el dataset

Descarga partidos historicos de regular season y genera las features en:

`data/dataset_ml.csv`

```bash
python scripts/build_dataset.py
```

## 2. Entrenar el modelo

Entrena un `XGBoost` con validacion temporal y guarda el modelo en:

`models/nba_model.pkl`

```bash
python scripts/train_model.py
```

## 3. Predecir partidos de las proximas 24 horas

Muestra por pantalla las predicciones y guarda una copia en:

`outputs/next_24h_predictions.csv`

```bash
python scripts/predict_next_24h.py
```

## Flujo diario recomendado

Si quieres actualizar todo de una vez y obtener la prediccion mas reciente:

```bash
python scripts/daily_update.py
```

Este script:

- regenera `data/dataset_ml.csv`
- reentrena `models/nba_model.pkl`
- genera las predicciones en pantalla y en `outputs/next_24h_predictions.csv`

Si quieres evitar depender de la API para el calendario futuro, crea:

`data/remaining_schedule.csv`

con este formato:

```csv
DATE,HOME_TEAM,AWAY_TEAM,SEASON
2026-03-16,Boston Celtics,Miami Heat,2025-26
2026-03-16,Los Angeles Lakers,Denver Nuggets,2025-26
```

## Notas

- El dataset usa solo partidos de `Regular Season`.
- El modelo trabaja con variables temporales como ELO, forma reciente, descanso y rendimiento home/away.
- `dataset_ml.csv`, `nba_model.pkl` y los CSV de salida pueden regenerarse en cualquier momento.

## Siguientes mejoras

- anadir lesiones y bajas de jugadores
- integrar cuotas de mercado
- exportar predicciones a una interfaz web o dashboard
