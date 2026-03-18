# NBA Predictions Pipeline

Proyecto en Python para construir un pipeline diario de prediccion NBA.

Objetivos actuales:

- construir y actualizar un dataset historico de regular season
- entrenar un modelo de prediccion de ganador local
- incorporar lesiones actuales al flujo diario
- predecir los partidos de las proximas 24 horas

Objetivo final:

- evolucionar este pipeline diario hacia un modelo final capaz de simular y proyectar el bracket completo de playoffs

## Estructura

```text
Predictor-del-Bracket-de-Playoff/
|-- data/
|   |-- dataset_ml.csv
|   |-- injuries_current.csv
|   |-- player_stats_reference.csv
|   `-- remaining_schedule_template.csv
|-- models/
|   `-- nba_model.pkl
|-- outputs/
|   |-- next_24h_predictions.csv
|   |-- predictions_history.csv
|   `-- daily_predictions/
|       `-- predictions_YYYY-MM-DD.csv
|-- templates/
|   `-- index.html
|-- scripts/
|   |-- build_dataset.py
|   |-- daily_update.py
|   |-- predict_next_24h.py
|   |-- show_injuries.py
|   |-- train_model.py
|   |-- update_injuries.py
|   `-- dev/
|       `-- test_player_lookup.py
|-- .gitignore
|-- app.py
|-- requirements.txt
`-- README.md
```

## Requisitos

- Python 3.9+
- Instalar dependencias:

```bash
pip install -r requirements.txt
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

## 3. Actualizar lesiones

Regenera el fichero de lesiones actual:

`data/injuries_current.csv`

```bash
python scripts/update_injuries.py
```

Tambien puedes inspeccionarlo por pantalla:

```bash
python scripts/show_injuries.py
```

## 4. Predecir partidos de las proximas 24 horas

Muestra por pantalla las predicciones y guarda una copia en:

`outputs/next_24h_predictions.csv`

```bash
python scripts/predict_next_24h.py
```

La prediccion muestra:

- probabilidad base del modelo
- probabilidad ajustada por lesiones
- score de lesiones de local y visitante
- favorito final del partido

Ademas se guarda historico en:

- `outputs/daily_predictions/predictions_YYYY-MM-DD.csv`
- `outputs/predictions_history.csv`

## 5. Interfaz web

Puedes abrir una interfaz visual con Flask:

```bash
python app.py
```

Y luego visitar:

```text
http://127.0.0.1:5000
```

La interfaz muestra:

- selector de dia por jornada
- prediccion de los partidos del dia elegido
- comparacion con resultado real en dias pasados
- accuracy diaria cuando ya existen resultados
- probabilidades base y ajustadas
- lesiones activas
- historico diario de archivos generados

## Fichero de lesiones

El fichero actual es:

`data/injuries_current.csv`

Columnas:

```csv
TEAM,PLAYER,STATUS,MINUTES_PER_GAME,POINTS_PER_GAME,IS_STARTER,RETURN_ESTIMATE,NOTES
```

Estados recomendados:

- `Out`
- `Doubtful`
- `Questionable`
- `Probable`

## Flujo diario recomendado

Si quieres actualizar todo de una vez y obtener la prediccion mas reciente:

```bash
python scripts/daily_update.py
```

Este script:

- regenera `data/dataset_ml.csv`
- actualiza `data/injuries_current.csv`
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
- Las lesiones actuales se integran en la prediccion diaria como ajuste sobre la probabilidad base.
- `dataset_ml.csv`, `nba_model.pkl` y los CSV de salida pueden regenerarse en cualquier momento.
- `data/player_stats_reference.csv` actua como cache local de stats de jugadores para enriquecer lesiones.

## Roadmap

1. Construir un historico de lesiones por fecha y unirlo al dataset de entrenamiento.
2. Reentrenar el modelo con features de lesiones historicas, no solo con ajuste post-modelo.
3. Integrar cuotas de mercado y compararlas con la probabilidad del modelo.
4. Generar un modelo final para simulacion de play-in y bracket de playoffs.
