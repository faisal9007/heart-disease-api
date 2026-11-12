<<<<<<< HEAD
# Heart Disease Prediction API

FastAPI service that predicts the presence of heart disease from the
[Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset),
containerized with Docker and deployable to Render.

**Live URL:** `https://<your-service>.onrender.com`  (docs at `/docs`)

## Project structure

```
app/            FastAPI app (main.py, schemas.py)
model/          Trained model (heart_model.joblib)
train.py        Training script
Dockerfile
docker-compose.yml
requirements.txt
```

## 1. Train the model

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt kagglehub
python train.py            # or: python train.py --csv data/heart.csv
```

This writes `model/heart_model.joblib`. Train with the same `requirements.txt`
used by Docker so the scikit-learn versions match.

## 2. Run with Docker

```bash
docker-compose build
docker-compose up
```

Swagger UI: http://localhost:8000/docs

## 3. Endpoints

| Method | Path       | Description                                  |
|--------|------------|----------------------------------------------|
| GET    | `/health`  | Service status and whether the model loaded  |
| GET    | `/info`    | Model type and feature list                  |
| POST   | `/predict` | Returns `heart_disease: true/false` + probability |

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":57,"sex":1,"cp":0,"trestbps":140,"chol":241,"fbs":0,"restecg":1,"thalach":123,"exang":1,"oldpeak":0.2,"slope":1,"ca":0,"thal":3}'
```

Example response:

```json
{"heart_disease": true, "probability": 0.87}
```

## 4. Deploy on Render

1. Make sure `model/heart_model.joblib` is committed (not git-ignored).
2. Push to GitHub.
3. Render dashboard → **New → Web Service** → connect the repo.
4. **Language/Environment:** Docker. Dockerfile path `./Dockerfile`, root directory blank (repo root).
5. Create the service, wait for the build, then test `/health`, `/docs`, and `/predict` on the live URL.

Note: the free tier sleeps after inactivity, so the first request may take ~30-60 seconds.

## Disclaimer

Educational project only; not for medical use.
=======
# heart-disease-api
heart-disease-api
>>>>>>> d6c7018b81e0f56db999be2a2df953aa22b0b347
