
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import xgboost as xgb
import os

app = FastAPI(
    title="SatQuery Model-3 API",
    description="Model-3 land-cover prediction API",
    version="1.0.0"
)

MODEL_DIR = "SatQuery_model3"

CLASS_NAMES = [
    "Agro-forestry",
    "Arable land",
    "Beaches dunes sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial/commercial",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland shrub",
    "Urban fabric"
]

models = []

for i in range(19):
    path = os.path.join(
        MODEL_DIR,
        f"xgb_class_{i:02d}.json"
    )

    model = xgb.XGBClassifier()
    model.load_model(path)
    models.append(model)

print("Loaded", len(models), "XGBoost models")


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=26,
        max_length=26,
        description="Exactly 26 Model-3 features"
    )


@app.get("/")
def root():
    return {
        "service": "SatQuery Model-3",
        "status": "running",
        "models": 19,
        "features": 26
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": len(models) == 19,
        "number_of_models": len(models),
        "number_of_features": 26
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        features = np.array(
            request.features,
            dtype=np.float32
        ).reshape(1, -1)

        probabilities = []

        for model in models:
            prob = float(
                model.predict_proba(features)[0][1]
            )
            probabilities.append(prob)

        predictions = [
            CLASS_NAMES[i]
            for i, p in enumerate(probabilities)
            if p >= 0.5
        ]

        ranked = sorted(
            zip(CLASS_NAMES, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        top5 = [
            {
                "class": name,
                "probability": round(float(prob), 6)
            }
            for name, prob in ranked[:5]
        ]

        confidence = max(probabilities)

        return {
            "predicted_classes": predictions,
            "confidence": round(float(confidence), 6),
            "top5": top5,
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]), 6)
                for i in range(19)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
