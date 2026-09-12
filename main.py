
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# import numpy as np
# import xgboost as xgb
# import os

# app = FastAPI(
#     title="SatQuery Model-3 API",
#     description="Model-3 land-cover prediction API",
#     version="1.0.0"
# )

# MODEL_DIR = "SatQuery_model3"

# CLASS_NAMES = [
#     "Agro-forestry",
#     "Arable land",
#     "Beaches dunes sands",
#     "Broad-leaved forest",
#     "Coastal wetlands",
#     "Complex cultivation patterns",
#     "Coniferous forest",
#     "Industrial/commercial",
#     "Inland waters",
#     "Inland wetlands",
#     "Land principally occupied by agriculture, with significant areas of natural vegetation",
#     "Marine waters",
#     "Mixed forest",
#     "Moors heathland and sclerophyllous vegetation",
#     "Natural grassland and sparsely vegetated areas",
#     "Pastures",
#     "Permanent crops",
#     "Transitional woodland shrub",
#     "Urban fabric"
# ]

# models = []

# for i in range(19):
#     path = os.path.join(
#         MODEL_DIR,
#         f"xgb_class_{i:02d}.json"
#     )

#     model = xgb.XGBClassifier()
#     model.load_model(path)
#     models.append(model)

# print("Loaded", len(models), "XGBoost models")


# class PredictionRequest(BaseModel):
#     features: list[float] = Field(
#         ...,
#         min_length=26,
#         max_length=26,
#         description="Exactly 26 Model-3 features"
#     )


# @app.get("/")
# def root():
#     return {
#         "service": "SatQuery Model-3",
#         "status": "running",
#         "models": 19,
#         "features": 26
#     }


# @app.get("/health")
# def health():
#     return {
#         "status": "healthy",
#         "model_loaded": len(models) == 19,
#         "number_of_models": len(models),
#         "number_of_features": 26
#     }


# @app.post("/predict")
# def predict(request: PredictionRequest):

#     try:
#         features = np.array(
#             request.features,
#             dtype=np.float32
#         ).reshape(1, -1)

#         probabilities = []

#         for model in models:
#             prob = float(
#                 model.predict_proba(features)[0][1]
#             )
#             probabilities.append(prob)

#         predictions = [
#             CLASS_NAMES[i]
#             for i, p in enumerate(probabilities)
#             if p >= 0.5
#         ]

#         ranked = sorted(
#             zip(CLASS_NAMES, probabilities),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         top5 = [
#             {
#                 "class": name,
#                 "probability": round(float(prob), 6)
#             }
#             for name, prob in ranked[:5]
#         ]

#         confidence = max(probabilities)

#         return {
#             "predicted_classes": predictions,
#             "confidence": round(float(confidence), 6),
#             "top5": top5,
#             "probabilities": {
#                 CLASS_NAMES[i]: round(float(probabilities[i]), 6)
#                 for i in range(19)
#             }
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=str(e)
#         )
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date as Date, datetime, timedelta
import os
import io
import json
import numpy as np
import requests
import xgboost as xgb
import ee
from google.oauth2 import service_account


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SatQuery Model-3 API",
    description="Sentinel-1 + Sentinel-2 land-cover prediction API",
    version="2.0.0"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "SatQuery_model3"

GEE_PROJECT_ID = os.getenv(
    "GEE_PROJECT_ID",
    "dynamic-camp-476214-s3"
)

GEE_SERVICE_ACCOUNT_FILE = "/etc/secrets/gee-service-account.json"

TARGET_SIZE = 120
PATCH_SIZE_METERS = 1200
SEARCH_DAYS = 15
EPS = 1e-8


# ============================================================
# CLASS NAMES
# ============================================================

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


# ============================================================
# MODEL 3 FEATURE ORDER
# ============================================================

FEATURE_NAMES = [
    "B02_mean",
    "B02_std",
    "B03_mean",
    "B03_std",
    "B04_mean",
    "B04_std",
    "B08_mean",
    "B08_std",
    "B11_mean",
    "B11_std",
    "B12_mean",
    "B12_std",
    "NDVI_mean",
    "NDVI_std",
    "NDWI_mean",
    "NDWI_std",
    "NDBI_mean",
    "NDBI_std",
    "VV_mean",
    "VV_std",
    "VH_mean",
    "VH_std",
    "VV_minus_VH_mean",
    "VV_minus_VH_std",
    "VV_VH_ratio_mean",
    "VV_VH_ratio_std"
]


# ============================================================
# LOAD XGBOOST MODELS
# ============================================================

models = []

for i in range(19):

    path = os.path.join(
        MODEL_DIR,
        f"xgb_class_{i:02d}.json"
    )

    if not os.path.exists(path):
        raise RuntimeError(
            f"Model file not found: {path}"
        )

    model = xgb.XGBClassifier()
    model.load_model(path)
    models.append(model)

print("Loaded", len(models), "XGBoost models")


# ============================================================
# INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

def initialize_earth_engine():

    if not os.path.exists(GEE_SERVICE_ACCOUNT_FILE):
        raise RuntimeError(
            "GEE service-account secret file not found: "
            f"{GEE_SERVICE_ACCOUNT_FILE}"
        )

    with open(
        GEE_SERVICE_ACCOUNT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        service_account_info = json.load(f)

    credentials = (
        service_account.Credentials
        .from_service_account_info(
            service_account_info,
            scopes=[
                "https://www.googleapis.com/auth/earthengine"
            ]
        )
    )

    ee.Initialize(
        credentials=credentials,
        project=GEE_PROJECT_ID
    )

    print(
        "Google Earth Engine initialized:",
        GEE_PROJECT_ID
    )


try:
    initialize_earth_engine()
    GEE_READY = True

except Exception as e:

    print(
        "WARNING: Earth Engine initialization failed:",
        str(e)
    )

    GEE_READY = False


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude in decimal degrees"
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude in decimal degrees"
    )

    date: Date = Field(
        ...,
        description="Requested imagery date YYYY-MM-DD"
    )

# ============================================================
# BASIC HELPERS
# ============================================================

def image_stats(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    valid = np.isfinite(image)

    if not np.any(valid):
        return 0.0, 0.0

    values = image[valid]

    return (
        float(np.mean(values)),
        float(np.std(values))
    )


def normalized_difference(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    denominator = a + b

    result = np.full(
        a.shape,
        np.nan,
        dtype=np.float32
    )

    valid = (
        np.isfinite(a)
        &
        np.isfinite(b)
        &
        np.isfinite(denominator)
        &
        (np.abs(denominator) > EPS)
    )

    result[valid] = (
        (a[valid] - b[valid])
        /
        denominator[valid]
    )

    return result


# ============================================================
# CREATE 1200m × 1200m PATCH
# ============================================================

def create_patch(latitude, longitude):

    point = ee.Geometry.Point(
        [
            float(longitude),
            float(latitude)
        ]
    )

    patch = (
        point
        .buffer(PATCH_SIZE_METERS / 2)
        .bounds()
    )

    return patch


# ============================================================
# FIND CLOSEST SENTINEL-2 IMAGE
# ============================================================

def find_sentinel2_image(
    latitude,
    longitude,
    requested_date
):

    point = ee.Geometry.Point(
        [
            float(longitude),
            float(latitude)
        ]
    )

    start_date = (
        requested_date
        - timedelta(days=SEARCH_DAYS)
    ).isoformat()

    end_date = (
        requested_date
        + timedelta(days=SEARCH_DAYS + 1)
    ).isoformat()

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(point)
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                90
            )
        )
        .select(
            [
                "B2",
                "B3",
                "B4",
                "B8",
                "B11",
                "B12"
            ]
        )
    )

    count = collection.size().getInfo()

    if count == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "No Sentinel-2 imagery found within "
                f"{SEARCH_DAYS} days of {requested_date}."
            )
        )

    image_list = collection.toList(count)

    candidates = []

    for i in range(count):

        image = ee.Image(
            image_list.get(i)
        )

        timestamp = (
            image
            .get("system:time_start")
            .getInfo()
        )

        image_date = datetime.fromtimestamp(
            timestamp / 1000
        ).date()

        cloud = image.get(
            "CLOUDY_PIXEL_PERCENTAGE"
        ).getInfo()

        candidates.append(
            (
                abs(
                    image_date
                    - requested_date
                ).days,
                float(cloud or 100),
                image_date,
                image
            )
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )

    _, _, matched_date, selected_image = (
        candidates[0]
    )

    return selected_image, matched_date


# ============================================================
# FIND CLOSEST SENTINEL-1 IMAGE
# ============================================================

def find_sentinel1_image(
    latitude,
    longitude,
    requested_date
):

    point = ee.Geometry.Point(
        [
            float(longitude),
            float(latitude)
        ]
    )

    start_date = (
        requested_date
        - timedelta(days=SEARCH_DAYS)
    ).isoformat()

    end_date = (
        requested_date
        + timedelta(days=SEARCH_DAYS + 1)
    ).isoformat()

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S1_GRD"
        )
        .filterBounds(point)
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.eq(
                "instrumentMode",
                "IW"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation",
                "VH"
            )
        )
        .select(
            [
                "VV",
                "VH"
            ]
        )
    )

    count = collection.size().getInfo()

    if count == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "No Sentinel-1 imagery found within "
                f"{SEARCH_DAYS} days of {requested_date}."
            )
        )

    image_list = collection.toList(count)

    candidates = []

    for i in range(count):

        image = ee.Image(
            image_list.get(i)
        )

        timestamp = (
            image
            .get("system:time_start")
            .getInfo()
        )

        image_date = datetime.fromtimestamp(
            timestamp / 1000
        ).date()

        candidates.append(
            (
                abs(
                    image_date
                    - requested_date
                ).days,
                image_date,
                image
            )
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    _, matched_date, selected_image = (
        candidates[0]
    )

    return selected_image, matched_date


# ============================================================
# DOWNLOAD IMAGE AS NUMPY
# ============================================================

def download_numpy_image(
    image,
    bands,
    region,
    scale=10
):

    image = image.resample(
        "bilinear"
    )

    url = image.getDownloadURL(
        {
            "bands": bands,
            "region": region,
            "scale": scale,
            "format": "NPY"
        }
    )

    response = requests.get(
        url,
        timeout=120
    )

    response.raise_for_status()

    data = np.load(
        io.BytesIO(
            response.content
        )
    )

    return data


# ============================================================
# GET SATELLITE ARRAYS
# ============================================================

def get_satellite_arrays(
    latitude,
    longitude,
    requested_date
):

    patch = create_patch(
        latitude,
        longitude
    )

    s2_image, s2_date = (
        find_sentinel2_image(
            latitude,
            longitude,
            requested_date
        )
    )

    s1_image, s1_date = (
        find_sentinel1_image(
            latitude,
            longitude,
            requested_date
        )
    )

    s2_data = download_numpy_image(
        s2_image,
        [
            "B2",
            "B3",
            "B4",
            "B8",
            "B11",
            "B12"
        ],
        patch,
        scale=10
    )

    s1_data = download_numpy_image(
        s1_image,
        [
            "VV",
            "VH"
        ],
        patch,
        scale=10
    )

    return (
        s2_data,
        s1_data,
        s2_date,
        s1_date
    )


# ============================================================
# EXTRACT EXACT 26 MODEL FEATURES
# ============================================================

def extract_model3_features(
    s2_data,
    s1_data
):

    features = []

    # --------------------------------------------------------
    # Sentinel-2
    # --------------------------------------------------------

    B02 = np.asarray(s2_data["B2"], dtype=np.float32)
    B03 = np.asarray(s2_data["B3"], dtype=np.float32)
    B04 = np.asarray(s2_data["B4"], dtype=np.float32)
    B08 = np.asarray(s2_data["B8"], dtype=np.float32)
    B11 = np.asarray(s2_data["B11"], dtype=np.float32)
    B12 = np.asarray(s2_data["B12"], dtype=np.float32)

    for band in [
        B02,
        B03,
        B04,
        B08,
        B11,
        B12
    ]:

        mean_value, std_value = (
            image_stats(band)
        )

        features.extend(
            [
                mean_value,
                std_value
            ]
        )

    # --------------------------------------------------------
    # Spectral indices
    # --------------------------------------------------------

    NDVI = normalized_difference(
        B08,
        B04
    )

    NDWI = normalized_difference(
        B03,
        B08
    )

    NDBI = normalized_difference(
        B11,
        B08
    )

    for index_array in [
        NDVI,
        NDWI,
        NDBI
    ]:

        mean_value, std_value = (
            image_stats(index_array)
        )

        features.extend(
            [
                mean_value,
                std_value
            ]
        )

    # --------------------------------------------------------
    # Sentinel-1
    # --------------------------------------------------------

    VV = np.asarray(
        s1_data["VV"],
        dtype=np.float32
    )

    VH = np.asarray(
        s1_data["VH"],
        dtype=np.float32
    )

    mean_value, std_value = (
        image_stats(VV)
    )

    features.extend(
        [
            mean_value,
            std_value
        ]
    )

    mean_value, std_value = (
        image_stats(VH)
    )

    features.extend(
        [
            mean_value,
            std_value
        ]
    )

    # --------------------------------------------------------
    # VV - VH
    # --------------------------------------------------------

    VV_minus_VH = VV - VH

    mean_value, std_value = (
        image_stats(
            VV_minus_VH
        )
    )

    features.extend(
        [
            mean_value,
            std_value
        ]
    )

    # --------------------------------------------------------
    # VV / VH ratio
    #
    # SAR values are dB.
    # Convert dB -> linear power first.
    # --------------------------------------------------------

    VV_linear = np.power(
        10.0,
        VV / 10.0
    )

    VH_linear = np.power(
        10.0,
        VH / 10.0
    )

    SAR_ratio = np.full_like(
        VV_linear,
        np.nan,
        dtype=np.float32
    )

    valid = (
        np.isfinite(VV_linear)
        &
        np.isfinite(VH_linear)
        &
        (VH_linear > EPS)
    )

    SAR_ratio[valid] = (
        VV_linear[valid]
        /
        VH_linear[valid]
    )

    mean_value, std_value = (
        image_stats(
            SAR_ratio
        )
    )

    features.extend(
        [
            mean_value,
            std_value
        ]
    )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if len(features) != 26:
        raise RuntimeError(
            f"Expected 26 features, got {len(features)}"
        )

    return features


# ============================================================
# MODEL PREDICTION
# ============================================================

def run_model_prediction(
    features
):

    feature_array = np.asarray(
        features,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    probabilities = []

    for model in models:

        probability = float(
            model.predict_proba(
                feature_array
            )[0][1]
        )

        probabilities.append(
            probability
        )

    ranked = sorted(
        zip(
            CLASS_NAMES,
            probabilities
        ),
        key=lambda x: x[1],
        reverse=True
    )

    predictions = [
        CLASS_NAMES[i]
        for i, probability
        in enumerate(probabilities)
        if probability >= 0.5
    ]

    top5 = [
        {
            "class": name,
            "probability": round(
                float(probability),
                6
            )
        }
        for name, probability
        in ranked[:5]
    ]

    return {
        "predicted_classes": predictions,
        "confidence": round(
            float(max(probabilities)),
            6
        ),
        "top5": top5,
        "probabilities": {
            CLASS_NAMES[i]:
                round(
                    float(probabilities[i]),
                    6
                )
            for i in range(19)
        }
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "SatQuery Model-3",
        "status": "running",
        "version": "2.0.0",
        "models": 19,
        "features": 26,
        "gee_ready": GEE_READY
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": (
            "healthy"
            if GEE_READY
            else "degraded"
        ),
        "model_loaded": (
            len(models) == 19
        ),
        "number_of_models": len(models),
        "number_of_features": 26,
        "gee_ready": GEE_READY,
        "gee_project": GEE_PROJECT_ID
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    if not GEE_READY:

        raise HTTPException(
            status_code=503,
            detail=(
                "Google Earth Engine is not initialized."
            )
        )

    try:

        requested_date = request.date

        # ----------------------------------------------------
        # Get satellite data
        # ----------------------------------------------------

        (
            s2_data,
            s1_data,
            s2_date,
            s1_date
        ) = get_satellite_arrays(
            request.latitude,
            request.longitude,
            requested_date
        )

        # ----------------------------------------------------
        # Extract 26 features
        # ----------------------------------------------------

        features = extract_model3_features(
            s2_data,
            s1_data
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        result = run_model_prediction(
            features
        )

        return {
            "location": {
                "latitude": request.latitude,
                "longitude": request.longitude
            },

            "requested_date": (
                requested_date.isoformat()
            ),

            "sentinel2_date": (
                s2_date.isoformat()
            ),

            "sentinel1_date": (
                s1_date.isoformat()
            ),

            "primary_class": (
                result["top5"][0]["class"]
            ),

            "primary_probability": (
                result["top5"][0]["probability"]
            ),

            "predicted_classes": (
                result["predicted_classes"]
            ),

            "confidence": (
                result["confidence"]
            ),

            "top5": result["top5"],

            "probabilities": (
                result["probabilities"]
            ),

            "num_features": 26,

            "feature_names": FEATURE_NAMES
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )