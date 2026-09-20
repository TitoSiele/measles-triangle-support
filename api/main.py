"""
FastAPI backend for the measles detection system.
Framed as a CLINICAL DECISION-SUPPORT / TRIAGE AID, not a diagnostic tool.
Reports confidence levels and always defers final judgment to a clinician.

v2: symptom inputs may be UNKNOWN (null / omitted). The model knows the difference
between "no" and "not recorded", and when too many key findings are unknown the
API declines to score and recommends clinical assessment instead.
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="Measles Triage Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],  # TODO: add Render URL once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = "This is decision SUPPORT, not a diagnosis. A qualified clinician must confirm any result."

BUNDLE_PATH = "models/symptom_bundle_v2.joblib"
bundle = None
if os.path.exists(BUNDLE_PATH):
    bundle = joblib.load(BUNDLE_PATH)
    print("Symptom model (v2, unknown-aware) loaded.")
else:
    print("WARNING: models/symptom_bundle_v2.joblib not found. Run src/train_symptom_model_v2.py.")

IMAGE_MODEL_PATH = "models/image_classifier_combined/final_model.keras"
IMAGE_CLASS_NAMES_PATH = "models/image_classifier_combined/class_names.txt"

image_model = None
image_class_names = None
if os.path.exists(IMAGE_MODEL_PATH) and os.path.exists(IMAGE_CLASS_NAMES_PATH):
    import tensorflow as tf
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    with open(IMAGE_CLASS_NAMES_PATH) as f:
        image_class_names = [line.strip() for line in f if line.strip()]
    print("Image model loaded.")
else:
    print("WARNING: Image model files not found. /predict/image will be unavailable.")


class SymptomInput(BaseModel):
    # Every field is optional: null / omitted means UNKNOWN (not recorded), not "no".
    fever_temp_c: Optional[float] = None
    fever_duration_days: Optional[int] = None
    cough: Optional[int] = None
    coryza: Optional[int] = None
    conjunctivitis: Optional[int] = None
    koplik_spots: Optional[int] = None
    rash_present: Optional[int] = None
    rash_onset_day: Optional[int] = None
    cephalocaudal_spread: Optional[int] = None
    sore_throat: Optional[int] = None
    itchy_rash: Optional[int] = None
    lymphadenopathy: Optional[int] = None
    age_years: Optional[float] = None
    vaccinated: Optional[int] = None


def confidence_from_margin(probs: np.ndarray) -> str:
    """
    Confidence is based on how far the top prediction is ahead of the
    second-best guess, not on the raw probability alone. A model can be
    'very sure' between two similar-looking diseases and still be wrong,
    so a close margin should always read as lower confidence.
    """
    sorted_probs = np.sort(probs)[::-1]
    top, second = sorted_probs[0], sorted_probs[1] if len(sorted_probs) > 1 else 0
    margin = top - second
    if margin >= 0.5:
        return "High"
    elif margin >= 0.2:
        return "Moderate"
    else:
        return "Low"


def run_symptom_model(data: SymptomInput) -> dict:
    """Score one record with the unknown-aware model, or decline if key fields are unknown."""
    b = bundle
    vals, flags, unknown = {}, {}, []
    for c in b["original_cols"]:
        v = getattr(data, c, None)
        is_unknown = v is None
        if is_unknown:
            unknown.append(c)
            vals[c] = b["medians"][c] if c in b["numeric_cols"] else 0.0
        else:
            vals[c] = float(v)
        flags[c + "_missing"] = 1.0 if is_unknown else 0.0

    unknown_key = [c for c in b["key_fields"] if c in unknown]
    if len(unknown_key) >= b["abstain_if_key_missing"]:
        return {
            "insufficient_information": True,
            "unknown_fields": unknown,
            "unknown_key_fields": unknown_key,
            "predicted_class": None,
            "measles_probability": None,
            "confidence": "Insufficient",
            "all_probabilities": {},
            "message": "Insufficient information: key findings are unknown ("
                       + ", ".join(unknown_key) + "). Recommend clinical assessment.",
        }

    X = pd.DataFrame([{**vals, **flags}])[b["feature_names"]]
    probs = b["model"].predict_proba(X)[0]
    classes = b["class_names"]
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    return {
        "insufficient_information": False,
        "unknown_fields": unknown,
        "unknown_key_fields": unknown_key,
        "predicted_class": classes[int(np.argmax(probs))],
        "measles_probability": round(prob_dict.get("Measles", 0.0), 4),
        "confidence": confidence_from_margin(probs),
        "all_probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
        "message": None,
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Measles Triage Support API is running",
        "disclaimer": "This tool provides decision SUPPORT only. It does not diagnose. All results require clinical confirmation.",
    }


@app.post("/predict/symptoms")
def predict_symptoms(data: SymptomInput):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Symptom model not loaded.")
    return run_symptom_model(data)


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded.")

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    probs = image_model.predict(arr, verbose=0)[0]
    prob_dict = {cls: float(p) for cls, p in zip(image_class_names, probs)}

    predicted_idx = int(np.argmax(probs))
    predicted_class = image_class_names[predicted_idx]
    measles_prob = prob_dict.get("Measles", 0.0)
    confidence = confidence_from_margin(probs)

    return {
        "predicted_class": predicted_class,
        "measles_probability": round(measles_prob, 4),
        "confidence": confidence,
        "all_probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
        "note": "Image model was validated on a very small measles sample (6 test images). Treat with caution.",
    }


@app.post("/predict/combined")
async def predict_combined(
    fever_temp_c: Optional[float] = None, fever_duration_days: Optional[int] = None,
    cough: Optional[int] = None, coryza: Optional[int] = None,
    conjunctivitis: Optional[int] = None, koplik_spots: Optional[int] = None,
    rash_present: Optional[int] = None, rash_onset_day: Optional[int] = None,
    cephalocaudal_spread: Optional[int] = None, sore_throat: Optional[int] = None,
    itchy_rash: Optional[int] = None, lymphadenopathy: Optional[int] = None,
    age_years: Optional[float] = None, vaccinated: Optional[int] = None,
    file: UploadFile = File(None),
):
    """
    Combines symptom + image signals into a triage flag. Always phrased as
    a recommendation to seek clinical review, never as a diagnosis, and
    explicitly reports when the two signals disagree (a strong reason for
    a clinician to look closer, not for the tool to average the disagreement away).
    Omitted symptom fields are treated as UNKNOWN. If too many key fields are
    unknown, no score is produced and clinical assessment is recommended.
    """
    symptom_measles_prob = None
    image_measles_prob = None
    symptom_confidence = None
    image_confidence = None
    insufficient = False
    unknown_key = []

    if bundle is not None:
        data = SymptomInput(
            fever_temp_c=fever_temp_c, fever_duration_days=fever_duration_days,
            cough=cough, coryza=coryza, conjunctivitis=conjunctivitis,
            koplik_spots=koplik_spots, rash_present=rash_present,
            rash_onset_day=rash_onset_day, cephalocaudal_spread=cephalocaudal_spread,
            sore_throat=sore_throat, itchy_rash=itchy_rash,
            lymphadenopathy=lymphadenopathy, age_years=age_years, vaccinated=vaccinated,
        )
        symptom_result = run_symptom_model(data)
        unknown_key = symptom_result["unknown_key_fields"]
        if symptom_result["insufficient_information"]:
            insufficient = True
        else:
            symptom_measles_prob = symptom_result["measles_probability"]
            symptom_confidence = symptom_result["confidence"]

    if file is not None and image_model is not None:
        image_result = await predict_image(file)
        image_measles_prob = image_result["measles_probability"]
        image_confidence = image_result["confidence"]

    if insufficient:
        # The weak image model must not drive a triage flag on its own.
        return {
            "symptom_measles_probability": None,
            "image_measles_probability": image_measles_prob,
            "combined_measles_probability": None,
            "confidence": "Insufficient",
            "signals_disagree": False,
            "insufficient_information": True,
            "unknown_key_fields": unknown_key,
            "triage_flag": "Insufficient information: key findings are unknown ("
                           + ", ".join(unknown_key)
                           + "). Recommend clinical assessment rather than relying on this tool.",
            "disclaimer": DISCLAIMER,
        }

    if symptom_measles_prob is None and image_measles_prob is None:
        raise HTTPException(status_code=400, detail="Provide symptoms and/or an image.")

    if symptom_measles_prob is not None and image_measles_prob is not None:
        combined_prob = 0.6 * symptom_measles_prob + 0.4 * image_measles_prob
        disagreement = abs(symptom_measles_prob - image_measles_prob) > 0.4
    else:
        combined_prob = symptom_measles_prob if symptom_measles_prob is not None else image_measles_prob
        disagreement = False

    confidences_used = [c for c in [symptom_confidence, image_confidence] if c is not None]
    confidence_rank = {"Low": 0, "Moderate": 1, "High": 2}
    overall_confidence = min(confidences_used, key=lambda c: confidence_rank[c]) if confidences_used else "Low"
    if disagreement:
        overall_confidence = "Low"
    if disagreement:
        flag = "Conflicting signals between symptoms and image — recommend clinical review"
    elif combined_prob >= 0.6:
        flag = "Findings consistent with measles — recommend clinical review and confirmatory testing"
    elif combined_prob >= 0.3:
        flag = "Some measles-consistent features present — clinical judgment advised"
    else:
        flag = "Low measles-consistent features — continue routine clinical assessment"

    return {
        "symptom_measles_probability": symptom_measles_prob,
        "image_measles_probability": image_measles_prob,
        "combined_measles_probability": round(combined_prob, 4),
        "confidence": overall_confidence,
        "signals_disagree": disagreement,
        "insufficient_information": False,
        "unknown_key_fields": unknown_key,
        "triage_flag": flag,
        "disclaimer": DISCLAIMER,
    }


if os.path.isdir("frontend"):
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")
