from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ============================================================
# PATHS + MODEL LOADING
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")

model = None
preprocessor = None

try:
    print("Looking for model at:", MODEL_PATH)
    print("Looking for preprocessor at:", PREPROCESSOR_PATH)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Model loaded successfully.")
    print("Preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")


# ============================================================
# CONFIG
# ============================================================
REQUIRED_FIELDS = [
    "delivery_days",
    "delivery_vs_estimated",
    "price",
    "freight_value",
    "product_category",
    "seller_state",
    "payment_type"
]

NUMERIC_FIELDS = [
    "delivery_days",
    "delivery_vs_estimated",
    "price",
    "freight_value"
]

CATEGORICAL_FIELDS = [
    "product_category",
    "seller_state",
    "payment_type"
]

VALID_PAYMENT_TYPES = {
    "credit_card",
    "boleto",
    "voucher",
    "debit_card",
    "not_defined"
}


# ============================================================
# VALIDATION
# ============================================================
def validate_single_record(data):
    errors = {}

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        errors["missing_fields"] = missing
        return errors

    for field in NUMERIC_FIELDS:
        value = data.get(field)
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors[field] = "must be a numeric value"
            continue

        if field in ["price", "freight_value"] and value < 0:
            errors[field] = "must be a non-negative number"

    for field in CATEGORICAL_FIELDS:
        value = data.get(field)
        if value is None or str(value).strip() == "":
            errors[field] = "must be a non-empty string"

    payment_type = str(data.get("payment_type", "")).strip()
    if payment_type not in VALID_PAYMENT_TYPES:
        errors["payment_type"] = (
            f"unrecognized value '{payment_type}'. "
            f"Valid values: {sorted(VALID_PAYMENT_TYPES)}"
        )

    return errors


def make_prediction(df):
    X_processed = preprocessor.transform(df)
    preds = model.predict(X_processed)
    probas = model.predict_proba(X_processed)[:, 1]
    return preds, probas


# ============================================================
# ROUTES
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    if model is None or preprocessor is None:
        return jsonify({
            "status": "error",
            "message": "model or preprocessor not loaded"
        }), 503

    return jsonify({
        "status": "healthy",
        "model": "loaded"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or preprocessor is None:
        return jsonify({"error": "model not available"}), 503

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON provided"}), 400

    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    errors = validate_single_record(data)
    if errors:
        return jsonify({
            "error": "Invalid input",
            "details": errors
        }), 400

    df = pd.DataFrame([data])[REQUIRED_FIELDS].copy()

    for col in NUMERIC_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    pred, proba = make_prediction(df)

    prediction = int(pred[0])
    probability = float(proba[0])

    return jsonify({
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "positive" if prediction == 1 else "negative"
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if model is None or preprocessor is None:
        return jsonify({"error": "model not available"}), 503

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON provided"}), 400

    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400

    if len(data) > 100:
        return jsonify({"error": "Max 100 records per batch"}), 400

    batch_errors = {}

    for i, record in enumerate(data):
        if not isinstance(record, dict):
            batch_errors[f"record_{i}"] = "must be a JSON object"
            continue

        record_errors = validate_single_record(record)
        if record_errors:
            batch_errors[f"record_{i}"] = record_errors

    if batch_errors:
        return jsonify({
            "error": "Invalid batch input",
            "details": batch_errors
        }), 400

    df = pd.DataFrame(data)[REQUIRED_FIELDS].copy()

    for col in NUMERIC_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    preds, probas = make_prediction(df)

    results = []
    for pred, proba in zip(preds, probas):
        pred_int = int(pred)
        results.append({
            "prediction": pred_int,
            "probability": round(float(proba), 4),
            "label": "positive" if pred_int == 1 else "negative"
        })

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)