# HW4 — From Model to Production: End-to-End MLOps Pipeline

## Nicholas Condos  
**DATA 6545 — Data Science & MLOps**

---

## Project Overview
This repository documents my HW4 project for **DATA 6545**, which extends the Olist customer satisfaction prediction problem into a broader MLOps workflow. The project moves beyond training a single machine learning model and instead addresses several practical production-oriented tasks, including foundation model evaluation, experiment tracking, model serving, monitoring, and deployment preparation.

The business objective is to predict whether a customer review will be positive, where:

- `is_positive_review = 1` if `review_score >= 4`
- `is_positive_review = 0` otherwise

The structured prediction workflow uses order and delivery features to estimate likely customer satisfaction before a written review is available. In parallel, a foundation model is used to analyze review text directly after the review has been written. Together, these approaches highlight the difference between **proactive structured prediction** and **reactive text-based sentiment analysis**.

---

## Repository Structure

```text
HW4-MLOps/
├── model/
│   ├── model.pkl
│   └── preprocessor.pkl
├── screenshots/
├── full_notebook/
│   └── HW4_full_workflow_notebook.ipynb
├── .dockerignore
├── Dockerfile
├── README.md
├── app.py
├── part1_foundation_model_clean.ipynb
├── part2_mlflow_work.ipynb
├── part5_monitoring.ipynb
├── requirements.txt
└── test_api.py

Notebook Organization

This repository includes both cleaned section-specific notebooks and one full workflow notebook:

part1_foundation_model_clean.ipynb
Contains only Part 1, focused on foundation model exploration.
part2_mlflow_work.ipynb
Contains only Part 2, focused on MLflow experiment tracking and model registration.
part5_monitoring.ipynb
Contains only Part 5, focused on monitoring, drift detection, and retraining recommendation.
full_notebook/HW4_full_workflow_notebook.ipynb
Contains the full combined working notebook used during development.

This structure is intentional so that each major assignment section can be reviewed independently while still preserving one consolidated workflow notebook for reference.

## Part 1 — Foundation Model Exploration

In Part 1, I applied the Hugging Face multilingual sentiment model:

nlptown/bert-base-multilingual-uncased-sentiment

to a 500-record sample of Olist review text with non-empty review_comment_message values. The model predicts 1–5 stars from the review text, and those predictions were mapped into the binary customer satisfaction target used throughout the project:

4–5 stars → positive (1)
1–3 stars → negative (0)

I then compared the foundation model against my HW2 custom structured model on the same 500 records using:

Accuracy
Precision
Recall
F1 Score
Key purpose of Part 1

This section was designed to compare two very different modeling strategies:

a foundation model, which is fast to apply and directly uses written text
a custom structured model, which is more operationally useful when predictions are needed before review text exists
Output

See:

part1_foundation_model_clean.ipynb
Part 2 — Experiment Tracking with MLflow

In Part 2, I used MLflow to track and compare multiple runs for the Olist satisfaction problem.

Experiment name
olist-satisfaction
Models logged
Random Forest
Gradient Boosting

For each run, I logged:

model type and key hyperparameters
evaluation metrics
trained model artifact

I also used the MLflow registry to register the tracked model and assign the selected version as the production version.

Metrics tracked
Accuracy
Precision
Recall
F1
ROC-AUC
Output

See:

part2_mlflow_work.ipynb
Supporting screenshots

Stored in:

screenshots/HW4_mlflow_experiments.png
screenshots/HW4_mlflow_registry.png
Part 3 — Model Serving API

In Part 3, I built a Flask-based prediction API for the deployed model.

API files
app.py
test_api.py
requirements.txt
Required endpoints
GET /health
POST /predict
POST /predict/batch
Validation included

The API validates:

missing required fields
invalid numeric types
negative numeric values where not allowed
invalid or empty categorical values
unsupported payment_type values
Local testing

The local API was tested successfully using the required 5 checks:

health endpoint
valid single prediction
valid batch prediction
missing required field returns 400
invalid type returns 400
Supporting screenshot

Stored in:

screenshots/HW4_local_api_tests_pass.png
Part 5 — Monitoring and Drift Detection

Part 5 simulates a production monitoring workflow for the deployed satisfaction model.

Monitoring design

I simulated 6 months of production-like data:

Months 1–3: stable period with small random variation
Months 4–6: progressively drifted period

The drift simulation intentionally changes:

delivery times
freight costs
product category mix
label quality
Drift detection methods

I used:

Population Stability Index (PSI) for numeric features
Kolmogorov-Smirnov (KS) tests
monthly model performance tracking
Performance monitoring

For each month, I tracked:

Accuracy
F1
AUC
Dashboard

The notebook includes a 3-panel monitoring dashboard showing:

PSI heatmap
monthly performance trend
alert summary table
Recommendation

The notebook concludes with a retraining recommendation based on both:

drift thresholds
model performance degradation
Output

See:

part5_monitoring.ipynb
Model Artifacts

The repository includes the serialized deployed model artifacts used by the API:

model/model.pkl
model/preprocessor.pkl

These files are used by the Flask application at startup to transform incoming inputs and generate predictions.

Input Schema for the API

The API expects the following fields:

Field	Type	Description
delivery_days	numeric	Number of days between purchase and delivery
delivery_vs_estimated	numeric	Difference between actual and estimated delivery date
price	numeric	Total item price
freight_value	numeric	Shipping/freight cost
product_category	string	Product category
seller_state	string	Seller’s state
payment_type	string	Payment method

API Usage
1. Health Check

Endpoint

GET /health

Example response

{
  "status": "healthy",
  "model": "loaded"
}
2. Single Prediction

Endpoint

POST /predict

Example request

{
  "delivery_days": 12,
  "delivery_vs_estimated": 3,
  "price": 149.99,
  "freight_value": 25.50,
  "product_category": "electronics",
  "seller_state": "SP",
  "payment_type": "credit_card"
}

Example response

{
  "prediction": 1,
  "probability": 0.5766,
  "label": "positive"
}
3. Batch Prediction

Endpoint

POST /predict/batch

Example request

[
  {
    "delivery_days": 12,
    "delivery_vs_estimated": 3,
    "price": 149.99,
    "freight_value": 25.50,
    "product_category": "electronics",
    "seller_state": "SP",
    "payment_type": "credit_card"
  },
  {
    "delivery_days": 8,
    "delivery_vs_estimated": -1,
    "price": 89.90,
    "freight_value": 14.20,
    "product_category": "electronics",
    "seller_state": "RJ",
    "payment_type": "boleto"
  }
]
Local Setup (Without Docker)
1. Create and activate environment
conda create -n hw4api python=3.10 -y
conda activate hw4api
2. Install dependencies
pip install -r requirements.txt
3. Run the API
python app.py
4. Run tests
python test_api.py
Docker Setup

Docker support is included in the repository through:

Dockerfile
.dockerignore

Intended commands:

docker build -t hw4-api .
docker run -p 5000:5000 hw4-api

This portion is included as part of the HW4 production/deployment workflow.

Screenshots Included

The repository currently includes screenshots for:

MLflow experiments page
MLflow model registry page
local API tests passing

Additional Docker and deployment screenshots can be added once the remaining deployment work is finalized.

Limitations
The structured model depends on the quality and availability of order-level features.
The foundation model is reactive because it depends on review text that only exists after the customer experience.
Monitoring results are based on simulated production drift rather than real live production traffic.
Docker and deployment behavior can vary depending on machine-level permissions, networking, and certificate settings.
Current Status

Completed:

Part 1 — Foundation Model Exploration
Part 2 — MLflow Experiment Tracking
Part 3 — Local Model Serving API and Testing
Part 5 — Monitoring and Drift Detection

Remaining / finalization:

Docker validation
cloud deployment
final screenshot PDF assembly
Repository Link

GitHub repository:

nicholascondos/HW4-MLOps

