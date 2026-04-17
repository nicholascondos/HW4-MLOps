# HW4 — From Model to Production: End-to-End MLOps Pipeline

## Project Overview
This project takes an Olist customer satisfaction model through key stages of the MLOps lifecycle: foundation model exploration, experiment tracking, model serving, monitoring, and deployment preparation.

The business goal is to predict whether a customer review will be positive (`review_score >= 4`) or not, using structured e-commerce order and delivery features. The project combines both custom machine learning and foundation model experimentation to evaluate practical tradeoffs between proactive and reactive prediction approaches.

## Project Structure

```text
HW4-MLOps/
├── app.py
├── requirements.txt
├── test_api.py
├── Dockerfile
├── README.md
├── part1_foundation_model.ipynb
├── part5_monitoring.ipynb
├── model/
│   ├── model.pkl
│   └── preprocessor.pkl
└── screenshots/
