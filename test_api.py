import requests
import json

BASE_URL = "http://127.0.0.1:5000"

valid_record = {
    "delivery_days": 12,
    "delivery_vs_estimated": 3,
    "price": 149.99,
    "freight_value": 25.50,
    "product_category": "electronics",
    "seller_state": "SP",
    "payment_type": "credit_card"
}

valid_batch = [
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
    },
    {
        "delivery_days": 20,
        "delivery_vs_estimated": 5,
        "price": 45.00,
        "freight_value": 18.00,
        "product_category": "electronics",
        "seller_state": "MG",
        "payment_type": "voucher"
    },
    {
        "delivery_days": 5,
        "delivery_vs_estimated": -2,
        "price": 220.00,
        "freight_value": 30.00,
        "product_category": "electronics",
        "seller_state": "SP",
        "payment_type": "debit_card"
    },
    {
        "delivery_days": 15,
        "delivery_vs_estimated": 0,
        "price": 60.50,
        "freight_value": 12.75,
        "product_category": "electronics",
        "seller_state": "BA",
        "payment_type": "credit_card"
    }
]

def print_result(test_name, passed, response=None):
    status = "PASS" if passed else "FAIL"
    print(f"{test_name}: {status}")
    if response is not None:
        try:
            print("Response:", json.dumps(response.json(), indent=2))
        except Exception:
            print("Response text:", response.text)
    print("-" * 60)

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    passed = (
        response.status_code == 200 and
        "status" in response.json() and
        "model" in response.json()
    )
    print_result("Test 1 - GET /health", passed, response)

def test_single_prediction():
    response = requests.post(f"{BASE_URL}/predict", json=valid_record)
    data = response.json()
    passed = (
        response.status_code == 200 and
        "prediction" in data and
        "probability" in data and
        "label" in data
    )
    print_result("Test 2 - POST /predict valid request", passed, response)

def test_batch_prediction():
    response = requests.post(f"{BASE_URL}/predict/batch", json=valid_batch)
    data = response.json()
    passed = (
        response.status_code == 200 and
        "predictions" in data and
        len(data["predictions"]) == 5
    )
    print_result("Test 3 - POST /predict/batch valid batch", passed, response)

def test_missing_field():
    bad_record = valid_record.copy()
    del bad_record["price"]

    response = requests.post(f"{BASE_URL}/predict", json=bad_record)
    data = response.json()
    passed = (
        response.status_code == 400 and
        "error" in data
    )
    print_result("Test 4 - POST /predict missing required field", passed, response)

def test_invalid_type():
    bad_record = valid_record.copy()
    bad_record["price"] = "not_a_number"

    response = requests.post(f"{BASE_URL}/predict", json=bad_record)
    data = response.json()
    passed = (
        response.status_code == 400 and
        "error" in data
    )
    print_result("Test 5 - POST /predict invalid type", passed, response)

if __name__ == "__main__":
    print("Running local API tests...\n")
    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_missing_field()
    test_invalid_type()
    print("Finished running tests.")