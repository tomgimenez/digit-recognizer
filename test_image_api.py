import requests

with open("test_digit.png", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/predict-image",
        files={"file": f}
    )

print(response.json())