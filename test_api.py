import joblib
import requests
import random

X_test, y_test = joblib.load('test_data.joblib')

# elegí un índice al azar, o fijo si querés reproducibilidad
index = random.randint(0, len(X_test) - 1)

pixels = X_test[index].tolist()
real_label = int(y_test[index])

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"pixels": pixels}
)

result = response.json()

print(f"Índice: {index}")
print(f"Label real: {real_label}")
print(f"Predicción: {result['prediction']}")
print(f"Probabilidades: {result['probabilities']}")