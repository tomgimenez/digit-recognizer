from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import joblib
import numpy as np
import io

app = FastAPI(title='Digit Recognizer API')

model = joblib.load('../digit_model.joblib')

class DigitInput(BaseModel):
  pixels: list[float]

@app.get('/')
def root():
  return { 'status': 'ok', 'message': 'Digit Recognizer API' }

@app.post('/predict')
def predict(data: DigitInput):
  if len(data.pixels) != 784:
    return { 'error': f'784 pixels waited, arraived {len(data.pixels)}' }

  X = np.array(data.pixels).reshape(1, -1)
  prediction = model.predict(X)[0]
  probabilities = model.predict_proba(X)[0]

  return {
    'prediction': int(prediction),
    'probabilities': { str(i): round(float(p), 4) for i, p in enumerate(probabilities) }
  }

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # 1. Abrir imagen y convertir a escala de grises
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # 2. Convertir a array de numpy
    pixel_array = np.array(image).astype(np.float32)

    # 3. Invertir colores: MNIST espera fondo negro (0) y trazo blanco (255)
    pixel_array = 255.0 - pixel_array

    # 4. Eliminar ruido de fondo (valores muy bajos se van a 0)
    pixel_array[pixel_array < 30] = 0

    # 5. Encontrar el bounding box del dígito (donde hay píxeles no-cero)
    rows = np.any(pixel_array > 0, axis=1)
    cols = np.any(pixel_array > 0, axis=0)

    if not rows.any() or not cols.any():
      # No se dibujó nada
      return np.zeros((1, 784))

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    digit = pixel_array[rmin:rmax+1, cmin:cmax+1]

    # 6. Redimensionar el dígito recortado a 20x20 preservando proporción
    #    (dejamos margen de 4px alrededor, como en MNIST original)
    digit_image = Image.fromarray(digit.astype(np.uint8))
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(w * (20 / h)))
    else:
        new_w = 20
        new_h = max(1, int(h * (20 / w)))
    digit_image = digit_image.resize((new_w, new_h), Image.LANCZOS)

    # 7. Pegar el dígito centrado en un lienzo de 28x28
    canvas = Image.new("L", (28, 28), color=0)
    upper_left = ((28 - new_w) // 2, (28 - new_h) // 2)
    canvas.paste(digit_image, upper_left)

    pixel_array = np.array(canvas).astype(np.float32)

    # 8. Ajuste fino: centrado por centro de masa (como en MNIST real)
    total = pixel_array.sum()
    if total > 0:
      y_idx, x_idx = np.indices(pixel_array.shape)
      cy = (y_idx * pixel_array).sum() / total
      cx = (x_idx * pixel_array).sum() / total
      shift_y = int(round(14 - cy))
      shift_x = int(round(14 - cx))

      shifted = np.zeros_like(pixel_array)
      src_y0, src_y1 = max(0, -shift_y), min(28, 28 - shift_y)
      src_x0, src_x1 = max(0, -shift_x), min(28, 28 - shift_x)
      dst_y0, dst_y1 = max(0, shift_y), max(0, shift_y) + (src_y1 - src_y0)
      dst_x0, dst_x1 = max(0, shift_x), max(0, shift_x) + (src_x1 - src_x0)
      shifted[dst_y0:dst_y1, dst_x0:dst_x1] = pixel_array[src_y0:src_y1, src_x0:src_x1]
      pixel_array = shifted

    # 9. Estirar el contraste para que el máximo llegue a 255
    max_val = pixel_array.max()
    if max_val > 0:
      pixel_array = pixel_array * (255.0 / max_val)

    # 10. Normalizar (0 a 1), igual que en entrenamiento
    pixel_array = pixel_array / 255.0

    # 11. Aplanar a un vector de 784
    return pixel_array.reshape(1, -1)

@app.post("/predict-image")
async def predict_image(file: Annotated[UploadFile, File()]):
    image_bytes = await file.read()
    X = preprocess_image(image_bytes)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return {
        "prediction": int(prediction),
        "probabilities": {str(i): round(float(p), 4) for i, p in enumerate(probabilities)}
    }