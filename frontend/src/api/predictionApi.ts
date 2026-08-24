const API_URL= import.meta.env.VITE_API_URL;

export interface PredictionResponse {
  prediction: number;
  probabilities: Record<string, number>;
}

export const predictDigit = async (imageBlob: Blob): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append('file', imageBlob, 'digit.png');

  const response = await fetch(`${API_URL}/predict-image`, {
    method: 'POST',
    body: formData
  })

  if (!response.ok)
    throw new Error(`Prediction failed: ${response.statusText}`)

  console.log(response)

  return response.json();
}