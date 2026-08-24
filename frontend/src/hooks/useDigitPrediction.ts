import { useState } from "react"
import { predictDigit, type PredictionResponse } from "../api/predictionApi";

export const useDigitPrediction = () => {
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null);

  const predict = async (canvas: HTMLCanvasElement) => {

    setScanning(true);
    setError(null);

    try {
      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, 'image/png')
      })
      if (!blob)
        throw new Error('Error capturing canvas')
      
      const start = performance.now()
      const response = await predictDigit(blob);
      const elapsed = performance.now() - start;

      setResult(response);
      setLatencyMs(Math.round(elapsed))
    } catch (error) {
      setError(error instanceof Error ? error.message : "Unknown Error");
    } finally {
      setScanning(false);
    }
  }

  const reset = () => {
    setResult(null);
    setError(null);
    setLatencyMs(null)
  }

  return { scanning, result, error, predict, reset, latencyMs }

}