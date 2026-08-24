import { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, Cpu, Eraser, Gauge, ScanLine, Sparkles, Zap } from 'lucide-react'
import { useDigitPrediction } from './hooks/useDigitPrediction'

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawingRef = useRef(false)
  const [hasInk, setHasInk] = useState(false)

  const { scanning, result, predict: runPrediction, reset, latencyMs } = useDigitPrediction();

  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const ratio = Math.max(window.devicePixelRatio || 1, 1)
    canvas.width = rect.width * ratio
    canvas.height = rect.height * ratio
    const context = canvas.getContext('2d')
    if (context) {

      context.fillStyle = '#000000'
      context.fillRect(0, 0, canvas.width, canvas.height)

      context.scale(ratio, ratio)
      context.lineCap = 'round'
      context.lineJoin = 'round'
      context.lineWidth = 18
      context.strokeStyle = '#d9fbff'
      context.shadowColor = '#55d6e8'
      context.shadowBlur = 14
    }
  }, [])

  useEffect(() => {
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)
    return () => window.removeEventListener('resize', resizeCanvas)
  }, [resizeCanvas])

  const point = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    return { x: event.clientX - rect.left, y: event.clientY - rect.top }
  }

  const startDrawing = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    const position = point(event)
    if (!context || !position) return
    canvas?.setPointerCapture(event.pointerId)
    drawingRef.current = true
    context.beginPath()
    context.moveTo(position.x, position.y)
    setHasInk(true)
  }

  const draw = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return
    const context = canvasRef.current?.getContext('2d')
    const position = point(event)
    if (!context || !position) return
    context.lineTo(position.x, position.y)
    context.stroke()
  }

  const stopDrawing = () => {
    drawingRef.current = false
    canvasRef.current?.getContext('2d')?.closePath()
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    const context = canvas?.getContext('2d')
    if (!canvas || !context) return;
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.fillStyle = '#000000'
    context.fillRect(0, 0, canvas.width, canvas.height)
    setHasInk(false)
    reset();
  }

  const predict = () => {
    const canvas = canvasRef.current;
    if (!hasInk || scanning || !canvas) return;
    runPrediction(canvas)
  }

  return (
    <main className="min-h-screen overflow-hidden bg-background text-foreground">
      <div className="scanlines pointer-events-none fixed inset-0 z-20" aria-hidden="true" />
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between border-b border-border/70 px-5 py-5 md:px-8">
        <div className="flex items-center gap-3">
          <div className="brand-mark"><ScanLine size={21} strokeWidth={1.8} /></div>
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-cyan">Neural Vision Lab</p>
            <h1 className="font-mono text-base font-bold tracking-[0.12em] text-foreground">DIGIT SCANNER <span className="text-cyan">// 01</span></h1>
          </div>
        </div>
        <div className="hidden items-center gap-6 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground sm:flex">
          <span className="flex items-center gap-2"><Activity size={13} className="text-cyan" /> System online</span>
          <span className="flex items-center gap-2"><Cpu size={13} /> MNIST / v4.2</span>
        </div>
      </header>

      <div className="mx-auto w-full max-w-6xl px-5 py-10 md:px-8 md:py-14">
        <section className="mb-10 max-w-2xl">
          <div className="mb-4 flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.28em] text-amber"><span className="status-dot" /> Inference chamber ready</div>
          <h2 className="text-balance font-mono text-3xl font-bold leading-tight tracking-[-0.04em] text-foreground md:text-5xl">Draw a digit.<br /><span className="text-cyan">Let the machine see.</span></h2>
          <p className="mt-5 max-w-lg text-pretty font-sans text-sm leading-6 text-muted-foreground">A compact neural network trained on 70,000 handwritten samples. Your gesture enters the chamber; a probability field returns.</p>
        </section>

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(320px,0.75fr)]">
          <section className="panel relative overflow-hidden p-4 md:p-6" aria-labelledby="canvas-title">
            <div className="mb-4 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              <span id="canvas-title" className="flex items-center gap-2 text-foreground"><span className="text-cyan">01</span> Input surface</span><span>28 × 28 px / normalized</span>
            </div>
            <div className="canvas-frame">
              <div className="corner corner-tl" /><div className="corner corner-tr" /><div className="corner corner-bl" /><div className="corner corner-br" />
              <canvas ref={canvasRef} onPointerDown={startDrawing} onPointerMove={draw} onPointerUp={stopDrawing} onPointerCancel={stopDrawing} onPointerLeave={stopDrawing} aria-label="Draw a handwritten digit" />
              {!hasInk && <div className="canvas-hint"><Sparkles size={18} /><span>Draw inside the frame</span></div>}
              <div className="canvas-label">LIVE CAPTURE <span>•</span> POINTER ACTIVE</div>
            </div>
            <div className="mt-5 flex flex-col gap-3 sm:flex-row">
              <button className="action-button primary" onClick={predict} disabled={!hasInk || scanning}><Zap size={16} /> {scanning ? 'Analyzing signal...' : 'Run prediction'}</button>
              <button className="action-button" onClick={clearCanvas}><Eraser size={16} /> Clear surface</button>
            </div>
          </section>

          <section className="panel p-5 md:p-6" aria-labelledby="result-title">
            <div className="mb-8 flex items-center justify-between border-b border-border/70 pb-4 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              <span id="result-title" className="flex items-center gap-2 text-foreground">
                <span className="text-cyan">02</span> Inference result
              </span>
                <Gauge size={15} className="text-cyan" />
            </div>
            <div className={`result-display ${result ? 'result-live' : ''}`}>
              <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Top classification</span>
              <strong>{result ? result.prediction : '—'}</strong>
              <span className="font-mono text-xs text-cyan">{result ? `${(result.probabilities[String(result.prediction)] * 100).toFixed(1)}% confidence` : 'Awaiting input'}</span>
            </div>
            <div className="mt-8">
              <div className="mb-4 flex justify-between font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
                <span>Probability field</span><span>Score</span>
              </div>
              <div className="space-y-4">
                {
                  Object.entries(result?.probabilities ?? {})
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 5)
                    .map(([digit, score], index) => (
                      <div key={digit} className="confidence-row">
                        <span className="w-4 font-mono text-xs text-foreground">{digit}</span>
                        <div className="bar-track">
                          <div className={`bar-fill ${index === 0 ? 'bar-active' : ''}`} style={{ width: `${score * 100}%` }} />
                        </div>
                        <span className="w-12 text-right font-mono text-[11px] text-muted-foreground">{`${(score * 100).toFixed(1)}%`}</span>
                      </div>
                    ))
                }
              </div>
            </div>
            <div className="mt-9 border-t border-border/70 pt-4 font-mono text-[10px] uppercase tracking-[0.17em] text-muted-foreground">
              <div className="flex justify-between">
                <span>Latency</span>
                <span className="text-foreground">{latencyMs !== null ? `${latencyMs} ms` : '—'}</span>
              </div>
              <div className="mt-3 flex justify-between">
                <span>Model state</span>
                <span className="text-amber">{scanning ? 'Computing' : 'Standby'}</span>
              </div>
            </div>
          </section>
        </div>
      </div>
      <footer className="mx-auto flex w-full max-w-6xl items-center justify-between border-t border-border/70 px-5 py-6 font-mono text-[9px] uppercase tracking-[0.2em] text-muted-foreground md:px-8">
        <span>Experimental interface / Developed by Tomas Gimenez</span>
        <span className="hidden sm:block">Local inference protocol <span className="text-cyan">●</span></span>
      </footer>
    </main>
  )
}
