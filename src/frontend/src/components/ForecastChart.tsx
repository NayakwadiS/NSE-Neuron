import { useEffect, useRef } from 'react'
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type LineData,
  type Time,
} from 'lightweight-charts'
import type { OHLCPoint, ForecastDay } from '../types'

interface Props {
  historical:  OHLCPoint[]
  forecast:    ForecastDay[]
  algoLabel:   string
}

export default function ForecastChart({ historical, forecast, algoLabel }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef     = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    // Destroy any previous chart instance before creating a new one
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    let rafId: number
    let ro: ResizeObserver

    const initChart = () => {
      // If the container still has no width (e.g. tab hidden), retry next frame
      const containerWidth = container.clientWidth
      if (containerWidth === 0) {
        rafId = requestAnimationFrame(initChart)
        return
      }

    // ── Create chart ───────────────────────────────────────────────────────
    const chart = createChart(container, {
      layout: {
        background:  { type: ColorType.Solid, color: '#0f172a' },
        textColor:   '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#1e293b' },
      timeScale: {
        borderColor:     '#1e293b',
        timeVisible:     true,
        secondsVisible:  false,
      },
      width:  containerWidth,
      height: 420,
    })
    chartRef.current = chart

    // ── Historical candlestick series ──────────────────────────────────────
    const candleSeries = chart.addCandlestickSeries({
      upColor:       '#22c55e',
      downColor:     '#ef4444',
      borderVisible: false,
      wickUpColor:   '#22c55e',
      wickDownColor: '#ef4444',
    })

    // Use close as fallback for open so candlesticks always render,
    // even when the API omits the open column for some symbols.
    const candleData: CandlestickData[] = historical
      .filter(d => d.close != null && d.date)
      .map(d => ({
        time:  d.date as Time,
        open:  d.open  ?? d.close,   // fallback: doji-style candle
        high:  d.high  ?? d.close,
        low:   d.low   ?? d.close,
        close: d.close,
      }))
    candleSeries.setData(candleData)

    // ── Forecast close price line ──────────────────────────────────────────
    const forecastSeries = chart.addLineSeries({
      color:     '#818cf8',
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      title:     `${algoLabel} Forecast`,
    })

    const forecastData: LineData[] = forecast.map(d => ({
      time:  d.date as Time,
      value: d.close,
    }))

    // Connect last historical close to first forecast point
    if (candleData.length > 0 && forecastData.length > 0) {
      const bridge: LineData[] = [
        { time: candleData[candleData.length - 1].time, value: candleData[candleData.length - 1].close },
        ...forecastData,
      ]
      forecastSeries.setData(bridge)
    } else {
      forecastSeries.setData(forecastData)
    }

    // ── Signal markers ─────────────────────────────────────────────────────
    const markerMap: Record<string, { color: string; shape: 'arrowUp' | 'arrowDown' | 'circle'; position: 'belowBar' | 'aboveBar' | 'inBar' }> = {
      BUY:  { color: '#22c55e', shape: 'arrowUp',   position: 'belowBar' },
      SELL: { color: '#ef4444', shape: 'arrowDown',  position: 'aboveBar' },
      HOLD: { color: '#f59e0b', shape: 'circle',     position: 'inBar'    },
    }

    const markers = forecast
      .filter(d => d.signal !== null)
      .map(d => {
        const cfg = markerMap[d.signal!.label] ?? markerMap.HOLD
        return {
          time:     d.date as Time,
          ...cfg,
          text:     `${d.signal!.label} ${d.signal!.confidence}%`,
          size:     1.2,
        }
      })

    if (markers.length > 0) {
      forecastSeries.setMarkers(markers)
    }

    // ── Resize observer ────────────────────────────────────────────────────
    ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        chart.applyOptions({ width: entry.contentRect.width })
      }
    })
    ro.observe(container)

    chart.timeScale().fitContent()
    } // end initChart

    rafId = requestAnimationFrame(initChart)

    return () => {
      cancelAnimationFrame(rafId)
      if (ro) ro.disconnect()
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [historical, forecast, algoLabel])

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/80 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-700 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white">
          {forecast.length > 0 ? 'Candlestick + Forecast' : 'Historical Candlestick'}
        </h3>
        <div className="flex items-center gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="w-6 h-0.5 bg-emerald-500 inline-block" /> Historical
          </span>
          {forecast.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="w-6 h-0.5 bg-indigo-400 inline-block border-dashed border" /> {algoLabel}
            </span>
          )}
        </div>
      </div>
      <div ref={containerRef} className="w-full" style={{ minHeight: 420 }} />
    </div>
  )
}

