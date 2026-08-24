import { useEffect, useRef, useState } from 'react'
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
import { Maximize2, Download } from 'lucide-react'
import type { OHLCPoint, ForecastDay } from '../types'

interface Props {
  historical:  OHLCPoint[]
  forecast:    ForecastDay[]
  algoLabel:   string
}

function isDarkMode(): boolean {
  return document.documentElement.classList.contains('dark')
}

interface HoverOhlc {
  date: string
  open: number
  high: number
  low: number
  close: number
}

export default function ForecastChart({ historical, forecast, algoLabel }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef     = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const [isDark, setIsDark] = useState(isDarkMode)
  const [hover, setHover] = useState<HoverOhlc | null>(null)

  const hasData = historical.some(d => d.close != null && d.date)

  // Track theme toggle so the chart colors update without a full page reload
  useEffect(() => {
    const observer = new MutationObserver(() => setIsDark(isDarkMode()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])


  useEffect(() => {
    if (!hasData) return
    if (!containerRef.current) return
    const container = containerRef.current

    // Destroy any previous chart instance before creating a new one
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    let rafId: number
    let ro: ResizeObserver

    const colors = isDark
      ? { bg: '#0f172a', text: '#94a3b8', grid: '#1e293b', border: '#1e293b' }
      : { bg: '#ffffff', text: '#475569', grid: '#e2e8f0', border: '#cbd5e1' }

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
        background:  { type: ColorType.Solid, color: colors.bg },
        textColor:   colors.text,
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: colors.border },
      timeScale: {
        borderColor:     colors.border,
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
    candleSeriesRef.current = candleSeries

    // Use close as fallback for open so candlesticks always render,
    // even when the API omits the open column for some symbols.
    // lightweight-charts requires strictly ascending, unique timestamps —
    // sort + de-duplicate (keep last) defensively so a bad/duplicate date
    // from the API never throws and silently blanks the whole chart.
    const dedupeSortByTime = <T extends { time: Time }>(rows: T[]): T[] => {
      const sorted = [...rows].sort((a, b) => {
        const ta = a.time as unknown as string
        const tb = b.time as unknown as string
        return ta < tb ? -1 : ta > tb ? 1 : 0
      })
      const map = new Map<string, T>()
      for (const row of sorted) map.set(row.time as unknown as string, row) // keep last
      return Array.from(map.values())
    }

    const candleData: CandlestickData[] = dedupeSortByTime(
      historical
        .filter(d => d.close != null && d.date)
        .map(d => ({
          time:  d.date as Time,
          open:  d.open  ?? d.close,   // fallback: doji-style candle
          high:  d.high  ?? d.close,
          low:   d.low   ?? d.close,
          close: d.close,
        }))
    )
    candleSeries.setData(candleData)

    // ── Forecast close price line ──────────────────────────────────────────
    const forecastSeries = chart.addLineSeries({
      color:     '#818cf8',
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      title:     `${algoLabel} Forecast`,
    })

    const forecastData: LineData[] = dedupeSortByTime(
      forecast.map(d => ({
        time:  d.date as Time,
        value: d.close,
      }))
    )

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

    // ── Hover legend: show OHLC values under the crosshair ─────────────────
    chart.subscribeCrosshairMove(param => {
      if (!param.time || !param.seriesData.has(candleSeries)) {
        setHover(null)
        return
      }
      const bar = param.seriesData.get(candleSeries) as CandlestickData | undefined
      if (!bar) { setHover(null); return }
      setHover({
        date:  param.time as unknown as string,
        open:  bar.open,
        high:  bar.high,
        low:   bar.low,
        close: bar.close,
      })
    })

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
  }, [historical, forecast, algoLabel, hasData, isDark])

  const handleFit = () => chartRef.current?.timeScale().fitContent()

  const handleExportPng = () => {
    if (!chartRef.current) return
    const canvas = chartRef.current.takeScreenshot()
    const url = canvas.toDataURL('image/png')
    const a = document.createElement('a')
    a.href = url
    a.download = `${algoLabel.replace(/\s+/g, '_')}_chart.png`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  return (
    <div className="rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900/80 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-300 dark:border-slate-700 flex items-center justify-between flex-wrap gap-2">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
          {forecast.length > 0 ? 'Candlestick + Forecast' : 'Historical Candlestick'}
        </h3>
        <div className="flex items-center gap-4 text-xs text-slate-500 dark:text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="w-6 h-0.5 bg-emerald-500 inline-block" /> Historical
          </span>
          {forecast.length > 0 && (
            <span className="flex items-center gap-1.5">
              <span className="w-6 h-0.5 bg-indigo-400 inline-block border-dashed border" /> {algoLabel}
            </span>
          )}
          {hasData && (
            <span className="flex items-center gap-1.5 border-l border-slate-300 dark:border-slate-700 pl-3">
              <button
                onClick={handleFit}
                title="Fit chart to screen"
                className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-indigo-500 dark:hover:text-indigo-400 transition"
              >
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={handleExportPng}
                title="Export chart as PNG"
                className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-indigo-500 dark:hover:text-indigo-400 transition"
              >
                <Download className="w-3.5 h-3.5" />
              </button>
            </span>
          )}
        </div>
      </div>
      {hasData && (
        <div className="px-4 py-1.5 border-b border-slate-200 dark:border-slate-800/70 text-xs font-mono
                        text-slate-500 dark:text-slate-400 flex items-center gap-3 min-h-[26px]">
          {hover ? (
            <>
              <span className="text-slate-400 dark:text-slate-600">{hover.date}</span>
              <span>O <span className="text-slate-700 dark:text-slate-300">{hover.open.toFixed(2)}</span></span>
              <span>H <span className="text-emerald-600 dark:text-emerald-400">{hover.high.toFixed(2)}</span></span>
              <span>L <span className="text-red-600 dark:text-red-400">{hover.low.toFixed(2)}</span></span>
              <span>C <span className="font-semibold text-slate-900 dark:text-white">{hover.close.toFixed(2)}</span></span>
            </>
          ) : (
            <span className="text-slate-400 dark:text-slate-600">Hover the chart for OHLC values</span>
          )}
        </div>
      )}
      <div ref={containerRef} className="w-full" style={{ minHeight: 420 }}>
        {!hasData && (
          <div className="flex items-center justify-center h-[420px] text-sm text-slate-500">
            No historical price data available to chart.
          </div>
        )}
      </div>
    </div>
  )
}
