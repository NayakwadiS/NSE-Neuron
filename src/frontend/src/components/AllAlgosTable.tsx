import { useState } from 'react'
import type { ForecastDay } from '../types'
import { downloadCsv } from '../utils/export'
import { Download, ArrowUpDown, Trophy } from 'lucide-react'

const ALGO_DISPLAY: Record<string, string> = {
  lstm: 'LSTM', bilstm: 'BiLSTM', gru: 'GRU', cnn_lstm: 'CNN-LSTM',
}

interface Props {
  algoForecasts: Record<string, ForecastDay[]>
  allRmse:       Record<string, number>
  bestAlgo:      string
  regime?:       { recommended_model: string; sufficient_data: boolean }
}

const fmt = (v: number) => v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

export default function AllAlgosTable({ algoForecasts, allRmse, bestAlgo, regime }: Props) {
  const algos = Object.keys(algoForecasts)
  const days  = algoForecasts[algos[0]] ?? []
  const [rmseAsc, setRmseAsc] = useState(true)

  const rankedAlgos = algos.slice().sort((a, b) => rmseAsc ? allRmse[a] - allRmse[b] : allRmse[b] - allRmse[a])

  const exportComparison = () => {
    downloadCsv(
      'all_algorithms_forecast.csv',
      algos.map(algo => {
        const row: Record<string, unknown> = { algorithm: ALGO_DISPLAY[algo] ?? algo, rmse: allRmse[algo] }
        algoForecasts[algo].forEach((d, i) => { row[`day_${i + 1}_${d.date}`] = d.close })
        return row
      }),
    )
  }

  return (
    <div className="space-y-4">
      {/* Comparison table */}
      <div className="rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800/40 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-300 dark:border-slate-700 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Close Price Forecast — All Algorithms</h3>
          <button
            onClick={exportComparison}
            title="Export as CSV"
            className="p-1.5 rounded hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-indigo-500 dark:hover:text-indigo-400 transition text-slate-500 dark:text-slate-400"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10 bg-slate-50 dark:bg-slate-800">
              <tr className="border-b border-slate-300 dark:border-slate-700/50">
                <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase">Algorithm</th>
                {days.map((d, i) => (
                  <th key={i} className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500">
                    Day {i + 1}<br />
                    <span className="font-mono font-normal text-slate-400 dark:text-slate-600">{d.date}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {algos.map(algo => {
                const isRecommended = regime?.sufficient_data && regime.recommended_model.toLowerCase().replace('-','_') === algo
                return (
                  <tr key={algo} className={`border-b border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700/20 ${isRecommended ? 'bg-indigo-500/5' : ''}`}>
                    <td className="px-4 py-2.5 font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-2">
                      {ALGO_DISPLAY[algo] ?? algo}
                      {isRecommended && (
                        <span className="text-xs bg-indigo-500/20 text-indigo-500 dark:text-indigo-400 px-1.5 py-0.5 rounded">Recommended</span>
                      )}
                    </td>
                    {algoForecasts[algo].map((d, i) => (
                      <td key={i} className="px-4 py-2.5 font-mono text-slate-900 dark:text-white">{fmt(d.close)}</td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* RMSE Benchmark */}
      <div className="rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800/40 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-300 dark:border-slate-700">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Model Benchmark — RMSE</h3>
        </div>
        <table className="w-full text-sm">
          <thead className="sticky top-0 z-10 bg-slate-50 dark:bg-slate-800">
            <tr className="border-b border-slate-300 dark:border-slate-700/50">
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase">Algorithm</th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase">
                <button
                  onClick={() => setRmseAsc(v => !v)}
                  className="flex items-center gap-1 hover:text-indigo-500 dark:hover:text-indigo-400 transition"
                  title="Sort by RMSE"
                >
                  RMSE <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase">Rank</th>
            </tr>
          </thead>
          <tbody>
            {rankedAlgos
              .map((algo, i) => {
                const rank = rmseAsc ? i : rankedAlgos.length - 1 - i
                return (
                <tr key={algo} className="border-b border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700/20">
                  <td className="px-4 py-2.5 font-semibold text-slate-700 dark:text-slate-300">{ALGO_DISPLAY[algo] ?? algo}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-700 dark:text-slate-300">{allRmse[algo].toFixed(6)}</td>
                  <td className="px-4 py-2.5">
                    {rank === 0 ? (
                      <span className="flex items-center gap-1 text-xs bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 px-2 py-0.5 rounded-full">
                        <Trophy className="w-3 h-3" /> Best
                      </span>
                    ) : (
                      <span className="text-xs text-slate-400 dark:text-slate-600">#{rank + 1}</span>
                    )}
                  </td>
                </tr>
              )})}
          </tbody>
        </table>
      </div>
    </div>
  )
}





