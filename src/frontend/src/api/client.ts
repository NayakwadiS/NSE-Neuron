import axios from 'axios'
import type {
  JobResponse,
  RegimeAnalysisResult,
  SymbolResult,
} from '../types'

const api = axios.create({ baseURL: '/api' })

// ── Symbol search ─────────────────────────────────────────────────────────────
export const searchSymbols = async (q: string): Promise<SymbolResult[]> => {
  const { data } = await api.get<{ symbols: SymbolResult[] }>('/symbols', { params: { q } })
  return data.symbols
}

// ── Start forecast job ────────────────────────────────────────────────────────
export const startForecast = async (
  symbol: string,
  algorithm: string,
): Promise<string> => {
  const { data } = await api.post<{ job_id: string }>('/forecast', { symbol, algorithm })
  return data.job_id
}

// ── Poll job ──────────────────────────────────────────────────────────────────
export const pollJob = async (jobId: string): Promise<JobResponse> => {
  const { data } = await api.get<JobResponse>(`/jobs/${jobId}`)
  return data
}

// ── Regime analysis ───────────────────────────────────────────────────────────
export const fetchRegimeAnalysis = async (
  symbol: string,
): Promise<RegimeAnalysisResult> => {
  const { data } = await api.post<RegimeAnalysisResult>(`/regime/${symbol}`)
  return data
}

export default api

