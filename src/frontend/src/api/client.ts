import axios from 'axios'
import type {
  JobResponse,
  RegimeAnalysisResult,
  SymbolResult,
  CacheListResponse,
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
  forceRetrain = false,
): Promise<string> => {
  const { data } = await api.post<{ job_id: string }>('/forecast', {
    symbol,
    algorithm,
    force_retrain: forceRetrain,
  })
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

// ── Model weight cache ────────────────────────────────────────────────────────
export const fetchCachedModels = async (): Promise<CacheListResponse> => {
  const { data } = await api.get<CacheListResponse>('/models/cached')
  return data
}

export const fetchCachedModelsForSymbol = async (symbol: string) => {
  const { data } = await api.get(`/models/cached/${symbol}`)
  return data
}

export const deleteCachedModels = async (symbol: string) => {
  const { data } = await api.delete(`/models/cached/${symbol}`)
  return data
}

export const deleteCachedModel = async (symbol: string, model: string) => {
  const { data } = await api.delete(`/models/cached/${symbol}/${model}`)
  return data
}

export default api

