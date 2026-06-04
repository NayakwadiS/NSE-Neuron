// ── Domain types ──────────────────────────────────────────────────────────────

export interface SignalInfo {
  label:            'BUY' | 'HOLD' | 'SELL'
  confidence:       number
  confidence_orig?: number
  confidence_delta?: string
  regime_adjusted?: boolean
  regime_direction?: string
}

export interface ForecastDay {
  date:       string
  high:       number
  low:        number
  close:      number
  prev_close: number
  signal:     SignalInfo | null
}

export interface RegimeInfo {
  regime:            'BULL' | 'BEAR' | 'SIDEWAYS' | 'UNKNOWN'
  recommended_model: string
  sufficient_data:   boolean
  rows:              number
  sma_fast:          number | null
  sma_slow:          number | null
  description:       string
}

export interface PatternInfo {
  name:      string
  value:     number
  direction: 'Bullish' | 'Bearish'
  date:      string
}

export interface OHLCPoint {
  date:  string
  open?: number
  high:  number
  low:   number
  close: number
}

// ── Forecast result shapes ────────────────────────────────────────────────────

export interface SingleForecastResult {
  symbol:       string
  company_name: string
  algorithm:    string
  display_name: string
  forecast:     ForecastDay[]
  rmse:         number
  regime:       RegimeInfo
  historical:   OHLCPoint[]
}

export interface AllForecastResult {
  symbol:         string
  company_name:   string
  algorithm:      'all'
  display_name:   string
  algo_forecasts: Record<string, ForecastDay[]>
  all_rmse:       Record<string, number>
  best_algo:      string
  regime:         RegimeInfo
  historical:     OHLCPoint[]
}

export type ForecastResult = SingleForecastResult | AllForecastResult

export interface RegimeAnalysisResult {
  symbol:       string
  company_name: string
  regime:       RegimeInfo
  patterns:     PatternInfo[]
  insight:      string | null
  historical:   OHLCPoint[]
}

// ── Job polling ───────────────────────────────────────────────────────────────

export type JobStatus = 'pending' | 'running' | 'done' | 'error'

export interface JobResponse {
  job_id:      string
  status:      JobStatus
  progress:    string
  description: string
  result:      ForecastResult | null
  error:       string | null
  created_at:  string
}

// ── Symbol search ─────────────────────────────────────────────────────────────

export interface SymbolResult {
  symbol: string
  name:   string
}

