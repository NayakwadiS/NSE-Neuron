import type { CacheStatus, CacheInfo } from '../types'
import { Zap, Recycle, RefreshCw } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

interface Props {
  status?: CacheStatus
  label?:  string
  info?:   CacheInfo
}

const STYLES: Record<CacheStatus, { icon: LucideIcon; cls: string; title: string }> = {
  fresh: {
    icon: Zap,
    cls:  'bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400',
    title: 'Cached model',
  },
  warm: {
    icon: Recycle,
    cls:  'bg-amber-500/10 border-amber-500/30 text-amber-600 dark:text-amber-400',
    title: 'Fine-tuned',
  },
  miss: {
    icon: RefreshCw,
    cls:  'bg-indigo-500/10 border-indigo-500/30 text-indigo-600 dark:text-indigo-400',
    title: 'Trained',
  },
}

function relativeAge(iso?: string): string | null {
  if (!iso) return null
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return null
  const days = Math.floor((Date.now() - then) / 86_400_000)
  if (days <= 0) return 'today'
  if (days === 1) return 'yesterday'
  return `${days} days ago`
}

export default function CacheBadge({ status, label, info }: Props) {
  if (!status) return null
  const s   = STYLES[status] ?? STYLES.miss
  const age = relativeAge(info?.trained_at)
  const Icon = s.icon

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs ${s.cls}`}>
      <Icon className="w-3.5 h-3.5" />
      <span className="font-medium">{label ?? s.title}</span>
      {age && status !== 'miss' && (
        <span className="opacity-70">· trained {age}</span>
      )}
    </div>
  )
}



