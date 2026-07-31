import type { CacheStatus, CacheInfo } from '../types'

interface Props {
  status?: CacheStatus
  label?:  string
  info?:   CacheInfo
}

const STYLES: Record<CacheStatus, { icon: string; cls: string; title: string }> = {
  fresh: {
    icon: '⚡',
    cls:  'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
    title: 'Cached model',
  },
  warm: {
    icon: '♻️',
    cls:  'bg-amber-500/10 border-amber-500/30 text-amber-400',
    title: 'Fine-tuned',
  },
  miss: {
    icon: '🔄',
    cls:  'bg-indigo-500/10 border-indigo-500/30 text-indigo-400',
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

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs ${s.cls}`}>
      <span>{s.icon}</span>
      <span className="font-medium">{label ?? s.title}</span>
      {age && status !== 'miss' && (
        <span className="opacity-70">· trained {age}</span>
      )}
    </div>
  )
}

