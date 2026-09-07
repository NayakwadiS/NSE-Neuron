export function SkeletonChart() {
  return (
    <div className="rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900/80 overflow-hidden animate-pulse">
      <div className="px-4 py-3 border-b border-slate-300 dark:border-slate-700 flex items-center justify-between">
        <div className="h-4 w-40 bg-slate-200 dark:bg-slate-700 rounded" />
        <div className="h-4 w-24 bg-slate-200 dark:bg-slate-700 rounded" />
      </div>
      <div className="p-4 flex items-end gap-1.5" style={{ height: 420 }}>
        {Array.from({ length: 48 }).map((_, i) => (
          <div
            key={i}
            className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-sm"
            style={{ height: `${20 + ((i * 37) % 70)}%` }}
          />
        ))}
      </div>
    </div>
  )
}

export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800/40 overflow-hidden animate-pulse">
      <div className="px-4 py-3 border-b border-slate-300 dark:border-slate-700">
        <div className="h-4 w-56 bg-slate-200 dark:bg-slate-700 rounded" />
      </div>
      <div className="p-4 space-y-3">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex gap-3">
            {Array.from({ length: 5 }).map((__, j) => (
              <div key={j} className="h-3.5 flex-1 bg-slate-200 dark:bg-slate-700 rounded" />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

