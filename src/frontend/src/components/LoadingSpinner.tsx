export default function LoadingSpinner({ message }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-20 text-slate-400">
      <div className="relative w-14 h-14">
        <div className="absolute inset-0 rounded-full border-4 border-slate-700" />
        <div className="absolute inset-0 rounded-full border-4 border-t-indigo-500 animate-spin" />
      </div>
      <p className="text-sm font-medium">{message ?? 'Training model…'}</p>
      <p className="text-xs text-slate-600">This may take 1–3 minutes depending on the algorithm</p>
    </div>
  )
}

