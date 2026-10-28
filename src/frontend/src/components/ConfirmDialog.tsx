import { useEffect } from 'react'
import { X } from 'lucide-react'
interface Props {
  open:      boolean
  title:     string
  message:   string
  confirmLabel?: string
  danger?:   boolean
  onConfirm: () => void
  onCancel:  () => void
}
export default function ConfirmDialog({ open, title, message, confirmLabel = 'Confirm', danger, onConfirm, onCancel }: Props) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
      if (e.key === 'Enter') onConfirm()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onCancel, onConfirm])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="w-full max-w-sm rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-white">{title}</h3>
          <button onClick={onCancel} className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="px-4 py-4 text-sm text-slate-600 dark:text-slate-300">
          {message}
        </div>
        <div className="flex justify-end gap-2 px-4 py-3 border-t border-slate-200 dark:border-slate-800">
          <button
            onClick={onCancel}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold text-white transition ${
              danger ? 'bg-red-600 hover:bg-red-500' : 'bg-indigo-600 hover:bg-indigo-500'
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
