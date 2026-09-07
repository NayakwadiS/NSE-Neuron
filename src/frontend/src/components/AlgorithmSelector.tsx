const ALGORITHMS = [
  { id: 'lstm',     label: 'LSTM',         desc: 'Long Short-Term Memory' },
  { id: 'bilstm',   label: 'BiLSTM',       desc: 'Bidirectional LSTM' },
  { id: 'gru',      label: 'GRU',          desc: 'Gated Recurrent Unit' },
  { id: 'cnn_lstm', label: 'CNN-LSTM',     desc: 'Convolutional + LSTM' },
  { id: 'all',      label: 'All & Compare',desc: 'Run all 4 algorithms' },
]

interface Props {
  value:    string
  onChange: (algo: string) => void
}

export default function AlgorithmSelector({ value, onChange }: Props) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1.5">
        Algorithm
      </label>
      <div className="grid grid-cols-1 gap-1.5">
        {ALGORITHMS.map(a => (
          <button
            key={a.id}
            onClick={() => onChange(a.id)}
            className={`
              flex items-center gap-3 px-3 py-2.5 rounded-lg border text-left transition
              ${value === a.id
                ? 'border-indigo-500 bg-indigo-500/10 text-slate-900 dark:text-white'
                : 'border-slate-300 dark:border-slate-700 bg-slate-100/60 dark:bg-slate-800/50 text-slate-500 dark:text-slate-400 hover:border-slate-400 dark:hover:border-slate-600 hover:text-slate-700 dark:hover:text-slate-200'}
            `}
          >
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${value === a.id ? 'bg-indigo-400' : 'bg-slate-400 dark:bg-slate-600'}`} />
            <span>
              <span className="block text-sm font-medium">{a.label}</span>
              <span className="block text-xs opacity-60">{a.desc}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

