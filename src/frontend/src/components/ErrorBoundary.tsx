import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { AlertTriangle } from 'lucide-react'
interface Props {
  children: ReactNode
  label?:   string
}
interface State {
  hasError: boolean
}
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }
  static getDerivedStateFromError(): State {
    return { hasError: true }
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', this.props.label ?? '', error, info)
  }
  componentDidUpdate(prevProps: Props) {
    if (prevProps.children !== this.props.children && this.state.hasError) {
      this.setState({ hasError: false })
    }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center gap-2 py-16 text-center rounded-xl border border-red-500/30 bg-red-500/10 text-red-500 dark:text-red-400">
          <AlertTriangle className="w-8 h-8" />
          <p className="text-sm font-medium">
            {this.props.label ?? 'Something went wrong rendering this section.'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false })}
            className="text-xs underline opacity-80 hover:opacity-100"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
