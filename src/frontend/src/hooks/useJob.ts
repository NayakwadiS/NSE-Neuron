import { useState, useEffect, useRef } from 'react'
import { pollJob } from '../api/client'
import type { JobResponse } from '../types'

/**
 * useJob — polls /api/jobs/{jobId} every 3 seconds until done/error.
 */
export function useJob(jobId: string | null) {
  const [job, setJob]       = useState<JobResponse | null>(null)
  const intervalRef          = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!jobId) { setJob(null); return }

    const tick = async () => {
      try {
        const res = await pollJob(jobId)
        setJob(res)
        if (res.status === 'done' || res.status === 'error') {
          if (intervalRef.current) clearInterval(intervalRef.current)
        }
      } catch {/* network error — keep polling */}
    }

    tick()
    intervalRef.current = setInterval(tick, 3000)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [jobId])

  return job
}

