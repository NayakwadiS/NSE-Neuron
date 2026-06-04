"""
services/job_manager.py
━━━━━━━━━━━━━━━━━━━━━━━
Simple in-memory async job queue backed by ThreadPoolExecutor.
Each job runs a blocking function in a worker thread and stores the result.
"""
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from datetime import datetime
from typing import Any, Optional, Callable


class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    ERROR    = "error"


class Job:
    def __init__(self, job_id: str, description: str = ""):
        self.job_id      = job_id
        self.status      = JobStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
        self.progress    = "Queued…"
        self.description = description
        self.created_at  = datetime.now()


class JobManager:
    """Thread-safe in-memory job manager."""

    def __init__(self, max_workers: int = 2):
        self._jobs: dict[str, Job] = {}
        self._lock       = threading.Lock()
        self._executor   = ThreadPoolExecutor(max_workers=max_workers)

    def submit(
        self,
        func: Callable,
        *args,
        description: str = "",
        **kwargs,
    ) -> str:
        """Submit a blocking function, returns a job_id for polling."""
        job_id = str(uuid.uuid4())
        job    = Job(job_id, description)

        with self._lock:
            self._jobs[job_id] = job

        def _run():
            with self._lock:
                job.status   = JobStatus.RUNNING
                job.progress = "Training model…"
            try:
                result = func(job, *args, **kwargs)
                with self._lock:
                    job.result   = result
                    job.status   = JobStatus.DONE
                    job.progress = "Done"
            except Exception as exc:
                with self._lock:
                    job.error    = str(exc)
                    job.status   = JobStatus.ERROR
                    job.progress = f"Error: {exc}"

        self._executor.submit(_run)
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def to_dict(self, job: Job) -> dict:
        with self._lock:
            return {
                "job_id":      job.job_id,
                "status":      job.status,
                "progress":    job.progress,
                "description": job.description,
                "result":      job.result,
                "error":       job.error,
                "created_at":  job.created_at.isoformat(),
            }


# Singleton
job_manager = JobManager()

