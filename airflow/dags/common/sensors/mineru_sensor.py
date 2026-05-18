from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from airflow.providers.http.hooks.http import HttpHook
from airflow.sdk import BaseOperator, BaseSensorOperator, Context  # noqa: F401
from airflow.triggers.base import BaseTrigger, TriggerEvent
from asgiref.sync import sync_to_async


class MineruBatchTrigger(BaseTrigger):
    """
    Deferrable trigger that polls MinerU until all tasks in a batch complete.

    Runs inside the Airflow triggerer process. Yields a single
    ``TriggerEvent`` when every task ID reaches ``completed`` status, or
    immediately on the first ``failed`` / ``task_not_found`` result.

    Parameters
    ----------
    task_ids : list of str
        MinerU task IDs to monitor.
    conn_id : str, optional
        Airflow connection ID for the MinerU API. Default is ``"mineru"``.
    poll_interval : int, optional
        Seconds to sleep between polling rounds. Default is ``60``.
    """

    def __init__(
        self,
        task_ids: list[str],
        conn_id: str = "mineru",
        poll_interval: int = 60,
    ):
        super().__init__()
        self.task_ids = task_ids
        self.conn_id = conn_id
        self.poll_interval = poll_interval

    def serialize(self) -> tuple[str, dict[str, Any]]:
        """
        Serialise the trigger so the triggerer process can reconstruct it.

        Returns
        -------
        tuple of (str, dict)
            Fully-qualified class path and a dict of constructor kwargs.
        """
        self.log.info(
            f"===== MineruBatchTrigger.serialize() called with {len(self.task_ids)} task_ids"
        )
        self.log.info(f"===== Task IDs to monitor: {self.task_ids}")
        return (
            "common.sensors.mineru_sensor.MineruBatchTrigger",
            {
                "task_ids": self.task_ids,
                "conn_id": self.conn_id,
                "poll_interval": self.poll_interval,
            },
        )

    @sync_to_async
    def _check_status(self, task_id: str) -> dict:
        """
        Fetch the current status of a single MinerU task (sync → async bridge).

        Wraps a synchronous ``HttpHook`` call with ``sync_to_async`` so it
        can be awaited inside ``asyncio.gather``.

        Parameters
        ----------
        task_id : str
            MinerU task ID to query via ``GET /tasks/{task_id}``.

        Returns
        -------
        dict
            Parsed JSON response, or ``{"status": "failed", "reason": "task_not_found"}``
            when the endpoint returns ``404``.
        """
        hook = HttpHook(method="GET", http_conn_id=self.conn_id)
        resp = hook.run(f"/tasks/{task_id}")

        if resp.status_code == 404:
            return {"status": "failed", "reason": "task_not_found"}

        return resp.json()

    async def run(self) -> AsyncIterator[TriggerEvent]:
        """
        Poll all task IDs concurrently until the batch is done or fails.

        Runs ``_check_status`` for every pending task in parallel via
        ``asyncio.gather``. Completed tasks are removed from the pending
        list; a single failure immediately yields a failure event and
        returns. When all tasks complete, yields one success event carrying
        ``task_ids`` and per-task ``file_names``.

        Yields
        ------
        TriggerEvent
            ``{"status": "completed", "task_ids": [...], "results": {...}}``
            on full success, or ``{"status": "failed", "error": "..."}`` on
            the first failure.
        """
        self.log.info(
            f"===== MineruBatchTrigger.run() starting with {len(self.task_ids)} task_ids"
        )
        self.log.info(f"===== Task IDs to monitor: {self.task_ids}")
        pending = list(self.task_ids)
        completed_results = {}

        while pending:
            checks = await asyncio.gather(
                *[self._check_status(tid) for tid in pending],
                return_exceptions=True,
            )

            for task_id, result in zip(list(pending), checks):
                if isinstance(result, Exception):
                    self.log.error(f"----- Polling error for {task_id}: {result}")
                    continue

                status = result.get("status")
                if status == "failed":
                    failure_reason = result.get("reason", "remote_failure")

                    if failure_reason == "task_not_found":
                        self.log.error(
                            f"----- Task {task_id} not found (pod restart). Marking failed."
                        )

                    yield TriggerEvent(
                        {
                            "status": "failed",
                            "error": f"MinerU task {task_id} failed: {failure_reason}",
                        }
                    )
                    return

                if status == "completed":
                    completed_results[task_id] = result.get("file_names", [])
                    pending.remove(task_id)
                    self.log.info(f"----- Batch task {task_id} completed")

            if pending:
                self.log.info(
                    f"----- Waiting for {len(pending)}/{len(self.task_ids)} tasks. Sleeping {self.poll_interval}s."
                )
                await asyncio.sleep(self.poll_interval)

        self.log.info(f"----- Batch completed: {list(completed_results.keys())}")
        yield TriggerEvent(
            {
                "status": "completed",
                "task_ids": self.task_ids,
                "results": completed_results,  # {task_id: [file_names]}
            }
        )


class MineruBatchStatusSensor(BaseSensorOperator):
    """
    Sensor that waits for a batch of MinerU tasks to reach terminal status.

    In deferrable mode (default) the operator suspends itself via
    ``MineruBatchTrigger``, freeing the worker slot while polling runs
    inside the triggerer process. A non-deferrable fallback mode is
    provided for environments that do not run a triggerer.

    Parameters
    ----------
    external_task_ids : list of str
        MinerU task IDs to monitor (``/tasks/{id}`` endpoint).
    mineru_conn_id : str, optional
        Airflow connection ID for the MinerU API. Default is ``"mineru"``.
    poll_interval : int, optional
        Seconds between status checks. Default is ``60``.
    deferrable : bool, optional
        Use the async deferrable path when ``True`` (default).
        Set to ``False`` to fall back to synchronous polling.
    **kwargs
        Passed through to ``BaseSensorOperator``.
    """

    ui_color = "#73deff"
    template_fields = ("external_task_ids", "poll_interval")

    def __init__(
        self,
        *,
        external_task_ids: list[str],  # task_id from MinerU-endpoint /tasks/{id}
        mineru_conn_id: str = "mineru",
        poll_interval: int = 60,
        deferrable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.external_task_ids = external_task_ids
        self.mineru_conn_id = mineru_conn_id
        self.poll_interval = poll_interval
        self.deferrable = deferrable

    def execute(self, context: Context):
        """
        Start monitoring the batch, deferring or blocking based on ``deferrable``.

        In deferrable mode, suspends the task via ``MineruBatchTrigger`` and
        resumes at ``execute_complete`` once the trigger fires. In fallback
        mode, synchronously polls ``GET /tasks/{id}`` in a loop.

        Parameters
        ----------
        context : Context
            Airflow task execution context.

        Returns
        -------
        list of str
            ``external_task_ids`` (fallback mode only; deferrable mode
            returns via ``execute_complete``).

        Raises
        ------
        RuntimeError
            If any task reports ``failed`` status (fallback mode).
        """
        if self.deferrable:
            self.log.info(
                f"===== MineruBatchStatusSensor.execute() - Deferring batch trigger for {len(self.external_task_ids)} tasks"
            )
            self.log.info(f"===== Task IDs to monitor: {self.external_task_ids}")
            self.log.info(
                f"===== conn_id: {self.mineru_conn_id}, poll_interval: {self.poll_interval}"
            )
            self.defer(
                trigger=MineruBatchTrigger(
                    task_ids=self.external_task_ids,
                    conn_id=self.mineru_conn_id,
                    poll_interval=self.poll_interval,
                ),
                method_name="execute_complete",
            )
        else:
            # Fallback sensor mode (без defer)
            hook = HttpHook(method="GET", http_conn_id=self.mineru_conn_id)
            import time

            pending = set(self.external_task_ids)
            while pending:
                for task_id in list(pending):
                    resp = hook.run(f"/tasks/{task_id}")
                    status = resp.json().get("status")

                    if status == "completed":
                        pending.remove(task_id)
                    elif status == "failed":
                        raise RuntimeError(f"MinerU task {task_id} failed")
                if pending:
                    time.sleep(self.poll_interval)
            return self.external_task_ids

    def execute_complete(
        self, context: Context, event: dict[str, Any] | None = None
    ) -> list[str]:
        """
        Resume execution after ``MineruBatchTrigger`` fires.

        Called automatically by Airflow when the deferred trigger yields a
        ``TriggerEvent``. Raises on failure; returns task IDs on success so
        they can be consumed by downstream tasks via XCom.

        Parameters
        ----------
        context : Context
            Airflow task execution context.
        event : dict or None
            Payload from ``TriggerEvent``. Expected keys:
            ``status`` (``"completed"`` | ``"failed"``),
            ``task_ids`` (list of str), ``error`` (str, failure only).

        Returns
        -------
        list of str
            Completed MinerU task IDs passed to the next task.

        Raises
        ------
        RuntimeError
            If ``event`` is ``None`` or ``event["status"] == "failed"``.
        """
        if not event or event.get("status") == "failed":
            raise RuntimeError(event.get("error", "Unknown batch error"))

        task_ids = event["task_ids"]
        self.log.info(
            f"===== execute_complete() - Batch completed: {len(task_ids)} tasks"
        )
        self.log.info(f"===== Completed task IDs: {task_ids}")
        if "results" in event:
            self.log.info(f"===== Results: {event['results']}")
        return task_ids  # передается в следующую задачу
