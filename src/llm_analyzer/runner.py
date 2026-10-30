import uuid
from datetime import datetime, timezone
from pathlib import Path

from llm_analyzer.adapters.base import LLMAdapter
from llm_analyzer.models import LLMResponse, Prompt, RunRecord, RunStatus
from llm_analyzer.storage import init_db, save_run


def run_experiment(
    adapters: list[LLMAdapter],
    prompts: list[Prompt],
    repetitions: int,
    db_path: str | Path,
    experiment_id: str | None = None,
) -> str:
    """Wysyła każdy prompt do każdego adaptera i zapisuje wyniki do bazy.

    Kolejność: powtórzenie -> prompt -> adapter, więc modele są przeplatane.
    Zwraca identyfikator eksperymentu.
    """
    if not adapters:
        raise ValueError("At least one adapter is required")
    if not prompts:
        raise ValueError("At least one prompt is required")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")

    init_db(db_path)
    if experiment_id is None:
        experiment_id = _new_experiment_id()

    for repetition in range(repetitions):
        for prompt in prompts:
            for adapter in adapters:
                response = _safe_complete(adapter, prompt)
                record = RunRecord(
                    experiment_id=experiment_id,
                    timestamp=_now_iso(),
                    provider=adapter.provider,
                    model=adapter.model,
                    prompt_id=prompt.id,
                    repetition=repetition,
                    response=response,
                )
                save_run(db_path, record)

    return experiment_id


def _safe_complete(adapter: LLMAdapter, prompt: Prompt) -> LLMResponse:
    try:
        return adapter.complete(prompt)
    except Exception as exc:
        return LLMResponse(
            status=RunStatus.ERROR,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def _new_experiment_id() -> str:
    now = datetime.now(timezone.utc)
    return f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")