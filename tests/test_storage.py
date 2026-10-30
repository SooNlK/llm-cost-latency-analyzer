import dataclasses

import pytest

from llm_analyzer.models import LLMResponse, RunRecord, RunStatus
from llm_analyzer.storage import init_db, load_runs, save_run


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def make_record(
    experiment_id: str = "exp1",
    prompt_id: str = "p1",
    repetition: int = 0,
    response: LLMResponse | None = None,
) -> RunRecord:
    if response is None:
        response = LLMResponse(
            status=RunStatus.SUCCESS,
            text="hello",
            input_tokens=1000,
            output_tokens=500,
            cached_input_tokens=200,
            ttft_ms=123.5,
            total_latency_ms=2000.25,
        )
    return RunRecord(
        experiment_id=experiment_id,
        timestamp="2026-10-29T12:00:00",
        provider="mock",
        model="test-model",
        prompt_id=prompt_id,
        repetition=repetition,
        response=response,
    )


def make_failed_record(status: RunStatus = RunStatus.ERROR) -> RunRecord:
    return make_record(
        response=LLMResponse(status=status, error_message="boom")
    )


# --- init_db ---

def test_init_db_creates_file(tmp_path):
    path = tmp_path / "new.db"
    init_db(path)
    assert path.exists()


def test_init_db_is_idempotent(db_path):
    # drugie wywołanie nie może rzucić wyjątku ani skasować danych
    save_run(db_path, make_record())
    init_db(db_path)
    assert len(load_runs(db_path)) == 1


#save_run

def test_save_run_returns_id(db_path):
    run_id = save_run(db_path, make_record())
    assert isinstance(run_id, int)


def test_save_run_ids_are_unique(db_path):
    first = save_run(db_path, make_record(repetition=0))
    second = save_run(db_path, make_record(repetition=1))
    assert first != second


#load_runs

def test_load_empty_db_returns_empty_list(db_path):
    assert load_runs(db_path) == []


def test_round_trip_successful_record(db_path):
    record = make_record()
    run_id = save_run(db_path, record)

    loaded = load_runs(db_path)

    assert loaded == [dataclasses.replace(record, run_id=run_id)]


def test_round_trip_failed_record_keeps_none(db_path):
    # nieudane wywołanie nie ma tokenów ani czasu: None musi wrócić jako None
    record = make_failed_record()
    run_id = save_run(db_path, record)

    loaded = load_runs(db_path)[0]

    assert loaded.run_id == run_id
    assert loaded.response.input_tokens is None
    assert loaded.response.output_tokens is None
    assert loaded.response.ttft_ms is None
    assert loaded.response.total_latency_ms is None
    assert loaded.response.error_message == "boom"


def test_none_and_zero_cached_tokens_are_distinguished(db_path):
    # None = dostawca nie raportuje, 0 = raportuje i było zero
    none_response = LLMResponse(
        status=RunStatus.SUCCESS, input_tokens=10, output_tokens=5,
        cached_input_tokens=None,
    )
    zero_response = LLMResponse(
        status=RunStatus.SUCCESS, input_tokens=10, output_tokens=5,
        cached_input_tokens=0,
    )
    save_run(db_path, make_record(repetition=0, response=none_response))
    save_run(db_path, make_record(repetition=1, response=zero_response))

    loaded = load_runs(db_path)

    assert loaded[0].response.cached_input_tokens is None
    assert loaded[1].response.cached_input_tokens == 0


def test_status_is_restored_as_enum(db_path):
    save_run(db_path, make_failed_record(RunStatus.TIMEOUT))

    loaded = load_runs(db_path)[0]

    assert loaded.response.status is RunStatus.TIMEOUT


def test_load_preserves_insertion_order(db_path):
    for repetition in range(3):
        save_run(db_path, make_record(repetition=repetition))

    loaded = load_runs(db_path)

    assert [r.repetition for r in loaded] == [0, 1, 2]


def test_load_filters_by_experiment(db_path):
    save_run(db_path, make_record(experiment_id="exp1"))
    save_run(db_path, make_record(experiment_id="exp2"))
    save_run(db_path, make_record(experiment_id="exp1", repetition=1))

    loaded = load_runs(db_path, experiment_id="exp1")

    assert len(loaded) == 2
    assert all(r.experiment_id == "exp1" for r in loaded)


def test_data_persists_across_connections(db_path):
    # każde wywołanie otwiera własne połączenie, więc dane muszą być na dysku
    save_run(db_path, make_record())
    assert len(load_runs(db_path)) == 1