from datetime import datetime

import pytest

from llm_analyzer.adapters.base import LLMAdapter
from llm_analyzer.adapters.mock_adapter import MockAdapter
from llm_analyzer.models import LLMResponse, Prompt, RunStatus
from llm_analyzer.runner import run_experiment
from llm_analyzer.storage import load_runs


def make_prompt(prompt_id: str) -> Prompt:
    return Prompt(
        id=prompt_id,
        category="qa",
        text="What is the capital of France?",
        max_output_tokens=100,
    )


class RaisingAdapter(LLMAdapter):
    """Adapter, który łamie umowę i rzuca wyjątek."""

    def __init__(self) -> None:
        super().__init__(provider="broken", model="broken-model")

    def complete(self, prompt: Prompt) -> LLMResponse:
        raise RuntimeError("network down")


@pytest.fixture
def db_path(tmp_path):
    # plik jeszcze nie istnieje: runner sam ma utworzyć bazę
    return tmp_path / "runner.db"


#liczba i zawartość rekordów

def test_record_count(db_path):
    adapters = [MockAdapter(model="a"), MockAdapter(model="b")]
    prompts = [make_prompt("p1"), make_prompt("p2"), make_prompt("p3")]

    run_experiment(adapters, prompts, repetitions=2, db_path=db_path)

    assert len(load_runs(db_path)) == 2 * 3 * 2


def test_records_contain_correct_context(db_path):
    experiment_id = run_experiment(
        [MockAdapter(model="a")], [make_prompt("p1")],
        repetitions=1, db_path=db_path,
    )

    record = load_runs(db_path)[0]

    assert record.experiment_id == experiment_id
    assert record.provider == "mock"
    assert record.model == "a"
    assert record.prompt_id == "p1"
    assert record.repetition == 0
    assert record.response.status is RunStatus.SUCCESS


def test_timestamp_is_iso_format(db_path):
    run_experiment(
        [MockAdapter()], [make_prompt("p1")], repetitions=1, db_path=db_path
    )

    record = load_runs(db_path)[0]

    datetime.fromisoformat(record.timestamp)  # rzuca wyjątek, jeśli zły format


#kolejność (przeplatanie modeli)

def test_models_are_interleaved_per_prompt(db_path):
    adapters = [MockAdapter(model="a"), MockAdapter(model="b")]
    prompts = [make_prompt("p1"), make_prompt("p2")]

    run_experiment(adapters, prompts, repetitions=1, db_path=db_path)

    order = [(r.prompt_id, r.model) for r in load_runs(db_path)]
    assert order == [("p1", "a"), ("p1", "b"), ("p2", "a"), ("p2", "b")]


def test_repetitions_are_outermost_loop(db_path):
    prompts = [make_prompt("p1"), make_prompt("p2")]

    run_experiment([MockAdapter()], prompts, repetitions=2, db_path=db_path)

    order = [(r.repetition, r.prompt_id) for r in load_runs(db_path)]
    assert order == [(0, "p1"), (0, "p2"), (1, "p1"), (1, "p2")]


#identyfikator eksperymentu

def test_generated_experiment_ids_are_unique(db_path):
    first = run_experiment(
        [MockAdapter()], [make_prompt("p1")], repetitions=1, db_path=db_path
    )
    second = run_experiment(
        [MockAdapter()], [make_prompt("p1")], repetitions=1, db_path=db_path
    )

    assert first != second
    assert len(load_runs(db_path, experiment_id=first)) == 1
    assert len(load_runs(db_path, experiment_id=second)) == 1


def test_custom_experiment_id_is_used(db_path):
    result = run_experiment(
        [MockAdapter()], [make_prompt("p1")],
        repetitions=1, db_path=db_path, experiment_id="my-exp",
    )

    assert result == "my-exp"
    assert load_runs(db_path)[0].experiment_id == "my-exp"


#błędy

def test_failed_call_is_recorded_and_experiment_continues(db_path):
    adapter = MockAdapter(failing_prompt_ids={"p2"})
    prompts = [make_prompt("p1"), make_prompt("p2"), make_prompt("p3")]

    run_experiment([adapter], prompts, repetitions=1, db_path=db_path)

    statuses = [r.response.status for r in load_runs(db_path)]
    assert statuses == [RunStatus.SUCCESS, RunStatus.ERROR, RunStatus.SUCCESS]


def test_adapter_exception_is_recorded_as_error(db_path):
    adapters = [RaisingAdapter(), MockAdapter()]

    run_experiment(
        adapters, [make_prompt("p1")], repetitions=1, db_path=db_path
    )

    records = load_runs(db_path)
    assert len(records) == 2
    assert records[0].provider == "broken"
    assert records[0].response.status is RunStatus.ERROR
    assert "network down" in records[0].response.error_message
    assert records[0].response.input_tokens is None
    assert records[1].response.status is RunStatus.SUCCESS


#walidacja

def test_no_adapters_raises(db_path):
    with pytest.raises(ValueError):
        run_experiment([], [make_prompt("p1")], repetitions=1, db_path=db_path)


def test_no_prompts_raises(db_path):
    with pytest.raises(ValueError):
        run_experiment([MockAdapter()], [], repetitions=1, db_path=db_path)


@pytest.mark.parametrize("repetitions", [0, -1])
def test_non_positive_repetitions_raises(db_path, repetitions):
    with pytest.raises(ValueError):
        run_experiment(
            [MockAdapter()], [make_prompt("p1")],
            repetitions=repetitions, db_path=db_path,
        )