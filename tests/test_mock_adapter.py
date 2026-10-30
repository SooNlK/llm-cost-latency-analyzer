import time

import pytest

from llm_analyzer.adapters.base import LLMAdapter
from llm_analyzer.adapters.mock_adapter import MockAdapter
from llm_analyzer.models import LLMResponse, Prompt, RunStatus


def make_prompt(
    prompt_id: str = "p1",
    text: str = "What is the capital of France?",
    max_output_tokens: int = 100,
) -> Prompt:
    return Prompt(
        id=prompt_id,
        category="qa",
        text=text,
        max_output_tokens=max_output_tokens,
    )


#interfejs

def test_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LLMAdapter()


def test_mock_adapter_implements_interface():
    assert isinstance(MockAdapter(), LLMAdapter)


def test_adapter_exposes_provider_and_model():
    adapter = MockAdapter(model="my-mock")
    assert adapter.provider == "mock"
    assert adapter.model == "my-mock"


#udane wywołanie

def test_complete_returns_successful_response():
    response = MockAdapter().complete(make_prompt())

    assert isinstance(response, LLMResponse)
    assert response.status is RunStatus.SUCCESS
    assert response.text
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.error_message is None


def test_output_tokens_do_not_exceed_limit():
    prompt = make_prompt(max_output_tokens=5)
    response = MockAdapter().complete(prompt)
    assert response.output_tokens <= 5


def test_latency_is_consistent():
    response = MockAdapter().complete(make_prompt())

    assert response.ttft_ms > 0
    assert response.total_latency_ms >= response.ttft_ms


def test_mock_reports_no_cache_information():
    # mock nie symuluje cache: None = "dostawca nie raportuje"
    response = MockAdapter().complete(make_prompt())
    assert response.cached_input_tokens is None


# determinizm

def test_same_prompt_gives_same_response():
    adapter = MockAdapter()
    prompt = make_prompt()

    assert adapter.complete(prompt) == adapter.complete(prompt)


def test_two_adapters_give_same_response():
    prompt = make_prompt()

    assert MockAdapter().complete(prompt) == MockAdapter().complete(prompt)


def test_longer_prompt_has_more_input_tokens():
    short = make_prompt(text="Hi")
    long = make_prompt(text="word " * 200)
    adapter = MockAdapter()

    assert (
        adapter.complete(long).input_tokens
        > adapter.complete(short).input_tokens
    )


#błędy

def test_configured_failure_returns_error_response():
    adapter = MockAdapter(failing_prompt_ids={"bad"})

    response = adapter.complete(make_prompt(prompt_id="bad"))

    assert response.status is RunStatus.ERROR
    assert response.error_message


def test_failed_response_has_no_measurements():
    # nieudane wywołanie nie może wyglądać jak sukces z zerem tokenów
    adapter = MockAdapter(failing_prompt_ids={"bad"})

    response = adapter.complete(make_prompt(prompt_id="bad"))

    assert response.text is None
    assert response.input_tokens is None
    assert response.output_tokens is None
    assert response.cached_input_tokens is None
    assert response.ttft_ms is None
    assert response.total_latency_ms is None


def test_failure_does_not_raise_and_other_prompts_still_work():
    adapter = MockAdapter(failing_prompt_ids={"bad"})

    adapter.complete(make_prompt(prompt_id="bad"))
    ok = adapter.complete(make_prompt(prompt_id="good"))

    assert ok.status is RunStatus.SUCCESS


#szybkość

def test_mock_does_not_actually_sleep():
    # czas jest symulowany, więc testy i runner na mocku mają być szybkie
    start = time.perf_counter()
    for i in range(50):
        MockAdapter().complete(make_prompt(prompt_id=f"p{i}"))

    assert time.perf_counter() - start < 1.0