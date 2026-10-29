import pytest

from llm_analyzer.analysis.cost import request_cost
from llm_analyzer.models import LLMResponse, Price, RunStatus


def make_price(cached: float | None = 0.30) -> Price:
    return Price(
        model="test-model",
        input_per_1m=3.00,
        output_per_1m=15.00,
        cached_input_per_1m=cached,
        price_date="2026-10-29",
    )


def make_response(
    input_tokens: int | None = 1000,
    output_tokens: int | None = 500,
    cached: int | None = None,
    status: RunStatus = RunStatus.SUCCESS,
) -> LLMResponse:
    return LLMResponse(
        status=status,
        text="ok" if status == RunStatus.SUCCESS else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
    )


def test_cost_with_cached_tokens():
    response = make_response(cached=200)
    assert request_cost(response, make_price()) == pytest.approx(0.00996)


def test_cost_without_cache_info():
    response = make_response(cached=None)
    assert request_cost(response, make_price()) == pytest.approx(0.0105)


def test_cost_with_zero_cached_tokens():
    response = make_response(cached=0)
    assert request_cost(response, make_price()) == pytest.approx(0.0105)


def test_cost_with_zero_output_tokens():
    response = make_response(output_tokens=0)
    assert request_cost(response, make_price()) == pytest.approx(0.003)


def test_cached_tokens_billed_at_normal_price_when_no_cache_price():
    response = make_response(cached=200)
    price = make_price(cached=None)
    assert request_cost(response, price) == pytest.approx(0.0105)


def test_failed_response_raises():
    response = make_response(
        input_tokens=None, output_tokens=None, status=RunStatus.ERROR
    )
    with pytest.raises(ValueError):
        request_cost(response, make_price())


def test_success_without_token_counts_raises():
    response = make_response(input_tokens=None, output_tokens=None)
    with pytest.raises(ValueError):
        request_cost(response, make_price())