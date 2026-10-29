import pytest

from llm_analyzer.models import LLMResponse, RunStatus
from llm_analyzer.analysis.stats import error_rate, mean, median, p95, tokens_per_second

LATENCIES = [100.0, 200.0, 300.0, 400.0, 500.0]

def make_response(status: RunStatus) -> LLMResponse:
    return LLMResponse(status=status)

#mean
def test_mean():
    assert mean(LATENCIES) == pytest.approx(300.0)


def test_mean_single_value():
    assert mean(250.0) == pytest.approx(250.0)


def test_mean_empty_raises():
    with pytest.raises(ValueError):
        mean([])


#median 
def test_median_odd_length():
    assert median(LATENCIES) == pytest.approx(300.0)


def test_median_even_length():
    assert median([100.0, 200.0, 300.0, 400.0]) == pytest.approx(250.0)


def test_median_unsorted_input():
    assert median([500.0, 100.0, 300.0, 200.0, 400.0]) == pytest.approx(300.0)


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median([])


#p95
def test_p95():
    # pozycja 0.95 * (5 - 1) = 3.8 -> 400 + 0.8 * (500 - 400) = 480
    assert p95(LATENCIES) == pytest.approx(480.0)


def test_p95_unsorted_input():
    assert p95([500.0, 100.0, 300.0, 200.0, 400.0]) == pytest.approx(480.0)


def test_p95_single_value():
    assert p95([250.0]) == pytest.approx(250.0)


def test_p95_empty_raises():
    with pytest.raises(ValueError):
        p95([])


def test_functions_do_not_modify_input():
    data = [500.0, 100.0, 300.0]
    median(data)
    p95(data)
    assert data == [500.0, 100.0, 300.0]


#token per second
def test_token_per_second():
    assert tokens_per_second(500, 2000.0) == pytest.approx(250.0)


def test_tokens_per_second_zero_latency_raises():
    with pytest.raises(ValueError):
        tokens_per_second(500, 0.0)


def test_error_rate():
    responses = [
        make_response(RunStatus.SUCCESS),
        make_response(RunStatus.SUCCESS),
        make_response(RunStatus.SUCCESS),
        make_response(RunStatus.ERROR),
    ]
    assert error_rate(responses) == pytest.approx(0.25)


def test_error_rate_no_failures():
    responses = [make_response(RunStatus.SUCCESS)] * 3
    assert error_rate(responses) == pytest.approx(0.0)


def test_error_rate_timeout_counts_as_failure():
    responses = [
        make_response(RunStatus.SUCCESS),
        make_response(RunStatus.TIMEOUT),
    ]
    assert error_rate(responses) == pytest.approx(0.5)


def test_error_rate_empty_raises():
    with pytest.raises(ValueError):
        error_rate([])