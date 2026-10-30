import numpy as np

from llm_analyzer.models import RunStatus

def mean(latencies: list) -> float:
    if not latencies:
        raise ValueError("Cannot calculate mean from empty list")

    return np.mean(latencies)

def median(latencies: list) -> float:
    if not latencies:
        raise ValueError("Cannot calculate meidan from empty list")

    return np.median(latencies)

def p95(latencies: list) -> float:
    if not latencies:
        raise ValueError("Cannot calcualte p95 from empty list")
    
    return np.percentile(latencies, 95)

def tokens_per_second(tokens: float, time: float) -> float:
    if time == 0:
        raise ValueError("Cannot devide by zero")

    return tokens / (time / 1000) 

def error_rate(responses: list[RunStatus]) -> float:
    if not responses:
        raise ValueError("Cannot calculate error rate from empty list")

    failures = sum(1 for r in responses if r.status != RunStatus.SUCCESS)

    return failures / len(responses)