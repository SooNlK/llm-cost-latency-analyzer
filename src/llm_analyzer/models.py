from dataclasses import dataclass
from enum import Enum

class RunStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    text: str
    max_output_tokens: int
    system: str | None = None
    shared_prefix_id: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    status: RunStatus
    text: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None   
    ttft_ms: float | None = None             
    total_latency_ms: float | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class RunRecord:
    experiment_id: str
    timestamp: str                     
    provider: str
    model: str
    prompt_id: str
    repetition: int
    response: LLMResponse
    run_id: int | None = None


@dataclass(frozen=True)
class Price:
    model: str
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float
    price_date: str 


@dataclass(frozen=True)
class Scenario:
    name: str
    requests_per_day: int
    category_mix: dict[str, float]
    days_per_month: int = 30

    def __post_init__(self) -> None:
        total = sum(self.category_mix.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"category_mix must sum to 1, got {total}")