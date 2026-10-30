from llm_analyzer.adapters.base import LLMAdapter
from llm_analyzer.models import LLMResponse, Prompt, RunStatus

# Parametry symulacji: deterministyczne, bez losowości i bez sleep.
DEFAULT_OUTPUT_TOKENS = 50
TTFT_BASE_MS = 100.0
TTFT_PER_INPUT_TOKEN_MS = 0.5
PER_OUTPUT_TOKEN_MS = 10.0


class MockAdapter(LLMAdapter):
    """Symuluje model: wynik zależy tylko od promptu."""

    def __init__(
        self,
        model: str = "mock-model",
        failing_prompt_ids: set[str] | None = None,
    ) -> None:
        super().__init__(provider="mock", model=model)
        self.failing_prompt_ids = set(failing_prompt_ids or ())

    def complete(self, prompt: Prompt) -> LLMResponse:
        if prompt.id in self.failing_prompt_ids:
            return LLMResponse(
                status=RunStatus.ERROR,
                error_message=f"Simulated failure for prompt {prompt.id}",
            )

        input_tokens = self._count_input_tokens(prompt)
        output_tokens = min(prompt.max_output_tokens, DEFAULT_OUTPUT_TOKENS)

        ttft_ms = TTFT_BASE_MS + input_tokens * TTFT_PER_INPUT_TOKEN_MS
        total_latency_ms = ttft_ms + output_tokens * PER_OUTPUT_TOKEN_MS

        return LLMResponse(
            status=RunStatus.SUCCESS,
            text=" ".join(["token"] * output_tokens),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=None,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
        )

    @staticmethod
    def _count_input_tokens(prompt: Prompt) -> int:
        text = prompt.text if prompt.system is None else f"{prompt.system} {prompt.text}"
        return max(1, len(text.split()))