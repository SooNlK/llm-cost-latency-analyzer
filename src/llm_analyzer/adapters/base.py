from abc import ABC, abstractmethod

from llm_analyzer.models import LLMResponse, Prompt


class LLMAdapter(ABC):
    """Wspólny interfejs adapterów. Runner zna tylko ten interfejs."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    @abstractmethod
    def complete(self, prompt: Prompt) -> LLMResponse:
        """Wysyła prompt do modelu i zwraca wynik.

        Błędy (sieć, limity, timeout) adapter zgłasza jako LLMResponse
        ze statusem ERROR lub TIMEOUT, a nie jako wyjątek.
        """