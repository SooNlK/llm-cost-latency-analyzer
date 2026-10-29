from llm_analyzer.models import RunStatus,  LLMResponse, Price

def request_cost(response: LLMResponse, price: Price) -> float:
    if response.status != RunStatus.SUCCESS:
        raise ValueError("Cannot compute cost of a failed request")
    
    if response.output_tokens is None or response.input_tokens is None:
        raise ValueError("Cannot compute cost without input and output tokens number")
    
    cached = response.cached_input_tokens
    cached_price = price.cached_input_per_1m

    if cached is None:
        cached = 0

    if cached_price is None:
        cached_price = price.input_per_1m
    
    cost = ((response.input_tokens - cached) * price.input_per_1m + cached * cached_price + response.output_tokens * price.output_per_1m) / 1000000

    return cost