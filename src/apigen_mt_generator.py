from .llm_client import LLMClient


class APIGenMTGenerator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, q: str) -> str:
        prompt = f"{q}\napigen_mt의 부족한 부분을 채워줘. 백트레이싱 포함."
        result = self.llm.generate(prompt)
        return result
