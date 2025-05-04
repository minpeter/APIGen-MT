from pydantic import BaseModel
import requests, os, json
from transformers import AutoTokenizer


class LLMClient:
    def __init__(
        self,
        url: str = "https://api.friendli.ai/serverless/v1",
        api_key: str = None,
        api_model: str = "deepseek-r1",
        hf_tokenizer_id: str = "deepseek-ai/deepseek-v3",
    ):
        self.url = url
        token = api_key or os.getenv("FRIENDLI_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.api_model = api_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_tokenizer_id,
            token=os.environ.get("HF_TOKEN"),
            legacy=False,
        )

    def __apply_chat_template(self, messages, prefill: bool = True) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=prefill
        )
        return prompt

    def chat(self, messages, kwargs) -> str:
        if messages[-1]["role"] == "assistant":
            # prefill = True
            prompt = self.__apply_chat_template(messages, prefill=True)
            return self.completions(
                prompt=prompt,
                kwargs=kwargs,
            )
        else:
            # prefill = False
            payload = {
                "model": self.api_model,
                "messages": messages,
                **kwargs,
            }
            response = requests.request(
                "POST",
                url=f"{self.url}/chat/completions",
                headers=self.headers,
                json=payload,
            ).json()["choices"][0]["message"]["content"]

            return response

    def completions(self, prompt: str, kwargs) -> str:
        payload = {
            "model": self.api_model,
            "prompt": prompt,
            **kwargs,
        }
        response = requests.request(
            "POST",
            url=f"{self.url}/completions",
            headers=self.headers,
            json=payload,
        ).json()["choices"][0]["text"]

        return response

    def structured_output(
        self,
        extraction_target: str,
        extraction_schema: BaseModel,
        reasoning: bool = True,
    ):
        extraction_system_prompt = f"""Extract the information.
        follow the schema: {extraction_schema.model_json_schema()}
        """
        messages = [
            {"role": "system", "content": extraction_system_prompt},
            {"role": "user", "content": extraction_target},
        ]
        if reasoning:
            reasoning_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": "<think>\n",
                },
            ]
            reasoning = self.chat(
                messages=reasoning_messages,
                kwargs={
                    "stop": ["</think>"],
                },
            )
        else:
            reasoning = ""

        final_messages = [
            *messages,
            {"role": "assistant", "content": "<think>\n" + reasoning + "\n</think>\n"},
        ]

        raw_json = self.chat(
            messages=final_messages,
            kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": extraction_schema.model_json_schema(),
                    },
                }
            },
        )
        parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        return parsed, reasoning


if __name__ == "__main__":
    # Example usage
    llm_client = LLMClient()

    # response = llm_client.chat(
    #     messages=[{"role": "user", "content": "Hello, how are you?"}],
    #     kwargs={"temperature": 0.7},
    # )
    # response = llm_client.completions(
    #     prompt="Hello, how are you?",
    #     kwargs={"stop": ["\n"]},
    # )

    class extraction_schema(BaseModel):
        name: str
        age: int

    response = llm_client.structured_output(
        extraction_target="Extract the name and age from this text: John Doe, 30 years old.",
        extraction_schema=extraction_schema,
        reasoning=True,
    )
    print(response)
