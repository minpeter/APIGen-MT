from pydantic import BaseModel
import requests, os, json, re
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
            response = self.completions(
                prompt=prompt,
                kwargs=kwargs,
            )
            # reasoning parser not working, if prefill is True
            return response, ""
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

            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            reasoning = think_match.group(1).strip() if think_match else ""
            response_wo_think = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL
            ).strip()

            return response_wo_think, reasoning

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

    def json_output(
        self,
        prompt: str,
        system_prompt: str = None,
        schema: BaseModel = None,
        reasoning: bool = True,
    ):
        if not system_prompt and schema is not None:
            system_prompt = f"""Extract the information.
            follow the schema: {schema.model_json_schema()}
            """

        if system_prompt is None:
            system_prompt = (
                "You are an information extraction assistant. "
                "Extract the required information from the user's input and respond ONLY with a valid, minified JSON object. "
                "Do not include any explanations or extra text. "
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if reasoning:
            reasoning_messages = [
                *messages,
                {
                    "role": "assistant",
                    "content": "<think>\n",
                },
            ]
            # if prefill is True, reasoning parser not working
            reasoning_str, _ = self.chat(
                messages=reasoning_messages,
                kwargs={
                    "stop": ["</think>"],
                },
            )
        else:
            reasoning_str = ""

        final_messages = [
            *messages,
            {
                "role": "assistant",
                "content": "<think>\n" + reasoning_str + "\n</think>\n",
            },
        ]

        if schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "schema": schema.model_json_schema(),
                },
            }
        else:
            response_format = {
                "type": "json_object",
            }

        # if prefill is True, reasoning parser not working
        raw_json, _ = self.chat(
            messages=final_messages,
            kwargs={"response_format": response_format},
        )
        parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        return parsed, reasoning_str


if __name__ == "__main__":
    # Example usage
    llm_client = LLMClient()

    # response = llm_client.chat(
    #     messages=[{"role": "user", "content": "1+1=?"}],
    #     kwargs={"temperature": 0.7},
    # )
    # response = llm_client.completions(
    #     prompt="Hello, how are you?",
    #     kwargs={"stop": ["\n"]},
    # )

    class schema(BaseModel):
        name: str
        age: int

    response = llm_client.json_output(
        prompt="Extract the name and age from this text: John Doe, 30 years old.",
        # schema=schema,
        reasoning=True,
    )

    print(response)
