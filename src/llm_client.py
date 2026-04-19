from pydantic import BaseModel
import requests
import os
import json
import re
import time
from transformers import AutoTokenizer
from llm_debug_logger import log_llm_call


class LLMClient:
    def __init__(
        self,
        url: str = "https://api.friendli.ai/serverless/v1",
        api_key: str = None,
        api_model: str = "deepseek-r1",
        hf_tokenizer_id: str = "deepseek-ai/deepseek-v3",
        debug_mode: bool = False,
    ):
        self.url = url
        token = api_key or os.getenv("FRIENDLI_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self.api_model = api_model
        self.debug_mode = debug_mode
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
        # Log the call if debug mode is enabled
        if self.debug_mode:
            log_llm_call(
                debug_mode=True,
                call_type='chat',
                model=self.api_model,
                endpoint=f'{self.url}/chat/completions',
                messages=messages,
                kwargs=kwargs
            )
        
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
            response_obj = requests.request(
                "POST",
                url=f"{self.url}/chat/completions",
                headers=self.headers,
                json=payload,
            ).json()
            
            response = response_obj["choices"][0]["message"]["content"]

            # Log the raw response if debug mode is enabled  
            if self.debug_mode:
                log_llm_call(
                    debug_mode=True,
                    call_type='chat_response',
                    model=self.api_model,
                    endpoint=f'{self.url}/chat/completions',
                    messages=messages,
                    kwargs=kwargs,
                    response=response
                )

        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        response_wo_think = re.sub(
            r"<think>.*?</think>", "", response, flags=re.DOTALL
        ).strip()

        return response_wo_think, reasoning

    def generate(self, messages, **kwargs) -> str:
        """
        Generate a response from the LLM given a list of messages.
        This method is a simplified interface that returns only the response string.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional keyword arguments to pass to the chat method

        Returns:
            str: The generated response text
        """
        response, _ = self.chat(messages, kwargs)
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

    def json_output(
        self,
        prompt: str,
        system_prompt: str = None,
        schema: BaseModel = None,
        reasoning: bool = True,
    ):
        # Prepare messages for potential logging
        messages_for_log = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt}
        ]
        
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
        
        # Log the final output if debug mode is enabled
        if self.debug_mode:
            log_llm_call(
                debug_mode=True,
                call_type='json_output',
                model=self.api_model,
                endpoint=f'{self.url}/chat/completions',
                messages=final_messages,
                kwargs={"response_format": response_format},
                response=raw_json,
                reasoning=reasoning_str,
                parsed_output=parsed
            )
        
        return parsed, reasoning_str


class LocalOpenAILLMClient(LLMClient):
    """
    An alternative LLMClient designed to work with local LLM studio servers
    or any OpenAI-compatible API endpoints.
    """
    def __init__(
        self,
        url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        api_model: str = "local-model",
        hf_tokenizer_id: str = None,
        debug_mode: bool = False,
    ):
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.api_model = api_model
        
        if hf_tokenizer_id:
            token = os.environ.get("HF_TOKEN")
            kwargs = {"legacy": False}
            if token:
                kwargs["token"] = token
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_tokenizer_id,
                **kwargs
            )
        else:
            self.tokenizer = None

    def chat(self, messages, kwargs) -> str:
        if messages[-1]["role"] == "assistant" and self.tokenizer is not None:
            # Fall back to base class prefill logic using tokenizer and /completions
            return super().chat(messages, kwargs)

        payload = {
            "model": self.api_model,
            "messages": messages,
            **kwargs,
        }

        max_retries = 5
        base_delay = 2
        request_timeout = kwargs.get("timeout", 120)
        rate_limit_retry_count = 0
        attempt = 0

        while attempt < max_retries:
            try:
                response = requests.request(
                    "POST",
                    url=f"{self.url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=request_timeout,
                )
                response_obj = response.json()

                if "choices" not in response_obj:
                    # Check for rate limiting (429) and retry infinitely
                    if response.status_code == 429:
                        rate_limit_retry_count += 1
                        delay = min(base_delay * (2 ** min(rate_limit_retry_count, 10)), 300)  # Cap at 300s
                        print(f"[LLMClient] Rate limited (429), retrying in {delay}s... (rate limit retry #{rate_limit_retry_count})")
                        time.sleep(delay)
                        continue  # Don't increment attempt, just retry
                    raise RuntimeError(f"Unexpected response from API: {response_obj}")
                break
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[LLMClient] Request timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
                    attempt += 1
                    continue
                else:
                    print(f"[LLMClient] Request failed after {max_retries} attempts due to timeout")
                    raise
            except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[LLMClient] Request error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay}s...")
                    time.sleep(delay)
                    attempt += 1
                    continue
                else:
                    raise
            attempt += 1

        response_text = response_obj["choices"][0]["message"]["content"]

        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        response_wo_think = re.sub(
            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
        ).strip()

        return response_wo_think, reasoning

    def json_output(
        self,
        prompt: str,
        system_prompt: str = None,
        schema: BaseModel = None,
        reasoning: bool = True,
    ):
        # Prepare messages for potential logging
        messages_for_log = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt}
        ]
        
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
                    "name": "json_schema_response",
                    "schema": schema.model_json_schema(),
                    "strict": True
                },
            }
        else:
            response_format = {
                "type": "json_object",
            }

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
