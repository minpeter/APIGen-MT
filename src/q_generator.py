import random
import re
from typing import List
from tool_manager import ToolManager
from llm_client import LLMClient


class QGenerator:
    def __init__(self, tool_manager: ToolManager, llm: LLMClient):
        self.tools = tool_manager.get_tools()
        self.llm = llm

    def generate(self) -> str:
        prompt = f"도구 목록: {self.tools}\nq를 생성해줘"
        q = self.llm.generate(prompt)
        return q

    def generate_candidate_qs_with_llm(
        self,
        num_to_generate: int,
        existing_qs_list: List[str],
    ) -> List[str]:
        """Generates candidate 'q' strings using the LLM, removing think blocks."""
        system_prompt = """Your ONLY task is to generate realistic, diverse user queries or requests ('q') suitable for an AI assistant with access to specific tools.
        These queries should be answerable using the provided tools. Vary the complexity, phrasing (questions, commands), and the tools potentially required.

        **CRITICAL INSTRUCTIONS:**
        1.  Output ONLY the raw user queries.
        2.  Each query MUST be on a new line.
        3.  **ABSOLUTELY DO NOT** include:
            * Explanations, comments, or justifications.
            * Numbered lists, bullet points, or any formatting other than one query per line.
            * Any text before the first query or after the last query.
        """

        examples_prompt = ""
        if existing_qs_list:
            sample_existing = random.sample(
                existing_qs_list, min(len(existing_qs_list), 5)
            )
            examples_prompt = (
                "Critically, avoid generating queries too similar to these examples:\n- "
                + "\n- ".join(sample_existing)
                + "\n\n"
            )

        user_prompt = f"""Based on the following available tools: {self.tools}
    Generate exactly {num_to_generate} diverse user queries ('q'). Remember to vary the required tools, complexity, and phrasing. {examples_prompt}Output ONLY the queries, one per line:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            completion, _ = self.llm.chat(
                messages=messages,
                kwargs={
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            )
        except Exception as e:
            print(f"An unexpected error occurred during LLM call: {e}")
            return []
        candidate_qs = []
        raw_lines = completion.split("\n")
        for line in raw_lines:
            clean_line = line.strip()
            if self.__is_valid_query_line(clean_line):
                candidate_qs.append(clean_line)
            elif clean_line:
                print(f"Filtered out invalid line: '{clean_line}'")

        return candidate_qs

    def __is_valid_query_line(self, q_text: str) -> bool:
        """Checks if a single line looks like a valid user query."""
        q_text = q_text.strip()
        if not q_text:
            return False
        # More robust filtering based on common LLM reasoning/meta-commentary patterns
        if q_text.startswith(
            (
                "- ",
                "* ",
                "Okay,",
                "First,",
                "Next,",
                "Now,",
                "Let me",
                "Wait,",
                "Also,",
                "###",
                "//",
                "```",
                "Queries",
                "That's",
                "This should",
                "Avoid",
                "Check for",
                "Example",
                "User Query:",
                "Generated Query:",
            )
        ):
            return False
        if q_text.endswith(":") or "→" in q_text:
            return False
        if re.match(r"^\d+\.", q_text):
            return False
        if len(q_text.split()) <= 1 and not re.search(r"[a-zA-Z]", q_text):
            return False
        if "/" in q_text and "." in q_text and " " not in q_text:
            return False
        # Filter lines that are likely descriptions of tools or parameters
        if any(
            tool_name in q_text.lower()
            for tool_name in ["get_weather", "get_news", "get_current_location"]
        ):
            if (
                "parameter" in q_text.lower()
                or "require" in q_text.lower()
                or "tool" in q_text.lower()
            ):
                return False
        return True


if __name__ == "__main__":
    llm_client = LLMClient()
    tool_manager = ToolManager(llm=llm_client)
    q_generator = QGenerator(tool_manager=tool_manager, llm=llm_client)

    # Generate multiple candidate queries
    existing_qs = []
    candidate_qs = q_generator.generate_candidate_qs_with_llm(5, existing_qs)
    print("Generated candidate queries:", candidate_qs)
