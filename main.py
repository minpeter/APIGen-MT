import os
from pathlib import Path
from src.tool_manager import ToolManager
from src.llm_client import LLMClient
from src.q_generator import QGenerator
from src.apigen_mt_generator import APIGenMTGenerator
from src.executor import VirtualToolExecutor


def main():
    base = Path(__file__).parent
    tools_path = base / "data/tools.yaml"
    api_key = os.getenv("UV_API_KEY")

    tool_mgr = ToolManager(tools_path)
    llm = LLMClient(api_key)
    qg = QGenerator(tool_mgr, llm)
    q = qg.generate()

    gen = APIGenMTGenerator(llm)
    output = gen.generate(q)

    executor = VirtualToolExecutor(tool_mgr.load_tools())
    if executor.verify(output):
        print("검증 통과:\n", output)
    else:
        print("검증 실패")


if __name__ == "__main__":
    main()
