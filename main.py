import os
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).parent / "src" / "generate_step_by_step.py"
    os.execv(sys.executable, [sys.executable, str(script), *sys.argv[1:]])


if __name__ == "__main__":
    main()
