"""Read and navigation operations for the virtual filesystem."""

import os

from .gorilla_file_system_core import GorillaFileSystemCore


class GorillaFileSystemReadOperations(GorillaFileSystemCore):
    """Provide non-mutating filesystem tool operations."""

    def cat(self, file_name: str) -> dict[str, object]:
        """Read and return file contents."""
        if err := self._validate_local_name(file_name, "file_name"):
            return {"file_content": f"Error: {err['error']}"}
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {"file_content": f"Error: File '{file_name}' not found"}
        if path.is_dir():
            return {"file_content": f"Error: '{file_name}' is not a file"}
        return {"file_content": path.read_text()}

    def cd(self, folder: str) -> dict[str, object]:
        """Change current working directory by one folder level."""
        folder = folder.rstrip("/") or "/"
        if folder == "..":
            current = self._get_relative_path(".")
            if current != self._get_relative_path("/"):
                self.current_dir: str = str(current.parent)
            return {
                "current_working_directory": self._get_virtual_current_dir()
            }
        if folder not in {".", "/"} and ("/" in folder or "\\" in folder):
            return {
                "current_working_directory": (
                    f"Error: Unsupported path '{folder}'"
                )
            }

        path = self._get_relative_path(folder)
        if not path.exists():
            return {
                "current_working_directory": (
                    f"Error: Directory '{folder}' not found"
                )
            }
        if not path.is_dir():
            return {
                "current_working_directory": (
                    f"Error: '{folder}' is not a directory"
                )
            }

        self.current_dir = str(path)
        return {"current_working_directory": self._get_virtual_current_dir()}

    def du(
        self,
        path: str = ".",
        human_readable: bool = False,
    ) -> dict[str, object]:
        """Calculate disk usage."""
        target_path = self._get_relative_path(path)
        if not target_path.exists():
            return {"disk_usage": f"Error: Path '{path}' not found"}

        if target_path.is_file():
            total = target_path.stat().st_size
        else:
            total = sum(
                file.stat().st_size
                for file in target_path.rglob("*")
                if file.is_file()
            )

        if human_readable:
            if total < 1024:
                return {"disk_usage": f"{total}B"}
            if total < 1024 * 1024:
                return {"disk_usage": f"{total / 1024:.1f}KB"}
            return {"disk_usage": f"{total / (1024 * 1024):.1f}MB"}
        return {"disk_usage": str(total)}

    def find(
        self,
        path: str = ".",
        name: str | None = None,
    ) -> dict[str, object]:
        """Find files and directories whose names contain the requested text."""
        path = path or "."
        search_root = self._get_relative_path(path)
        if not search_root.exists() or not search_root.is_dir():
            return {"matches": []}

        pattern = name if name and name != "None" else None
        paths = sorted(
            search_root.rglob("*"),
            key=lambda item: item.relative_to(search_root).as_posix(),
        )
        matches = [
            item.relative_to(search_root).as_posix()
            for item in paths
            if pattern is None or pattern in item.name
        ]
        return {"matches": matches}

    def grep(self, file_name: str, pattern: str) -> dict[str, object]:
        """Search for a pattern in a local file."""
        if self._validate_local_name(file_name, "file_name"):
            return {"matching_lines": []}
        path = self._get_relative_path(file_name)
        if not path.exists() or path.is_dir():
            return {"matching_lines": []}
        matching = [
            line for line in path.read_text().splitlines() if pattern in line
        ]
        return {"matching_lines": matching}

    def head(self, file_name: str, lines: int = 10) -> dict[str, object]:
        """Return first n lines of file."""
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {
                "first_n_lines": "",
                "error": f"File '{file_name}' not found",
            }
        all_lines = path.read_text().splitlines()
        return {"first_n_lines": "\n".join(all_lines[:lines])}

    def ls(self, a: bool = False) -> dict[str, object]:
        """List directory contents.

        Args:
            a: If True, show hidden files (starting with .).
        """
        try:
            target_path = self._get_relative_path(".")
            if not target_path.exists() or not target_path.is_dir():
                return {"current_directory_content": []}
            entries = os.listdir(target_path)
            if not a:
                entries = [
                    entry for entry in entries if not entry.startswith(".")
                ]
            return {"current_directory_content": sorted(entries)}
        except OSError:
            return {"current_directory_content": []}

    def pwd(self) -> dict[str, object]:
        """Return current working directory."""
        return {"current_working_directory": self._get_virtual_current_dir()}

    def tail(self, file_name: str, lines: int = 10) -> dict[str, object]:
        """Return the last n lines of a local file."""
        if err := self._validate_local_name(file_name, "file_name"):
            return {"last_lines": f"Error: {err['error']}"}
        path = self._get_relative_path(file_name)
        if not path.exists() or path.is_dir():
            return {"last_lines": f"Error: File '{file_name}' not found"}
        all_lines = path.read_text().splitlines()
        selected = all_lines[-lines:] if lines > 0 else []
        return {"last_lines": "\n".join(selected)}

    def wc(self, file_name: str, mode: str = "l") -> dict[str, object]:
        """Count lines, words, or characters in a local file."""
        mode = mode or "l"
        unit = {"l": "lines", "w": "words", "c": "characters"}.get(
            mode,
            "lines",
        )
        if self._validate_local_name(file_name, "file_name"):
            return {"count": 0, "type": unit}
        path = self._get_relative_path(file_name)
        if not path.exists() or path.is_dir():
            return {"count": 0, "type": unit}
        content = path.read_text()

        if mode == "w":
            count = len(content.split())
        elif mode == "c":
            count = len(content)
        else:
            count = len(content.splitlines())
        return {"count": count, "type": unit}

    def diff(self, file_name1: str, file_name2: str) -> dict[str, object]:
        """Compare two local files line by line."""
        if err := self._validate_local_name(file_name1, "file_name1"):
            return {"diff_lines": f"Error: {err['error']}"}
        if err := self._validate_local_name(file_name2, "file_name2"):
            return {"diff_lines": f"Error: {err['error']}"}
        path1 = self._get_relative_path(file_name1)
        path2 = self._get_relative_path(file_name2)

        if not path1.exists() or path1.is_dir():
            return {"diff_lines": f"Error: File '{file_name1}' not found"}
        if not path2.exists() or path2.is_dir():
            return {"diff_lines": f"Error: File '{file_name2}' not found"}

        lines1 = path1.read_text().splitlines()
        lines2 = path2.read_text().splitlines()
        differences: list[str] = []
        for index in range(max(len(lines1), len(lines2))):
            line1 = lines1[index] if index < len(lines1) else ""
            line2 = lines2[index] if index < len(lines2) else ""
            if line1 != line2:
                differences.append(f"- {line1}\n+ {line2}")
        return {"diff_lines": "\n".join(differences)}

    def sort(self, file_name: str, reverse: bool = False) -> dict[str, object]:
        """Sort and return local file contents."""
        if err := self._validate_local_name(file_name, "file_name"):
            return {"sorted_content": f"Error: {err['error']}"}
        path = self._get_relative_path(file_name)

        if not path.exists() or path.is_dir():
            return {"sorted_content": f"Error: File '{file_name}' not found"}

        lines = sorted(path.read_text().splitlines(), reverse=reverse)
        return {"sorted_content": "\n".join(lines)}
