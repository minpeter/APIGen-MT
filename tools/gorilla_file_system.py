"""Auto-generated GorillaFileSystem implementation."""

import copy
import json
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class GorillaFileSystem:
    """A simple file system that allows users to perform basic file operations."""

    def __init__(self, initial_config: dict) -> None:
        """Initialize the GorillaFileSystem with the given configuration."""
        self.root = copy.deepcopy(initial_config)
        self.current_dir: List[str] = []

    def _get_current_directory_node(self) -> Optional[Dict[str, Any]]:
        """Helper to get the node of the current working directory."""
        node = self.root
        for part in self.current_dir:
            if part in node and node[part]["type"] == "directory":
                node = node[part]["contents"]
            else:
                return None
        return node

    def _get_node_at_path(self, path_parts: List[str]) -> Optional[Dict[str, Any]]:
        """Helper to get a node at a specific path from root."""
        node = self.root
        for part in path_parts:
            if part in node and node[part]["type"] == "directory":
                node = node[part]["contents"]
            elif part in node and node[part]["type"] == "file":
                return node[part]
            else:
                return None
        return node

    def _get_parent_and_target(self, path_parts: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
        """Helper to get parent node and target name."""
        if not path_parts:
            return None, None, ""
        parent_path = path_parts[:-1]
        target_name = path_parts[-1]
        parent_node = self._get_node_at_path(parent_path) if parent_path else self._get_current_directory_node()
        if parent_node is None:
            return None, None, target_name
        return parent_node, parent_node.get(target_name), target_name

    def cat(self, file_name: str) -> Dict[str, Any]:
        """Display the contents of a file of any extension from current directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"file_content": f"Error: Current directory does not exist."}
        if file_name not in current_node:
            return {"file_content": f"Error: File '{file_name}' not found."}
        file_entry = current_node[file_name]
        if file_entry["type"] != "file":
            return {"file_content": f"Error: '{file_name}' is not a file."}
        return {"file_content": file_entry.get("content", "")}

    def cd(self, folder: str) -> Dict[str, Any]:
        """Change the current working directory to the specified folder."""
        if folder == "..":
            if self.current_dir:
                self.current_dir.pop()
            path_str = "/" + "/".join(self.current_dir) if self.current_dir else "/"
            return {"current_working_directory": path_str}
        
        current_node = self._get_current_directory_node()
        if current_node is None:
            path_str = "/" + "/".join(self.current_dir) if self.current_dir else "/"
            return {"current_working_directory": path_str}
            
        if folder in current_node and current_node[folder]["type"] == "directory":
            self.current_dir.append(folder)
            path_str = "/" + "/".join(self.current_dir)
            return {"current_working_directory": path_str}
        else:
            path_str = "/" + "/".join(self.current_dir) if self.current_dir else "/"
            return {"current_working_directory": path_str}

    def cp(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy a file or directory from one location to another."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"result": "Error: Current directory does not exist."}
        if source not in current_node:
            return {"result": f"Error: Source '{source}' not found."}
        
        source_entry = current_node[source]
        source_copy = copy.deepcopy(source_entry)
        
        if destination in current_node:
            dest_entry = current_node[destination]
            if dest_entry["type"] == "directory":
                dest_entry["contents"][source] = source_copy
                return {"result": f"Copied '{source}' into directory '{destination}'."}
            else:
                return {"result": f"Error: Destination '{destination}' already exists as a file."}
        else:
            current_node[destination] = source_copy
            return {"result": f"Copied '{source}' to '{destination}'."}

    def diff(self, file_name1: str, file_name2: str) -> Dict[str, Any]:
        """Compare two files of any extension line by line at the current directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"diff_lines": "Error: Current directory does not exist."}
        
        if file_name1 not in current_node or current_node[file_name1]["type"] != "file":
            return {"diff_lines": f"Error: File '{file_name1}' not found or is not a file."}
        if file_name2 not in current_node or current_node[file_name2]["type"] != "file":
            return {"diff_lines": f"Error: File '{file_name2}' not found or is not a file."}
        
        lines1 = current_node[file_name1]["content"].splitlines()
        lines2 = current_node[file_name2]["content"].splitlines()
        
        diff_output = []
        max_len = max(len(lines1), len(lines2))
        for i in range(max_len):
            line1 = lines1[i] if i < len(lines1) else None
            line2 = lines2[i] if i < len(lines2) else None
            if line1 != line2:
                if line1 is not None:
                    diff_output.append(f"< {line1}")
                if line2 is not None:
                    diff_output.append(f"> {line2}")
        
        return {"diff_lines": "\n".join(diff_output)}

    def du(self, human_readable: bool = False) -> Dict[str, Any]:
        """Estimate the disk usage of a directory and its contents."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"disk_usage": "Error: Current directory does not exist."}
        
        def calculate_size(node: Dict[str, Any]) -> int:
            total = 0
            for name, entry in node.items():
                if isinstance(entry, dict):
                    if entry.get("type") == "file":
                        total += len(entry.get("content", ""))
                    elif entry.get("type") == "directory":
                        total += calculate_size(entry.get("contents", {}))
                    elif "contents" in entry:
                        total += calculate_size(entry["contents"])
                    elif "content" in entry:
                        total += len(entry["content"])
            return total
        
        size_bytes = calculate_size(current_node)
        
        if human_readable:
            if size_bytes < 1024:
                return {"disk_usage": f"{size_bytes}B"}
            elif size_bytes < 1024 * 1024:
                return {"disk_usage": f"{size_bytes / 1024:.1f}KB"}
            else:
                return {"disk_usage": f"{size_bytes / (1024 * 1024):.1f}MB"}
        else:
            return {"disk_usage": str(size_bytes)}

    def echo(self, content: str, file_name: str = "None") -> Dict[str, Any]:
        """Write content to a file at current directory or display it in the terminal."""
        if file_name == "None" or not file_name:
            return {"terminal_output": content}
        
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"terminal_output": "Error: Current directory does not exist."}
        
        if file_name in current_node and current_node[file_name]["type"] == "directory":
            return {"terminal_output": f"Error: '{file_name}' is a directory."}
        
        current_node[file_name] = {
            "type": "file",
            "content": content
        }
        return {"terminal_output": "None"}

    def find(self, path: str = ".", name: str = "None") -> Dict[str, Any]:
        """Find files or directories under a specific path that contain name in its file name."""
        start_node = None
        start_path_parts = []
        
        if path == ".":
            start_node = self._get_current_directory_node()
            start_path_parts = list(self.current_dir)
        else:
            path_parts = path.split("/")
            start_path_parts = path_parts
            start_node = self._get_node_at_path(path_parts)
            if start_node is not None and isinstance(start_node, dict) and start_node.get("type") == "directory":
                start_node = start_node["contents"]
            elif start_node is not None and isinstance(start_node, dict):
                return {"matches": []}
        
        if start_node is None:
            return {"matches": []}
            
        matches = []
        search_name = name if name != "None" else None
        
        def recursive_search(node: Dict[str, Any], current_path: List[str]) -> None:
            for entry_name, entry in node.items():
                if not isinstance(entry, dict):
                    continue
                entry_path = "/".join(current_path + [entry_name]) if current_path else entry_name
                if search_name is None or search_name in entry_name:
                    matches.append(entry_path)
                if entry.get("type") == "directory":
                    recursive_search(entry.get("contents", {}), current_path + [entry_name])
                elif "contents" in entry:
                    recursive_search(entry["contents"], current_path + [entry_name])
        
        recursive_search(start_node, start_path_parts if path == "." else start_path_parts)
        return {"matches": matches}

    def grep(self, file_name: str, pattern: str) -> Dict[str, Any]:
        """Search for lines in a file of any extension at current directory that contain the specified pattern."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"matching_lines": []}
        if file_name not in current_node or current_node[file_name]["type"] != "file":
            return {"matching_lines": []}
        
        content = current_node[file_name].get("content", "")
        lines = content.splitlines()
        matching = [line for line in lines if pattern in line]
        return {"matching_lines": matching}

    def ls(self, a: bool = False) -> Dict[str, Any]:
        """List the contents of the current directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"current_directory_content": []}
        
        contents = []
        for name, entry in current_node.items():
            if not a and name.startswith("."):
                continue
            contents.append(name)
        return {"current_directory_content": contents}

    def mkdir(self, dir_name: str) -> Dict[str, Any]:
        """Create a new directory in the current directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {}
        if dir_name in current_node:
            return {}
        
        current_node[dir_name] = {
            "type": "directory",
            "contents": {}
        }
        return {}

    def mv(self, source: str, destination: str) -> Dict[str, Any]:
        """Move a file or directory from one location to another."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"result": "Error: Current directory does not exist."}
        if source not in current_node:
            return {"result": f"Error: Source '{source}' not found."}
        
        source_entry = current_node[source]
        
        if destination in current_node:
            dest_entry = current_node[destination]
            if dest_entry["type"] == "directory":
                dest_entry["contents"][source] = source_entry
                del current_node[source]
                return {"result": f"Moved '{source}' into directory '{destination}'."}
            else:
                return {"result": f"Error: Destination '{destination}' already exists as a file."}
        else:
            current_node[destination] = source_entry
            del current_node[source]
            return {"result": f"Moved '{source}' to '{destination}'."}

    def rm(self, file_name: str) -> Dict[str, Any]:
        """Remove a file or directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"result": "Error: Current directory does not exist."}
        if file_name not in current_node:
            return {"result": f"Error: '{file_name}' not found."}
        
        del current_node[file_name]
        return {"result": f"Removed '{file_name}'."}

    def rmdir(self, dir_name: str) -> Dict[str, Any]:
        """Remove a directory at current directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"result": "Error: Current directory does not exist."}
        if dir_name not in current_node:
            return {"result": f"Error: Directory '{dir_name}' not found."}
        
        entry = current_node[dir_name]
        if entry["type"] != "directory":
            return {"result": f"Error: '{dir_name}' is not a directory."}
        if entry.get("contents", {}):
            return {"result": f"Error: Directory '{dir_name}' is not empty."}
        
        del current_node[dir_name]
        return {"result": f"Removed directory '{dir_name}'."}

    def sort(self, file_name: str) -> Dict[str, Any]:
        """Sort the contents of a file line by line."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"sorted_content": "Error: Current directory does not exist."}
        if file_name not in current_node or current_node[file_name]["type"] != "file":
            return {"sorted_content": f"Error: File '{file_name}' not found or is not a file."}
        
        content = current_node[file_name].get("content", "")
        lines = content.splitlines()
        lines.sort()
        return {"sorted_content": "\n".join(lines)}

    def tail(self, file_name: str, lines: int = 10) -> Dict[str, Any]:
        """Display the last part of a file of any extension."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"last_lines": "Error: Current directory does not exist."}
        if file_name not in current_node or current_node[file_name]["type"] != "file":
            return {"last_lines": f"Error: File '{file_name}' not found or is not a file."}
        
        content = current_node[file_name].get("content", "")
        all_lines = content.splitlines()
        last_lines = all_lines[-lines:] if lines > 0 else []
        return {"last_lines": "\n".join(last_lines)}

    def touch(self, file_name: str) -> Dict[str, Any]:
        """Create a new file of any extension in the current directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {}
        if file_name in current_node:
            return {}
        
        current_node[file_name] = {
            "type": "file",
            "content": ""
        }
        return {}

    def wc(self, file_name: str, mode: str = "l") -> Dict[str, Any]:
        """Count the number of lines, words, and characters in a file of any extension from current directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"count": 0, "type": "lines"}
        if file_name not in current_node or current_node[file_name]["type"] != "file":
            return {"count": 0, "type": "lines"}
        
        content = current_node[file_name].get("content", "")
        
        if mode == "l":
            count = len(content.splitlines())
            return {"count": count, "type": "lines"}
        elif mode == "w":
            count = len(content.split())
            return {"count": count, "type": "words"}
        elif mode == "c":
            count = len(content)
            return {"count": count, "type": "characters"}
        else:
            return {"count": 0, "type": "lines"}