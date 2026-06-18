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
            # Skip "root" as it's the top of the tree, not a directory name
            if part == "root":
                continue
            if part in node and node[part]["type"] == "directory":
                node = node[part]["contents"]
            else:
                return None
        return node

    def _get_node_at_path(self, path_parts: List[str]) -> Optional[Dict[str, Any]]:
        """Helper to get a node at a specific path from root."""
        node = self.root
        for part in path_parts:
            # Skip "root" as it's the top of the tree, not a directory name
            if part == "root":
                continue
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

    def _resolve_path(self, path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        """
        Resolve a path that may contain slashes to a (parent_node, filename, error) tuple.

        Handles paths like 'foo/bar/file.txt' by navigating from current_dir.
        Returns (parent_node, filename, None) on success or (None, None, error_msg) on failure.
        """
        if not path:
            return None, None, "Empty path"

        parts = path.replace("\\", "/").split("/")
        filename = parts[-1]
        dir_parts = parts[:-1]

        parent_node = self._get_current_directory_node()
        if parent_node is None:
            return None, None, f"Current directory does not exist"

        # Navigate additional path components if present
        for part in dir_parts:
            if part not in parent_node:
                return None, None, f"Directory '{part}' not found"
            if parent_node[part]["type"] != "directory":
                return None, None, f"'{part}' is not a directory"
            parent_node = parent_node[part]["contents"]

        return parent_node, filename, None

    def _find_file_entry(self, file_name: str) -> Optional[Dict[str, Any]]:
        """Find a file entry by name, with fallback for underscore/dot naming differences.
        
        When state modifications create files with underscores (project_plan_txt) but tools
        look for them with dots (project_plan.txt), this tries both naming conventions.
        """
        current_node = self._get_current_directory_node()
        if current_node is None:
            return None
        
        # Try exact name first
        if file_name in current_node:
            return current_node[file_name]
        
        # Try fallback: replace dots with underscores (for root-level files with extensions)
        fallback_name = file_name.replace('.', '_')
        if fallback_name in current_node:
            return current_node[fallback_name]
        
        # Try reverse fallback: replace underscores with dots (for when file was created with dots but lookup uses underscores)
        for key in current_node.keys():
            if key.replace('.', '_') == fallback_name or key.replace('_', '.') == file_name:
                return current_node[key]
        
        return None

    def cat(self, file_name: str) -> Dict[str, Any]:
        """Display the contents of a file of any extension from current directory."""
        if "/" in file_name or "\\" in file_name:
            parent, basename, error = self._resolve_path(file_name)
            if error:
                return {"file_content": f"Error: {error}."}
            file_name = basename

        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"file_content": f"Error: Current directory does not exist."}
        
        file_entry = self._find_file_entry(file_name)
        if file_entry is None:
            return {"file_content": f"Error: File '{file_name}' not found."}
        if file_entry.get("type") != "file":
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
        # Resolve source path if it contains slashes
        if "/" in source or "\\" in source:
            source_parent, source_basename, source_error = self._resolve_path(source)
            if source_error:
                return {"result": f"Error: {source_error}."}
        else:
            source_parent = self._get_current_directory_node()
            source_basename = source
            source_error = None

        if source_parent is None:
            return {"result": "Error: Current directory does not exist."}
        if source_error or source_basename not in source_parent:
            return {"result": f"Error: Source '{source}' not found."}

        source_entry = source_parent[source_basename]
        source_copy = copy.deepcopy(source_entry)

        # Resolve destination path if it contains slashes
        if "/" in destination or "\\" in destination:
            dest_parent, dest_basename, dest_error = self._resolve_path(destination)
            if dest_error:
                return {"result": f"Error: {dest_error}."}
            # If destination is a path ending in a directory name, use that
            if dest_basename in dest_parent and dest_parent[dest_basename]["type"] == "directory":
                dest_parent[dest_basename]["contents"][source_basename] = source_copy
                del source_parent[source_basename]
                return {"result": f"Copied '{source}' into directory '{destination}'."}
        else:
            dest_parent = source_parent
            dest_basename = destination

        if dest_basename in dest_parent:
            dest_entry = dest_parent[dest_basename]
            if dest_entry["type"] == "directory":
                dest_entry["contents"][source_basename] = source_copy
                return {"result": f"Copied '{source_basename}' into directory '{dest_basename}'."}
            else:
                return {"result": f"Error: Destination '{dest_basename}' already exists as a file."}
        else:
            dest_parent[dest_basename] = source_copy
            return {"result": f"Copied '{source_basename}' to '{dest_basename}'."}

    def diff(self, file_name1: str, file_name2: str) -> Dict[str, Any]:
        """Compare two files of any extension line by line at the current directory."""
        # Resolve file_name1 path if it contains slashes
        if "/" in file_name1 or "\\" in file_name1:
            parent1, basename1, error1 = self._resolve_path(file_name1)
            if error1:
                return {"diff_lines": f"Error: {error1}."}
            file_name1 = basename1
        else:
            parent1 = self._get_current_directory_node()

        # Resolve file_name2 path if it contains slashes
        if "/" in file_name2 or "\\" in file_name2:
            parent2, basename2, error2 = self._resolve_path(file_name2)
            if error2:
                return {"diff_lines": f"Error: {error2}."}
            file_name2 = basename2
        else:
            parent2 = self._get_current_directory_node()

        if parent1 is None or parent2 is None:
            return {"diff_lines": "Error: Current directory does not exist."}

        if file_name1 not in parent1 or parent1[file_name1]["type"] != "file":
            return {"diff_lines": f"Error: File '{file_name1}' not found or is not a file."}
        if file_name2 not in parent2 or parent2[file_name2]["type"] != "file":
            return {"diff_lines": f"Error: File '{file_name2}' not found or is not a file."}

        lines1 = parent1[file_name1]["content"].splitlines()
        lines2 = parent2[file_name2]["content"].splitlines()

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

        # Handle paths with slashes by creating parent directories if needed
        if "/" in file_name or "\\" in file_name:
            parts = file_name.replace("\\", "/").split("/")
            file_name = parts[-1]
            parent_parts = parts[:-1]

            # Navigate to root first, then through parent path
            self.current_dir = []
            for dir_name in parent_parts:
                if not dir_name:
                    continue
                current_node = self._get_current_directory_node()
                if current_node is None:
                    return {"terminal_output": f"Error: Cannot create path - current directory does not exist."}
                if dir_name not in current_node:
                    current_node[dir_name] = {
                        "type": "directory",
                        "contents": {}
                    }
                self.current_dir.append(dir_name)

        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"terminal_output": "Error: Current directory does not exist."}

        if file_name in current_node and current_node[file_name]["type"] == "directory":
            return {"terminal_output": f"Error: '{file_name}' is a directory."}

        current_node[file_name] = {
            "type": "file",
            "content": content
        }
        return {"terminal_output": f"File '{file_name}' created successfully."}

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
        if "/" in file_name or "\\" in file_name:
            parent, basename, error = self._resolve_path(file_name)
            if error:
                return {"matching_lines": []}
            file_name = basename
        else:
            parent = self._get_current_directory_node()

        if parent is None:
            return {"matching_lines": []}
        
        file_entry = self._find_file_entry(file_name)
        if file_entry is None or file_entry.get("type") != "file":
            return {"matching_lines": []}

        content = file_entry.get("content", "")
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
            return {"error": "Current directory does not exist.", "created": False}
        if dir_name in current_node:
            entry = current_node[dir_name]
            if entry.get("type") == "directory":
                return {"error": f"Directory '{dir_name}' already exists.", "created": False}
            else:
                return {"error": f"'{dir_name}' already exists as a file.", "created": False}

        current_node[dir_name] = {
            "type": "directory",
            "contents": {}
        }
        return {"created": True, "directory": dir_name}

    def mv(self, source: str, destination: str) -> Dict[str, Any]:
        """Move a file or directory from one location to another."""
        # Resolve source path if it contains slashes
        if "/" in source or "\\" in source:
            source_parent, source_basename, source_error = self._resolve_path(source)
            if source_error:
                return {"result": f"Error: {source_error}."}
        else:
            source_parent = self._get_current_directory_node()
            source_basename = source
            source_error = None

        if source_parent is None:
            return {"result": "Error: Current directory does not exist."}
        if source_error or source_basename not in source_parent:
            return {"result": f"Error: Source '{source}' not found."}

        source_entry = source_parent[source_basename]

        # Resolve destination path if it contains slashes
        if "/" in destination or "\\" in destination:
            dest_parent, dest_basename, dest_error = self._resolve_path(destination)
            if dest_error:
                return {"result": f"Error: {dest_error}."}
            # If destination is a path ending in a directory, move into it
            if dest_basename in dest_parent and dest_parent[dest_basename]["type"] == "directory":
                dest_parent[dest_basename]["contents"][source_basename] = source_entry
                del source_parent[source_basename]
                return {"result": f"Moved '{source}' into directory '{destination}'."}
        else:
            dest_parent = source_parent
            dest_basename = destination

        if dest_basename in dest_parent:
            dest_entry = dest_parent[dest_basename]
            if dest_entry["type"] == "directory":
                dest_entry["contents"][source_basename] = source_entry
                del source_parent[source_basename]
                return {"result": f"Moved '{source_basename}' into directory '{dest_basename}'."}
            else:
                return {"result": f"Error: Destination '{dest_basename}' already exists as a file."}
        else:
            dest_parent[dest_basename] = source_entry
            del source_parent[source_basename]
            return {"result": f"Moved '{source_basename}' to '{dest_basename}'."}

    def rm(self, file_name: str) -> Dict[str, Any]:
        """Remove a file or directory."""
        if "/" in file_name or "\\" in file_name:
            parent, basename, error = self._resolve_path(file_name)
            if error:
                return {"result": f"Error: {error}."}
            file_name = basename
        else:
            parent = self._get_current_directory_node()

        if parent is None:
            return {"result": "Error: Current directory does not exist."}
        if file_name not in parent:
            return {"result": f"Error: '{file_name}' not found."}

        del parent[file_name]
        return {"result": f"Removed '{file_name}'."}

    def rmdir(self, dir_name: str) -> Dict[str, Any]:
        """Remove a directory at current directory."""
        if "/" in dir_name or "\\" in dir_name:
            parent, basename, error = self._resolve_path(dir_name)
            if error:
                return {"result": f"Error: {error}."}
            dir_name = basename
        else:
            parent = self._get_current_directory_node()

        if parent is None:
            return {"result": "Error: Current directory does not exist."}
        if dir_name not in parent:
            return {"result": f"Error: Directory '{dir_name}' not found."}

        entry = parent[dir_name]
        if entry["type"] != "directory":
            return {"result": f"Error: '{dir_name}' is not a directory."}
        if entry.get("contents", {}):
            return {"result": f"Error: Directory '{dir_name}' is not empty."}

        del parent[dir_name]
        return {"result": f"Removed directory '{dir_name}'."}

    def sort(self, file_name: str) -> Dict[str, Any]:
        """Sort the contents of a file line by line."""
        if "/" in file_name or "\\" in file_name:
            parent, basename, error = self._resolve_path(file_name)
            if error:
                return {"sorted_content": f"Error: {error}."}
            file_name = basename
        else:
            parent = self._get_current_directory_node()

        if parent is None:
            return {"sorted_content": "Error: Current directory does not exist."}
        if file_name not in parent or parent[file_name]["type"] != "file":
            return {"sorted_content": f"Error: File '{file_name}' not found or is not a file."}

        content = parent[file_name].get("content", "")
        lines = content.splitlines()
        lines.sort()
        return {"sorted_content": "\n".join(lines)}

    def tail(self, file_name: str, lines: int = 10) -> Dict[str, Any]:
        """Display the last part of a file of any extension."""
        if "/" in file_name or "\\" in file_name:
            parent, basename, error = self._resolve_path(file_name)
            if error:
                return {"last_lines": f"Error: {error}."}
            file_name = basename
        else:
            parent = self._get_current_directory_node()

        if parent is None:
            return {"last_lines": "Error: Current directory does not exist."}
        if file_name not in parent or parent[file_name]["type"] != "file":
            return {"last_lines": f"Error: File '{file_name}' not found or is not a file."}

        content = parent[file_name].get("content", "")
        all_lines = content.splitlines()
        last_lines = all_lines[-lines:] if lines > 0 else []
        return {"last_lines": "\n".join(last_lines)}

    def touch(self, file_name: str) -> Dict[str, Any]:
        """Create a new file of any extension in the current directory."""
        current_node = self._get_current_directory_node()
        if current_node is None:
            return {"error": "Current directory does not exist.", "created": False}
        if file_name in current_node:
            entry = current_node[file_name]
            if entry.get("type") == "directory":
                return {"error": f"'{file_name}' already exists as a directory.", "created": False}
            else:
                return {"error": f"'{file_name}' already exists.", "created": False}

        current_node[file_name] = {
            "type": "file",
            "content": ""
        }
        return {"created": True, "file": file_name}

    def wc(self, file_name: str, mode: str = "l") -> Dict[str, Any]:
        """Count the number of lines, words, and characters in a file of any extension from current directory."""
        if "/" in file_name or "\\" in file_name:
            parent, basename, error = self._resolve_path(file_name)
            if error:
                return {"count": 0, "type": "lines"}
            file_name = basename
        else:
            parent = self._get_current_directory_node()

        if parent is None:
            return {"count": 0, "type": "lines"}
        
        file_entry = self._find_file_entry(file_name)
        if file_entry is None or file_entry.get("type") != "file":
            return {"count": 0, "type": "lines"}

        content = file_entry.get("content", "")

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