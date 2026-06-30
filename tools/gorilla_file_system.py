"""GorillaFileSystem - A proper file system implementation with actual state."""

import copy
from typing import Any, Dict, List, Optional


class GorillaFileSystem:
    """
    A simple file system that maintains actual file/folder dict structure.
    
    State structure:
    {
        "root": {
            "dirname": {"type": "directory", "contents": {...}},
            "file.txt": {"type": "file", "content": "..."}
        },
        "current_dir": ["path", "to", "dir"]
    }
    """

    def __init__(self, initial_config: dict) -> None:
        if "root" in initial_config and isinstance(initial_config["root"], dict):
            self.root = copy.deepcopy(initial_config["root"])
        else:
            self.root = copy.deepcopy(initial_config)
        self.current_dir: List[str] = []

    def _get_current_node(self) -> Dict[str, Any]:
        """Get the current directory node."""
        node = self.root
        for part in self.current_dir:
            if part not in node:
                return None
            if node[part]["type"] != "directory":
                return None
            node = node[part]["contents"]
        return node

    def _resolve_path(self, path: str) -> tuple[Optional[Dict], Optional[str], Optional[str]]:
        """
        Resolve a path to (parent_node, basename, error).
        Handles both absolute paths (from root) and relative paths (from current_dir).
        """
        if not path:
            return None, None, "Empty path"
        
        parts = path.replace("\\", "/").split("/")
        if parts[0] == "":
            return None, None, "Absolute paths not supported"
        
        current = self._get_current_node()
        if current is None:
            return None, None, "Current directory does not exist"
        
        for name in parts[:-1]:
            if name not in current:
                return None, None, f"Directory '{name}' not found"
            if current[name]["type"] != "directory":
                return None, None, f"'{name}' is not a directory"
            current = current[name]["contents"]
        
        return current, parts[-1], None

    def cat(self, file_name: str) -> Dict[str, Any]:
        """Display the contents of a file. Returns content as a string."""
        parent, basename, error = self._resolve_path(file_name)
        if error:
            return {"error": error}
        if basename not in parent:
            return {"error": f"File '{basename}' not found"}
        if parent[basename]["type"] != "file":
            return {"error": f"'{basename}' is not a file"}
        return {"content": parent[basename]["content"]}

    def cd(self, folder: str) -> Dict[str, Any]:
        """Change the current working directory."""
        if folder == "..":
            if self.current_dir:
                self.current_dir.pop()
            return {"success": True, "current_path": "/" + "/".join(self.current_dir) if self.current_dir else "/"}
        
        current = self._get_current_node()
        if current is None:
            return {"error": "Current directory does not exist"}
        
        if folder not in current:
            return {"error": f"Directory '{folder}' not found"}
        
        if current[folder]["type"] != "directory":
            return {"error": f"'{folder}' is not a directory"}
        
        self.current_dir.append(folder)
        return {"success": True, "current_path": "/" + "/".join(self.current_dir)}

    def cp(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy a file or directory to destination."""
        src_parent, src_name, error = self._resolve_path(source)
        if error:
            return {"error": f"Source error: {error}"}
        if src_name not in src_parent:
            return {"error": f"Source '{source}' not found"}
        
        src_entry = copy.deepcopy(src_parent[src_name])
        
        dst_parent, dst_name, _ = self._resolve_path(destination)
        if dst_parent is None:
            return {"error": "Destination error: Current directory does not exist"}
        
        if dst_name in dst_parent:
            return {"error": f"Destination '{destination}' already exists"}
        
        dst_parent[dst_name] = src_entry
        return {"success": True, "message": f"Copied '{src_name}' to '{dst_name}'"}

    def du(self, human_readable: bool = False) -> Dict[str, Any]:
        """Calculate disk usage of current directory."""
        current = self._get_current_node()
        if current is None:
            return {"error": "Current directory does not exist"}
        
        def calc_size(node: Dict) -> int:
            total = 0
            for entry in node.values():
                if entry["type"] == "file":
                    total += len(entry.get("content", ""))
                elif entry["type"] == "directory":
                    total += calc_size(entry["contents"])
            return total
        
        size = calc_size(current)
        if human_readable:
            if size < 1024:
                return {"disk_usage": f"{size}B"}
            elif size < 1024 * 1024:
                return {"disk_usage": f"{size / 1024:.1f}KB"}
            else:
                return {"disk_usage": f"{size / (1024 * 1024):.1f}MB"}
        return {"disk_usage": str(size)}

    def echo(self, content: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """Write content to file or display in terminal."""
        if not file_name or file_name == "None":
            return {"content": content}
        
        parent, basename, error = self._resolve_path(file_name)
        if error and "not found" in error:
            parent = self._get_current_node()
            basename = file_name
        elif error:
            return {"error": error}
        
        if parent is None:
            return {"error": "Cannot access current directory"}
        
        parent[basename] = {"type": "file", "content": content}
        return {"success": True, "message": f"File '{basename}' written successfully"}

    def find(self, path: str = ".", name: Optional[str] = None) -> Dict[str, Any]:
        """Find files matching name pattern. Returns list of paths."""
        if path == ".":
            current = self._get_current_node()
            start_parts = list(self.current_dir)
        else:
            parent, _, error = self._resolve_path(path)
            if error:
                return {"files": [], "error": error}
            current = parent
            start_parts = path.split("/")
        
        if current is None:
            return {"files": [], "error": "Path not found"}
        
        matches = []
        def search(node: Dict, path_parts: List[str]):
            for entry_name, entry in node.items():
                full_path = "/".join(path_parts + [entry_name]) if path_parts else entry_name
                if name is None or name in entry_name:
                    matches.append(full_path)
                if entry["type"] == "directory":
                    search(entry["contents"], path_parts + [entry_name])
        
        search(current, start_parts if path != "." else [])
        return {"files": matches}

    def grep(self, file_name: str, pattern: str) -> Dict[str, Any]:
        """Search for pattern in file. Returns list of matching lines."""
        parent, basename, error = self._resolve_path(file_name)
        if error:
            return {"lines": [], "error": error}
        if basename not in parent:
            return {"lines": [], "error": f"File '{basename}' not found"}
        
        content = parent[basename]["content"]
        lines = content.splitlines()
        matching = [line for line in lines if pattern in line]
        return {"lines": matching}

    def head(self, file_name: str, lines: int = 10) -> Dict[str, Any]:
        """Return first n lines of file."""
        parent, basename, error = self._resolve_path(file_name)
        if error:
            return {"first_n_lines": "", "error": error}
        if basename not in parent:
            return {"first_n_lines": "", "error": f"File '{basename}' not found"}
        
        content = parent[basename]["content"]
        all_lines = content.splitlines()
        return {"first_n_lines": "\n".join(all_lines[:lines])}

    def ls(self, a: bool = False) -> Dict[str, Any]:
        """List directory contents. Returns list of names."""
        current = self._get_current_node()
        if current is None:
            return {"files": [], "error": "Current directory does not exist"}
        
        names = []
        for name in current.keys():
            if not a and name.startswith("."):
                continue
            names.append(name)
        return {"files": names}

    def mkdir(self, dir_name: str) -> Dict[str, Any]:
        """Create a directory. Returns success message or error."""
        current = self._get_current_node()
        if current is None:
            return {"error": "Current directory does not exist", "success": False}
        
        if dir_name in current:
            if current[dir_name]["type"] == "directory":
                return {"error": f"Directory '{dir_name}' already exists", "success": False}
            return {"error": f"'{dir_name}' already exists as a file", "success": False}
        
        current[dir_name] = {"type": "directory", "contents": {}}
        return {"success": True, "message": f"Directory {dir_name} created successfully.", "dir_name": dir_name}

    def mv(self, source: str, destination: str) -> Dict[str, Any]:
        """Move file or directory to destination."""
        src_parent, src_name, error = self._resolve_path(source)
        if error:
            return {"error": f"Source error: {error}"}
        if src_name not in src_parent:
            return {"error": f"Source '{source}' not found"}
        
        src_entry = src_parent.pop(src_name)
        
        dst_parent, dst_name, _ = self._resolve_path(destination)
        if dst_parent is None:
            src_parent[src_name] = src_entry
            return {"error": "Destination error: Current directory does not exist"}
        
        dst_parent[dst_name] = src_entry
        return {"success": True, "message": f"Moved '{source}' to '{destination}'", "source": source, "destination": destination}

    def rm(self, file_name: str) -> Dict[str, Any]:
        """Remove a file. Returns True/False."""
        parent, basename, error = self._resolve_path(file_name)
        if error:
            return {"success": False, "error": error}
        if basename not in parent:
            return {"success": False, "error": f"File '{basename}' not found"}
        
        del parent[basename]
        return {"success": True}

    def rmdir(self, dir_name: str) -> Dict[str, Any]:
        """Remove an empty directory."""
        parent, basename, error = self._resolve_path(dir_name)
        if error:
            return {"success": False, "error": error}
        if basename not in parent:
            return {"success": False, "error": f"Directory '{basename}' not found"}
        
        entry = parent[basename]
        if entry["type"] != "directory":
            return {"success": False, "error": f"'{basename}' is not a directory"}
        if entry["contents"]:
            return {"success": False, "error": f"Directory '{basename}' is not empty"}
        
        del parent[basename]
        return {"success": True, "message": f"Directory '{basename}' removed"}

    def tail(self, file_name: str, lines: int = 10) -> Dict[str, Any]:
        """Return last n lines of file."""
        parent, basename, error = self._resolve_path(file_name)
        if error:
            return {"last_lines": "", "error": error}
        if basename not in parent:
            return {"last_lines": "", "error": f"File '{basename}' not found"}
        
        content = parent[basename]["content"]
        all_lines = content.splitlines()
        return {"last_lines": "\n".join(all_lines[-lines:])}

    def touch(self, file_name: str) -> Dict[str, Any]:
        """Create an empty file. Returns success message or error."""
        current = self._get_current_node()
        if current is None:
            return {"error": "Current directory does not exist", "success": False}
        
        if file_name in current:
            if current[file_name]["type"] == "directory":
                return {"error": f"'{file_name}' already exists as a directory", "success": False}
            return {"error": f"'{file_name}' already exists", "success": False}
        
        current[file_name] = {"type": "file", "content": ""}
        return {"success": True, "message": f"File created successfully.", "file_name": file_name}

    def wc(self, file_name: str, mode: str = "l") -> Dict[str, Any]:
        """Count lines, words, or characters in file."""
        parent, basename, error = self._resolve_path(file_name)
        if error:
            return {"error": error}
        if basename not in parent:
            return {"error": f"File '{basename}' not found"}
        
        content = parent[basename]["content"]
        
        if mode == "l":
            return {"lines": len(content.splitlines()), "words": len(content.split()), "characters": len(content)}
        elif mode == "w":
            return {"words": len(content.split())}
        elif mode == "c":
            return {"characters": len(content)}
        return {"lines": len(content.splitlines())}