"""Virtual in-memory filesystem for tool simulation."""

import fnmatch
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class GorillaFileSystem:
    """
    A virtual in-memory filesystem that persists state between tool calls.

    State is maintained as both actual files on disk AND as a JSON-serializable
    dict attribute (_fs_state) that mirrors the filesystem structure. This dict
    is updated with every operation to maintain accurate state representation.
    """

    def __init__(self, initial_config: Optional[Dict] = None):
        """Initialize with optional initial state."""
        self._temp_dir = tempfile.mkdtemp(prefix="vfs_")
        self.current_dir = self._temp_dir
        self._fs_state: Dict[str, Any] = {}  # Filesystem state as dict

        if initial_config:
            self._load_from_config(initial_config)
        else:
            self._setup_default_structure()

    def _setup_default_structure(self):
        """Set up a default directory structure."""
        workspace = Path(self._temp_dir) / "workspace"
        workspace.mkdir(exist_ok=True)
        self._fs_state = {
            "workspace": {"type": "directory", "contents": {}}
        }

    def _load_from_config(self, config: Dict):
        """Load filesystem structure from config dict."""
        if "root" in config and isinstance(config["root"], dict):
            self._build_from_dict(config["root"], Path(self._temp_dir))
        elif isinstance(config, dict):
            self._build_from_dict(config, Path(self._temp_dir))

    def _build_from_dict(self, node: Dict, path: Path, fs_state_parent: Optional[Dict] = None):
        """Recursively build filesystem from nested dict."""
        if fs_state_parent is None:
            fs_state_parent = self._fs_state
        for name, entry in node.items():
            entry_path = path / name
            if isinstance(entry, dict):
                if entry.get("type") == "directory":
                    entry_path.mkdir(exist_ok=True)
                    contents = entry.get("contents", {})
                    dir_state = {"type": "directory", "contents": {}}
                    fs_state_parent[name] = dir_state
                    self._build_from_dict(contents, entry_path, dir_state["contents"])
                elif entry.get("type") == "file":
                    entry_path.parent.mkdir(parents=True, exist_ok=True)
                    content = entry.get("content", "")
                    entry_path.write_text(content)
                    fs_state_parent[name] = {"type": "file", "content": content}

    def _get_relative_path(self, name: str) -> Path:
        """Get absolute path from current directory."""
        if name.startswith("/"):
            return Path(name.lstrip("/"))
        return Path(self.current_dir) / name

    def _rebuild_fs_state(self):
        """Rebuild _fs_state from the actual filesystem on disk."""
        self._fs_state = self._dict_from_path(Path(self._temp_dir), "")

    def _dict_from_path(self, p: Path, rel_prefix: str) -> Dict:
        """Recursively build dict from a path."""
        result = {}
        if not p.exists():
            return result
        for item in p.iterdir():
            item_rel = f"{rel_prefix}/{item.name}" if rel_prefix else item.name
            if item.is_dir():
                result[item.name] = {
                    "type": "directory",
                    "path": item_rel
                }
                sub = self._dict_from_path(item, item_rel)
                if sub:
                    result[item.name]["contents"] = sub
            else:
                result[item.name] = {
                    "type": "file",
                    "path": item_rel,
                    "size": item.stat().st_size,
                    "content": item.read_text()
                }
        return result

    def _validate_local_name(self, name: str, param_name: str = "name") -> Optional[Dict[str, Any]]:
        """Validate that a name is local to current directory (no path separators, except .. for parent).
        
        Returns error dict if name contains path separators (except .. prefix), None if valid.
        """
        if name.startswith(".."):
            return None
        if "/" in name:
            return {"error": f"Invalid {param_name} '{name}': must be local to current directory, not a path. Use cd to navigate first."}
        return None

    def _get_node_from_cwd(self, name: str) -> Optional[Dict]:
        """Get file/dir info relative to current working directory."""
        p = self._get_relative_path(name)
        if not p.exists():
            return None
        stat = p.stat()
        return {
            "type": "directory" if p.is_dir() else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime
        }

    def cat(self, file_name: str) -> Dict[str, Any]:
        """Read and return file contents."""
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {"error": f"File '{file_name}' not found"}
        if path.is_dir():
            return {"error": f"'{file_name}' is a directory"}
        return {"content": path.read_text()}

    def cd(self, folder: str) -> Dict[str, Any]:
        """Change current working directory."""
        if folder == "..":
            parent = Path(self.current_dir).parent
            if parent == Path(self._temp_dir).parent:
                return {"error": "Already at root"}
            self.current_dir = str(parent)
            return {"success": True, "current_path": self.current_dir}
        
        path = self._get_relative_path(folder)
        if not path.exists():
            return {"error": f"Directory '{folder}' not found"}
        if not path.is_dir():
            return {"error": f"'{folder}' is not a directory"}
        
        self.current_dir = str(path)
        return {"success": True, "current_path": self.current_dir}

    def cp(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy file or directory."""
        if err := self._validate_local_name(source, "source"):
            return err
        if err := self._validate_local_name(destination, "destination"):
            return err
        src_path = self._get_relative_path(source)
        if not src_path.exists():
            return {"error": f"Source '{source}' not found"}
        
        dst_path = self._get_relative_path(destination)
        
        if dst_path.is_dir():
            dst_path = dst_path / src_path.name
        
        if dst_path.exists():
            return {"error": f"Destination '{destination}' already exists"}
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            self._rebuild_fs_state()
            return {"success": True, "message": f"Copied '{source}' to '{str(dst_path)}'"}
        except Exception as e:
            return {"error": str(e)}

    def du(self, path: str = ".", human_readable: bool = False) -> Dict[str, Any]:
        """Calculate disk usage."""
        target_path = self._get_relative_path(path)
        if not target_path.exists():
            return {"error": f"Path '{path}' not found"}
        
        if target_path.is_file():
            total = target_path.stat().st_size
        else:
            total = sum(
                f.stat().st_size 
                for f in target_path.rglob("*") 
                if f.is_file()
            )
        
        if human_readable:
            if total < 1024:
                return {"disk_usage": f"{total}B"}
            elif total < 1024 * 1024:
                return {"disk_usage": f"{total / 1024:.1f}KB"}
            else:
                return {"disk_usage": f"{total / (1024 * 1024):.1f}MB"}
        return {"disk_usage": str(total)}

    def echo(self, content: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """Write content to file or display in terminal."""
        if not file_name or file_name == "None":
            return {"content": content}
        if err := self._validate_local_name(file_name, "file_name"):
            return err
        path = self._get_relative_path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self._rebuild_fs_state()
        return {"success": True, "message": f"File '{file_name}' written successfully"}

    def find(self, path: str = ".", name: Optional[str] = None) -> Dict[str, Any]:
        """Find files matching pattern."""
        if path == ".":
            search_root = Path(self.current_dir)
        else:
            search_root = self._get_relative_path(path)
            if not search_root.exists():
                return {"files": [], "error": f"Path '{path}' not found"}
        
        matches = []
        pattern = name if name and name != "None" else ""
        
        for p in search_root.rglob("*"):
            if pattern and not fnmatch.fnmatch(p.name, pattern):
                continue
            rel = p.relative_to(search_root)
            matches.append(str(rel))
        
        return {"files": matches}

    def grep(self, file_name: str, pattern: str) -> Dict[str, Any]:
        """Search for pattern in file."""
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {"lines": [], "error": f"File '{file_name}' not found"}
        if path.is_dir():
            return {"lines": [], "error": f"'{file_name}' is a directory"}
        content = path.read_text()
        lines = content.splitlines()
        matching = [line for line in lines if pattern in line]
        return {"lines": matching}

    def head(self, file_name: str, lines: int = 10) -> Dict[str, Any]:
        """Return first n lines of file."""
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {"first_n_lines": "", "error": f"File '{file_name}' not found"}
        all_lines = path.read_text().splitlines()
        return {"first_n_lines": "\n".join(all_lines[:lines])}

    def ls(self, a: bool = False) -> Dict[str, Any]:
        """List directory contents.
        
        Args:
            a: If True, show hidden files (starting with .).
        """
        try:
            target_path = self._get_relative_path(".")
            if not target_path.exists():
                return {"error": "Current directory does not exist"}
            if not target_path.is_dir():
                return {"error": "Current path is not a directory"}
            entries = os.listdir(target_path)
            if not a:
                entries = [e for e in entries if not e.startswith(".")]
            return {"files": sorted(entries)}
        except Exception as e:
            return {"files": [], "error": str(e)}

    def mkdir(self, dir_name: str) -> Dict[str, Any]:
        """Create a directory."""
        path = self._get_relative_path(dir_name)
        if path.exists():
            if path.is_dir():
                return {"error": f"Directory '{dir_name}' already exists", "success": False}
            return {"error": f"'{dir_name}' already exists as a file", "success": False}
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            self._rebuild_fs_state()
            return {"success": True, "message": f"Directory {dir_name} created successfully.", "dir_name": dir_name}
        except Exception as e:
            return {"error": str(e), "success": False}

    def mv(self, source: str, destination: str) -> Dict[str, Any]:
        """Move file or directory."""
        src_path = self._get_relative_path(source)
        if not src_path.exists():
            return {"error": f"Source '{source}' not found"}
        
        dst_path = self._get_relative_path(destination)
        
        # If destination is a directory, move source into it (preserving filename)
        if dst_path.is_dir():
            dst_path = dst_path / src_path.name

        if dst_path.exists():
            return {"error": f"Destination '{destination}' already exists"}

        try:
            shutil.move(str(src_path), str(dst_path))
            self._rebuild_fs_state()
            return {"success": True, "message": f"Moved '{source}' to '{str(dst_path)}'", "source": source, "destination": str(dst_path)}
        except Exception as e:
            return {"error": str(e)}

    def pwd(self) -> Dict[str, Any]:
        """Return current working directory."""
        return {"path": self.current_dir}

    def rm(self, file_name: str) -> Dict[str, Any]:
        """Remove file or directory."""
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {"success": False, "error": f"'{file_name}' not found"}
        
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            self._rebuild_fs_state()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rmdir(self, dir_name: str) -> Dict[str, Any]:
        """Remove empty directory."""
        path = self._get_relative_path(dir_name)
        if not path.exists():
            return {"success": False, "error": f"Directory '{dir_name}' not found"}
        if not path.is_dir():
            return {"success": False, "error": f"'{dir_name}' is not a directory"}
        
        try:
            path.rmdir()
            self._rebuild_fs_state()
            return {"success": True}
        except OSError:
            return {"success": False, "error": f"Directory '{dir_name}' is not empty"}

    def tail(self, file_name: str, lines: int = 10) -> Dict[str, Any]:
        """Return last n lines of file."""
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {"last_lines": "", "error": f"File '{file_name}' not found"}
        all_lines = path.read_text().splitlines()
        return {"last_lines": "\n".join(all_lines[-lines:])}

    def touch(self, file_name: str) -> Dict[str, Any]:
        """Create or update a file."""
        path = self._get_relative_path(file_name)
        if path.exists() and path.is_dir():
            return {"error": f"'{file_name}' already exists as a directory", "success": False}
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            self._rebuild_fs_state()
            return {"success": True, "message": "File created successfully.", "file_name": file_name}
        except Exception as e:
            return {"error": str(e), "success": False}

    def wc(self, file_name: str, mode: str = "l") -> Dict[str, Any]:
        """Count lines, words, or characters in file."""
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {"error": f"File '{file_name}' not found"}
        content = path.read_text()
        
        if mode == "l":
            return {"lines": len(content.splitlines()), "words": len(content.split()), "characters": len(content)}
        elif mode == "w":
            return {"words": len(content.split())}
        elif mode == "c":
            return {"characters": len(content)}
        else:
            return {"lines": len(content.splitlines())}

    def diff(self, file_name1: str, file_name2: str) -> Dict[str, Any]:
        """Compare two files and show differences."""
        path1 = self._get_relative_path(file_name1)
        path2 = self._get_relative_path(file_name2)
        
        if not path1.exists():
            return {"error": f"File '{file_name1}' not found"}
        if not path2.exists():
            return {"error": f"File '{file_name2}' not found"}
        
        if path1.is_dir() or path2.is_dir():
            return {"error": "diff is for files only"}
        
        lines1 = path1.read_text().splitlines()
        lines2 = path2.read_text().splitlines()
        
        diff_lines = []
        max_len = max(len(lines1), len(lines2))
        for i in range(max_len):
            l1 = lines1[i] if i < len(lines1) else ""
            l2 = lines2[i] if i < len(lines2) else ""
            if l1 != l2:
                prefix = "CHG"
            else:
                prefix = "SAME"
            diff_lines.append(f"{prefix} {i+1}: {l1} | {l2}")
        
        return {"diff": diff_lines, "identical": len(diff_lines) == 0}

    def sort(self, file_name: str, reverse: bool = False) -> Dict[str, Any]:
        """Sort and return file contents."""
        path = self._get_relative_path(file_name)
        
        if not path.exists():
            return {"error": f"File '{file_name}' not found"}
        if path.is_dir():
            return {"error": f"'{file_name}' is a directory"}
        
        lines = path.read_text().splitlines()
        sorted_lines = sorted(lines, reverse=reverse)
        return {"lines": sorted_lines, "count": len(sorted_lines)}

    def get_state(self) -> Dict[str, Any]:
        """Export current filesystem state."""
        def dict_from_path(p: Path) -> Dict:
            result = {}
            for item in p.iterdir():
                if item.is_dir():
                    result[item.name] = {
                        "type": "directory",
                        "contents": dict_from_path(item)
                    }
                else:
                    result[item.name] = {
                        "type": "file",
                        "content": item.read_text()
                    }
            return result
        
        return {
            "root": dict_from_path(Path(self._temp_dir)),
            "current_dir": self.current_dir
        }

    def cleanup(self):
        """Remove temp directory."""
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()