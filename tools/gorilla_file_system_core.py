"""Shared state and path handling for the virtual filesystem."""

import shutil
import tempfile
from pathlib import Path
from typing import TypedDict

from .type_utils import Config, Record, get_record, get_str, is_record


class PathValidationError(TypedDict):
    """Description of an invalid virtual path component."""

    error: str


class GorillaFileSystemCore:
    """Manage virtual filesystem state and its isolated temporary root."""

    def __init__(self, initial_config: Config | None = None) -> None:
        """Initialize with optional serialized filesystem state."""
        self._temp_dir: str = tempfile.mkdtemp(prefix="vfs_")
        self.current_dir: str = self._temp_dir
        self._fs_state: Record = {}
        if initial_config:
            self._load_from_config(initial_config)
        else:
            self._setup_default_structure()

    def _setup_default_structure(self) -> None:
        """Set up a default directory structure."""
        workspace = Path(self._temp_dir) / "workspace"
        workspace.mkdir(exist_ok=True)
        self._fs_state = {"workspace": {"type": "directory", "contents": {}}}

    def _load_from_config(self, config: Config) -> None:
        """Load filesystem structure from serialized state."""
        root = config.get("root")
        node = root if is_record(root) else config
        self._build_from_dict(node, Path(self._temp_dir))
        configured_directory = get_str(config, "current_dir", "/")
        directory = self._get_relative_path(configured_directory)
        if directory.exists() and directory.is_dir():
            self.current_dir = str(directory)

    def _build_from_dict(
        self,
        node: Config,
        path: Path,
        fs_state_parent: Record | None = None,
    ) -> None:
        """Recursively build the isolated filesystem from serialized state."""
        parent = self._fs_state if fs_state_parent is None else fs_state_parent
        for name, raw_entry in node.items():
            if name == "current_dir" or not is_record(raw_entry):
                continue
            if error := self._validate_local_name(name):
                raise ValueError(error["error"])
            entry_path = path / name
            entry_type = get_str(raw_entry, "type")
            if entry_type == "directory":
                entry_path.mkdir(exist_ok=True)
                contents_state: Record = {}
                parent[name] = {
                    "type": "directory",
                    "contents": contents_state,
                }
                self._build_from_dict(
                    get_record(raw_entry, "contents"),
                    entry_path,
                    contents_state,
                )
            elif entry_type == "file":
                entry_path.parent.mkdir(parents=True, exist_ok=True)
                content = get_str(raw_entry, "content")
                _ = entry_path.write_text(content)
                parent[name] = {"type": "file", "content": content}

    def _get_relative_path(self, name: str) -> Path:
        """Resolve a virtual path without allowing it to escape the temp root."""
        root = Path(self._temp_dir)
        path = root if name.startswith("/") else Path(self.current_dir)
        for part in name.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                if path != root:
                    path = path.parent
            else:
                path = path / part
        return path

    def _get_virtual_current_dir(self) -> str:
        """Return the deterministic virtual current directory."""
        relative = Path(self.current_dir).relative_to(self._temp_dir)
        return "/" if str(relative) == "." else f"/{relative.as_posix()}"

    def _rebuild_fs_state(self) -> None:
        """Rebuild serialized state from the isolated filesystem."""
        self._fs_state = self._dict_from_path(Path(self._temp_dir), "")

    def _dict_from_path(self, path: Path, relative_prefix: str) -> Record:
        """Recursively serialize a filesystem path."""
        result: Record = {}
        if not path.exists():
            return result
        for item in path.iterdir():
            relative = (
                f"{relative_prefix}/{item.name}"
                if relative_prefix
                else item.name
            )
            if item.is_dir():
                entry: Record = {"type": "directory", "path": relative}
                contents = self._dict_from_path(item, relative)
                if contents:
                    entry["contents"] = contents
            else:
                entry = {
                    "type": "file",
                    "path": relative,
                    "size": item.stat().st_size,
                    "content": item.read_text(),
                }
            result[item.name] = entry
        return result

    def _validate_local_name(
        self,
        name: str,
        param_name: str = "name",
    ) -> PathValidationError | None:
        """Validate that a name identifies one local directory entry."""
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            return {
                "error": (
                    f"Invalid {param_name} '{name}': must be local to current "
                    "directory, not a path. Use cd to navigate first."
                )
            }
        return None

    def _get_node_from_cwd(self, name: str) -> Record | None:
        """Get file or directory metadata relative to the current directory."""
        path = self._get_relative_path(name)
        if not path.exists():
            return None
        stat = path.stat()
        return {
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
        }

    def get_state(self) -> Record:
        """Export JSON-serializable filesystem state."""
        def dict_from_path(path: Path) -> Record:
            result: Record = {}
            for item in path.iterdir():
                if item.is_dir():
                    result[item.name] = {
                        "type": "directory",
                        "contents": dict_from_path(item),
                    }
                else:
                    result[item.name] = {
                        "type": "file",
                        "content": item.read_text(),
                    }
            return result

        return {
            "root": dict_from_path(Path(self._temp_dir)),
            "current_dir": self._get_virtual_current_dir(),
        }

    def cleanup(self) -> None:
        """Remove the isolated temporary directory."""
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __del__(self) -> None:
        """Release the temporary directory when the API is collected."""
        self.cleanup()
