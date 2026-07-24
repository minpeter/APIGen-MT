"""Mutation operations for the virtual filesystem."""

import shutil

from .gorilla_file_system_core import GorillaFileSystemCore


class GorillaFileSystemWriteOperations(GorillaFileSystemCore):
    """Provide mutating filesystem tool operations."""

    def cp(self, source: str, destination: str) -> dict[str, object]:
        """Copy a local file or directory."""
        if err := self._validate_local_name(source, "source"):
            return {"result": f"Error: {err['error']}"}
        if err := self._validate_local_name(destination, "destination"):
            return {"result": f"Error: {err['error']}"}
        src_path = self._get_relative_path(source)
        if not src_path.exists():
            return {"result": f"Error: Source '{source}' not found"}

        dst_path = self._get_relative_path(destination)
        display_destination = destination
        if dst_path.is_dir():
            dst_path = dst_path / src_path.name
            display_destination = f"{destination}/{src_path.name}"

        if dst_path.exists():
            return {
                "result": (
                    f"Error: Destination '{display_destination}' already exists"
                )
            }

        try:
            if src_path.is_dir():
                _ = shutil.copytree(src_path, dst_path)
            else:
                _ = shutil.copy2(src_path, dst_path)
            self._rebuild_fs_state()
            return {
                "result": f"Copied '{source}' to '{display_destination}'"
            }
        except (OSError, shutil.Error) as error:
            return {"result": f"Error: {error}"}

    def echo(
        self,
        content: str,
        file_name: str | None = None,
    ) -> dict[str, object]:
        """Write content to a local file or display it in the terminal."""
        if not file_name or file_name == "None":
            return {"terminal_output": content}
        if err := self._validate_local_name(file_name, "file_name"):
            return {"terminal_output": f"Error: {err['error']}"}
        path = self._get_relative_path(file_name)
        if path.is_dir():
            return {
                "terminal_output": f"Error: '{file_name}' is a directory"
            }
        _ = path.write_text(content)
        self._rebuild_fs_state()
        return {"terminal_output": None}

    def mkdir(self, dir_name: str) -> dict[str, object]:
        """Create a directory in the current directory."""
        if self._validate_local_name(dir_name, "dir_name"):
            return {}
        path = self._get_relative_path(dir_name)
        if path.exists():
            return {}
        path.mkdir()
        self._rebuild_fs_state()
        return {}

    def mv(self, source: str, destination: str) -> dict[str, object]:
        """Move a local file or directory."""
        if err := self._validate_local_name(source, "source"):
            return {"result": f"Error: {err['error']}"}
        if err := self._validate_local_name(destination, "destination"):
            return {"result": f"Error: {err['error']}"}

        src_path = self._get_relative_path(source)
        if not src_path.exists():
            return {"result": f"Error: Source '{source}' not found"}

        dst_path = self._get_relative_path(destination)
        display_destination = destination
        if dst_path.is_dir():
            dst_path = dst_path / src_path.name
            display_destination = f"{destination}/{src_path.name}"

        if dst_path.exists():
            return {
                "result": (
                    f"Error: Destination '{display_destination}' already exists"
                )
            }

        try:
            _ = shutil.move(str(src_path), str(dst_path))
            self._rebuild_fs_state()
            return {
                "result": f"Moved '{source}' to '{display_destination}'"
            }
        except (OSError, shutil.Error) as error:
            return {"result": f"Error: {error}"}

    def rm(self, file_name: str) -> dict[str, object]:
        """Remove a local file or directory."""
        if err := self._validate_local_name(file_name, "file_name"):
            return {"result": f"Error: {err['error']}"}
        path = self._get_relative_path(file_name)
        if not path.exists():
            return {"result": f"Error: '{file_name}' not found"}

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            self._rebuild_fs_state()
            return {"result": f"Removed '{file_name}'"}
        except (OSError, shutil.Error) as error:
            return {"result": f"Error: {error}"}

    def rmdir(self, dir_name: str) -> dict[str, object]:
        """Remove an empty local directory."""
        if err := self._validate_local_name(dir_name, "dir_name"):
            return {"result": f"Error: {err['error']}"}
        path = self._get_relative_path(dir_name)
        if not path.exists():
            return {"result": f"Error: Directory '{dir_name}' not found"}
        if not path.is_dir():
            return {"result": f"Error: '{dir_name}' is not a directory"}

        try:
            path.rmdir()
            self._rebuild_fs_state()
            return {"result": f"Removed '{dir_name}'"}
        except OSError:
            return {"result": f"Error: Directory '{dir_name}' is not empty"}

    def touch(self, file_name: str) -> dict[str, object]:
        """Create or update a file in the current directory."""
        if self._validate_local_name(file_name, "file_name"):
            return {}
        path = self._get_relative_path(file_name)
        if path.is_dir():
            return {}
        path.touch()
        self._rebuild_fs_state()
        return {}
