import pytest
import json
from tools.gorilla_file_system import GorillaFileSystem

INITIAL_CONFIG = {
    "temp": {
        "type": "directory",
        "contents": {
            "dev_summary.txt": {
                "type": "file",
                "content": "This is a summary of the development process. No server error occurred during the initial phase. However, a server error was detected in the final testing phase. The team is working on resolving the server error. The server error is expected to be fixed by next week. Additional testing will be conducted to ensure no further server errors. The project is on track for completion. The final report will be submitted by the end of the month. The server error has been a major focus. The team is confident in resolving the server error soon."
            }
        }
    }
}


@pytest.fixture
def fs():
    config = json.loads(json.dumps(INITIAL_CONFIG))
    return GorillaFileSystem(initial_config=config)


def test_cat_normal(fs):
    fs.cd(folder="temp")
    result = fs.cat(file_name="dev_summary.txt")
    assert "development process" in result.get("file_content", "")


def test_cat_missing_file(fs):
    result = fs.cat(file_name="nonexistent.txt")
    assert "error" in result.get("file_content", "").lower() or "not found" in result.get("file_content", "").lower()


def test_cat_directory(fs):
    result = fs.cat(file_name="temp")
    assert "error" in result.get("file_content", "").lower() or "not a file" in result.get("file_content", "").lower() or "not found" in result.get("file_content", "").lower()


def test_cd_normal(fs):
    result = fs.cd(folder="temp")
    assert "temp" in result.get("current_working_directory", "")


def test_cd_nonexistent_folder(fs):
    result = fs.cd(folder="nonexistent")
    assert isinstance(result, dict) and "current_working_directory" in result


def test_cd_parent_directory(fs):
    fs.cd(folder="temp")
    result = fs.cd(folder="..")
    assert "temp" not in result.get("current_working_directory", "") or result.get("current_working_directory") == "/"


def test_cp_file_to_directory(fs):
    fs.cd(folder="temp")
    fs.mkdir(dir_name="backup")
    result = fs.cp(source="dev_summary.txt", destination="backup")
    assert "error" not in result.get("result", "").lower() or "copied" in result.get("result", "").lower()


def test_cp_nonexistent_source(fs):
    result = fs.cp(source="missing.txt", destination="backup")
    assert "error" in result.get("result", "").lower() or "not found" in result.get("result", "").lower()


def test_cp_file_to_new_name(fs):
    fs.cd(folder="temp")
    result = fs.cp(source="dev_summary.txt", destination="dev_summary_backup.txt")
    assert "copied" in result.get("result", "").lower()


def test_diff_identical_files(fs):
    fs.cd(folder="temp")
    fs.cp(source="dev_summary.txt", destination="copy.txt")
    result = fs.diff(file_name1="dev_summary.txt", file_name2="copy.txt")
    assert result.get("diff_lines", "") == ""


def test_diff_different_files(fs):
    fs.cd(folder="temp")
    fs.echo(content="Different content", file_name="other.txt")
    result = fs.diff(file_name1="dev_summary.txt", file_name2="other.txt")
    assert result.get("diff_lines", "") != ""


def test_diff_missing_file(fs):
    fs.cd(folder="temp")
    result = fs.diff(file_name1="dev_summary.txt", file_name2="missing.txt")
    assert "error" in result.get("diff_lines", "").lower() or "not found" in result.get("diff_lines", "").lower()


def test_du_human_readable(fs):
    result = fs.du(human_readable=True)
    assert "KB" in result.get("disk_usage", "") or "B" in result.get("disk_usage", "") or "MB" in result.get("disk_usage", "")


def test_du_not_human_readable(fs):
    result = fs.du(human_readable=False)
    disk_usage = result.get("disk_usage", "")
    assert any(c.isdigit() for c in disk_usage)


def test_du_empty_directory(fs):
    fs.mkdir(dir_name="empty_dir")
    fs.cd(folder="empty_dir")
    result = fs.du(human_readable=True)
    assert "0" in result.get("disk_usage", "") or "0B" in result.get("disk_usage", "")


def test_echo_to_file(fs):
    fs.cd(folder="temp")
    fs.echo(content="Hello World", file_name="hello.txt")
    cat_result = fs.cat(file_name="hello.txt")
    assert "Hello World" in cat_result.get("file_content", "")


def test_echo_to_terminal(fs):
    result = fs.echo(content="Display this")
    assert "Display this" in result.get("terminal_output", "")


def test_echo_overwrite_file(fs):
    fs.cd(folder="temp")
    fs.echo(content="First", file_name="test.txt")
    fs.echo(content="Second", file_name="test.txt")
    result = fs.cat(file_name="test.txt")
    assert "Second" in result.get("file_content", "") and "First" not in result.get("file_content", "")


def test_find_by_name(fs):
    fs.cd(folder="temp")
    result = fs.find(path=".", name="dev_summary")
    assert any("dev_summary.txt" in m for m in result.get("matches", []))


def test_find_all(fs):
    result = fs.find(path=".", name="None")
    assert any("temp" in m for m in result.get("matches", []))


def test_find_nonexistent_name(fs):
    result = fs.find(path=".", name="nonexistent_xyz")
    assert result.get("matches", []) == []


def test_grep_pattern_found(fs):
    fs.cd(folder="temp")
    result = fs.grep(file_name="dev_summary.txt", pattern="server error")
    assert any("server error" in line.lower() for line in result.get("matching_lines", []))


def test_grep_pattern_not_found(fs):
    fs.cd(folder="temp")
    result = fs.grep(file_name="dev_summary.txt", pattern="unicorn rainbow")
    assert result.get("matching_lines", []) == []


def test_grep_missing_file(fs):
    result = fs.grep(file_name="missing.txt", pattern="test")
    assert result.get("matching_lines", []) == []


def test_ls_normal(fs):
    result = fs.ls(a=False)
    assert "temp" in result.get("current_directory_content", [])


def test_ls_show_hidden(fs):
    result = fs.ls(a=True)
    assert "temp" in result.get("current_directory_content", [])


def test_ls_empty_directory(fs):
    fs.mkdir(dir_name="empty_dir")
    fs.cd(folder="empty_dir")
    result = fs.ls(a=False)
    assert result.get("current_directory_content", []) == []


def test_mkdir_normal(fs):
    fs.mkdir(dir_name="new_folder")
    ls_result = fs.ls(a=False)
    assert "new_folder" in ls_result.get("current_directory_content", [])


def test_mkdir_already_exists(fs):
    result = fs.mkdir(dir_name="temp")
    assert result == {}


def test_mkdir_nested_invalid(fs):
    result = fs.mkdir(dir_name="parent/child")
    assert isinstance(result, dict)


def test_mv_file_to_directory(fs):
    fs.cd(folder="temp")
    fs.mkdir(dir_name="archive")
    result = fs.mv(source="dev_summary.txt", destination="archive")
    assert "error" not in result.get("result", "").lower()


def test_mv_rename_file(fs):
    fs.cd(folder="temp")
    fs.mv(source="dev_summary.txt", destination="renamed.txt")
    ls_result = fs.ls(a=False)
    assert "renamed.txt" in ls_result.get("current_directory_content", []) and "dev_summary.txt" not in ls_result.get("current_directory_content", [])


def test_mv_nonexistent_source(fs):
    result = fs.mv(source="missing.txt", destination="new_name.txt")
    assert "error" in result.get("result", "").lower() or "not found" in result.get("result", "").lower()


def test_rm_file(fs):
    fs.cd(folder="temp")
    fs.rm(file_name="dev_summary.txt")
    ls_result = fs.ls(a=False)
    assert "dev_summary.txt" not in ls_result.get("current_directory_content", [])


def test_rm_directory(fs):
    fs.rm(file_name="temp")
    ls_result = fs.ls(a=False)
    assert "temp" not in ls_result.get("current_directory_content", [])


def test_rm_nonexistent(fs):
    result = fs.rm(file_name="nonexistent.txt")
    assert "error" in result.get("result", "").lower() or "not found" in result.get("result", "").lower()


def test_rmdir_empty(fs):
    fs.mkdir(dir_name="empty_dir")
    fs.rmdir(dir_name="empty_dir")
    ls_result = fs.ls(a=False)
    assert "empty_dir" not in ls_result.get("current_directory_content", [])


def test_rmdir_non_empty(fs):
    result = fs.rmdir(dir_name="temp")
    assert "error" in result.get("result", "").lower() or "not empty" in result.get("result", "").lower() or "not found" in result.get("result", "").lower()


def test_rmdir_nonexistent(fs):
    result = fs.rmdir(dir_name="nonexistent")
    assert "error" in result.get("result", "").lower() or "not found" in result.get("result", "").lower()


def test_sort_normal(fs):
    fs.echo(content="Banana\nApple\nCherry", file_name="fruits.txt")
    result = fs.sort(file_name="fruits.txt")
    output = result.get("sorted_content", "")
    assert output.index("Apple") < output.index("Banana")


def test_sort_missing_file(fs):
    result = fs.sort(file_name="missing.txt")
    assert "error" in result.get("sorted_content", "").lower() or "not found" in result.get("sorted_content", "").lower()


def test_sort_single_line(fs):
    fs.echo(content="Only line", file_name="single.txt")
    result = fs.sort(file_name="single.txt")
    assert "Only line" in result.get("sorted_content", "")


def test_tail_default_lines(fs):
    fs.cd(folder="temp")
    result = fs.tail(file_name="dev_summary.txt", lines=10)
    output_lines = result.get("last_lines", "").strip().split("\n")
    assert len(output_lines) <= 10


def test_tail_one_line(fs):
    fs.cd(folder="temp")
    result = fs.tail(file_name="dev_summary.txt", lines=1)
    output_lines = result.get("last_lines", "").strip().split("\n")
    assert len(output_lines) == 1


def test_tail_missing_file(fs):
    result = fs.tail(file_name="missing.txt", lines=5)
    assert "error" in result.get("last_lines", "").lower() or "not found" in result.get("last_lines", "").lower()


def test_touch_new_file(fs):
    fs.touch(file_name="new_file.txt")
    ls_result = fs.ls(a=False)
    assert "new_file.txt" in ls_result.get("current_directory_content", [])


def test_touch_existing_file(fs):
    fs.cd(folder="temp")
    result = fs.touch(file_name="dev_summary.txt")
    assert isinstance(result, dict)


def test_touch_no_extension(fs):
    fs.touch(file_name="Makefile")
    ls_result = fs.ls(a=False)
    assert "Makefile" in ls_result.get("current_directory_content", [])


def test_wc_words(fs):
    fs.cd(folder="temp")
    result = fs.wc(file_name="dev_summary.txt", mode="w")
    assert result.get("count", 0) > 0


def test_wc_characters(fs):
    fs.cd(folder="temp")
    result = fs.wc(file_name="dev_summary.txt", mode="c")
    assert result.get("count", 0) > 0


def test_wc_missing_file(fs):
    result = fs.wc(file_name="missing.txt", mode="l")
    assert result.get("count", 0) == 0
