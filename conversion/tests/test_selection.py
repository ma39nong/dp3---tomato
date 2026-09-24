import pytest

from convest.config import validate_repo_id
from convest.selection import read_episode_list, select_episodes


def test_list_bom_comments_commas_and_canonical_ids(tmp_path):
    path = tmp_path / "group.txt"
    path.write_text("\ufeff# comment\n18，episode19, 020\n\n22 # note\n23\n")
    parsed = read_episode_list(path)
    assert parsed["requested_episodes"] == [
        "episode18", "episode19", "episode20", "episode22", "episode23"
    ]


def test_list_inclusive_ranges_mixed_with_single_ids(tmp_path):
    path = tmp_path / "group.txt"
    path.write_text("34-39，episode041-episode043, 045-episode046\n48-48 50\n")
    assert read_episode_list(path)["requested_episodes"] == [
        f"episode{n}" for n in [34, 35, 36, 37, 38, 39, 41, 42, 43, 45, 46, 48, 50]
    ]


def test_arbitrary_safe_bag_directory_names(tmp_path):
    path = tmp_path / "group.txt"
    path.write_text("trial_A\n2026-09-24-run-03\n")
    assert read_episode_list(path)["requested_episodes"] == [
        "trial_A", "2026-09-24-run-03"
    ]


@pytest.mark.parametrize("content,error", [
    ("18\nepisode018", "duplicate"), ("# comment\n", "empty"),
    ("18-20-22", "invalid numeric range"), ("../episode18", "invalid bag name"),
    ("39-34", "reversed range"), ("34-39\n039", "duplicate episode39"),
])
def test_invalid_list(tmp_path, content, error):
    path = tmp_path / "group.txt"
    path.write_text(content)
    with pytest.raises(ValueError, match=error):
        read_episode_list(path)


def test_selection_is_explicit_and_never_falls_back_to_all(tmp_path):
    path = tmp_path / "group.txt"
    candidates = [
        {"path": "/bags/episode18", "eligible": True, "reason": None},
        {"path": "/bags/episode64", "eligible": False, "reason": "incomplete"},
    ]
    path.write_text("18\n64")
    with pytest.raises(ValueError, match="episode64"):
        select_episodes(candidates, path)
    selected, report = select_episodes(candidates, path, skip_ineligible=True)
    assert [item["path"] for item in selected] == ["/bags/episode18"]
    assert report["skipped"][0]["episode"] == "episode64"


@pytest.mark.parametrize("value", ["../data", "a/../b", "/a/b", "a/b/c", "name", "a/"])
def test_repo_id_cannot_be_a_filesystem_path(value):
    with pytest.raises(ValueError, match="namespace/dataset"):
        validate_repo_id(value)
