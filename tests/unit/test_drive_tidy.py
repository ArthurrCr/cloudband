from cloudband.colab import drive_tidy
from cloudband.colab.drive_tidy import ROOT, SHARED, Copy, plan_tidy

MB = 1_000_000


def copy(id, name="run.state.pt", size=500 * MB, modified="2026-10-10T00:00:00Z",
         mine=True, where=SHARED):
    return Copy(id, name, size, modified, mine, where)


def ids(copies):
    return sorted(c.id for c in copies)


def test_the_newest_copy_stays_and_the_older_ones_are_trashed():
    copies = [copy("old", modified="2026-10-10T00:00:00Z"),
              copy("new", modified="2026-10-12T00:00:00Z"),
              copy("mid", modified="2026-10-11T00:00:00Z")]

    plan = plan_tidy(copies)

    assert ids(plan.keep) == ["new"]
    assert ids(plan.trash) == ["mid", "old"]


def test_a_copy_cut_short_never_replaces_a_complete_one():
    copies = [copy("good", size=500 * MB, modified="2026-10-10T00:00:00Z"),
              copy("cut", size=40 * MB, modified="2026-10-12T00:00:00Z")]

    plan = plan_tidy(copies)

    assert ids(plan.keep) == ["good"]
    assert ids(plan.trash) == ["cut"]


def test_a_copy_in_the_project_folders_beats_a_newer_one_loose_in_the_root():
    copies = [copy("inside", modified="2026-10-10T00:00:00Z", where=SHARED),
              copy("loose", modified="2026-10-12T00:00:00Z", where=ROOT)]

    plan = plan_tidy(copies)

    assert ids(plan.keep) == ["inside"]
    assert ids(plan.trash) == ["loose"]
    assert plan.last_in_root == []


def test_the_last_copy_of_a_name_is_kept_even_when_it_is_loose_in_the_root():
    copies = [copy("a", where=ROOT, modified="2026-10-10T00:00:00Z"),
              copy("b", where=ROOT, modified="2026-10-11T00:00:00Z")]

    plan = plan_tidy(copies)

    assert ids(plan.keep) == ["b"]
    assert ids(plan.last_in_root) == ["b"]
    assert ids(plan.trash) == ["a"]


def test_copies_owned_by_another_account_are_reported_and_not_trashed():
    copies = [copy("theirs", mine=False, modified="2026-10-10T00:00:00Z"),
              copy("mine", mine=True, modified="2026-10-09T00:00:00Z"),
              copy("new", mine=False, modified="2026-10-12T00:00:00Z")]

    plan = plan_tidy(copies)

    assert ids(plan.keep) == ["new"]
    assert ids(plan.trash) == ["mine"]
    assert ids(plan.others) == ["theirs"]


def test_different_names_are_planned_on_their_own():
    copies = [copy("a1", name="a"), copy("a2", name="a", modified="2026-10-11T00:00:00Z"),
              copy("b1", name="b")]

    plan = plan_tidy(copies)

    assert ids(plan.keep) == ["a2", "b1"]
    assert ids(plan.trash) == ["a1"]


def test_a_small_file_that_grows_keeps_its_newest_version():
    copies = [copy("v1", name="s.json", size=446, modified="2026-10-10T00:00:00Z"),
              copy("v2", name="s.json", size=668, modified="2026-10-12T00:00:00Z")]

    plan = plan_tidy(copies)

    assert ids(plan.keep) == ["v2"]


class FakeFiles:
    def __init__(self, tree):
        self.tree = tree          # parent id -> list of file entries
        self.trashed = []

    def list(self, q, fields, pageSize, pageToken):
        parent = q.split("'")[1] if "name =" not in q else q.split("' in parents")[0].split("'")[-1]
        entries = self.tree.get(parent, [])
        if "name = '" in q:
            wanted = q.split("name = '")[1].split("'")[0]
            entries = [e for e in entries if e["name"] == wanted]
        elif "mimeType != " in q:
            entries = [e for e in entries if e.get("mimeType") != drive_tidy.FOLDER]
        return _Reply({"files": entries})

    def update(self, fileId, body):
        self.trashed.append((fileId, body))
        return _Reply({})


class _Reply:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeService:
    def __init__(self, tree):
        self._files = FakeFiles(tree)

    def files(self):
        return self._files


def test_a_shortcut_leads_to_the_shared_folder_and_files_are_listed_with_their_owner():
    tree = {
        "root": [
            {"id": "sc", "name": "cloudband_treino", "mimeType": drive_tidy.SHORTCUT,
             "shortcutDetails": {"targetId": "shared-id"}},
            {"id": "x1", "name": "run.state.pt", "size": "10", "modifiedTime": "2026-10-10T00:00:00Z",
             "ownedByMe": True, "mimeType": "application/x-zip"},
            {"id": "x2", "name": "unrelated.txt", "size": "1", "modifiedTime": "2026-10-10T00:00:00Z",
             "ownedByMe": True, "mimeType": "text/plain"},
        ],
        "shared-id": [{"id": "res", "name": "results", "mimeType": drive_tidy.FOLDER}],
        "res": [{"id": "p1", "name": "run.state.pt", "size": "10", "mimeType": "application/x-zip",
                 "modifiedTime": "2026-10-11T00:00:00Z", "ownedByMe": False}],
    }
    service = FakeService(tree)

    results = drive_tidy.resolve_folder(service, ("cloudband_treino", "results"))
    copies = drive_tidy.list_copies(service, {results: SHARED, "root": ROOT}, ("run",))

    assert results == "res"
    assert sorted((c.id, c.where, c.mine) for c in copies) == [
        ("p1", SHARED, False), ("x1", ROOT, True)]


def test_trash_marks_files_as_trashed_and_does_not_delete_them():
    service = FakeService({})

    count = drive_tidy.trash(service, [copy("a"), copy("b")])

    assert count == 2
    assert service.files().trashed == [("a", {"trashed": True}), ("b", {"trashed": True})]


def test_folder_parts_are_taken_from_the_mounted_path():
    assert drive_tidy.folder_parts("/content/drive/MyDrive/cloudband_treino/results/progress") == (
        "cloudband_treino", "results", "progress")
