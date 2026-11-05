import subprocess
import sys

from cloudband.molab.workers import Worker


def test_a_worker_reports_its_state_and_the_last_lines_it_showed(tmp_path):
    log = tmp_path / "worker1.log"
    with open(log, "w") as handle:
        process = subprocess.Popen(
            [sys.executable, "-c", "print('a'); print('epoch 1\\repoch 2'); raise SystemExit(3)"],
            stdout=handle,
        )
        process.wait()
    worker = Worker(1, process, log)

    assert worker.state == "stopped with error (exit 3)"
    assert worker.tail() == "a\nepoch 2"


def test_the_worker_command_line_parses(tmp_path):
    reply = subprocess.run(
        [sys.executable, "-m", "cloudband.molab.workers", "--help"],
        capture_output=True, text=True, env={"PYTHONPATH": "src"},
    )
    assert reply.returncode == 0
    assert "--parent-training-dir" in reply.stdout
