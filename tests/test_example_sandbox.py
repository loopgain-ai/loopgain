"""Synthetic sandbox checks; integration requires an existing local image."""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

import importlib.util

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SPEC = importlib.util.spec_from_file_location(
    "example_sandbox", EXAMPLES / "_sandbox.py"
)
sandbox = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sandbox)

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = pytest.mark.skipif(
    os.environ.get("LOOPGAIN_SANDBOX_INTEGRATION") != "1",
    reason="opt in with LOOPGAIN_SANDBOX_INTEGRATION=1; never pulls images",
)


def test_container_has_mandatory_boundaries(tmp_path):
    cmd = sandbox._command("docker", "sha256:" + "a" * 64, "test", tmp_path)
    for flag in (
        "--pull=never",
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=128m",
        "--memory-swap=128m",
        "--cpus=1",
        "--log-driver=none",
    ):
        assert flag in cmd
    assert cmd.count("--mount") == 1
    assert f"type=bind,src={tmp_path},dst=/input,readonly" in cmd
    assert "-i" in cmd
    assert not any("docker.sock" in x for x in cmd)


def test_missing_docker_fails_closed(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
    with pytest.raises(sandbox.SandboxUnavailable, match="no host fallback"):
        sandbox.ensure_available()


def test_missing_image_does_not_pull(monkeypatch):
    calls = []

    def unavailable(cmd, **kwargs):
        calls.append(cmd)
        raise sandbox.subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(sandbox.subprocess, "run", unavailable)
    with pytest.raises(sandbox.SandboxUnavailable, match="no automatic pull"):
        sandbox._local_image("docker")
    assert len(calls) == 1
    assert calls[0][1:3] == ["image", "inspect"]


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.setitem(__import__("sys").modules, "_sandbox", sandbox)
    spec = importlib.util.spec_from_file_location(
        "code_example", EXAMPLES / "01_code_pytest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_prevents_model_request(example, monkeypatch, tmp_path):
    call = Mock(side_effect=AssertionError("model must not be called"))
    monkeypatch.setattr(example, "call_claude", call)
    monkeypatch.setattr(
        example,
        "ensure_available",
        Mock(side_effect=sandbox.SandboxUnavailable("offline")),
    )
    with pytest.raises(sandbox.SandboxUnavailable):
        example.one_iteration(None, tmp_path, "", "")
    call.assert_not_called()


def test_cleanup_failure_stops_baseline(example, monkeypatch, tmp_path):
    iteration = Mock(side_effect=sandbox.SandboxUnavailable("cleanup unconfirmed"))
    monkeypatch.setattr(example, "one_iteration", iteration)
    with pytest.raises(sandbox.SandboxUnavailable):
        example.baseline_run(None, tmp_path)
    iteration.assert_called_once()


@INTEGRATION
def test_benign_candidate_pass_fail():
    status, output = sandbox.execute_pytest(
        "def add(a,b): return a+b",
        "from solution import add\ndef test_pass(): assert add(1,2)==3\ndef test_fail(): assert add(1,2)==4",
    )
    assert status == 1
    assert "1 failed, 1 passed" in output


@INTEGRATION
def test_effective_container_boundaries(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_HOST_SECRET", "must-not-reach-candidate")
    worker = """import os,json,pathlib
try:
 pathlib.Path('/input/request.json').write_text('changed')
 writable=True
except OSError:
 writable=False
pathlib.Path('/tmp/scratch').write_text('ok')
print(json.dumps(dict(uid=os.getuid(), env=dict(os.environ), writable=writable,
 memory=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip(),
 pids=pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip(),
 routes=pathlib.Path('/proc/net/route').read_text().splitlines(),
 status=pathlib.Path('/proc/self/status').read_text())))"""
    data = json.loads(sandbox._run(sandbox.ensure_available(), worker, {}, 10))
    assert data["uid"] == 65534
    assert "SYNTHETIC_HOST_SECRET" not in data["env"]
    assert set(data["env"]) <= {
        "PATH",
        "HOME",
        "LC_CTYPE",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    }
    assert not data["writable"]
    assert data["memory"] == "134217728"
    assert data["pids"] == "32"
    assert len(data["routes"]) == 1  # no routes beyond the table header
    assert "CapEff:\t0000000000000000" in data["status"]
    assert "NoNewPrivs:\t1" in data["status"]


@INTEGRATION
@pytest.mark.parametrize(
    "worker,timeout,match",
    [
        ("import time; time.sleep(30)", 0.5, "timeout"),
        ("print('x'*70000)", 10, "output limit"),
    ],
)
def test_limits_remove_container(monkeypatch, worker, timeout, match):
    from types import SimpleNamespace

    name = "loopgain-example-candidate-synthetic-limit-test"
    monkeypatch.setattr(
        sandbox.uuid, "uuid4", lambda: SimpleNamespace(hex="synthetic-limit-test")
    )
    image = sandbox._local_image(sandbox._docker())
    with pytest.raises(sandbox.SandboxExecutionError, match=match):
        sandbox._run(image, worker, {}, timeout)
    result = sandbox.subprocess.run(
        [sandbox._docker(), "container", "inspect", name],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert b"No such" in result.stderr


def test_cleanup_failure_stops_run(monkeypatch):
    import subprocess

    monkeypatch.setattr(sandbox, "_docker", lambda: "docker")

    def failed_create_and_cleanup(cmd, **kwargs):
        if cmd[1] == "create":
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 1, b"", b"daemon unavailable")

    monkeypatch.setattr(sandbox.subprocess, "run", failed_create_and_cleanup)
    with pytest.raises(sandbox.SandboxUnavailable, match="cleanup"):
        sandbox._run("sha256:" + "a" * 64, "print('unused')", {}, 1)


@INTEGRATION
def test_worker_exit_removes_background_child(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        sandbox.uuid, "uuid4", lambda: SimpleNamespace(hex="synthetic-child-test")
    )
    image = sandbox._local_image(sandbox._docker())
    result = sandbox._run(
        image,
        "import subprocess; subprocess.Popen(['sleep','30']); print('done')",
        {},
        10,
    )
    assert result.strip() == "done"
    check = sandbox.subprocess.run(
        [
            sandbox._docker(),
            "container",
            "inspect",
            "loopgain-example-candidate-synthetic-child-test",
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert check.returncode != 0
    assert b"No such" in check.stderr


@INTEGRATION
def test_actual_example_fixed_fixture_runs_in_container(example, tmp_path):
    (tmp_path / "solution.py").write_text(
        'def format_duration(seconds):\n if seconds < 0: raise ValueError()\n return "now"'
    )
    (tmp_path / "test_solution.py").write_text(example.TESTS)
    failures, output = example.run_pytest(tmp_path)
    assert failures == 13
    assert "13 failed, 2 passed" in output


def test_main_preflights_before_client_creation(example, monkeypatch):
    client = Mock(side_effect=AssertionError("client must not be created"))
    monkeypatch.setattr(example, "get_client", client)
    monkeypatch.setattr(
        example,
        "ensure_available",
        Mock(side_effect=sandbox.SandboxUnavailable("offline")),
    )
    with pytest.raises(sandbox.SandboxUnavailable):
        example.main()
    client.assert_not_called()


def test_cleanup_failure_stops_loopgain(example, monkeypatch, tmp_path):
    from types import SimpleNamespace

    iteration = Mock(side_effect=sandbox.SandboxUnavailable("cleanup unconfirmed"))
    observe = Mock(side_effect=AssertionError("fatal failure must not be scored"))
    monkeypatch.setattr(example, "one_iteration", iteration)
    monkeypatch.setattr(
        example,
        "LoopGain",
        lambda **kwargs: SimpleNamespace(should_continue=lambda: True, observe=observe),
    )
    with pytest.raises(sandbox.SandboxUnavailable):
        example.loopgain_run(None, tmp_path)
    iteration.assert_called_once()
    observe.assert_not_called()
