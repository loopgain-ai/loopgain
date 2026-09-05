"""Fail-closed Docker execution for untrusted candidates (stdlib only).

Images must already exist locally; this module never pulls or builds one.
The image ID is resolved before execution so a mutable tag cannot change mid-run.
"""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024


class SandboxUnavailable(RuntimeError):
    """Infrastructure failure: stop rather than spend on an unexecutable trial."""


class SandboxExecutionError(RuntimeError):
    """Candidate exceeded a limit, crashed, or returned invalid output."""


def _docker() -> str:
    executable = shutil.which("docker")
    if os.name != "posix" or not executable:
        raise SandboxUnavailable(
            "Candidate execution requires Docker on a POSIX host; no host fallback."
        )
    return executable


def _docker_env() -> dict[str, str]:
    # These configure the CLI only; the candidate environment is cleared by env -i.
    return {
        key: os.environ[key]
        for key in ("PATH", "HOME", "DOCKER_CONFIG")
        if key in os.environ
    }


def _local_image(docker: str) -> str:
    image = os.environ.get("LOOPGAIN_SANDBOX_IMAGE", "loopgain-python-pytest:local")
    try:
        result = subprocess.run(
            [docker, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=10,
            env=_docker_env(),
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxUnavailable(
            "Docker or local Python sandbox image unavailable; no automatic pull."
        ) from exc
    image_id = result.stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise SandboxUnavailable("Docker did not return an immutable local image ID.")
    return image_id


def _command(docker: str, image: str, name: str, directory: Path) -> list[str]:
    return [
        docker,
        "create",
        "--pull=never",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=128m",
        "--memory-swap=128m",
        "--cpus=1",
        "--ulimit=cpu=20:20",
        "--ulimit=nofile=64:64",
        "--ulimit=core=0:0",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--mount",
        f"type=bind,src={directory},dst=/input,readonly",
        "--workdir=/tmp",
        "--log-driver=none",
        "--entrypoint=/usr/bin/env",
        image,
        "-i",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "HOME=/tmp",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "python3",
        "-I",
        "-B",
        "-c",
        (
            "import runpy,sys; sys.stdin=open('/input/request.json'); "
            "runpy.run_path('/input/worker.py',run_name='__main__')"
        ),
    ]


def _run(image: str, worker: str, payload: dict, timeout: float, files=None) -> str:
    encoded = json.dumps(payload).encode()
    files = files or {}
    if any(name not in {"solution.py", "test_solution.py"} for name in files):
        raise ValueError("Unexpected sandbox fixture name")
    if (
        len(encoded) + sum(len(text.encode()) for text in files.values())
        > MAX_INPUT_BYTES
    ):
        raise SandboxExecutionError("Sandbox input exceeds 2 MiB")
    docker = _docker()
    name = f"loopgain-example-candidate-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="loopgain-example-candidate-") as temporary:
        directory = Path(temporary)
        # Only staged worker/request/fixtures are mounted; never repo, home or Docker socket.
        directory.chmod(0o755)
        for filename, contents in (
            ("worker.py", worker.encode()),
            ("request.json", encoded),
        ):
            path = directory / filename
            path.write_bytes(contents)
            path.chmod(0o444)
        for filename, contents in files.items():
            path = directory / filename
            path.write_text(contents)
            path.chmod(0o444)
        proc = None
        try:
            subprocess.run(
                _command(docker, image, name, directory),
                capture_output=True,
                timeout=15,
                env=_docker_env(),
                check=True,
            )
            proc = subprocess.Popen(
                [docker, "start", "--attach", name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_docker_env(),
                start_new_session=True,
            )
            output = bytearray()
            error = bytearray()
            deadline = time.monotonic() + timeout
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ, output)
                selector.register(proc.stderr, selectors.EVENT_READ, error)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise SandboxExecutionError("Sandbox wall-clock timeout")
                    for key, _ in selector.select(min(remaining, 0.1)):
                        chunk = os.read(key.fileobj.fileno(), 4096)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        key.data.extend(chunk)
                        if len(output) + len(error) > MAX_OUTPUT_BYTES:
                            raise SandboxExecutionError("Sandbox output limit exceeded")
            proc.wait(timeout=max(0.01, deadline - time.monotonic()))
            if proc.returncode:
                raise SandboxExecutionError(
                    f"Sandbox worker exited {proc.returncode}: {error[:200]!r}"
                )
            try:
                return output.decode("utf8")
            except UnicodeError as exc:
                raise SandboxExecutionError("Invalid sandbox output encoding") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxUnavailable("Docker sandbox process failed") from exc
        finally:
            # Remove the container explicitly, including every descendant process.
            # Do not rely on terminating the attached docker CLI to stop a container.
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                cleanup = subprocess.run(
                    [docker, "rm", "--force", name],
                    capture_output=True,
                    timeout=10,
                    env=_docker_env(),
                    check=False,
                )
                if cleanup.returncode and b"No such container" not in cleanup.stderr:
                    raise SandboxUnavailable(
                        "Could not confirm sandbox container cleanup; stop the run."
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise SandboxUnavailable(
                    "Sandbox cleanup unavailable; stop the run."
                ) from exc
            finally:
                if proc is not None:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                    proc.stdout.close()
                    proc.stderr.close()


def ensure_available() -> str:
    """Exercise the actual isolation flags before any model request is made."""
    image = _local_image(_docker())
    try:
        result = _run(image, 'import pytest; print("sandbox-ready")', {}, 15)
    except SandboxExecutionError as exc:
        raise SandboxUnavailable(
            "Sandbox preflight failed; do not start model requests."
        ) from exc
    if result.strip() != "sandbox-ready":
        raise SandboxUnavailable("Unexpected sandbox preflight result")
    return image


_PYTEST_WORKER = """import contextlib,io,json,pytest
output=io.StringIO()
with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    result=pytest.main(['/input/test_solution.py','-q','--tb=short','--no-header','-p','no:cacheprovider'])
print(json.dumps({'returncode':int(result),'output':output.getvalue()}))
"""


def execute_pytest(solution: str, tests: str) -> tuple[int, str]:
    """Run the fixed pytest fixture and candidate exclusively inside Docker."""
    image = _local_image(_docker())
    raw = _run(
        image,
        _PYTEST_WORKER,
        {},
        30,
        {"solution.py": solution, "test_solution.py": tests},
    )
    try:
        result = json.loads(raw)
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("returncode"), int)
            or not isinstance(result.get("output"), str)
        ):
            raise ValueError("Invalid pytest result")
        return result["returncode"], result["output"]
    except (TypeError, ValueError) as exc:
        raise SandboxExecutionError("Invalid sandbox result") from exc
