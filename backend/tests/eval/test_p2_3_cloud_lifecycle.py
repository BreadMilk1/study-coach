from __future__ import annotations

import asyncio
import errno
import fcntl
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from traceback import walk_tb

import pytest

from app.eval.p2_3_cloud_capability.lifecycle import (
    LifecycleBusy,
    LifecycleLeaseInvalid,
    LifecycleUnavailable,
    acquire,
    hold,
    lock_path_for,
)

_BACKEND = Path(__file__).resolve().parents[2]
_CHILD = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from app.eval.p2_3_cloud_capability.lifecycle import acquire
lease = acquire(Path(sys.argv[2]))
sys.stdout.write("HELD\\n")
sys.stdout.flush()
sys.stdin.readline()
lease.release()
sys.stdout.write("RELEASED\\n")
sys.stdout.flush()
"""
_HANDSHAKE_TIMEOUT = 10.0


def _readline_with_deadline(proc: subprocess.Popen, deadline: float):
    fd = proc.stdout.fileno()
    buf = bytearray()
    while True:
        if b"\n" in buf:
            line, _, _ = bytes(buf).partition(b"\n")
            return line + b"\n"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return bytes(buf) if buf else None
        ready, _, _ = select.select([fd], [], [], min(remaining, 0.2))
        if not ready:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            return bytes(buf) if buf else None
        buf.extend(chunk)


def _reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=_HANDSHAKE_TIMEOUT)
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _hold_process(
    output_dir: Path, timeout: float = _HANDSHAKE_TIMEOUT, child: str = _CHILD
) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", child, str(_BACKEND), str(output_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        line = _readline_with_deadline(proc, time.monotonic() + timeout)
        if line != b"HELD\n":
            raise AssertionError(f"child handshake failed before HELD: {line!r}")
    except BaseException:
        _reap(proc)
        raise
    return proc


def _release_process(proc: subprocess.Popen, timeout: float = _HANDSHAKE_TIMEOUT) -> None:
    try:
        assert proc.stdin is not None
        proc.stdin.write(b"\n")
        proc.stdin.flush()
        line = _readline_with_deadline(proc, time.monotonic() + timeout)
        if line != b"RELEASED\n":
            raise AssertionError(f"child handshake failed before RELEASED: {line!r}")
        assert proc.wait(timeout=timeout) == 0
    finally:
        _reap(proc)


def _waitpid_bounded(pid: int, timeout: float = _HANDSHAKE_TIMEOUT) -> int:
    deadline = time.monotonic() + timeout
    while True:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return status
        if time.monotonic() >= deadline:
            os.kill(pid, signal.SIGKILL)
            return os.waitpid(pid, 0)[1]
        time.sleep(0.005)


def test_same_output_aliases_share_one_lock_and_other_output_does_not(tmp_path):
    real = tmp_path / "out"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real, target_is_directory=True)
    other = tmp_path / "other"
    other.mkdir()
    first = acquire(real)
    try:
        with pytest.raises(LifecycleBusy):
            acquire(link)
        with pytest.raises(LifecycleBusy):
            acquire(real.resolve())
        second = acquire(other)
        second.release()
        assert lock_path_for(real) == lock_path_for(link)
        assert lock_path_for(real) != lock_path_for(other)
        assert lock_path_for(real).read_bytes() == b""
    finally:
        first.release()
    again = acquire(link)
    again.release()


def test_cross_process_lock_is_nonblocking(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    proc = _hold_process(out)
    try:
        with pytest.raises(LifecycleBusy):
            acquire(out)
    finally:
        _release_process(proc)
    lease = acquire(out)
    lease.release()


def test_missing_parent_and_missing_fcntl_fail_closed(tmp_path, monkeypatch):
    missing = tmp_path / "nope" / "out"
    with pytest.raises(LifecycleUnavailable):
        acquire(missing)
    assert not (tmp_path / "nope").exists()
    import app.eval.p2_3_cloud_capability.lifecycle as life

    monkeypatch.setattr(life, "fcntl", None)
    ready = tmp_path / "ready"
    ready.mkdir()
    with pytest.raises(LifecycleUnavailable):
        life.acquire(ready)


def test_invalid_or_foreign_lease_is_rejected_before_use(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    lease = acquire(left)
    lease.release()
    with pytest.raises(LifecycleLeaseInvalid):
        with hold(left, lease):
            pass
    fresh = acquire(left)
    try:
        with pytest.raises(LifecycleLeaseInvalid):
            with hold(right, fresh):
                pass
        still = acquire(right)
        still.release()
    finally:
        fresh.release()


def test_borrower_does_not_release_owner_lock(tmp_path):
    lease = acquire(tmp_path)
    try:
        with hold(tmp_path, lease) as borrowed:
            assert borrowed is lease
        with pytest.raises(LifecycleBusy):
            acquire(tmp_path)
    finally:
        lease.release()
    nxt = acquire(tmp_path)
    nxt.release()


def test_inherited_lease_cleanup_does_not_unlock_parent(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    lease = acquire(out)
    pid = os.fork()
    if pid == 0:
        try:
            lease.release()
        except BaseException:
            os._exit(1)
        os._exit(0)
    status = _waitpid_bounded(pid)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    third = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from app.eval.p2_3_cloud_capability.lifecycle import LifecycleBusy, acquire\n"
            "try:\n"
            "    lease = acquire(Path(sys.argv[2]))\n"
            "    lease.release()\n"
            "    print('GOT')\n"
            "except LifecycleBusy:\n"
            "    print('BUSY')\n",
            str(_BACKEND),
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert third.stdout.strip() == "BUSY", third.stderr
    lease.release()
    third_after = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from app.eval.p2_3_cloud_capability.lifecycle import acquire\n"
            "lease = acquire(Path(sys.argv[2]))\n"
            "lease.release()\n"
            "print('GOT')\n",
            str(_BACKEND),
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert third_after.stdout.strip() == "GOT", third_after.stderr


def test_fork_child_exit_code_reports_release_failure(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    out = tmp_path / "out"
    out.mkdir()
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    shim.fail_next_close = True
    lease = acquire(out)
    try:
        pid = os.fork()
        if pid == 0:
            try:
                lease.release()
            except BaseException:
                os._exit(1)
            os._exit(0)
        status = _waitpid_bounded(pid)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 1
    finally:
        shim.fail_next_close = False
        lease.release()
    follow = acquire(out)
    follow.release()


def test_existing_lock_bytes_are_not_truncated(tmp_path):
    path = lock_path_for(tmp_path)
    path.write_bytes(b"keep")
    lease = acquire(tmp_path)
    try:
        assert path.read_bytes() == b"keep"
    finally:
        lease.release()
    assert path.is_file()
    assert path.read_bytes() == b"keep"
    assert os.path.basename(path).startswith(".")


# ---------------------------------------------------------------------------
# Release/cleanup failure contract
# ---------------------------------------------------------------------------

_RELEASE_SENTINEL = "top-secret-io-detail"
_PATH_SENTINEL = "/secret/injected/path"
_CLOSE_NOTE = f"lifecycle_release_failed close errno={errno.EIO}"
_UNLOCK_NOTE = f"lifecycle_release_failed unlock errno={errno.EIO}"


def _unlock_io_error() -> OSError:
    return OSError(errno.EIO, f"injected unlock failure {_RELEASE_SENTINEL} {_PATH_SENTINEL}")


def _unlock_programming_error() -> RuntimeError:
    return RuntimeError("injected unlock programming error")


class _OsCloseShim:
    """Replaces the os module seen by lifecycle only; delegates everything else."""

    def __init__(self, real_os):
        self._real = real_os
        self.fail_close_fds: set[int] = set()
        self.fail_next_close = False
        self.close_calls: list[int] = []

    def __getattr__(self, name):
        return getattr(self._real, name)

    def close(self, fd):
        self.close_calls.append(fd)
        if fd in self.fail_close_fds or self.fail_next_close:
            self.fail_next_close = False
            raise OSError(errno.EIO, f"injected close failure {_RELEASE_SENTINEL} {_PATH_SENTINEL}")
        return self._real.close(fd)


class _FcntlShim:
    """Replaces the fcntl module seen by lifecycle only; fails the selected stage."""

    LOCK_UN = fcntl.LOCK_UN
    LOCK_EX = fcntl.LOCK_EX
    LOCK_NB = fcntl.LOCK_NB

    def __init__(self, real_fcntl, unlock_error=None, acquire_error=None):
        self._real = real_fcntl
        self._unlock_error = unlock_error
        self._acquire_error = acquire_error
        self.calls: list[tuple[int, int]] = []

    def flock(self, fd, cmd):
        self.calls.append((fd, cmd))
        if cmd == self.LOCK_UN and self._unlock_error is not None:
            raise self._unlock_error()
        if cmd != self.LOCK_UN and self._acquire_error is not None:
            raise self._acquire_error()
        return self._real.flock(fd, cmd)


def _scratch_fd(tmp_path: Path, name: str) -> int:
    return os.open(tmp_path / name, os.O_RDWR | os.O_CREAT, 0o644)


def _reclaim_failed_close(shim: _OsCloseShim) -> None:
    # an injected close failure leaves the real descriptor open; close it if
    # it is still alive so tests do not leak descriptors
    for fd in shim.close_calls:
        try:
            os.fstat(fd)
        except OSError:
            continue
        os.close(fd)


def _assert_sanitized(stable, notes) -> None:
    assert _RELEASE_SENTINEL not in str(stable)
    assert all(_RELEASE_SENTINEL not in note and _PATH_SENTINEL not in note for note in notes)


def test_release_io_failure_via_hold_raises_stable_exception(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    monkeypatch.setattr(life, "fcntl", _FcntlShim(fcntl, unlock_error=_unlock_io_error))
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(life.LifecycleReleaseFailed) as excinfo:
        with hold(out):
            pass
    failed = excinfo.value
    assert failed.reason_code == "lifecycle_release_failed"
    assert str(failed) == "lifecycle_release_failed"
    assert type(failed.__cause__) is OSError
    assert failed.__cause__.errno == errno.EIO
    assert "unlock failure" in str(failed.__cause__)
    notes = getattr(failed, "__notes__", [])
    assert notes == [_UNLOCK_NOTE]
    _assert_sanitized(failed, notes)
    assert len(shim.close_calls) == 1
    _reclaim_failed_close(shim)


def test_release_close_failure_raises_stable_exception(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    out = tmp_path / "out"
    out.mkdir()
    fd = _scratch_fd(tmp_path, "scratch-fd")
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    shim.fail_close_fds.add(fd)
    lease = life.LifecycleLease(out.resolve(), fd)
    with pytest.raises(life.LifecycleReleaseFailed) as excinfo:
        lease.release()
    failed = excinfo.value
    assert failed.reason_code == "lifecycle_release_failed"
    assert str(failed) == "lifecycle_release_failed"
    assert type(failed.__cause__) is OSError
    assert failed.__cause__.errno == errno.EIO
    assert "close failure" in str(failed.__cause__)
    notes = getattr(failed, "__notes__", [])
    assert notes == [_CLOSE_NOTE]
    _assert_sanitized(failed, notes)
    assert len(shim.close_calls) == 1
    _reclaim_failed_close(shim)


def test_both_cleanup_failures_keep_first_cause_and_all_evidence(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    out = tmp_path / "out"
    out.mkdir()
    fd = _scratch_fd(tmp_path, "scratch-fd")
    monkeypatch.setattr(life, "fcntl", _FcntlShim(fcntl, unlock_error=_unlock_io_error))
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    shim.fail_close_fds.add(fd)
    lease = life.LifecycleLease(out.resolve(), fd)
    with pytest.raises(life.LifecycleReleaseFailed) as excinfo:
        lease.release()
    failed = excinfo.value
    assert failed.reason_code == "lifecycle_release_failed"
    assert str(failed) == "lifecycle_release_failed"
    assert type(failed.__cause__) is OSError
    assert failed.__cause__.errno == errno.EIO
    assert "unlock failure" in str(failed.__cause__)
    notes = getattr(failed, "__notes__", [])
    assert notes == [_UNLOCK_NOTE, _CLOSE_NOTE]
    _assert_sanitized(failed, notes)
    assert len(shim.close_calls) == 1
    _reclaim_failed_close(shim)


def test_unlock_programming_error_stays_primary_with_close_evidence(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(
        life, "fcntl", _FcntlShim(fcntl, unlock_error=_unlock_programming_error)
    )

    fd = _scratch_fd(tmp_path, "scratch-fd-1")
    lease = life.LifecycleLease(out.resolve(), fd)
    with pytest.raises(RuntimeError) as excinfo:
        lease.release()
    err = excinfo.value
    assert type(err) is RuntimeError
    assert not isinstance(err, life.LifecycleError)
    assert not getattr(err, "__notes__", [])
    with pytest.raises(OSError):
        os.close(fd)

    fd2 = _scratch_fd(tmp_path, "scratch-fd-2")
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    shim.fail_close_fds.add(fd2)
    lease2 = life.LifecycleLease(out.resolve(), fd2)
    with pytest.raises(RuntimeError) as excinfo2:
        lease2.release()
    err2 = excinfo2.value
    assert type(err2) is RuntimeError
    assert not isinstance(err2, life.LifecycleError)
    notes = getattr(err2, "__notes__", [])
    assert notes == [_CLOSE_NOTE]
    _assert_sanitized(err2, notes)
    assert len(shim.close_calls) == 1
    _reclaim_failed_close(shim)


def test_unlock_reraised_active_exception_keeps_close_evidence(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    out = tmp_path / "out"
    out.mkdir()
    fd = _scratch_fd(tmp_path, "scratch-fd")
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    shim.fail_close_fds.add(fd)
    holder: dict[str, BaseException] = {}
    monkeypatch.setattr(
        life, "fcntl", _FcntlShim(fcntl, unlock_error=lambda: holder["active"])
    )
    lease = life.LifecycleLease(out.resolve(), fd)
    active = RuntimeError("injected active failure")
    try:
        raise active
    except RuntimeError:
        holder["active"] = active
        with pytest.raises(RuntimeError) as excinfo:
            lease.release()
    err = excinfo.value
    assert err is active
    assert type(err) is RuntimeError
    assert not isinstance(err, life.LifecycleError)
    notes = getattr(err, "__notes__", [])
    assert notes == [_CLOSE_NOTE]
    _assert_sanitized(err, notes)
    assert len(shim.close_calls) == 1
    _reclaim_failed_close(shim)


def test_direct_release_in_except_handler_does_not_touch_handled_exception(
    tmp_path, monkeypatch
):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    out = tmp_path / "out"
    out.mkdir()
    fd = _scratch_fd(tmp_path, "scratch-fd")
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    shim.fail_close_fds.add(fd)
    lease = life.LifecycleLease(out.resolve(), fd)
    handled = ValueError("handled upstream")
    try:
        raise handled
    except ValueError:
        with pytest.raises(life.LifecycleReleaseFailed) as excinfo:
            lease.release()
    failed = excinfo.value
    assert failed.reason_code == "lifecycle_release_failed"
    assert str(failed) == "lifecycle_release_failed"
    assert type(failed.__cause__) is OSError
    assert failed.__cause__.errno == errno.EIO
    notes = getattr(failed, "__notes__", [])
    assert notes == [_CLOSE_NOTE]
    assert not getattr(handled, "__notes__", [])
    assert len(shim.close_calls) == 1
    _reclaim_failed_close(shim)


def test_body_error_with_release_failure_preserves_business_exception(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    out = tmp_path / "out"
    out.mkdir()
    business = RuntimeError("business-failure")
    with pytest.raises(RuntimeError) as excinfo:
        with hold(out):
            shim.fail_next_close = True
            raise business
    assert excinfo.value is business
    assert type(excinfo.value) is RuntimeError
    assert excinfo.value.__traceback__ is not None
    assert any(
        "test_p2_3_cloud_lifecycle" in str(frame.f_code.co_filename)
        for frame, _lineno in walk_tb(excinfo.value.__traceback__)
    )
    notes = getattr(excinfo.value, "__notes__", [])
    assert notes == [_CLOSE_NOTE]
    _assert_sanitized(excinfo.value, notes)
    _reclaim_failed_close(shim)


def test_cancelled_task_with_release_failure_stays_cancelled(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    out = tmp_path / "out"
    out.mkdir()
    captured: dict[str, BaseException] = {}

    async def _scenario():
        started = asyncio.Event()

        async def _body():
            with hold(out):
                shim.fail_next_close = True
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancel_error:
                    captured["cancelled"] = cancel_error
                    raise

        task = asyncio.create_task(_body())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await task
        assert task.cancelled()
        return excinfo.value

    err = asyncio.run(_scenario())
    assert captured["cancelled"] is not None
    assert err is captured["cancelled"]
    assert type(err) is asyncio.CancelledError
    notes = getattr(err, "__notes__", [])
    assert notes == [_CLOSE_NOTE]
    _assert_sanitized(err, notes)
    _reclaim_failed_close(shim)


def test_release_failure_is_idempotent_and_check_rejected(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    out = tmp_path / "out"
    out.mkdir()
    fd = _scratch_fd(tmp_path, "scratch-fd")
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    shim.fail_close_fds.add(fd)
    lease = life.LifecycleLease(out.resolve(), fd)
    with pytest.raises(life.LifecycleReleaseFailed):
        lease.release()
    lease.release()
    assert len(shim.close_calls) == 1
    with pytest.raises(life.LifecycleLeaseInvalid):
        lease.check(out)
    _reclaim_failed_close(shim)


def test_acquire_close_failure_keeps_busy_classification(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    out = tmp_path / "out"
    out.mkdir()
    proc = _hold_process(out)
    shim = _OsCloseShim(os)
    try:
        monkeypatch.setattr(life, "os", shim)
        shim.fail_next_close = True
        with pytest.raises(life.LifecycleBusy) as excinfo:
            life.acquire(out)
        busy = excinfo.value
        assert busy.reason_code == "lifecycle_busy"
        assert str(busy) == "lifecycle_busy"
        assert isinstance(busy.__cause__, OSError)
        assert busy.__cause__.errno in (errno.EAGAIN, errno.EACCES)
        notes = getattr(busy, "__notes__", [])
        assert notes == [_CLOSE_NOTE]
        _assert_sanitized(busy, notes)
        assert len(shim.close_calls) == 1
    finally:
        _release_process(proc)
    _reclaim_failed_close(shim)


def test_acquire_close_failure_keeps_unavailable_classification(tmp_path, monkeypatch):
    import app.eval.p2_3_cloud_capability.lifecycle as life

    ready = tmp_path / "ready"
    ready.mkdir()
    monkeypatch.setattr(life, "fcntl", _FcntlShim(fcntl, acquire_error=_unlock_io_error))
    shim = _OsCloseShim(os)
    monkeypatch.setattr(life, "os", shim)
    shim.fail_next_close = True
    with pytest.raises(life.LifecycleUnavailable) as excinfo:
        life.acquire(ready)
    unavailable = excinfo.value
    assert unavailable.reason_code == "lifecycle_unavailable"
    assert str(unavailable) == "lifecycle_unavailable"
    assert type(unavailable.__cause__) is OSError
    assert unavailable.__cause__.errno == errno.EIO
    notes = getattr(unavailable, "__notes__", [])
    assert notes == [_CLOSE_NOTE]
    _assert_sanitized(unavailable, notes)
    assert len(shim.close_calls) == 1
    _reclaim_failed_close(shim)


def test_without_cleanup_failure_lock_is_reacquirable(tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    with hold(out):
        pass
    lease = acquire(out)
    lease.release()

    with pytest.raises(RuntimeError):
        with hold(out):
            raise RuntimeError("business-failure")
    lease = acquire(out)
    lease.release()

    async def _cancelled():
        started = asyncio.Event()

        async def _body():
            with hold(out):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(_body())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_cancelled())
    lease = acquire(out)
    lease.release()


_SILENT_CHILD = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from app.eval.p2_3_cloud_capability.lifecycle import acquire
lease = acquire(Path(sys.argv[2]))
sys.stdin.readline()
lease.release()
"""

_PARTIAL_CHILD = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from app.eval.p2_3_cloud_capability.lifecycle import acquire
lease = acquire(Path(sys.argv[2]))
sys.stdout.write("HE")
sys.stdout.flush()
sys.stdin.readline()
lease.release()
"""


@pytest.mark.parametrize("child", [_SILENT_CHILD, _PARTIAL_CHILD], ids=["silent", "partial"])
def test_child_handshake_deadline_covers_silent_and_partial_output(tmp_path, child):
    out = tmp_path / "out"
    out.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-c", child, str(_BACKEND), str(out)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        got = _readline_with_deadline(proc, time.monotonic() + 1.0)
        assert got != b"HELD\n"
    finally:
        _reap(proc)
    assert proc.poll() is not None


def _record_popen(monkeypatch) -> list[subprocess.Popen]:
    created: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    return created


def test_hold_process_reaps_child_when_handshake_read_raises(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    created = _record_popen(monkeypatch)

    def _boom(proc, deadline):
        raise OSError(errno.EIO, "injected handshake read failure")

    monkeypatch.setitem(globals(), "_readline_with_deadline", _boom)
    try:
        with pytest.raises(OSError, match="injected handshake read failure"):
            _hold_process(out, timeout=5.0)
        victim = created[0]
        assert victim.poll() is not None, "child leaked after handshake failure"
        assert victim.stdin is not None and victim.stdin.closed
        assert victim.stdout is not None and victim.stdout.closed
    finally:
        for proc in created:
            _reap(proc)


@pytest.mark.parametrize("child", [_SILENT_CHILD, _PARTIAL_CHILD], ids=["silent", "partial"])
def test_hold_process_timeout_failure_reaps_child(tmp_path, monkeypatch, child):
    out = tmp_path / "out"
    out.mkdir()
    created = _record_popen(monkeypatch)
    try:
        with pytest.raises(AssertionError, match="handshake failed"):
            _hold_process(out, timeout=1.0, child=child)
        victim = created[0]
        assert victim.poll() is not None, "child leaked after handshake timeout"
        assert victim.stdin is not None and victim.stdin.closed
        assert victim.stdout is not None and victim.stdout.closed
    finally:
        for proc in created:
            _reap(proc)
