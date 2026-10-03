from __future__ import annotations

import errno
import os
import sys
from pathlib import Path


try:
    import fcntl
except ImportError:  # pragma: no cover - Unix stdlib
    fcntl = None  # type: ignore[assignment]

_BUSY = {errno.EAGAIN, errno.EACCES, getattr(errno, "EWOULDBLOCK", errno.EAGAIN)}

_CLEANUP_NOTE = "lifecycle_release_failed {} errno={}"


class LifecycleError(Exception):
    reason_code = "lifecycle_unavailable"


class LifecycleBusy(LifecycleError):
    reason_code = "lifecycle_busy"


class LifecycleUnavailable(LifecycleError):
    reason_code = "lifecycle_unavailable"


class LifecycleLeaseInvalid(LifecycleError):
    reason_code = "lifecycle_lease_invalid"


class LifecycleReleaseFailed(LifecycleError):
    reason_code = "lifecycle_release_failed"


def _cleanup_note(stage: str, exc: OSError) -> str:
    code = exc.errno if type(exc.errno) is int else "unknown"
    return _CLEANUP_NOTE.format(stage, code)


class LifecycleLease:
    def __init__(self, output_dir: Path, fd: int) -> None:
        self.output_dir = output_dir
        self._fd = fd
        self._pid = os.getpid()
        self._released = False

    def check(self, output_dir: Path | str) -> None:
        if self._released or self._fd < 0 or os.getpid() != self._pid:
            raise LifecycleLeaseInvalid("lifecycle_lease_invalid")
        if Path(output_dir).resolve() != self.output_dir:
            raise LifecycleLeaseInvalid("lifecycle_lease_invalid")

    def release(self) -> None:
        if self._released:
            return
        fd = self._fd
        creator = self._pid == os.getpid()
        self._fd = -1
        self._released = True

        unlock_settled = False
        unlock_error: OSError | None = None
        close_error: OSError | None = None
        try:
            if creator and fcntl is not None and fd >= 0:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError as exc:
                    unlock_error = exc
            unlock_settled = True
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError as exc:
                    close_error = exc
            if not unlock_settled:
                unwinding = sys.exc_info()[1]
                if unwinding is not None and close_error is not None:
                    unwinding.add_note(_cleanup_note("close", close_error))

        first_error = unlock_error if unlock_error is not None else close_error
        if first_error is not None:
            failed = LifecycleReleaseFailed("lifecycle_release_failed")
            if unlock_error is not None:
                failed.add_note(_cleanup_note("unlock", unlock_error))
            if close_error is not None:
                failed.add_note(_cleanup_note("close", close_error))
            raise failed from first_error


def lock_path_for(output_dir: Path | str) -> Path:
    resolved = Path(output_dir).resolve()
    parent = resolved.parent
    if not parent.is_dir():
        raise LifecycleUnavailable("lifecycle_unavailable")
    return parent / f".{resolved.name}.cloud-capability.lifecycle.lock"


def acquire(output_dir: Path | str) -> LifecycleLease:
    if fcntl is None:
        raise LifecycleUnavailable("lifecycle_unavailable")
    path = lock_path_for(output_dir)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        raise LifecycleUnavailable("lifecycle_unavailable") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        close_error: OSError | None = None
        try:
            os.close(fd)
        except OSError as close_exc:
            close_error = close_exc
        if exc.errno in _BUSY:
            refused: LifecycleError = LifecycleBusy("lifecycle_busy")
        else:
            refused = LifecycleUnavailable("lifecycle_unavailable")
        if close_error is not None:
            refused.add_note(_cleanup_note("close", close_error))
        raise refused from exc
    return LifecycleLease(Path(output_dir).resolve(), fd)


class hold:
    def __init__(self, output_dir: Path | str, lease: LifecycleLease | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.lease = lease
        self._owned: LifecycleLease | None = None

    def __enter__(self) -> LifecycleLease:
        if self.lease is None:
            self._owned = acquire(self.output_dir)
            return self._owned
        self.lease.check(self.output_dir)
        return self.lease

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if self._owned is None:
                return False
            try:
                self._owned.release()
            except LifecycleReleaseFailed as release_error:
                if exc is not None:
                    for note in getattr(release_error, "__notes__", ()):
                        exc.add_note(note)
                    return False
                raise
            return False
        finally:
            self._owned = None
