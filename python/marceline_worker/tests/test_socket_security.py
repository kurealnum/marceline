import os
import stat

import pytest

from marceline_worker.socket_security import prepare_socket_path, secure_socket_permissions


def test_prepare_socket_path_removes_a_stale_socket_owned_by_the_current_user(tmp_path):
    socket_path = tmp_path / "worker.sock"
    socket_path.touch()

    prepare_socket_path(str(socket_path))

    assert not socket_path.exists()


def test_prepare_socket_path_is_a_noop_when_nothing_is_there(tmp_path):
    socket_path = tmp_path / "worker.sock"

    prepare_socket_path(str(socket_path))  # must not raise


def test_prepare_socket_path_refuses_a_file_owned_by_another_uid(tmp_path, monkeypatch):
    socket_path = tmp_path / "worker.sock"
    socket_path.touch()

    real_lstat = os.lstat

    def fake_lstat(path):
        result = real_lstat(path)
        if str(path) == str(socket_path):
            other_uid = result.st_uid + 1
            return os.stat_result(
                (result.st_mode, result.st_ino, result.st_dev, result.st_nlink, other_uid, result.st_gid)
                + result[6:]
            )
        return result

    monkeypatch.setattr(os, "lstat", fake_lstat)

    with pytest.raises(PermissionError):
        prepare_socket_path(str(socket_path))
    assert socket_path.exists()


def test_secure_socket_permissions_sets_mode_0600(tmp_path):
    socket_path = tmp_path / "worker.sock"
    socket_path.touch()
    os.chmod(socket_path, 0o644)

    secure_socket_permissions(str(socket_path))

    mode = stat.S_IMODE(os.stat(socket_path).st_mode)
    assert mode == 0o600
