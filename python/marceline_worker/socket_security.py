"""Unix-socket bind hardening shared by every worker (SPEC.md §2.2, §11).

The worker socket carries the live microphone feed (STT) or spoken output
text (TTS), so it must not be readable or writable by another local user:
a socket left world-connectable in a shared directory like `/tmp` lets any
other account on the machine intercept audio or inject fake transcripts.
"""

from __future__ import annotations

import os
import stat


def prepare_socket_path(socket_path: str) -> None:
    """Clears `socket_path` for a fresh bind, refusing to touch a stale
    socket file this process doesn't own.

    A stale socket from this same user's previous, uncleanly-killed run is
    unlinked as before. A socket owned by a *different* user is left
    alone and raises instead of being silently replaced — unlinking it
    would let any local user squat the path first and have Marceline
    clear the way for them.
    """
    try:
        info = os.lstat(socket_path)
    except FileNotFoundError:
        return

    if info.st_uid != os.getuid():
        raise PermissionError(
            f"refusing to remove {socket_path}: it is owned by uid {info.st_uid}, "
            f"not the current user (uid {os.getuid()})"
        )
    os.unlink(socket_path)


def secure_socket_permissions(socket_path: str) -> None:
    """Chmods `socket_path` to 0600 after binding, so its permissions
    don't depend on the process umask.
    """
    os.chmod(socket_path, stat.S_IRUSR | stat.S_IWUSR)
