# Model utilities
import os


def win_long_path(path: str) -> str:
    """Prefix absolute paths with ``\\\\?\\`` on Windows to bypass the 260-char MAX_PATH limit.

    Safe to call unconditionally on any OS — returns the path unchanged on non-Windows.
    """
    if os.name == 'nt' and path and len(path) >= 260 and not path.startswith('\\\\?\\'):
        return '\\\\?\\' + os.path.abspath(path)
    return path
