from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

DISTRIBUTION_NAME = "dota-winprob"

UNKNOWN_VERSION = "0.0.0+unknown"


def _resolve_version() -> str:
    # Без фолбэка запуск из исходников (пакет не установлен) уронил бы
    # импорт, а вместе с ним всё приложение, включая liveness-пробу.
    try:
        return _distribution_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VERSION


__version__ = _resolve_version()

__all__ = ["DISTRIBUTION_NAME", "UNKNOWN_VERSION", "__version__"]
