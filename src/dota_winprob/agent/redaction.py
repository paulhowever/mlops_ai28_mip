import re
from collections.abc import Iterable, Mapping

from dota_winprob.logging_config import mask_secrets

MASK = "***"
MIN_SECRET_LENGTH = 8

_SENSITIVE = (
    r"(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|authorization)"
)
_SECRET_FIELD = re.compile(
    rf"(?P<key>[\"']?[\w-]{{0,40}}{_SENSITIVE}[\"']?\s*[:=]\s*)(?P<quote>[\"']?)(?P<value>[^\s\"',;&}}]+)",
    re.IGNORECASE,
)
_AUTH_SCHEME = re.compile(r"\b(?P<scheme>Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{6,}", re.IGNORECASE)
_KEY_LIKE = re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{8,}")
_SECRET_KEY = re.compile(rf"^[\w-]*{_SENSITIVE}$", re.IGNORECASE)


class Redactor:
    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        self._secrets = tuple(
            secret for secret in secrets if secret and len(secret) >= MIN_SECRET_LENGTH
        )

    def text(self, value: str) -> str:
        for secret in self._secrets:
            value = value.replace(secret, MASK)
        value = _AUTH_SCHEME.sub(rf"\g<scheme> {MASK}", value)
        value = mask_secrets(value)
        value = _SECRET_FIELD.sub(rf"\g<key>\g<quote>{MASK}", value)
        return _KEY_LIKE.sub(MASK, value)

    def data(self, value: Mapping[str, object]) -> dict[str, object]:
        return {self.text(str(key)): self._item(str(key), item) for key, item in value.items()}

    def _item(self, key: str, value: object) -> object:
        if isinstance(value, str) and _SECRET_KEY.match(key):
            return MASK
        return self._value(value)

    def _value(self, value: object) -> object:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return self.data(value)
        if isinstance(value, list | tuple):
            return [self._value(item) for item in value]
        return value
