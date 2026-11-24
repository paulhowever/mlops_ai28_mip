import re

WORDS = {
    0: "ноль|нулю|ни одн",
    1: "один|одна|одно|одного",
    2: "два|две|двух|двое",
    3: "три|трёх|трех|трое|трижды",
    4: "четыре|четырёх",
    5: "пять|пяти",
}
_NUMBER = re.compile(r"(?<![A-Za-z\d.,])\d+(?:[.,]\d+)?")
_FINAL = re.compile(r"ИТОГ\s*:\s*(.+)", re.IGNORECASE)
_CLOCK = re.compile(r"(\d{1,2}):(\d{2})")


def numbers(text: str) -> list[float]:
    spaceless = re.sub(r"(?<=\d)[   ](?=\d{3}(?!\d))", "", text)
    return [float(raw.replace(",", ".")) for raw in _NUMBER.findall(spaceless)]


def has_number(
    text: str, value: float, tolerance: float, scales: tuple[float, ...] = (1.0,)
) -> bool:
    if any(
        abs(found - value * scale) <= tolerance * scale
        for found in numbers(text)
        for scale in scales
    ):
        return True
    words = WORDS.get(int(value)) if value == int(value) else None
    return bool(words and re.search(rf"\b(?:{words})\b", text, re.IGNORECASE))


def has_probability(text: str, value: float) -> bool:
    return has_number(text, value, 0.0051, (1.0, 100.0))


def has_any(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


def final(text: str) -> str:
    found = _FINAL.findall(text)
    return found[-1].strip() if found else ""


def final_number(text: str) -> float | None:
    found = numbers(final(text))
    return found[0] if found else None


def final_minute(text: str) -> int | None:
    found = _CLOCK.search(final(text))
    return int(found[1]) * 60 + int(found[2]) if found else None
