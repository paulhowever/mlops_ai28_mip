import json
import math
from collections.abc import Sequence

from pydantic_ai.messages import BinaryContent, ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from dota_winprob.agent.images import ImageError, inspect

CHARS_PER_TOKEN = 2.5
PIXELS_PER_TOKEN = 750
UNKNOWN_IMAGE_TOKENS = 4000
MESSAGE_OVERHEAD = 8
DEFAULT_OUTPUT = 1000


def _image_tokens(content: BinaryContent) -> int:
    try:
        image = inspect(content.data)
    except ImageError:
        return UNKNOWN_IMAGE_TOKENS
    return math.ceil(image.width * image.height / PIXELS_PER_TOKEN)


def _content(content: object) -> tuple[int, int]:
    if isinstance(content, str):
        return len(content), 0
    if isinstance(content, BinaryContent):
        return 0, _image_tokens(content)
    if isinstance(content, Sequence):
        sizes = [_content(item) for item in content]
        return sum(chars for chars, _ in sizes), sum(tokens for _, tokens in sizes)
    return len(str(content)), 0


def estimate_tokens(
    messages: list[ModelMessage],
    settings: ModelSettings | None,
    parameters: ModelRequestParameters,
) -> int:
    chars, tokens = 0, 0
    for message in messages:
        tokens += MESSAGE_OVERHEAD
        chars += len(getattr(message, "instructions", None) or "")
        for part in message.parts:
            content = getattr(part, "content", None)
            part_chars, part_tokens = _content("" if content is None else content)
            chars += part_chars + len(str(getattr(part, "args", "") or ""))
            tokens += part_tokens
    for tool in parameters.function_tools:
        schema = json.dumps(tool.parameters_json_schema, ensure_ascii=False)
        chars += len(schema) + len(tool.description or "") + len(tool.name)
    output = (settings or {}).get("max_tokens") or DEFAULT_OUTPUT
    return tokens + math.ceil(chars / CHARS_PER_TOKEN) + output
