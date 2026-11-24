import base64
import binascii
import struct
import zlib
from dataclasses import dataclass

MAX_BYTES = 4 * 1024 * 1024
MAX_SIDE = 8192
MAX_PIXELS = 12_000_000
MAX_BASE64 = MAX_BYTES * 4 // 3 + 4

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
PNG_KEPT = frozenset({b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND", b"gAMA", b"sRGB", b"pHYs"})
PNG_DEPTHS = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
JPEG_FRAMES = frozenset({0xC0, 0xC1, 0xC2})
JPEG_UNSUPPORTED = frozenset({0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
JPEG_STANDALONE = frozenset({0x01, *range(0xD0, 0xD8)})
JPEG_DROPPED = frozenset({0xFE, *range(0xE1, 0xEE), 0xEF})
SOS, EOI = 0xDA, 0xD9


class ImageError(ValueError):
    pass


@dataclass(frozen=True)
class Image:
    data: bytes
    media_type: str
    width: int
    height: int


def decode(encoded: str) -> Image:
    if len(encoded) > MAX_BASE64:
        raise ImageError("картинка больше 4 МБ")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ImageError("картинка не в base64") from error
    return inspect(data)


def inspect(data: bytes) -> Image:
    if len(data) > MAX_BYTES:
        raise ImageError("картинка больше 4 МБ")
    if data.startswith(PNG_SIGNATURE):
        return _png(data)
    if data.startswith(JPEG_SIGNATURE):
        return _jpeg(data)
    raise ImageError("нужна картинка PNG или JPEG")


def _checked(width: int, height: int) -> tuple[int, int]:
    if not width or not height:
        raise ImageError("у картинки нулевой размер")
    if max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
        raise ImageError(
            f"картинка {width}x{height} слишком велика: до {MAX_SIDE} по стороне "
            f"и до {MAX_PIXELS // 1_000_000} мегапикселей"
        )
    return width, height


def _png(data: bytes) -> Image:
    size, kept, position = None, [PNG_SIGNATURE], len(PNG_SIGNATURE)
    pixels = False
    while position + 12 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, position)
        end = position + 12 + length
        if end > len(data):
            break
        chunk = data[position:end]
        if zlib.crc32(chunk[4:-4]) != struct.unpack(">I", chunk[-4:])[0]:
            raise ImageError("PNG повреждён")
        if size is None:
            size = _png_header(kind, chunk[8:-4])
        elif kind == b"acTL":
            raise ImageError("анимированный PNG не поддерживается")
        pixels = pixels or kind == b"IDAT"
        if kind in PNG_KEPT:
            kept.append(chunk)
        if kind == b"IEND":
            if not pixels:
                break
            return Image(b"".join(kept), "image/png", *size)
        position = end
    raise ImageError("PNG обрезан")


def _png_header(kind: bytes, body: bytes) -> tuple[int, int]:
    if kind != b"IHDR" or len(body) != 13:
        raise ImageError("PNG без заголовка")
    width, height, depth, colour = struct.unpack_from(">IIBB", body)
    if depth not in PNG_DEPTHS.get(colour, ()):
        raise ImageError("PNG с недопустимым форматом пикселей")
    return _checked(width, height)


def _jpeg(data: bytes) -> Image:
    size, kept, position = None, [data[:2]], 2
    while position + 4 <= len(data) and data[position] == 0xFF:
        marker = data[position + 1]
        if marker == 0xFF:
            position += 1
            continue
        if marker in JPEG_STANDALONE:
            position += 2
            continue
        length = struct.unpack_from(">H", data, position + 2)[0]
        end = position + 2 + length
        if length < 2 or end > len(data) or marker == EOI:
            break
        if marker in JPEG_UNSUPPORTED:
            raise ImageError("этот вид JPEG не поддерживается")
        if marker in JPEG_FRAMES:
            if length < 8 or size is not None:
                raise ImageError("JPEG повреждён")
            height, width = struct.unpack_from(">HH", data, position + 5)
            size = _checked(width, height)
        if marker == SOS:
            stop = data.rfind(b"\xff\xd9", end)
            if size is None or stop < 0:
                break
            kept.append(data[position : stop + 2])
            return Image(b"".join(kept), "image/jpeg", *size)
        if marker not in JPEG_DROPPED:
            kept.append(data[position:end])
        position = end
    raise ImageError("JPEG повреждён")
