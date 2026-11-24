import base64
import json
import struct
import zlib

import httpx
import httpx2
import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from dota_winprob.agent import app as agent_app
from dota_winprob.agent import images
from dota_winprob.agent.images import ImageError, decode, inspect
from dota_winprob.agent.llm.estimate import estimate_tokens
from tests.agent.conftest import BASE_URL
from tests.agent.fakes import say, script
from tests.agent.test_ask import ASK, recorded
from tests.agent.test_llm import completion


def chunk(kind, body=b""):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def png(width=4, height=2, extra=b"", colour=2, depth=8):
    header = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, 0))
    pixels = chunk(b"IDAT", zlib.compress(b"\x00" * 64))
    return images.PNG_SIGNATURE + header + extra + pixels + chunk(b"IEND")


def segment(marker, body):
    return bytes((0xFF, marker)) + struct.pack(">H", len(body) + 2) + body


def jpeg(width=4, height=2, extra=b"", tail=b""):
    frame = segment(0xC0, struct.pack(">BHHB", 8, height, width, 1) + b"\x01\x11\x00")
    scan = segment(0xDA, b"\x01\x01\x00\x00\x3f\x00") + b"\x12\xff\x00\x34"
    return b"\xff\xd8" + segment(0xE0, b"JFIF\x00") + extra + frame + scan + b"\xff\xd9" + tail


def encoded(data):
    return base64.b64encode(data).decode()


@pytest.mark.parametrize(("data", "kind"), [(png(), "image/png"), (jpeg(), "image/jpeg")])
def test_valid_image(data, kind):
    image = decode(encoded(data))

    assert (image.media_type, image.width, image.height) == (kind, 4, 2)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "PNG или JPEG"),
        (b"GIF89a", "PNG или JPEG"),
        (b"<svg onload=alert(1)>", "PNG или JPEG"),
        (b"RIFF\x00\x00\x00\x00WEBP", "PNG или JPEG"),
        (images.PNG_SIGNATURE, "обрезан"),
        (png()[:-12], "обрезан"),
        (png(50_000, 50_000), "слишком велика"),
        (png(8000, 8000), "слишком велика"),
        (png(0, 10), "нулевой"),
        (png(colour=5), "формат"),
        (png().replace(b"IHDR", b"IHDX"), "PNG повреждён"),
        (images.PNG_SIGNATURE + chunk(b"IDAT", b"x") + chunk(b"IEND"), "без заголовка"),
        (png(extra=chunk(b"acTL", b"\x00" * 8)), "анимированный"),
        (b"\xff\xd8\xff", "JPEG повреждён"),
        (jpeg(65_000, 65_000), "слишком велика"),
        (jpeg(0, 10), "нулевой"),
        (jpeg()[:-2], "JPEG повреждён"),
        (b"\xff\xd8" + segment(0xC3, b"\x08" * 8), "не поддерживается"),
        (b"\xff\xd8" + segment(0xDA, b"\x00" * 6) + b"\xff\xd9", "JPEG повреждён"),
        (b"\xff\xd8\xff\xe0\xff\xff", "JPEG повреждён"),
    ],
)
def test_bad_image(data, message):
    with pytest.raises(ImageError, match=message):
        inspect(data)


def test_oversized_and_undecodable_input():
    with pytest.raises(ImageError, match="4 МБ"):
        decode("A" * (images.MAX_BASE64 + 1))
    with pytest.raises(ImageError, match="4 МБ"):
        inspect(b"\x00" * (images.MAX_BYTES + 1))
    with pytest.raises(ImageError, match="base64"):
        decode("не base64")


def test_metadata_is_removed():
    marked = png(extra=chunk(b"tEXt", b"Comment\x00SECRET") + chunk(b"zTXt", b"SECRET"))
    photo = jpeg(
        extra=segment(0xE1, b"Exif SECRET") + segment(0xFE, b"SECRET"), tail=b"SECRET trailer"
    )

    for image in (inspect(marked), inspect(photo)):
        assert b"SECRET" not in image.data
        assert inspect(image.data) == image


async def test_image_reaches_the_gateway_as_data_uri(
    monkeypatch, settings, agent_settings, backends, pool
):
    sent = []

    def gateway(request):
        sent.append(json.loads(request.content))
        return completion("На графике очередь")

    async def fake_create_pool(_):
        return pool

    monkeypatch.setattr(agent_app.tasks_db, "create_pool", fake_create_pool)
    application = agent_app.create_app(
        agent_settings,
        settings,
        source_transport=httpx.MockTransport(backends),
        llm_transport=httpx2.MockTransport(gateway),
    )
    async with (
        agent_app.lifespan(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url=BASE_URL
        ) as client,
    ):
        response = await client.post(
            ASK, json={"question": "Что это?", "image": {"data": encoded(png())}}
        )

    assert response.status_code == 200
    content = sent[0]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "Что это?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{encoded(png())}"


@pytest.mark.parametrize(
    "image",
    [
        {"data": encoded(b"GIF89a")},
        {"data": "не base64"},
        {"data": ""},
        {"data": encoded(png(50_000, 50_000))},
        {"data": encoded(png()), "url": "http://example.com/x.png"},
        "строка",
    ],
)
async def test_bad_image_is_422_without_llm(make_agent_client, image):
    called = []

    def respond(messages, info):
        called.append(1)
        return say("ок")

    client = await make_agent_client(respond)
    response = await client.post(ASK, json={"question": "Что это?", "image": image})

    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]
    assert called == []


async def test_image_content_stays_out_of_traces_and_logs(make_agent_client, spans, capsys):
    client = await make_agent_client(script(say("ок")))
    data = encoded(png())

    await client.post(ASK, json={"question": "Что это?", "image": {"data": data}})

    text = recorded(spans) + capsys.readouterr().out
    assert data[:40] not in text
    assert "agent.question.image" in text


def test_image_tokens_depend_on_pixels():
    def estimate(*content):
        return estimate_tokens(
            [ModelRequest(parts=[UserPromptPart(list(content))])],
            {"max_tokens": 0},
            ModelRequestParameters(),
        )

    small = estimate("x", BinaryContent(png(100, 100), media_type="image/png"))
    large = estimate("x", BinaryContent(png(3000, 2000), media_type="image/png"))
    broken = estimate("x", BinaryContent(b"broken", media_type="image/png"))

    assert large - small > 7000
    assert broken > small
