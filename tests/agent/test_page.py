import re

from dota_winprob.agent.web.api import STATIC
from tests.agent.fakes import say, script

UNSAFE = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "createContextualFragment",
    "DOMParser",
    "eval(",
    "new Function",
    "srcdoc",
)


def source(name):
    return (STATIC / name).read_text(encoding="utf-8")


async def test_page_and_its_files_are_served(make_agent_client):
    client = await make_agent_client(script(say("ок")))

    page = await client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"] == "text/html; charset=utf-8"
    assert "Агент конвейера" in page.text
    linked = re.findall(r'(?:src|href)="([^"]+)"', page.text)
    assert len(linked) == 3
    for path in linked:
        response = await client.get(path)
        assert response.status_code == 200, path
        assert "content-security-policy" in response.headers


async def test_static_serves_nothing_else(make_agent_client):
    client = await make_agent_client(script(say("ок")))

    for path in ("/static/", "/static/../api.py", "/static/%2e%2e/api.py", "/static/.env"):
        assert (await client.get(path)).status_code == 404


def test_page_has_no_external_or_inline_code():
    page = source("index.html")

    assert not re.search(r"https?://", page)
    assert not re.search(r"\son\w+=|style=", page)
    assert re.findall(r"<script[^>]*>", page) == ['<script src="/static/app.js" defer>']
    assert "@import" not in source("style.css")
    assert "url(" not in source("style.css")


def test_script_never_parses_html():
    code = source("app.js")

    assert [name for name in UNSAFE if name in code] == []
    assert "createTextNode" in code


def test_script_posts_json_to_known_routes():
    code = source("app.js")

    assert set(re.findall(r'"(/agent/[a-z]+)"', code)) == {
        "/agent/ask",
        "/agent/actions",
        "/agent/incidents",
        "/agent/info",
    }
    assert '"Content-Type": "application/json"' in code
