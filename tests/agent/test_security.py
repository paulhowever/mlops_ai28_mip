import pytest

from dota_winprob.agent import security
from tests.agent.conftest import BASE_URL
from tests.agent.fakes import call, say, script
from tests.agent.test_ask import ASK, ask

ALERTS = "/agent/alerts"
OWN = {"Origin": BASE_URL}


@pytest.fixture
async def client(make_agent_client):
    return await make_agent_client(script(call("requeue_dlq"), say("готово")))


@pytest.fixture
async def confirm_path(client):
    [action] = (await ask(client)).json()["actions"]
    return f"/agent/actions/{action['id']}/confirm"


@pytest.mark.parametrize(
    ("headers", "status", "reason"),
    [
        ({"Origin": "https://evil.example"}, 403, "origin"),
        ({"Origin": "null"}, 403, "origin"),
        ({"Origin": "http://127.0.0.1:3000"}, 403, "origin"),
        ({"Origin": "http://localhost:8010"}, 403, "origin"),
        ({"Origin": f"{BASE_URL}.evil.example"}, 403, "origin"),
        ({"Sec-Fetch-Site": "cross-site"}, 403, "origin"),
        ({"Sec-Fetch-Site": "same-site"}, 403, "origin"),
        ({"Host": "evil.example"}, 403, "host"),
        ({"Host": "evil.example:8010"}, 403, "host"),
        ({"Host": "agent:8010"}, 403, "host"),
        ({"Host": "127.0.0.1.evil.example"}, 403, "host"),
        ({"Host": ""}, 403, "host"),
        ({"Content-Type": "text/plain"}, 415, "content_type"),
        ({"Content-Type": "application/x-www-form-urlencoded"}, 415, "content_type"),
        ({"Content-Type": "multipart/form-data; boundary=application/json"}, 415, "content_type"),
    ],
)
async def test_foreign_page_cannot_confirm(
    client, confirm_path, dead_letters, headers, status, reason
):
    response = await client.post(
        confirm_path, content="{}", headers={"Content-Type": "application/json"} | headers
    )

    assert response.status_code == status
    assert response.json()["reason"] == reason
    assert dead_letters.requeued == []


async def test_missing_content_type_is_refused(client, confirm_path, dead_letters):
    response = await client.post(confirm_path)

    assert response.status_code == 415
    assert dead_letters.requeued == []


@pytest.mark.parametrize("headers", [{}, OWN, OWN | {"Sec-Fetch-Site": "same-origin"}])
async def test_own_page_and_curl_can_confirm(client, confirm_path, dead_letters, headers):
    response = await client.post(confirm_path, json={}, headers=headers)

    assert response.status_code == 200
    assert len(dead_letters.requeued) == 1


async def test_reading_is_protected_from_dns_rebinding(client, confirm_path):
    response = await client.get("/agent/actions", headers={"Host": "evil.example:8010"})

    assert response.status_code == 403
    assert "batch:1" not in response.text


async def test_get_does_not_confirm(client, confirm_path, dead_letters):
    assert (await client.get(confirm_path)).status_code == 405
    assert dead_letters.requeued == []


async def test_preflight_gives_no_permission(client, confirm_path):
    response = await client.options(
        confirm_path,
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )

    assert not [name for name in response.headers if name.startswith("access-control")]


@pytest.mark.parametrize(
    ("path", "limit"),
    [
        (ASK, security.MAX_ASK_BODY),
        (ALERTS, security.MAX_ALERTS_BODY),
        ("/agent/actions/x/reject", security.MAX_BODY),
    ],
)
async def test_body_size_is_bounded(client, path, limit):
    async def chunks():
        for _ in range(limit // 65536 + 2):
            yield b" " * 65536

    headers = {"Content-Type": "application/json"}
    declared = await client.post(path, content=b" " * (limit + 1), headers=headers)

    assert declared.status_code == 413
    if path != "/agent/actions/x/reject":
        streamed = await client.post(path, content=chunks(), headers=headers)
        assert streamed.status_code == 413


async def test_headers_protect_the_page(client):
    response = await client.get("/")

    policy = response.headers["content-security-policy"]
    assert response.status_code == 200
    assert "default-src 'none'" in policy
    assert "script-src 'self'" in policy
    assert "unsafe" not in policy
    assert "frame-ancestors 'none'" in policy
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


async def test_private_answers_are_not_cached(client):
    assert (await client.get("/agent/actions")).headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("host", "name"),
    [
        ("localhost:8010", "localhost"),
        ("LOCALHOST", "localhost"),
        ("[::1]:8010", "[::1]"),
        ("[::1]", "[::1]"),
        ("127.0.0.1", "127.0.0.1"),
    ],
)
def test_host_name(host, name):
    assert security.host_name(host) == name


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer s3cret-value", True),
        ("bearer s3cret-value", True),
        ("Bearer wrong", False),
        ("Basic s3cret-value", False),
        ("s3cret-value", False),
        ("", False),
        (None, False),
    ],
)
def test_token(header, expected):
    assert security.token_matches(header, "s3cret-value") is expected
