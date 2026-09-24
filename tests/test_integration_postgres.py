import os
import re

import pytest

from dota_winprob.clients import postgres

DATABASE_URL = os.environ.get("INTEGRATION_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="INTEGRATION_DATABASE_URL не задан"),
]


async def test_fetch_version_from_real_postgres(settings):
    pool = await postgres.create_pool(settings.model_copy(update={"database_url": DATABASE_URL}))
    try:
        version = await postgres.fetch_version(pool)
    finally:
        await postgres.close_pool(pool)

    assert re.fullmatch(r"\d+(\.\d+)+", version)
