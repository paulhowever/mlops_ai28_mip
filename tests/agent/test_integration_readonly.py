import asyncpg
import pytest
from testcontainers.community.postgres import PostgresContainer

from dota_winprob.agent.errors import SourceUnavailable
from dota_winprob.agent.sources import tasks_db
from dota_winprob.clients import tasks_repo
from tests.fakes import task_message
from tests.test_integration_pipeline import POSTGRES_IMAGE, docker_available

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database_url():
    if not docker_available():
        pytest.skip("Docker недоступен")
    with PostgresContainer(POSTGRES_IMAGE, driver=None) as container:
        yield container.get_connection_url()


@pytest.fixture
async def pools(agent_settings, database_url):
    writer = await asyncpg.create_pool(database_url, min_size=0, max_size=1)
    reader = await tasks_db.create_pool(
        agent_settings.model_copy(update={"database_url": database_url})
    )
    yield writer, reader
    await reader.close()
    await writer.close()


async def test_missing_table_is_unavailable(pools):
    writer, reader = pools
    await writer.execute("DROP TABLE IF EXISTS tasks")

    with pytest.raises(SourceUnavailable, match="UndefinedTableError"):
        await tasks_db.TasksReader(reader).task("batch:1")


async def test_agent_reads_rows_but_cannot_write(pools):
    writer, reader = pools
    await tasks_repo.ensure_schema(writer)
    repository = tasks_repo.TaskRepository(writer)
    await repository.save_result(task_message("batch:1"), 0.62, "4", 1, "worker-1")
    await repository.mark_failed("batch:2", task_message().source, "OSError: disk", 3, "worker-1")
    tasks = tasks_db.TasksReader(reader)

    row = await tasks.task("batch:1")
    assert (row["status"], row["win_probability"], row["worker"]) == ("done", 0.62, "worker-1")
    assert await tasks.task("batch:404") is None
    assert [failed["task_id"] for failed in await tasks.failed(10)] == ["batch:2"]
    assert await tasks.totals() == {"done": 1, "failed": 1}
    await tasks.check()

    for statement in (
        "DELETE FROM tasks",
        "UPDATE tasks SET status = 'pending'",
        "DROP TABLE tasks",
    ):
        with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
            await reader.execute(statement)
        with pytest.raises(SourceUnavailable, match="ReadOnlySQLTransactionError"):
            await tasks._fetch(statement)
    assert await tasks.totals() == {"done": 1, "failed": 1}


async def test_injection_is_only_a_value(pools):
    writer, reader = pools
    await tasks_repo.ensure_schema(writer)
    repository = tasks_repo.TaskRepository(writer)
    await repository.save_result(task_message("batch:9"), 0.5, "4", 1, "worker-1")
    tasks = tasks_db.TasksReader(reader)

    assert await tasks.task("x' OR '1'='1") is None
    assert await tasks.task("batch:9") is not None
