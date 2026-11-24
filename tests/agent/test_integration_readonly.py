from pathlib import Path

import asyncpg
import pytest
from testcontainers.community.postgres import PostgresContainer

from dota_winprob.agent.errors import SourceUnavailable
from dota_winprob.agent.sources import tasks_db
from dota_winprob.clients import tasks_repo
from tests.fakes import task_message
from tests.test_integration_pipeline import POSTGRES_IMAGE, docker_available

pytestmark = pytest.mark.integration


ROLE_SQL = Path(__file__).parents[2] / "docker" / "agent-role.sql"
ROLE_PASSWORD = "pa'ss$1"


@pytest.fixture(scope="module")
def container():
    if not docker_available():
        pytest.skip("Docker недоступен")
    with PostgresContainer(POSTGRES_IMAGE, driver=None).with_volume_mapping(
        str(ROLE_SQL), "/agent-role.sql", "ro"
    ) as running:
        yield running


@pytest.fixture(scope="module")
def database_url(container):
    return container.get_connection_url()


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


async def test_role_cannot_write_even_with_read_only_mode_off(pools, container):
    writer, _ = pools
    await tasks_repo.ensure_schema(writer)
    apply = ["psql", "-q", "-U", container.username, "-d", container.dbname]
    apply += ["-v", "ON_ERROR_STOP=1", "-v", f"password={ROLE_PASSWORD}", "-f", "/agent-role.sql"]
    for _ in range(2):
        result = container.exec(apply)
        assert result.exit_code == 0, result.output
    agent = await asyncpg.connect(
        host=container.get_container_host_ip(),
        port=int(container.get_exposed_port(5432)),
        user="dota_agent",
        password=ROLE_PASSWORD,
        database=container.dbname,
    )
    try:
        with pytest.raises(asyncpg.ReadOnlySQLTransactionError):
            await agent.execute("DELETE FROM tasks")
        await agent.execute("SET default_transaction_read_only = off")

        assert await agent.fetchval("SELECT count(*) FROM tasks") >= 0
        for statement in (
            "INSERT INTO tasks (task_id) VALUES ('x')",
            "UPDATE tasks SET status = 'pending'",
            "DELETE FROM tasks",
            "TRUNCATE tasks",
            "DROP TABLE tasks",
            "CREATE TABLE public.other (id int)",
            "CREATE ROLE other",
            "SELECT passwd FROM pg_shadow",
            "SELECT pg_read_file('/etc/passwd')",
            "COPY tasks TO PROGRAM 'id'",
        ):
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await agent.execute(statement)
    finally:
        await agent.close()
