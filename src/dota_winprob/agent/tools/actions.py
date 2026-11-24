from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field

from dota_winprob.agent.actions.models import MAX_REQUEUE, RequeuePlan
from dota_winprob.agent.actions.service import ActionService
from dota_winprob.agent.actions.store import ActionsFull
from dota_winprob.agent.envelope import ToolResult
from dota_winprob.agent.sources.dead_letters import NAME, MessageAtRisk

RequeueLimit = Annotated[
    int, Field(ge=1, le=MAX_REQUEUE, description="Сколько задач предложить вернуть.")
]
SHOWN_TASKS = 10


def _shown(plan: RequeuePlan) -> dict[str, Any]:
    return {
        "dlq_depth": plan.depth,
        "scanned": plan.scanned,
        "valid": plan.valid,
        "garbage": plan.garbage,
        "planned": len(plan.tasks),
        "first_tasks": [task.model_dump() for task in plan.tasks[:SHOWN_TASKS]],
        "notes": plan.notes,
    }


@dataclass(frozen=True)
class ActionTools:
    actions: ActionService

    async def requeue_dlq(self, limit: RequeueLimit = MAX_REQUEUE) -> ToolResult:
        try:
            plan, action = await self.actions.plan_requeue(limit)
        except MessageAtRisk:
            message = (
                "у сообщения в DLQ почти исчерпан счётчик аварийных возвратов, осмотр остановлен: "
                "нужен разбор человеком в RabbitMQ Management"
            )
            return ToolResult(
                NAME, {"proposal_created": False, "error": message}, message, "failed"
            )
        except ActionsFull:
            message = "слишком много незавершённых действий"
            return ToolResult(
                NAME, {"proposal_created": False, "error": message}, message, "failed"
            )
        data = {"proposal_created": action is not None, "executed": False, **_shown(plan)}
        if action is None:
            return ToolResult(
                NAME, data, f"возвращать нечего: в DLQ сообщений {plan.depth}, задач нет", "empty"
            )
        data["next_step"] = (
            "предложение показано человеку, без его подтверждения ничего не произойдёт"
        )
        summary = f"предложено вернуть {len(plan.tasks)} из {plan.depth}, мусора {plan.garbage}"
        return ToolResult(NAME, data, summary, "proposed", action.view())
