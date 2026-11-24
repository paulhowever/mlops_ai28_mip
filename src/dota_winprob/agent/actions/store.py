import secrets
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from opentelemetry.trace import SpanContext

from dota_winprob.agent.actions.models import (
    ActionResult,
    ActionStatus,
    ActionView,
    RequeuePlan,
)
from dota_winprob.agent.sources.dead_letters import Key
from dota_winprob.agent.tracing import trace_id_of

ID_BYTES = 24


class ActionConflict(Exception):
    def __init__(self, status: ActionStatus) -> None:
        super().__init__(status.value)
        self.status = status


class ActionUnknown(Exception):
    pass


class ActionsFull(Exception):
    pass


@dataclass
class Action:
    kind: str
    plan: RequeuePlan
    keys: Counter[Key]
    deadline: float
    created_at: datetime
    expires_at: datetime
    origin: SpanContext | None
    id: str = field(default_factory=lambda: secrets.token_urlsafe(ID_BYTES))
    status: ActionStatus = ActionStatus.proposed
    decided_at: datetime | None = None
    result: ActionResult | None = None
    execution_trace_id: str | None = None

    def view(self) -> ActionView:
        return ActionView(
            id=self.id,
            kind=self.kind,
            status=self.status,
            plan=self.plan,
            created_at=self.created_at,
            expires_at=self.expires_at,
            decided_at=self.decided_at,
            result=self.result,
            trace_id=trace_id_of(self.origin),
            execution_trace_id=self.execution_trace_id,
        )


class ActionStore:
    def __init__(
        self,
        *,
        ttl: float,
        capacity: int,
        clock: Callable[[], float] = time.monotonic,
        on_expired: Callable[[Action], None] = lambda action: None,
    ) -> None:
        self._ttl = ttl
        self._capacity = capacity
        self._clock = clock
        self._on_expired = on_expired
        self._actions: OrderedDict[str, Action] = OrderedDict()

    def propose(
        self, kind: str, plan: RequeuePlan, keys: Counter[Key], origin: SpanContext | None
    ) -> Action:
        self._expire()
        self._make_room()
        now = datetime.now(UTC)
        action = Action(
            kind=kind,
            plan=plan,
            keys=Counter(keys),
            deadline=self._clock() + self._ttl,
            created_at=now,
            expires_at=now + timedelta(seconds=self._ttl),
            origin=origin,
        )
        self._actions[action.id] = action
        return action

    def decide(self, action_id: str, status: ActionStatus) -> Action:
        self._expire()
        action = self._actions.get(action_id)
        if action is None:
            raise ActionUnknown
        if action.status is not ActionStatus.proposed:
            raise ActionConflict(action.status)
        action.status = status
        action.decided_at = datetime.now(UTC)
        return action

    def all(self) -> list[Action]:
        self._expire()
        return list(reversed(self._actions.values()))

    def _expire(self) -> None:
        now = self._clock()
        for action in self._actions.values():
            if action.status is ActionStatus.proposed and now >= action.deadline:
                action.status = ActionStatus.expired
                self._on_expired(action)

    def _make_room(self) -> None:
        if len(self._actions) < self._capacity:
            return
        for wanted in (False, True):
            for action_id, action in self._actions.items():
                waiting = action.status is ActionStatus.proposed
                if action.status is not ActionStatus.executing and waiting is wanted:
                    del self._actions[action_id]
                    return
        raise ActionsFull
