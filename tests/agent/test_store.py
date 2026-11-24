import pytest

from dota_winprob.agent.actions.models import REQUEUE_DLQ, ActionStatus, RequeuePlan
from dota_winprob.agent.actions.store import (
    ActionConflict,
    ActionsFull,
    ActionStore,
    ActionUnknown,
)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


PLAN = RequeuePlan(
    queue="tasks.dlq", target="tasks.score", depth=1, scanned=1, valid=1, garbage=0, tasks=[]
)


def make_store(clock, **overrides):
    options = {"ttl": 300, "capacity": 3, "clock": clock}
    return ActionStore(**(options | overrides))


def propose(store):
    return store.propose(REQUEUE_DLQ, PLAN, {}, None)


def test_ids_are_long_and_unpredictable():
    store = make_store(Clock(), capacity=100)

    ids = {propose(store).id for _ in range(50)}

    assert len(ids) == 50
    assert min(len(action_id) for action_id in ids) >= 32


def test_decision_is_taken_once():
    store = make_store(Clock())
    action = propose(store)

    store.decide(action.id, ActionStatus.executing)

    for status in (ActionStatus.executing, ActionStatus.rejected):
        with pytest.raises(ActionConflict):
            store.decide(action.id, status)
    assert action.status is ActionStatus.executing


def test_proposal_expires_exactly_once():
    clock, expired = Clock(), []
    store = make_store(clock, on_expired=expired.append)
    action = propose(store)
    clock.now += 299
    assert store.all()[0].status is ActionStatus.proposed
    clock.now += 1

    with pytest.raises(ActionConflict) as conflict:
        store.decide(action.id, ActionStatus.executing)

    store.all()
    assert conflict.value.status is ActionStatus.expired
    assert expired == [action]


def test_unknown_id():
    with pytest.raises(ActionUnknown):
        make_store(Clock()).decide("nope", ActionStatus.executing)


def test_capacity_drops_finished_first_and_never_running():
    store = make_store(Clock())
    first, second, third = (propose(store) for _ in range(3))
    store.decide(first.id, ActionStatus.executing)
    store.decide(third.id, ActionStatus.rejected)

    fourth = propose(store)
    fifth = propose(store)

    kept = [action.id for action in store.all()]
    assert kept == [fifth.id, fourth.id, first.id]
    assert second.id not in kept


def test_store_full_of_running_actions_refuses():
    store = make_store(Clock(), capacity=1)
    store.decide(propose(store).id, ActionStatus.executing)

    with pytest.raises(ActionsFull):
        propose(store)
