"""Judgment-layer circuit breaker: Jev first, an LLM judge while Jev is down."""
import pytest

from cs_agent.agent import CustomerServiceAgent, build_judge
from cs_agent.judge import GatePolicy, Judgment, JudgeError, MockJudge
from cs_agent.llm import MockLLM
from cs_agent.resilience import ResilientJudge


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Primary:
    """Fails while `failing` is set; each call takes `cost` seconds of the fake clock."""

    def __init__(self, clock, failing=False, cost=0.2):
        self.clock, self.failing, self.cost, self.calls = clock, failing, cost, 0

    def judge(self, state):
        self.calls += 1
        self.clock.now += self.cost
        if self.failing:
            raise JudgeError("Jev HTTP 529: overloaded")
        return Judgment("logistics_query", 0.99, {}, "jev:jev-1.13.0", needs_human=0.05)


class Backup:
    def __init__(self, failing=False):
        self.failing, self.calls = failing, 0

    def judge(self, state):
        self.calls += 1
        if self.failing:
            raise JudgeError("LLM down")
        return Judgment("logistics_query", 0.9, {}, "llm", needs_human=0.1)


STATE = {"latest_customer_message": "快递到哪了", "recent_conversation": []}


def make(failing=False, cost=0.2, backup=None):
    clock = Clock()
    primary = Primary(clock, failing, cost)
    judge = ResilientJudge(primary, backup if backup is not None else Backup(), failure_threshold=3,
                           slow_ms=3000, cooldown_s=60, clock=clock)
    return judge, primary, clock


def test_healthy_primary_is_used_directly():
    judge, primary, _ = make()
    j = judge.judge(STATE)
    assert j.backend.startswith("jev") and j.degraded is None
    assert judge.status()["state"] == "closed" and judge.status()["primary_calls"] == 1


def test_failures_fall_back_at_once_and_open_the_circuit_after_three():
    judge, primary, _ = make(failing=True)
    for _ in range(3):
        j = judge.judge(STATE)
        assert j.backend == "llm" and "529" in j.degraded       # this customer is still served
    assert judge.status()["state"] == "open"
    j = judge.judge(STATE)
    assert primary.calls == 3                                     # open: Jev is not even tried
    assert j.degraded == "熔断中"


def test_half_open_probe_closes_on_success():
    judge, primary, clock = make(failing=True)
    for _ in range(3):
        judge.judge(STATE)
    clock.now += 61
    primary.failing = False
    assert judge.judge(STATE).degraded is None                    # one probe goes to Jev ...
    assert judge.status()["state"] == "closed"                    # ... and it is healthy again


def test_half_open_probe_reopens_on_failure():
    judge, primary, clock = make(failing=True)
    for _ in range(3):
        judge.judge(STATE)
    clock.now += 61
    judge.judge(STATE)                                            # probe fails
    status = judge.status()
    assert status["state"] == "open" and status["retry_in_s"] == pytest.approx(60, abs=1)


def test_slow_answers_are_used_but_count_as_strikes():
    judge, primary, _ = make(cost=4.0)                            # 4 s > 3 s budget
    for _ in range(3):
        assert judge.judge(STATE).backend.startswith("jev")
    assert judge.status()["state"] == "open" and "慢" in judge.status()["last_error"]


def test_a_success_resets_the_count():
    judge, primary, _ = make(failing=True)
    judge.judge(STATE)
    judge.judge(STATE)
    primary.failing = False
    judge.judge(STATE)
    primary.failing = True
    judge.judge(STATE)
    assert judge.status()["state"] == "closed" and judge.status()["consecutive_failures"] == 1


def test_without_a_working_backup_the_error_propagates():
    judge, _, _ = make(failing=True, backup=Backup(failing=True))
    with pytest.raises(JudgeError):
        judge.judge(STATE)
    judge = ResilientJudge(Primary(Clock(), failing=True), None)
    with pytest.raises(JudgeError):
        judge.judge(STATE)


def test_agent_keeps_serving_when_jev_is_down(make_session):
    events = []
    s = make_session("C001")
    s.on_event = lambda t, d: events.append((t, d))
    judge, _, _ = make(failing=True, backup=MockJudge())
    result = CustomerServiceAgent(MockLLM(), s, judge=judge, policy=GatePolicy(0.5)).handle("A1002 的快递到哪了")
    assert result.route == "agent" and result.reason != "judge_error"   # not dumped on humans
    judgment = next(d for t, d in events if t == "judgment")
    assert "529" in judgment["degraded"]


def test_factory_wraps_jev_only_when_asked():
    assert isinstance(build_judge("jev", MockLLM(), mock=True, resilient=True), ResilientJudge)
    assert isinstance(build_judge("jev", MockLLM(), mock=True), MockJudge)   # experiment: pure Jev


def test_product_jev_retries_once_because_the_backup_is_waiting(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    product = build_judge("jev", MockLLM(), resilient=True)
    assert product.primary.max_retries == 1 and product.primary.client.timeout.read == 5.0
    assert build_judge("jev", MockLLM()).max_retries == 4              # experiment: keep every sample
