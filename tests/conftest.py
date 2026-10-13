from datetime import date

import pytest

from cs_agent.session import new_session


class ConfirmRecorder:
    """Stands in for the customer answering the y/N confirmation prompt."""

    def __init__(self, answer=True):
        self.answer = answer
        self.prompts = []

    def __call__(self, text):
        self.prompts.append(text)
        return self.answer


@pytest.fixture
def confirm():
    return ConfirmRecorder(True)


@pytest.fixture
def make_session(tmp_path, confirm):
    def _make(customer_id="C001", confirm_fn=None):
        return new_session(customer_id, memory_root=tmp_path / "mem",
                           confirm_fn=confirm_fn or confirm, today=date(2026, 10, 13))
    return _make
