"""Native stop-search algorithm over clocked fake motor I/O; never constructs CAN."""

from types import SimpleNamespace

import numpy as np
import pytest

from i2rt.robots import utils


class SearchChain:
    def __init__(self, *, bounded: bool = True, stuck: bool = False) -> None:
        self.motor_list = [(7, "DM4310")]
        self.motor_direction = [-1]
        self.now = 100.0
        self.pos = 0.0
        self.torque = 0.0
        self.commands = []
        self.started = 0
        self.bounded = bounded
        self.stuck = stuck

    def read_states(self) -> list:
        return [SimpleNamespace(pos=self.pos, eff=0)]

    def set_commands(self, *, torques: np.ndarray, valid_until: float) -> None:
        assert valid_until > self.now
        self.torque = torques[0]
        self.commands.append((self.now, self.torque, valid_until))

    def start_thread(self) -> None:
        assert self.commands and self.commands[-1][2] > self.now
        self.started += 1

    def sleep(self, duration: float) -> None:
        assert duration >= 0
        self.now += duration
        if not self.stuck:
            self.pos += duration * np.sign(self.torque) * 5
            if self.bounded:
                self.pos = float(np.clip(self.pos, -0.5, 0.5))


@pytest.mark.parametrize("direction", (-1, 1))
def test_native_search_preserves_endpoint_order_and_renews_only_after_supervision(
    monkeypatch: pytest.MonkeyPatch, direction: int
) -> None:
    chain = SearchChain()
    chain.motor_direction = [direction]
    monkeypatch.setattr(utils, "time", SimpleNamespace(monotonic=lambda: chain.now, sleep=chain.sleep))
    callbacks = []
    result = utils.detect_gripper_limits(
        chain,
        0,
        command_validity_s=0.03,
        progress_callback=lambda: callbacks.append(chain.now),
        require_confirmed_stops=True,
    )
    assert result == ((-0.5, 0.5) if direction < 0 else (0.5, -0.5))
    assert chain.started == 1 and len(callbacks) == len(chain.commands)
    assert np.max(np.diff(callbacks)) <= 0.010000001
    assert all(expiry - stamp == pytest.approx(0.03) for stamp, _, expiry in chain.commands)
    assert chain.commands[-1][1] == 0


@pytest.mark.parametrize(("bounded", "stuck", "match"), ((False, False, "timed out"), (True, True, "distinct")))
def test_guarded_search_rejects_timeout_and_indistinguishable_endpoints(
    monkeypatch: pytest.MonkeyPatch, bounded: bool, stuck: bool, match: str
) -> None:
    chain = SearchChain(bounded=bounded, stuck=stuck)
    monkeypatch.setattr(utils, "time", SimpleNamespace(monotonic=lambda: chain.now, sleep=chain.sleep))
    with pytest.raises(RuntimeError, match=match):
        utils.detect_gripper_limits(chain, 0, command_validity_s=0.03, require_confirmed_stops=True)


def test_cancellation_does_not_renew_the_last_command(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = SearchChain()
    monkeypatch.setattr(utils, "time", SimpleNamespace(monotonic=lambda: chain.now, sleep=chain.sleep))

    def cancel() -> None:
        if len(chain.commands) == 3:
            raise PermissionError("cancelled")

    with pytest.raises(PermissionError, match="cancelled"):
        utils.detect_gripper_limits(chain, 0, command_validity_s=0.03, progress_callback=cancel)
    assert len(chain.commands) == 3
    assert chain.commands[-1][2] < chain.now + 0.03
