"""Actual driver methods over fake bus I/O: no physical enable or CAN import side effects."""

import inspect
import threading
import time
from types import SimpleNamespace
from typing import Any

import can
import numpy as np
import pytest

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, DMSingleMotorCanInterface
from i2rt.motor_drivers.utils import FeedbackFrameInfo, ReceiveMode


def fake_chain() -> DMChainCanInterface:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_list = [(1, "DM4310"), (2, "DM4310")]
    chain.motor_offset = np.array([0.2, -0.3])
    chain.motor_direction = np.array([1, -1])
    chain.absolute_positions = np.array([0.5, 0.8])
    chain.state_lock = threading.Lock()
    chain.command_lock = threading.RLock()
    chain.transport_lock = threading.Lock()
    chain._guarded_startup = False
    chain.start_thread_flag = False
    chain._sender_fault = None
    chain._command_id = 3
    chain._feedback_command_id = 2
    chain._sweep_id = 4
    chain._receive_sequence = 8
    chain._sweep_started_monotonic = 10.0
    chain._sweep_completed_monotonic = 10.01
    chain._command_valid_until = None
    chain.running = True
    chain.channel = "fake"
    chain.state = [
        FeedbackFrameInfo(i + 1, "0x1", "normal", 0.5, 0.1, 0.2, 30, 31, 10 + 0.001 * i, 20 + 0.001 * i, i + 7)
        for i in range(2)
    ]
    return chain


def test_guarded_expiry_is_named_without_changing_legacy_positional_commands() -> None:
    parameters = inspect.signature(DMChainCanInterface.set_commands).parameters
    assert parameters["valid_until"].kind == inspect.Parameter.KEYWORD_ONLY
    for name in ("torques", "pos", "vel", "kp", "kd", "get_state"):
        assert parameters[name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_cached_feedback_preserves_actual_receipt_times_and_axis_signs() -> None:
    chain = fake_chain()
    first = chain.read_states()
    time.sleep(0.005)
    second = chain.read_states()
    assert first == second
    assert first[0].timestamp == 20
    assert first[1].received_monotonic == 10.001
    assert first[1].receive_sequence == 8
    assert first[1].sweep_id == 4 and first[1].command_id == 2
    np.testing.assert_allclose([m.pos for m in first], [0.3, -1.1])
    np.testing.assert_allclose([m.vel for m in first], [0.1, -0.1])
    np.testing.assert_allclose([m.eff for m in first], [0.2, -0.2])


def test_native_command_expiry_is_atomic_and_stops_actual_sender() -> None:
    chain = fake_chain()
    calls = []

    def control(**kwargs: Any) -> FeedbackFrameInfo:
        calls.append(kwargs)
        return chain.state[len(calls) - 1]

    chain.motor_interface = SimpleNamespace(set_control=control)
    chain.set_commands(
        np.array([1.0, 2.0]), pos=np.array([0.1, 0.2]), vel=np.array([0.3, 0.4]), valid_until=time.monotonic() + 1
    )
    chain._set_commands(chain.commands)
    assert calls[0]["pos"] == pytest.approx(0.3)
    assert calls[1]["pos"] == pytest.approx(-0.5)
    assert calls[1]["vel"] == pytest.approx(-0.4)
    assert calls[1]["torque"] == -2
    chain._command_valid_until = time.monotonic() - 0.001
    with pytest.raises(RuntimeError, match="expired"):
        chain._set_commands(chain.commands)
    assert len(calls) == 2


def test_expired_native_producer_cannot_revive_its_command_lease() -> None:
    chain = fake_chain()
    chain.set_commands(np.zeros(2), valid_until=time.monotonic() + 1)
    previous_id = chain._command_id
    chain._command_valid_until = time.monotonic() - 0.01
    with pytest.raises(RuntimeError, match="expired native producer"):
        chain.set_commands(np.ones(2), valid_until=time.monotonic() + 1)
    assert not chain.running
    assert chain._command_id == previous_id


def test_disable_reports_partial_failure_and_does_not_call_enable() -> None:
    chain = fake_chain()
    calls = []

    def off(motor_id: int, motor_type: str) -> SimpleNamespace:
        calls.append((motor_id, motor_type))
        if motor_id == 2:
            raise TimeoutError("no response")
        return SimpleNamespace(error_code="0x0")

    chain.motor_interface = SimpleNamespace(motor_off=off)
    result = chain.disable_motors()
    assert not chain.running
    assert len(calls) == 2
    assert result[0]["confirmed"]
    assert result[1]["attempted"] and not result[1]["confirmed"]
    assert "no response" in result[1]["error"]


def test_motor_off_uses_native_packet_and_returns_status_without_fault_clear() -> None:
    driver = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    driver.cmd_idoffset = 0
    driver.receive_mode = ReceiveMode.p16
    driver.name = "fake"
    driver.bus = SimpleNamespace(channel_info="fake")
    calls = []

    def exchange(*args: Any, **kwargs: Any) -> can.Message:
        calls.append((args, kwargs))
        return can.Message(arbitration_id=0x11, data=[0x01, 0x80, 0, 0x80, 0, 0, 25, 26])

    driver._send_message_get_response = exchange
    result = driver.motor_off(1)
    assert calls == [((1, 1, [0xFF] * 7 + [0xFD]), {"bypass_command_guard": True})]
    assert result.error_code == "0x0"
    assert result.received_monotonic > 0


def test_guarded_motor_enable_never_implicitly_clears_faults() -> None:
    driver = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    driver._send_message_get_response = lambda *args: object()
    driver.parse_recv_message = lambda *args, **kwargs: SimpleNamespace(error_code="0xb", error_message="hot")
    driver.clean_error = lambda *args, **kwargs: pytest.fail("guarded enable must not clear motor faults")
    with pytest.raises(RuntimeError, match="no fault clear"):
        driver.motor_on(1, "DM4310", allow_error_recovery=False)


def test_staged_feedback_is_bounded_explicit_and_cannot_overwrite_installed_command() -> None:
    chain = fake_chain()
    chain._guarded_startup = True
    calls = []

    def control(**kwargs: Any) -> FeedbackFrameInfo:
        calls.append(kwargs)
        previous = chain.state[kwargs["motor_id"] - 1]
        previous.received_monotonic = time.monotonic()
        return previous

    chain.motor_interface = SimpleNamespace(set_control=control)
    with pytest.raises(RuntimeError, match="installed"):
        chain.start_thread()
    old = chain.read_states()
    fresh = chain.acquire_startup_feedback()
    assert len(calls) == 2 and not chain.start_thread_flag
    assert all(c["kp"] == c["kd"] == c["torque"] == 0 for c in calls)
    assert fresh[0].receive_sequence > old[0].receive_sequence
    assert fresh[0].sweep_id == old[0].sweep_id + 1
    chain.set_commands(np.zeros(2), valid_until=time.monotonic() + 1)
    with pytest.raises(RuntimeError, match="no longer permitted"):
        chain.acquire_startup_feedback()


def test_urgent_stop_prevents_next_transaction_without_waiting_for_publication_lock() -> None:
    chain = fake_chain()
    chain._guarded_startup = True
    calls = []

    def control(**kwargs: Any) -> FeedbackFrameInfo:
        calls.append(kwargs)
        chain.running = False
        return chain.state[0]

    chain.motor_interface = SimpleNamespace(set_control=control)
    chain.set_commands(np.zeros(2), valid_until=time.monotonic() + 1)
    with pytest.raises(RuntimeError, match="before next motor"):
        chain._set_commands(chain.commands)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="stopped guarded"):
        chain.set_commands(np.zeros(2), valid_until=time.monotonic() + 1)


def test_new_publication_cannot_extend_old_sweep_original_expiry() -> None:
    chain = fake_chain()
    calls = []

    def control(**kwargs: Any) -> FeedbackFrameInfo:
        calls.append(kwargs)
        chain.set_commands(np.zeros(2), valid_until=time.monotonic() + 1)
        time.sleep(0.005)
        return chain.state[0]

    chain.motor_interface = SimpleNamespace(set_control=control)
    chain.set_commands(np.zeros(2), valid_until=time.monotonic() + 1)
    with pytest.raises(RuntimeError, match="expired during"):
        chain._set_commands(chain.commands, valid_until=time.monotonic() + 0.002)
    assert len(calls) == 1


def test_guarded_can_bounds_send_and_retry_and_allows_explicit_disable() -> None:
    interface = CanInterface.__new__(CanInterface)
    interface.transaction_timeout_s = 0.01
    interface.command_deadline = time.monotonic() - 1
    interface.command_permitted = lambda: False
    interface.receive_mode = ReceiveMode.p16
    calls = []
    interface.bus = SimpleNamespace(send=lambda *args, **kwargs: calls.append(kwargs))
    interface._receive_message = lambda *args, **kwargs: can.Message(arbitration_id=0x11, data=[0] * 8)
    with pytest.raises(RuntimeError, match="authority stopped"):
        interface._send_message_get_response(1, 1, [0] * 8)
    assert not calls
    interface._send_message_get_response(1, 1, [0xFF] * 7 + [0xFD], bypass_command_guard=True)
    assert len(calls) == 1 and 0 < calls[0]["timeout"] <= 0.01

    interface.command_permitted = lambda: True
    with pytest.raises(TimeoutError, match="deadline"):
        interface._send_message_get_response(1, 1, [0] * 8)
    assert len(calls) == 1
