import logging
import time
from typing import List, Optional

import can

from i2rt.motor_drivers.utils import ReceiveMode


class CanInterface:
    def __init__(
        self,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "socketcan",
        bitrate: int = 1000000,
        name: str = "default_can_interface",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        use_buffered_reader: bool = False,
    ):
        self.channel = channel
        self.bus = can.interface.Bus(bustype=bustype, channel=channel, bitrate=bitrate)
        self.busstate = self.bus.state
        self.name = name
        self.receive_mode = receive_mode
        self.use_buffered_reader = use_buffered_reader
        logging.info(f"Can interface {self.name} use_buffered_reader: {use_buffered_reader}")
        if use_buffered_reader:
            # Initialize BufferedReader for asynchronous message handling
            self.buffered_reader = can.BufferedReader()
            self.notifier = can.Notifier(self.bus, [self.buffered_reader])

    def close(self) -> None:
        """Shut down the CAN bus."""
        if self.use_buffered_reader:
            self.notifier.stop()
        self.bus.shutdown()

    def _send_message_get_response(
        self,
        id: int,
        motor_id: int,
        data: List[int],
        max_retry: int = 5,
        expected_id: Optional[int] = None,
        *,
        bypass_command_guard: bool = False,
    ) -> can.Message:
        """Send a message over the CAN bus.

        Args:
            id (int): The arbitration ID of the message.
            data (List[int]): The data payload of the message.

        Returns:
            can.Message: The message that was sent.
        """
        message = can.Message(arbitration_id=id, data=data, is_extended_id=False)
        budget = getattr(self, "transaction_timeout_s", None)
        deadline = None if budget is None else time.monotonic() + budget
        original_expiry = None if bypass_command_guard else getattr(self, "command_deadline", None)
        if original_expiry is not None:
            deadline = original_expiry if deadline is None else min(deadline, original_expiry)
        for _ in range(max_retry):
            if not bypass_command_guard:
                permitted = getattr(self, "command_permitted", None)
                if permitted is not None and not permitted():
                    raise RuntimeError("guarded transport authority stopped")
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("guarded CAN transaction exceeded original deadline")
            try:
                # logging.info("Sending message: %s at %f", message, time.time())
                if remaining is None:
                    self.bus.send(message)
                else:
                    self.bus.send(message, timeout=remaining)
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("guarded CAN send exceeded original deadline")
                response = self._receive_message(motor_id, timeout=0.01 if remaining is None else min(0.01, remaining))
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("guarded CAN reply exceeded original deadline")
                # logging.info("Received response: %s at %f", response, time.time())

                if expected_id is None:
                    expected_id = self.receive_mode.get_receive_id(motor_id)
                if response and (expected_id == response.arbitration_id):
                    return response
                if deadline is None:
                    self.try_receive_message(id)
                elif time.monotonic() < deadline:
                    self.try_receive_message(id, timeout=min(0.009, deadline - time.monotonic()))
            except (can.CanError, AssertionError) as e:
                logging.warning(e)
                logging.warning(
                    "\033[91m"
                    + f"CAN Error {self.name}: Failed to communicate with motor {id} over can bus. Retrying..."
                    + "\033[0m"
                )
            time.sleep(0.001)
        raise AssertionError(
            f"fail to communicate with the motor {id} on {self.name} at can channel {self.bus.channel_info}"
        )

    def try_receive_message(self, motor_id: Optional[int] = None, timeout: float = 0.009) -> Optional[can.Message]:
        """Try to receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message, or None if no message is received.
        """
        try:
            return self._receive_message(motor_id, timeout, supress_warning=True)
        except AssertionError:
            return None

    def _drain_bus(self, timeout_s: float = 0.05, idle_count: int = 10) -> int:
        """Drain pending CAN frames until the bus is idle.

        Loops `try_receive_message(timeout=0.001)` until either `idle_count`
        consecutive 1 ms reads return None or `timeout_s` wall-clock has
        elapsed. Used at init handovers (e.g. between encoder validation and
        motor bring-up) to flush stale frames that would otherwise be misread
        as the next motor's reply. Returns the number of frames consumed.
        """
        drained = 0
        idle = 0
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and idle < idle_count:
            if self.try_receive_message(timeout=0.001) is None:
                idle += 1
            else:
                idle = 0
                drained += 1
        return drained

    def _receive_message(
        self, motor_id: Optional[int] = None, timeout: float = 0.009, supress_warning: bool = False
    ) -> Optional[can.Message]:
        """Receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message.

        Raises:
            AssertionError: If no message is received within the timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = min(0.001, max(0.0, deadline - time.monotonic()))
            if self.use_buffered_reader:
                message = self.buffered_reader.get_message(timeout=remaining)
            else:
                message = self.bus.recv(timeout=remaining)
            if message:
                return message
        if not supress_warning:
            logging.warning(
                "\033[91m"
                + f"Failed to receive message, {self.name} motor id {motor_id} motor timeout. Check if the motor is powered on or if the motor ID exists."
                + "\033[0m"
            )
