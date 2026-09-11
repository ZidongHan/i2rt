"""Finite, precomputed joint references consumed by native execution.

This module evaluates data; it does not plan, repair, clamp, or solve IK. The
supervisor owns admission of position/derivative limits and swept geometry.
Tuples keep references immutable and usable over a spawn-process connection.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JerkPiece:
    duration: float
    position: float
    velocity: float
    acceleration: float
    jerk: float

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(x) for x in (self.duration, self.position, self.velocity, self.acceleration, self.jerk)
            )
            or self.duration <= 0
        ):
            raise ValueError("reference piece must be finite with positive duration")

    def at(self, elapsed: float) -> tuple[float, float, float]:
        t = min(max(elapsed, 0.0), self.duration)
        return (
            self.position + t * (self.velocity + t * (self.acceleration / 2 + t * self.jerk / 6)),
            self.velocity + t * (self.acceleration + t * self.jerk / 2),
            self.acceleration + t * self.jerk,
        )


@dataclass(frozen=True)
class JointReference:
    """Per-coordinate exact constant-jerk polynomials, followed by fixed hold.

    All coordinates must finish at zero velocity and acceleration. A moving
    endpoint is never silently converted into a hold. Shorter coordinates may
    finish early (notably the independent gripper).
    """

    pieces: tuple[tuple[JerkPiece, ...], ...]
    stationary_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.pieces or len(self.pieces) != len(self.stationary_positions):
            raise ValueError("reference coordinate dimensions differ")
        if not all(math.isfinite(x) for x in self.stationary_positions):
            raise ValueError("reference positions must be finite")
        for axis, stationary in zip(self.pieces, self.stationary_positions, strict=True):
            previous = None
            for piece in axis:
                start = (piece.position, piece.velocity, piece.acceleration)
                if previous is not None and not np.allclose(previous, start, rtol=1e-8, atol=1e-8):
                    raise ValueError("reference contains a discontinuous polynomial boundary")
                previous = piece.at(piece.duration)
            if previous is not None:
                if not np.allclose(previous, (stationary, 0.0, 0.0), rtol=1e-8, atol=1e-8):
                    raise ValueError("reference must finish at its stationary position with zero derivatives")

    @property
    def duration(self) -> float:
        return max(sum(piece.duration for piece in axis) for axis in self.pieces)

    def at(self, elapsed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not math.isfinite(elapsed):
            raise ValueError("reference time must be finite")
        values = []
        for axis, stationary in zip(self.pieces, self.stationary_positions, strict=True):
            remaining = max(elapsed, 0.0)
            value = (stationary, 0.0, 0.0)
            for piece in axis:
                if remaining < piece.duration:
                    value = piece.at(remaining)
                    break
                remaining -= piece.duration
            values.append(value)
        return tuple(np.asarray(values, dtype=float).T)


@dataclass(frozen=True)
class AdmittedReference:
    """A bounded nominal horizon and its precomputed braking continuation.

    ``brake_at`` is an absolute monotonic deadline. Missing a replacement follows
    the brake without extending the nominal horizon. An execution owner admits
    the measured following corridor before publishing; native execution checks
    that corridor and feedback freshness again when taking over the brake.
    """

    sequence: int
    origin: float
    brake_at: float
    nominal: JointReference
    brake: JointReference
    following_error: tuple[float, ...]
    following_velocity_error: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            self.sequence < 0
            or not math.isfinite(self.origin)
            or not math.isfinite(self.brake_at)
            or self.brake_at < self.origin
        ):
            raise ValueError("invalid admitted reference sequence/time")
        n = len(self.nominal.pieces)
        if any(len(x) != n for x in (self.brake.pieces, self.following_error, self.following_velocity_error)):
            raise ValueError("admitted reference dimensions differ")
        if not all(math.isfinite(x) and x > 0 for x in (*self.following_error, *self.following_velocity_error)):
            raise ValueError("following corridor must contain positive finite limits")
        if not np.allclose(self.nominal.at(self.brake_at - self.origin), self.brake.at(0), atol=1e-8, rtol=1e-8):
            raise ValueError("fallback does not continue nominal position/velocity/acceleration")

    def at(self, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.nominal.at(now - self.origin) if now < self.brake_at else self.brake.at(now - self.brake_at)
