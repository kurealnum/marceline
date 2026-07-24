#!/usr/bin/env python3
"""Audio conditioning for the STT worker (SPEC.md §2.4.1, EPIC 3.1).

Incoming `AudioChunk`s are self-describing: sample rate and channel count
travel with the data (invariant 2), so the worker never assumes a format
out of band. Whisper, on the other hand, wants exactly mono f32 at
16 kHz. This module is that conversion, kept free of gRPC and model
imports so it stays cheap to test.
"""

from __future__ import annotations

import numpy as np


def to_mono(pcm: np.ndarray, channels: int) -> np.ndarray:
    """Downmixes interleaved f32 PCM to a single channel by averaging.

    Args:
        pcm: Interleaved f32 samples, `channels`-major.
        channels: Channel count declared by the chunk.

    Returns:
        Mono f32 samples. Returned unchanged when `channels == 1`.

    Raises:
        ValueError: If `channels` is not positive, or the sample count is
            not a whole number of frames — either means the producer and
            the declared format disagree, which is a bug worth surfacing
            rather than silently truncating.
    """
    if channels < 1:
        raise ValueError(f"channel count must be >= 1, got {channels}")
    if channels == 1:
        return pcm
    if pcm.size % channels != 0:
        raise ValueError(
            f"{pcm.size} samples is not a whole number of {channels}-channel frames"
        )
    return pcm.reshape(-1, channels).mean(axis=1)


def resample(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resamples mono f32 audio by linear interpolation.

    Linear interpolation is deliberate rather than lazy: the capture path
    (EPIC 1) already delivers 16 kHz mono, so in the normal case this is
    the `from_rate == to_rate` no-op below. This branch exists only as a
    correctness backstop for an off-rate producer, where a slightly soft
    resample beats a chipmunk-voice transcript.

    Args:
        pcm: Mono f32 samples.
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Mono f32 samples at `to_rate`.

    Raises:
        ValueError: If either rate is not positive.
    """
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {from_rate}->{to_rate}")
    if from_rate == to_rate or pcm.size == 0:
        return pcm

    duration = pcm.size / from_rate
    out_len = int(round(duration * to_rate))
    if out_len == 0:
        return np.zeros(0, dtype=np.float32)

    # Map output sample centers onto the input timeline. `endpoint=False`
    # keeps the step exactly `from_rate/to_rate` so repeated calls do not
    # accumulate a timing drift.
    positions = np.linspace(0.0, pcm.size, num=out_len, endpoint=False, dtype=np.float64)
    return np.interp(positions, np.arange(pcm.size, dtype=np.float64), pcm).astype(
        np.float32
    )
