#!/usr/bin/env python3
"""Tests for the shared worker audio conditioning (EPIC 3.1)."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from marceline_worker.audio import resample, to_mono  # noqa: E402


class ToMonoTest(unittest.TestCase):
    def test_passes_mono_through_untouched(self) -> None:
        pcm = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        self.assertIs(to_mono(pcm, 1), pcm)

    def test_averages_interleaved_channels(self) -> None:
        # Frames: (1.0, 0.0), (0.5, -0.5), (-1.0, 1.0)
        pcm = np.array([1.0, 0.0, 0.5, -0.5, -1.0, 1.0], dtype=np.float32)
        got = to_mono(pcm, 2)
        self.assertTrue(np.allclose(got, [0.5, 0.0, 0.0]))

    def test_rejects_partial_frames(self) -> None:
        pcm = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        with self.assertRaises(ValueError):
            to_mono(pcm, 2)

    def test_rejects_nonpositive_channel_count(self) -> None:
        with self.assertRaises(ValueError):
            to_mono(np.zeros(4, dtype=np.float32), 0)


class ResampleTest(unittest.TestCase):
    def test_is_a_noop_at_the_same_rate(self) -> None:
        pcm = np.array([0.1, 0.2], dtype=np.float32)
        self.assertIs(resample(pcm, 16_000, 16_000), pcm)

    def test_downsamples_to_the_expected_length(self) -> None:
        pcm = np.zeros(48_000, dtype=np.float32)
        self.assertEqual(resample(pcm, 48_000, 16_000).size, 16_000)

    def test_upsamples_to_the_expected_length(self) -> None:
        pcm = np.zeros(8_000, dtype=np.float32)
        self.assertEqual(resample(pcm, 8_000, 16_000).size, 16_000)

    def test_preserves_a_constant_signal(self) -> None:
        """Linear interpolation of a DC signal must not ripple."""
        pcm = np.full(4_800, 0.25, dtype=np.float32)
        got = resample(pcm, 48_000, 16_000)
        self.assertTrue(np.allclose(got, 0.25))

    def test_preserves_a_sine_within_interpolation_error(self) -> None:
        """A 220 Hz tone survives 48k -> 16k with modest error."""
        t = np.arange(48_000, dtype=np.float64) / 48_000
        pcm = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        got = resample(pcm, 48_000, 16_000)

        t_out = np.arange(got.size, dtype=np.float64) / 16_000
        want = np.sin(2 * np.pi * 220 * t_out)
        self.assertLess(float(np.max(np.abs(got - want))), 0.02)

    def test_returns_empty_for_empty_input(self) -> None:
        self.assertEqual(resample(np.zeros(0, dtype=np.float32), 48_000, 16_000).size, 0)

    def test_rejects_nonpositive_rates(self) -> None:
        with self.assertRaises(ValueError):
            resample(np.zeros(4, dtype=np.float32), 0, 16_000)


if __name__ == "__main__":
    unittest.main()
