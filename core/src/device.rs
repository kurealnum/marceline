//! Compute device abstraction (SPEC.md §1.2, §9.8/§9.9, EPIC 0.7).
//!
//! v1 hard-requires CUDA (with CPU for the lightweight embedder), but the
//! seam exists from day one: config `device` strings parse into this enum,
//! and it's the only thing worker-launch call sites ever pass around.
//! Adding Metal/ROCm later is a one-place change here, not a call-site
//! refactor.

use std::fmt;
use std::str::FromStr;

use serde::Deserialize;

/// A compute device a worker (or the embedder) can run on.
///
/// v1 only supports [`Device::Cuda`] and [`Device::Cpu`]; unsupported
/// strings (e.g. `"metal"`, `"rocm"`) fail to parse with a clear error
/// rather than being silently accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// NVIDIA CUDA. The only GPU backend v1 supports.
    Cuda,
    /// CPU. Used today by the lightweight embedder (`[memory].embed_device`).
    Cpu,
}

/// Error returned when a config/CLI string doesn't name a supported device.
#[derive(Debug, thiserror::Error)]
#[error("unsupported device {found:?}; v1 supports: {}", Device::SUPPORTED.join(", "))]
pub struct UnsupportedDevice {
    found: String,
}

impl Device {
    /// Device names v1 accepts, in the same casing `as_str` produces.
    const SUPPORTED: &'static [&'static str] = &["cuda", "cpu"];

    /// Canonical lowercase name, as passed on the worker's `--device` flag.
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cuda => "cuda",
            Device::Cpu => "cpu",
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Device {
    type Err = UnsupportedDevice;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cuda" => Ok(Device::Cuda),
            "cpu" => Ok(Device::Cpu),
            _ => Err(UnsupportedDevice {
                found: s.to_string(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        raw.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_devices_case_insensitively() {
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda);
        assert_eq!("CUDA".parse::<Device>().unwrap(), Device::Cuda);
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
    }

    #[test]
    fn rejects_unsupported_device_with_clear_error() {
        let err = "metal".parse::<Device>().unwrap_err();
        assert!(err.to_string().contains("metal"));
        assert!(err.to_string().contains("cuda"));
    }
}
