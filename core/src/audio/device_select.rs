//! Resolves a configured device name against `cpal`'s enumerated devices
//! (SPEC.md EPIC 1.3). An empty/absent name means "use the system
//! default"; an unrecognized name falls back to the default with a
//! warning rather than failing startup.

use cpal::traits::DeviceTrait;

/// Picks a device from `devices` whose name matches `wanted`, falling
/// back to `default` (with a warning) if `wanted` is empty/absent or
/// doesn't match any enumerated device.
pub fn resolve<D: DeviceTrait>(
    devices: impl Iterator<Item = D>,
    wanted: Option<&str>,
    kind: &str,
    default: impl FnOnce() -> Option<D>,
) -> Option<D> {
    if let Some(name) = wanted.filter(|s| !s.is_empty()) {
        for device in devices {
            if device.name().is_ok_and(|n| n == name) {
                return Some(device);
            }
        }
        tracing::warn!(
            requested = %name,
            kind,
            "configured device not found, falling back to default"
        );
    }
    default()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeDevice(&'static str);

    impl DeviceTrait for FakeDevice {
        type SupportedInputConfigs = std::iter::Empty<cpal::SupportedStreamConfigRange>;
        type SupportedOutputConfigs = std::iter::Empty<cpal::SupportedStreamConfigRange>;
        type Stream = cpal::Stream;

        fn name(&self) -> Result<String, cpal::DeviceNameError> {
            Ok(self.0.to_string())
        }
        fn supported_input_configs(
            &self,
        ) -> Result<Self::SupportedInputConfigs, cpal::SupportedStreamConfigsError> {
            Ok(std::iter::empty())
        }
        fn supported_output_configs(
            &self,
        ) -> Result<Self::SupportedOutputConfigs, cpal::SupportedStreamConfigsError> {
            Ok(std::iter::empty())
        }
        fn default_input_config(
            &self,
        ) -> Result<cpal::SupportedStreamConfig, cpal::DefaultStreamConfigError> {
            unimplemented!("not exercised by these tests")
        }
        fn default_output_config(
            &self,
        ) -> Result<cpal::SupportedStreamConfig, cpal::DefaultStreamConfigError> {
            unimplemented!("not exercised by these tests")
        }
        fn build_input_stream_raw<D2, E>(
            &self,
            _config: &cpal::StreamConfig,
            _sample_format: cpal::SampleFormat,
            _data_callback: D2,
            _error_callback: E,
            _timeout: Option<std::time::Duration>,
        ) -> Result<Self::Stream, cpal::BuildStreamError>
        where
            D2: FnMut(&cpal::Data, &cpal::InputCallbackInfo) + Send + 'static,
            E: FnMut(cpal::StreamError) + Send + 'static,
        {
            unimplemented!("not exercised by these tests")
        }
        fn build_output_stream_raw<D2, E>(
            &self,
            _config: &cpal::StreamConfig,
            _sample_format: cpal::SampleFormat,
            _data_callback: D2,
            _error_callback: E,
            _timeout: Option<std::time::Duration>,
        ) -> Result<Self::Stream, cpal::BuildStreamError>
        where
            D2: FnMut(&mut cpal::Data, &cpal::OutputCallbackInfo) + Send + 'static,
            E: FnMut(cpal::StreamError) + Send + 'static,
        {
            unimplemented!("not exercised by these tests")
        }
    }

    #[test]
    fn picks_matching_device_by_name() {
        let devices = vec![FakeDevice("a"), FakeDevice("b")].into_iter();
        let picked = resolve(devices, Some("b"), "input", || None);
        assert_eq!(picked.unwrap().name().unwrap(), "b");
    }

    #[test]
    fn falls_back_to_default_when_name_absent() {
        let devices = vec![FakeDevice("a")].into_iter();
        let picked = resolve(devices, None, "input", || Some(FakeDevice("default")));
        assert_eq!(picked.unwrap().name().unwrap(), "default");
    }

    #[test]
    fn falls_back_to_default_when_name_not_found() {
        let devices = vec![FakeDevice("a")].into_iter();
        let picked = resolve(devices, Some("nonexistent"), "input", || {
            Some(FakeDevice("default"))
        });
        assert_eq!(picked.unwrap().name().unwrap(), "default");
    }
}
