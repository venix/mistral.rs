//! Audio decoding (WAV, MP3, FLAC, Ogg/Vorbis, M4A/AAC via symphonia),
//! resampling to 16 kHz mono, and `Gemma4AudioFeatureExtractor`'s log-mel
//! front end:
//!
//! - truncate to 30 s, zero-pad to a multiple of 128 samples;
//! - semicausal padding (`frame_length / 2` zeros in front), frames of 320
//!   samples every 160 (10 ms), periodic Hann window, 512-point real FFT;
//! - **magnitude** spectrum × 128 HTK mel filters (0–8 kHz, no area
//!   normalisation), `ln(mel + 0.001)`;
//! - a frame is valid only if its whole analysis window is real audio;
//!   invalid frames are zeroed.

use super::AudioFeatures;
use crate::config::ProcessorConfig;
use crate::error::{EmbedError, Result};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decode a compressed or PCM audio file to mono f32 samples and its sample
/// rate. `extension` (`wav`, `mp3`, …) helps the format probe.
pub fn decode_audio(bytes: &[u8], extension: Option<&str>) -> Result<(Vec<f32>, u32)> {
    let err =
        |what: &str, e: SymError| EmbedError::Media(format!("cannot decode audio ({what}): {e}"));
    let source = std::io::Cursor::new(bytes.to_vec());
    let stream = MediaSourceStream::new(Box::new(source), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = extension {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| err("format", e))?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| EmbedError::Media("audio file has no track".into()))?;
    let track_id = track.id;
    let rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| EmbedError::Media("audio track has no sample rate".into()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| err("codec", e))?;

    let mut mono = Vec::new();
    let mut buffer: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(err("read", e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(err("decode", e)),
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let buf = buffer.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if buf.capacity() < decoded.capacity() * channels {
            *buf = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    if mono.is_empty() {
        return Err(EmbedError::Media("audio file has no samples".into()));
    }
    Ok((mono, rate))
}

/// Band-limited resampling (Hann-windowed sinc, 32 zero crossings, cut-off
/// at the lower Nyquist). Close to, not bit-identical with, the soxr / sinc
/// resamplers Python pipelines use.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.0) * 0.97;
    let zeros = 32.0;
    let half_width = zeros / cutoff; // in input samples
    let n_out = ((x.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let t = i as f64 / ratio;
        let lo = (t - half_width).ceil().max(0.0) as usize;
        let hi = ((t + half_width).floor() as usize).min(x.len() - 1);
        let (mut acc, mut norm) = (0.0f64, 0.0f64);
        for (j, &s) in x.iter().enumerate().take(hi + 1).skip(lo) {
            let d = j as f64 - t;
            let arg = d * cutoff;
            let sinc = if arg.abs() < 1e-12 {
                1.0
            } else {
                (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg)
            };
            let window = 0.5 + 0.5 * (std::f64::consts::PI * d / half_width).cos();
            let k = sinc * window;
            acc += s as f64 * k;
            norm += k;
        }
        out.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.0
        });
    }
    out
}

/// `mel_filter_bank(…, norm=None, mel_scale="htk")`: `[fft_bins][mel]`.
pub fn mel_filters(cfg: &ProcessorConfig) -> Vec<Vec<f64>> {
    let bins = cfg.fft_length / 2 + 1;
    let mels = cfg.feature_size;
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (lo, hi) = (hz_to_mel(cfg.min_frequency), hz_to_mel(cfg.max_frequency));
    let filter_freqs: Vec<f64> = (0..mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (mels + 1) as f64))
        .collect();
    let nyquist = (cfg.sampling_rate / 2) as f64;
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|i| nyquist * i as f64 / (bins - 1) as f64)
        .collect();
    fft_freqs
        .iter()
        .map(|&f| {
            (0..mels)
                .map(|m| {
                    let down = (f - filter_freqs[m]) / (filter_freqs[m + 1] - filter_freqs[m]);
                    let up =
                        (filter_freqs[m + 2] - f) / (filter_freqs[m + 2] - filter_freqs[m + 1]);
                    down.min(up).max(0.0)
                })
                .collect()
        })
        .collect()
}

/// 16 kHz mono samples → log-mel features (see the module docs).
pub fn log_mel(samples: &[f32], cfg: &ProcessorConfig) -> Result<AudioFeatures> {
    let real = samples.len().min(cfg.max_audio_samples);
    if real == 0 {
        return Err(EmbedError::Media("audio is empty".into()));
    }
    let multiple = cfg.pad_to_multiple_of.max(1);
    let padded_len = real.div_ceil(multiple) * multiple;
    let pad_left = cfg.frame_length / 2;
    let mut wave = vec![0f32; pad_left + padded_len];
    wave[pad_left..pad_left + real].copy_from_slice(&samples[..real]);

    let unfold = cfg.frame_length + 1;
    if wave.len() < unfold {
        return Err(EmbedError::Media(format!(
            "audio is too short ({real} samples; at least {} needed)",
            unfold - pad_left
        )));
    }
    let frames = (wave.len() - unfold) / cfg.hop_length + 1;
    let window: Vec<f32> = (0..cfg.frame_length)
        .map(|n| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / cfg.frame_length as f64).cos())
                as f32
        })
        .collect();
    let filters = mel_filters(cfg);
    let fft = FftPlanner::<f64>::new().plan_fft_forward(cfg.fft_length);
    let mut buf = vec![Complex::new(0.0f64, 0.0); cfg.fft_length];
    let mut mag = vec![0f64; cfg.fft_length / 2 + 1];
    let mut mel = Vec::with_capacity(frames * cfg.feature_size);
    let mut valid_frames = 0;
    for i in 0..frames {
        let start = i * cfg.hop_length;
        // Valid iff the last sample of the analysis window is real audio.
        let valid = start + unfold - 1 < pad_left + real;
        if !valid {
            mel.extend(std::iter::repeat_n(0.0, cfg.feature_size));
            continue;
        }
        valid_frames += 1;
        for (k, slot) in buf.iter_mut().enumerate() {
            *slot = if k < cfg.frame_length {
                // float32 product, like numpy's `frames * window`.
                Complex::new((wave[start + k] * window[k]) as f64, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }
        fft.process(&mut buf);
        for (m, c) in mag.iter_mut().zip(&buf) {
            *m = c.norm();
        }
        mel.extend((0..cfg.feature_size).map(|j| {
            let e: f64 = mag.iter().zip(&filters).map(|(m, row)| m * row[j]).sum();
            (e + cfg.mel_floor).ln() as f32
        }));
    }
    if valid_frames == 0 {
        return Err(EmbedError::Media(
            "audio is too short to form one frame".into(),
        ));
    }
    Ok(AudioFeatures {
        mel,
        frames,
        bins: cfg.feature_size,
        valid_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_bank_is_triangular_htk() {
        let cfg = ProcessorConfig::default();
        let f = mel_filters(&cfg);
        assert_eq!(f.len(), 257);
        assert_eq!(f[0].len(), 128);
        assert!(
            f[0].iter().all(|&x| x == 0.0),
            "DC bin is below the first filter peak"
        );
        // Each filter peaks at ≤ 1 and is non-negative.
        for m in 0..128 {
            let peak = f.iter().map(|row| row[m]).fold(0.0, f64::max);
            assert!(peak <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn frames_and_mask_follow_the_extractor() {
        let cfg = ProcessorConfig::default();
        // 1 s of a 440 Hz tone: 16000 samples (a multiple of 128) →
        // (16160 - 321) / 160 + 1 = 99 frames, all valid.
        let x: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let a = log_mel(&x, &cfg).unwrap();
        assert_eq!((a.frames, a.valid_frames, a.bins), (99, 99, 128));
        assert_eq!(a.soft_tokens(), 25);
        // 15840 samples pad to 15872: the last frame's window ends in padding.
        let a = log_mel(&vec![0.1; 15_840], &cfg).unwrap();
        assert_eq!((a.frames, a.valid_frames), (99, 98));
        assert!(a.mel[a.valid_frames * 128..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn resampling_preserves_a_tone() {
        let x: Vec<f32> = (0..44_100)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44_100.0).sin())
            .collect();
        let y = resample(&x, 44_100, 16_000);
        assert_eq!(y.len(), 16_000);
        let peak = y[1000..15_000].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 1.0).abs() < 0.02, "peak {peak}");
    }
}
