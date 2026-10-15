use std::io::{Cursor, ErrorKind};

use anyhow::{bail, ensure, Context, Result};
use embeddinggemma_core::media::audio::resample;
use image::{ImageFormat, ImageReader, Limits, RgbImage};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::DecoderOptions,
    errors::Error as AudioError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::{Limit, MetadataOptions},
    probe::Hint,
};

pub(super) const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_IMAGE_PIXELS: u64 = 32_000_000;
const MAX_IMAGE_DIMENSION: u32 = 32_768;
const MAX_IMAGE_ALLOC: u64 = 128 * 1024 * 1024;
pub(super) const MAX_AUDIO_BYTES: usize = 50 * 1024 * 1024;
pub(super) const AUDIO_SAMPLE_RATE: u32 = 16_000;
pub(super) const MAX_AUDIO_SAMPLES: usize = AUDIO_SAMPLE_RATE as usize * 30;
const MIN_SOURCE_SAMPLE_RATE: u32 = 8_000;
const MAX_SOURCE_SAMPLE_RATE: u32 = 192_000;
const MAX_AUDIO_CHANNELS: usize = 8;
const MAX_PACKET_FRAMES: u64 = 65_536;
const MAX_PACKET_BYTES: usize = 1024 * 1024;
const MAX_AUDIO_PACKETS: usize = 100_000;
const MAX_METADATA_BYTES: usize = 1024 * 1024;

fn image_reader(bytes: &[u8], format: ImageFormat) -> ImageReader<Cursor<&[u8]>> {
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_IMAGE_ALLOC);
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    reader.limits(limits);
    reader
}

fn inspect_image(bytes: &[u8]) -> Result<(ImageFormat, u32, u32)> {
    ensure!(
        bytes.len() <= MAX_IMAGE_BYTES,
        "encoded image exceeds 20 MiB"
    );
    let format = image::guess_format(bytes).context("cannot identify image format")?;
    ensure!(
        matches!(
            format,
            ImageFormat::Png
                | ImageFormat::Jpeg
                | ImageFormat::WebP
                | ImageFormat::Gif
                | ImageFormat::Bmp
        ),
        "unsupported image format; expected PNG, JPEG, WebP, GIF or BMP"
    );
    let (width, height) = image_reader(bytes, format)
        .into_dimensions()
        .context("cannot inspect image dimensions")?;
    ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
    ensure!(
        u64::from(width) * u64::from(height) <= MAX_IMAGE_PIXELS,
        "image exceeds 32 million pixels"
    );
    Ok((format, width, height))
}

pub(super) fn image_rgb_bytes(bytes: &[u8]) -> Result<u64> {
    let (_, width, height) = inspect_image(bytes)?;
    Ok(u64::from(width) * u64::from(height) * 3)
}

pub(super) fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    let (format, width, height) = inspect_image(bytes)?;
    let decoded = image_reader(bytes, format)
        .decode()
        .context("image decoding failed or exceeded decoder limits")?;
    ensure!(
        decoded.width() == width && decoded.height() == height,
        "decoded image dimensions disagree with header"
    );
    // The header pixel budget also bounds RGB expansion, which is outside the decoder's allocation budget.
    Ok(decoded.into_rgb8())
}

pub(super) fn decode_audio(bytes: Vec<u8>, max_samples: usize) -> Result<(Vec<f32>, bool)> {
    ensure!(
        bytes.len() <= MAX_AUDIO_BYTES,
        "encoded audio exceeds 50 MiB"
    );
    ensure!(
        (1..=MAX_AUDIO_SAMPLES).contains(&max_samples),
        "invalid audio sample budget"
    );
    let stream = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let metadata = MetadataOptions {
        limit_metadata_bytes: Limit::Maximum(MAX_METADATA_BYTES),
        limit_visual_bytes: Limit::Maximum(0),
    };
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            stream,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &metadata,
        )
        .context("cannot probe encoded audio")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("audio file has no default track")?;
    let track_id = track.id;
    let rate = track
        .codec_params
        .sample_rate
        .context("audio track has no sample rate")?;
    ensure!(
        (MIN_SOURCE_SAMPLE_RATE..=MAX_SOURCE_SAMPLE_RATE).contains(&rate),
        "audio sample rate must be between 8000 and 192000 Hz"
    );
    if let Some(channels) = track.codec_params.channels {
        ensure!(
            (1..=MAX_AUDIO_CHANNELS).contains(&channels.count()),
            "audio exceeds 8 channels"
        );
    }
    for frames in [
        track.codec_params.max_frames_per_packet,
        track.codec_params.frames_per_block,
    ]
    .into_iter()
    .flatten()
    {
        ensure!(
            frames <= MAX_PACKET_FRAMES,
            "audio declares an oversized codec block"
        );
    }
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("cannot create audio decoder")?;
    let clip_frames = max_samples as u64 * u64::from(rate) / u64::from(AUDIO_SAMPLE_RATE);
    // Match the upstream sinc resampler's one-second lookahead without decoding the whole recording.
    let keep = usize::try_from(clip_frames + u64::from(rate))?;
    let mut mono = Vec::with_capacity(keep);
    let mut reached_end = false;
    let mut truncated = false;
    for _ in 0..MAX_AUDIO_PACKETS {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(AudioError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                reached_end = true;
                break;
            }
            Err(error) => return Err(error).context("cannot read audio packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        ensure!(
            packet.buf().len() <= MAX_PACKET_BYTES,
            "encoded audio packet exceeds 1 MiB"
        );
        if let Some(time_base) = decoder.codec_params().time_base {
            let duration = time_base.calc_time(packet.dur());
            ensure!(
                duration.seconds as f64 + duration.frac
                    <= MAX_PACKET_FRAMES as f64 / f64::from(rate),
                "audio packet duration exceeds decoding budget"
            );
        }
        // Decode errors are not silently skipped: that would undercount duration and misreport clipping.
        let decoded = decoder
            .decode(&packet)
            .context("cannot decode audio packet")?;
        let spec = *decoded.spec();
        let channels = spec.channels.count();
        ensure!(spec.rate == rate, "audio sample rate changed mid-stream");
        ensure!(
            (1..=MAX_AUDIO_CHANNELS).contains(&channels),
            "decoded audio exceeds 8 channels"
        );
        ensure!(
            decoded.capacity() as u64 <= MAX_PACKET_FRAMES,
            "decoded audio block exceeds frame budget"
        );
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        for frame in buffer.samples().chunks_exact(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
            truncated |= mono.len() as u64 * u64::from(AUDIO_SAMPLE_RATE)
                > max_samples as u64 * u64::from(rate);
            if mono.len() == keep {
                reached_end = true;
                break;
            }
        }
        if mono.len() == keep {
            break;
        }
    }
    ensure!(reached_end, "audio exceeded packet processing budget");
    ensure!(!mono.is_empty(), "audio file has no decoded samples");
    let mut samples = resample(&mono, rate, AUDIO_SAMPLE_RATE);
    // Clip here as well as in the model so no oversized sample vector crosses the adapter boundary.
    samples.truncate(max_samples);
    if samples.is_empty() {
        bail!("audio resampling produced no samples");
    }
    Ok((samples, truncated))
}
