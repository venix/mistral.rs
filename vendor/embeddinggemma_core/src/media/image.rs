//! `Gemma4ImageProcessor`: aspect-ratio-preserving bicubic resize so that
//! both sides are multiples of `patch × pooling` (48 px) and the image has at
//! most `soft_tokens × pooling²` patches, rescale to `[0, 1]`, cut into
//! row-major 16×16×3 patches.

use super::ImagePatches;
use crate::error::{EmbedError, Result};
use image::{RgbImage, imageops::FilterType};

/// Decode PNG / JPEG / WebP / GIF (first frame) / BMP and convert to RGB
/// (alpha is dropped, as `convert("RGB")` does).
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    image::load_from_memory(bytes)
        .map(|img| img.to_rgb8())
        .map_err(|e| EmbedError::Media(format!("cannot decode image: {e}")))
}

/// `get_aspect_ratio_preserving_size`: the largest `(height, width)`, both
/// multiples of `patch_size × pooling_kernel_size`, with at most
/// `max_patches` patches.
pub fn target_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(EmbedError::Media("image has no pixels".into()));
    }
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let side = pooling_kernel_size * patch_size;
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    if th == 0 && tw == 0 {
        return Err(EmbedError::Media(format!(
            "cannot resize a {height}×{width} image to a multiple of {side} px"
        )));
    }
    let max_side = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side;
    if th == 0 {
        th = side;
        tw = ((width / height) * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height / width) * side).min(max_side);
    }
    if (th * tw) as f64 > target_px {
        return Err(EmbedError::Media(format!(
            "resizing {height}×{width} to {th}×{tw} exceeds {max_patches} patches"
        )));
    }
    Ok((th, tw))
}

/// Resize and patchify one image for a budget of `soft_tokens` (70, 140,
/// 280, 560 or 1120).
pub fn preprocess_image(
    img: &RgbImage,
    soft_tokens: usize,
    patch_size: usize,
    pooling_kernel_size: usize,
) -> Result<ImagePatches> {
    let max_patches = soft_tokens * pooling_kernel_size * pooling_kernel_size;
    let (h, w) = (img.height() as usize, img.width() as usize);
    let (th, tw) = target_size(h, w, patch_size, max_patches, pooling_kernel_size)?;
    let resized;
    let img = if (th, tw) == (h, w) {
        img
    } else {
        // Catmull-Rom with a support scaled by the downsampling ratio is
        // PIL's / torchvision's antialiased bicubic.
        resized = image::imageops::resize(img, tw as u32, th as u32, FilterType::CatmullRom);
        &resized
    };
    let (ph, pw) = (th / patch_size, tw / patch_size);
    let scale = 1.0f32 / 255.0;
    let mut pixels = Vec::with_capacity(th * tw * 3);
    for py in 0..ph {
        for px in 0..pw {
            for y in 0..patch_size {
                for x in 0..patch_size {
                    let p =
                        img.get_pixel((px * patch_size + x) as u32, (py * patch_size + y) as u32);
                    pixels.extend(p.0.iter().map(|&c| c as f32 * scale));
                }
            }
        }
    }
    Ok(ImagePatches {
        pixels,
        patches_h: ph,
        patches_w: pw,
        pooling_kernel_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_the_reference_processor() {
        // 280 soft tokens → 2520 patches of 16 px, sides multiples of 48.
        assert_eq!(target_size(480, 640, 16, 2520, 3).unwrap(), (672, 912));
        assert_eq!(target_size(1000, 1000, 16, 2520, 3).unwrap(), (768, 768));
        assert_eq!(target_size(48, 48, 16, 2520, 3).unwrap(), (768, 768));
        // Extreme aspect ratio: one side clamps to 48 px.
        let (h, w) = target_size(10, 4000, 16, 2520, 3).unwrap();
        assert_eq!(h, 48);
        assert!(w % 48 == 0 && (h / 16) * (w / 16) <= 2520);
        assert!(target_size(0, 10, 16, 2520, 3).is_err());
    }

    #[test]
    fn patches_are_row_major_with_channels_last() {
        let mut img = RgbImage::new(96, 48);
        img.put_pixel(16, 0, image::Rgb([255, 0, 0])); // patch (0, 1), first pixel
        let p = preprocess_image(&img, 70, 16, 3).unwrap();
        // 70 tokens → 630 patches; 48×96 grows to the budget keeping 1:2.
        assert_eq!(p.patches_h % 3, 0);
        assert_eq!(p.pixels.len(), p.patches_h * p.patches_w * 768);
        let p = preprocess_image(&RgbImage::new(96, 48), 70, 16, 3).unwrap();
        assert!(p.pixels.iter().all(|&x| x == 0.0));
        assert_eq!(p.soft_tokens(), (p.patches_h / 3) * (p.patches_w / 3));
    }
}
