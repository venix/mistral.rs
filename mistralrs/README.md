# mistralrs — Blazing-Fast LLM Inference in Rust

The Rust SDK for [mistral.rs](https://github.com/EricLBuehler/mistral.rs), a high-performance
LLM inference engine supporting text, multimodal, speech, image generation, and embedding models.

[API Docs](https://docs.rs/mistralrs) | [GitHub](https://github.com/EricLBuehler/mistral.rs) | [Examples](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples) | [Discord](https://discord.gg/SZrecqK8qw)

## Quick Start

```rust
use mistralrs::{IsqBits, ModelBuilder, TextMessages, TextMessageRole};

#[tokio::main]
async fn main() -> mistralrs::error::Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    let response = model.chat("What is Rust's ownership model?").await?;
    println!("{response}");
    Ok(())
}
```

## Capabilities

| Capability | Builder | Example |
|---|---|---|
| Any model (auto-detect) | `ModelBuilder` | `examples/getting_started/text_generation/` |
| Text generation | `TextModelBuilder` | `examples/getting_started/text_generation/` |
| Multimodal (image+text) | `MultimodalModelBuilder` | `examples/getting_started/multimodal/` |
| GGUF quantized models | `GgufModelBuilder` | `examples/getting_started/gguf/` |
| Image generation | `DiffusionModelBuilder` | `examples/models/diffusion/` |
| Speech synthesis | `SpeechModelBuilder` | `examples/models/speech/` |
| Embeddings | `EmbeddingModelBuilder` | `examples/getting_started/embedding/` |
| Structured output | `Model::generate_structured` | `examples/advanced/json_schema/` |
| Tool calling | `Tool`, `ToolChoice` | `examples/advanced/tools/` |
| Agents | `AgentBuilder` | `examples/advanced/agent/` |
| LoRA / X-LoRA | `LoraModelBuilder`, `XLoraModelBuilder` | `examples/advanced/lora/` |
| AnyMoE | `AnyMoeModelBuilder` | `examples/advanced/anymoe/` |
| MCP client | `McpClientConfig` | `examples/advanced/mcp_client/` |

## Choosing a Request Type

| Type | Use When | Sampling |
|---|---|---|
| `TextMessages` | Simple text-only chat | Deterministic |
| `MultimodalMessages` | Prompt includes images or audio | Deterministic |
| `RequestBuilder` | Tools, logprobs, custom sampling, constraints, adapters, or web search | Configurable |

`TextMessages` and `MultimodalMessages` convert into `RequestBuilder` via `Into<RequestBuilder>`.

## Feature Flags

| Flag | Effect |
|---|---|
| `cuda` | CUDA GPU support |
| `flash-attn` | Flash Attention 2 kernels (requires `cuda`) |
| `cudnn` | cuDNN acceleration (requires `cuda`) |
| `nccl` | Multi-GPU via NCCL (requires `cuda` and NCCL) |
| `metal` | Apple Metal GPU support |
| `accelerate` | Apple Accelerate framework |
| `mkl` | Intel MKL acceleration |

The default feature set (no flags) builds with pure Rust — no C compiler or system libraries required.

## License

MIT

## Optional local multimodal embeddings

Enable `multimodal-embeddings` on the `mistralrs` dependency (plus `metal`, `cuda`,
or `accelerate` as appropriate). This is a dedicated EmbeddingGemma 2 adapter,
not the existing `EmbeddingModelBuilder`, `Model`, runner, or scheduler.

```rust,no_run
use mistralrs::{MultimodalEmbeddingInput, MultimodalEmbeddingModelBuilder};

# async fn example() -> anyhow::Result<()> {
let model = MultimodalEmbeddingModelBuilder::new("/models/embeddinggemma-2")
    .with_force_cpu(false)
    .with_max_input_tokens(2048)
    .with_modalities(true, true) // vision and audio; text is always loaded
    .build()
    .await?;
let output = model.embed(vec![
    MultimodalEmbeddingInput::Text("Already formatted by the caller".into()),
    MultimodalEmbeddingInput::Parts(vec![
        MultimodalEmbeddingInput::Text("Describe this photo: ".into()),
        MultimodalEmbeddingInput::Image(std::fs::read("photo.jpg")?),
        MultimodalEmbeddingInput::Parts(vec![
            MultimodalEmbeddingInput::Audio(std::fs::read("speech.wav")?),
        ]),
    ]),
]).await?;
assert_eq!(output.vectors.len(), output.truncated.len());
# Ok(())
# }
```

- An explicit existing local snapshot directory is required. The adapter never
  resolves model IDs or downloads files. Supply native safetensors (single or
  sharded), `config.json`, `tokenizer.json`, and the checkpoint's processor
  configuration and other companion files. No UQFF, GGUF, ISQ, or MLX support.
- Default loading requests text, vision and audio; only towers present in the
  checkpoint are loaded. Disabled or absent towers produce errors on use.
- The default maximum input length is **2048 tokens**, including special tokens.
  The builder rejects values below 16 (the upstream minimum) instead of silently
  raising them. Choose a limit compatible with the checkpoint's context window.
  Pure-text rows are right-truncated by the checkpoint tokenizer, preserving
  its special tokens. Flags are determined with the same tokenizer without
  truncation or padding, not by guessing from the final sequence length.
- Each top-level input returns one full-dimensional, L2-normalized `Vec<f32>`
  in input order. `Parts(Vec<MultimodalEmbeddingInput>)` groups ordered text,
  image and audio parts into **one row and one vector**. Nested `Parts` are
  flattened depth-first in order; grouping does not insert separators or
  task prefixes. The caller owns task formatting. An empty top-level batch
  returns empty vectors/flags, but every `Parts` container must be nonempty.
- Rows containing only text parts concatenate those parts verbatim and use
  normal text truncation with an accurate row flag. Rows containing media
  never truncate text fragments: the adapter checks their untruncated text
  token counts, then the core rejects a combined sequence that exceeds the
  token limit, preserving all media placeholders and special tokens. The
  core's normal media-only spacing behavior is unchanged.
- A mixed/media row's truncation flag is true if **any** audio part was clipped
  to the processor's duration limit. Images do not set truncation flags.
- Images are encoded PNG/JPEG/WebP/GIF/BMP bytes. Audio is encoded WAV, MP3,
  FLAC, Ogg/Vorbis, or M4A/AAC; decoding mixes to mono and resamples to 16 kHz.
  Audio is clipped to the processor's duration limit (normally 30 seconds),
  reported by the row's truncation flag. Media sequences that exceed the token
  limit fail rather than dropping media tokens.
- Device selection tries compiled CUDA, then Metal, then CPU. Accelerator
  precision is BF16; CPU is F32. `with_force_cpu(true)` overrides selection.
- `build` and inference use Tokio `spawn_blocking`. The model is `Debug`,
  `Clone`, `Send + Sync` and can be shared in an `Arc`. Clones share a one-call
  semaphore; queue waits are async, and decoding and tokenization are also
  serialized to avoid piling blocking threads up behind a model lock.
  Dropping a waiting call cancels its queue wait. Once blocking work has started,
  dropping the future does **not** immediately cancel loading or inference;
  its permit remains held until that work finishes. Errors use `anyhow::Result`.
- This adapter has no HTTP endpoint, Python bindings, CLI/server integration,
  existing API scheduling, UQFF export/loading, or in-situ quantization support.
  Model weights and their license/terms must be obtained separately.

The MIT core is vendored from `ikcore/ai-embeddinggemma-2` at
`2369ca8eeebee30785b7a6eff1ab7968e5a56404`; see
[`vendor/embeddinggemma_core/UPSTREAM.md`](../vendor/embeddinggemma_core/UPSTREAM.md).
Its Candle dependencies use the exact same Git revision as this fork.

### Media decoding budgets

The adapter's own media boundary rejects encoded images over **20 MiB** and
encoded audio over **50 MiB** before decoding. Each batch is limited to **32
top-level rows** and **64 MiB aggregate UTF-8 text plus encoded image/audio
bytes**, summed across every leaf of every parts tree. Each row permits at most
**32 tree nodes** (including its root and every `Parts` container) and **four
nested `Parts` levels**. Empty `Parts` containers are rejected. The complete
batch/tree/encoded-byte preflight runs before queueing or any media decoding,
so an invalid later row cannot trigger decoding of earlier rows. These limits
do not constitute a total decoded-audio or process-memory quota.

Images are restricted to PNG, JPEG, WebP, GIF and BMP. Header dimensions are
inspected with `ImageReader::into_dimensions` before full decoding: at most
**32,000,000 pixels**, with neither dimension above **32,768**. Both header
inspection and full decoding receive image `Limits` with a **128 MiB** allocation
budget and strict dimension limits. RGB expansion is separately bounded by the
pixel limit (at most 96,000,000 RGB bytes); it is not charged to the decoder's
allocation budget. Image allocation limits are codec-dependent/best-effort, not
a hard limit on total decoder scratch space or process memory.

Before **any full image or audio decode**, a blocking header-preflight pass
walks every image leaf across all rows/parts and enforces a **128 MiB batch
image RGB reservation budget**. Each image reserves the larger of its header's
`width * height * 3` bytes and its maximum resized RGB footprint:
`image_soft_tokens * pooling_kernel_size^2 * patch_size^2 * 3`.
This bounds the sum of retained decoded RGB images to 128 MiB, while also
preventing many tiny images from bypassing the budget and expanding excessively
during preprocessing. Header inspection repeats when the image is decoded;
pixel decoding itself happens once. The 32-row limit is unchanged: 16/32-row
batches remain allowed when they satisfy these budgets.

The upstream processor's resize/patchification output is bounded by its model
soft-token budget, not the original resolution. With the standard processor
(280 soft tokens, pooling 3, patches 16), its maximum resized RGB footprint is
1,935,360 bytes per image. The batch reservation consequently also bounds the
sum of the float32 patch buffers to **512 MiB** (four bytes per RGB component).
These are distinct buffer bounds, not a combined peak-memory guarantee:
decoder/resizer scratch space, audio buffers, encoded inputs and model/device
tensors consume additional memory.

Audio is demuxed and decoded incrementally by the adapter, not by upstream's
whole-file `decode_audio`. Only the configured clip duration plus one second of
sinc-resampling lookahead is retained; the adapter then resamples and clips to
the model's exact sample limit. No complete recording is accumulated. The
processor must use 16 kHz and a positive duration limit of at most 30 seconds.
Source audio is limited to 8,000-192,000 Hz and 1-8 channels; declared codec
blocks and decoded buffers are limited to 65,536 frames, encoded packets to
1 MiB, and processing to 100,000 packets. Metadata tag allocations are limited
to 1 MiB and embedded visual metadata is disabled where supported by Symphonia.
Malformed streams, mid-stream rate changes, unsupported formats and exhausted
budgets return errors instead of silently discarding audio. Gapless trimming
is enabled for encoded delay/padding.

Audio `truncated` is based on actual decoded sample duration crossing the clip
limit, not container duration metadata or the already-clipped output length.
Exactly-at-limit audio is not marked truncated; longer audio is marked true
even though only its bounded prefix is decoded. Codec/demuxer internal
allocations are not a sandbox: some work/allocations occur while probing or
before a decoded packet can be checked. Use process isolation and aggregate
request limits for fully untrusted workloads.

Only compile checks validate this adapter here; no checkpoint inference or
runtime device/kernel compatibility is claimed. Candle operator/dtype support,
Metal kernel availability and memory requirements can still produce runtime
errors, including on the first image/audio forward pass.

### Classifying deterministic input failures

`MultimodalEmbeddingInputError` is a public `std::error::Error + Send + Sync`
marker exported at the `mistralrs` crate root. Methods still return
`anyhow::Result`; the original error is retained beneath the marker/context.
Callers can classify rejected input without parsing messages:

```rust
use mistralrs::MultimodalEmbeddingInputError;

fn is_invalid_input(error: &anyhow::Error) -> bool {
    error.downcast_ref::<MultimodalEmbeddingInputError>().is_some()
}
```

Inference input validation, tree/byte/image-budget/header failures, image/audio
decode failures, unavailable requested towers and tokenizer failures carry this
type. The pinned core's `EmbedError::InvalidRequest`, `Media` and `Tokenizer`
variants are also classified, including combined-context and media-placeholder
layout failures. Its special `Media("input preprocessing panicked")` failure
is deliberately excluded because it represents an internal worker panic.

Candle/model compute failures (`EmbedError::Candle`/`Backend`), Metal kernel/OOM
errors, device/configuration/weight-loading/I/O errors, queue failures and
blocking-task panics are **not** classified as invalid input. Build failures
also remain ordinary errors. The marker describes input rejection, not whether
retrying the identical bytes unchanged will succeed; callers may use it to
isolate a corrupt row or continue processing other requests rather than
starving their queue with an undifferentiated backend retry loop.
