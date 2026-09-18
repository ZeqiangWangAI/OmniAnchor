# Models and adapters

## Scoring policy (identical for every model)

1. The official chat template renders a user turn with the material and a generation
   prompt; the exact bridge prefix is appended as the start of the assistant turn.
2. Prefix and prefix + anchor are encoded with the same frozen media, without padding,
   truncation or added special tokens. The full token sequence must begin with the prefix
   tokens, otherwise the candidate fails with `BoundaryError`.
3. `token_prefix` scores the anchor's token chain; `turn_terminated` appends the exact
   turn-end token and scores it too. Both are recorded in the manifest.
4. One unpadded forward per candidate, `model.eval()`, inference mode, `use_cache=False`,
   full-vocabulary log-softmax in FP32, log probabilities summed in double precision.
5. Failures are rows with a status and message, never zeros.

## Adapters

An adapter declares what differs between model families; the policy above does not change.

| Adapter | Kind | `model_type` | Transformers class | Modalities | Template kwargs | Turn end |
|---|---|---|---|---|---|---|
| `qwen3_5` | native multimodal | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | text, image, video | `enable_thinking: false` | `<\|im_end\|>` |
| `qwen3_vl` | native multimodal | `qwen3_vl` | `Qwen3VLForConditionalGeneration` | text, image, video | `enable_thinking: false` | `<\|im_end\|>` |
| `qwen3` | causal LM | `qwen3` | `AutoModelForCausalLM` | text | `enable_thinking: false` | `<\|im_end\|>` |
| `causal_lm` | causal LM | any | `AutoModelForCausalLM` | text | none | must be given |

Selection order: `model.adapter` if set; otherwise the checkpoint's `config.model_type`;
otherwise the packaged registry entry for that model id. Text-only adapters wrap the
tokenizer in a small processor shim so the encoding path is the same; a sample with image
or video parts fails explicitly under a text-only adapter. The generic adapter never guesses
a turn-end token: scoring `turn_terminated` without `turn_end_token` is an error.

The manifest records `adapter`, `adapter_kind`, `chat_template_kwargs` and
`turn_end_token`, so caches and score tables from different adapters never mix.

## Registry

`src/omnianchor/registry/models.yaml` maps aliases to pinned revisions and adapters.
`omnianchor models` prints it. Aliases with a `validation` entry passed the native
acceptance check on real weights (see [validation.md](validation.md)).

## Using another model

Write the model mapping in full in the study file:

```yaml
model:
  backend: hf
  id: org/some-chat-model
  revision: <40-character commit>
  adapter: causal_lm            # omit for Qwen3 / Qwen3-VL / Qwen3.5 families
  turn_end_token: "<|eot_id|>"  # only needed for event: turn_terminated
  chat_template_kwargs: {}      # e.g. {enable_thinking: false} for thinking models
```

Then run the acceptance check on the GPU node before measuring anything:

```bash
omnianchor verify-model --config my_study.yaml --output verification.json --image sample.png
```

It compares the selected-logit scorer with an independent full-logit forward on a bounded
text input, checks the optional shared-prefill shortcut, and (with `--image`) checks that
scoring an image does not change later text scores. Treat a `failed` status as a blocker.

Two things to inspect in a new model's manifest: `prefix_token_count` and
`rendered_prefix_sha256`. Some published chat templates prepend a long system preamble,
sometimes including the current date (SmolLM3 does), which lengthens every prefix and
changes the rendered prompt from day to day. Raise `--max-full-sequence-tokens` for the
check if the preamble exceeds 256 tokens, and fix the preamble with `system_prompt` in the
study when the template allows it.

## Adding a model family

Add a `ModelAdapter` entry to `ADAPTERS` in `src/omnianchor/backends/adapters.py` with the
family's `model_type`, model class, modalities, template arguments and turn-end token, add a
CPU test in `tests/test_backends_adapters.py`, run `verify-model` on real weights, and
record the evidence in `validation.md` before adding a registry alias.

## Video

Videos are decoded with PyAV, frames are sampled uniformly within the optional clip, and
the real frame rate, timestamps and frame hashes are recorded. Variable-frame-rate
recordings are refused for the Qwen native adapters rather than re-timed silently.
