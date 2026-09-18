"""Optional text baselines. Imports/model loading are lazy; no implicit CPU fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .errors import BudgetExceeded, ResourceUnavailable


def cosine_scores(left, right) -> np.ndarray:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Cosine inputs require matching embedding dimensions.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Embedding values must be finite.")
    ln, rn = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
    if np.any(ln == 0) or np.any(rn == 0):
        raise ValueError("Cosine similarity is undefined for a zero embedding.")
    return (left / ln[:, None]) @ (right / rn[:, None]).T


@dataclass
class HFBaseline:
    model_id: str
    revision: str
    device: str = "cuda"
    batch_size: int = 8
    max_tokens: int = 512
    local_files_only: bool = True
    max_parameters: int = 4_000_000_000

    def __post_init__(self):
        if not self.model_id or not self.revision or self.batch_size < 1 or self.max_tokens < 1:
            raise ValueError("Specify model/revision and positive batch/token budgets.")
        self._model = None
        self._tokenizer = None
        self._torch = None

    @property
    def capabilities(self) -> dict:
        return {"modalities": ["text"], "supports_images": False, "supports_video": False}

    def _load(self, auto_class: str):
        if self._model is not None:
            return
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ImportError("Install vlanchor[hf] to use Hugging Face baselines.") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ResourceUnavailable("CUDA was requested but is unavailable; no CPU fallback performed.")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ResourceUnavailable("MPS was requested but is unavailable.")
        if self.device != "cpu" and self.device != "mps" and not self.device.startswith("cuda"):
            raise ValueError("Baseline device must be explicit cpu, cuda[:index], or mps.")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=self.local_files_only,
            trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            raise ValueError("Baseline tokenizer has no pad token; choose a compatible checkpoint.")
        model = getattr(transformers, auto_class).from_pretrained(
            self.model_id, revision=self.revision, local_files_only=self.local_files_only,
            trust_remote_code=False)
        count = sum(p.numel() for p in model.parameters())
        if count > self.max_parameters:
            raise BudgetExceeded(f"Baseline has {count} parameters, above {self.max_parameters}.")
        self._tokenizer, self._model, self._torch = tokenizer, model.to(self.device).eval(), torch

    def _inputs(self, texts: Sequence[str], pairs: Sequence[str] | None = None):
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("This baseline supports text strings only.")
        values = self._tokenizer(list(texts), text_pair=list(pairs) if pairs is not None else None,
                                 padding=True, truncation=False, return_tensors="pt")
        if values["input_ids"].shape[1] > self.max_tokens:
            raise BudgetExceeded("Baseline token budget exceeded; implicit truncation is disabled.")
        return {key: value.to(self.device) for key, value in values.items()}


class HFMeanEmbedding(HFBaseline):
    """Attention-mask mean pooling over AutoModel final states, with explicit normalization.

    This is a generic mean-pooling baseline, not a claim to reproduce every sentence
    embedding checkpoint's custom pooling, task prefixes, or normalization recipe.
    """

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text is required.")
        self._load("AutoModel")
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._inputs(texts[start:start + self.batch_size])
            with self._torch.inference_mode():
                output = self._model(**batch)
                hidden = output.last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                result.append(mean.float().cpu().numpy())
        return np.concatenate(result, axis=0)

    def score(self, texts: Sequence[str], anchors: Sequence[str]) -> np.ndarray:
        return cosine_scores(self.encode(texts), self.encode(anchors))


@dataclass
class HFCrossEncoder(HFBaseline):
    """Aligned text-pair logits from a sequence-classification reranker checkpoint."""

    positive_label_index: int | None = None

    def score_pairs(self, texts: Sequence[str], candidates: Sequence[str]) -> np.ndarray:
        if not texts or len(texts) != len(candidates):
            raise ValueError("Reranker requires nonempty, equally sized text/candidate sequences.")
        self._load("AutoModelForSequenceClassification")
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._inputs(texts[start:start + self.batch_size], candidates[start:start + self.batch_size])
            with self._torch.inference_mode():
                logits = self._model(**batch).logits
            if logits.shape[-1] == 1:
                values = logits[:, 0]
            elif self.positive_label_index is not None and 0 <= self.positive_label_index < logits.shape[-1]:
                values = logits[:, self.positive_label_index]
            else:
                raise ValueError("Multiclass reranker requires an explicit positive_label_index.")
            result.append(values.float().cpu().numpy())
        return np.concatenate(result)


class HFSingleTokenMLM(HFBaseline):
    """Exact masked-token log probabilities; multi-token anchors fail explicitly."""

    @property
    def capabilities(self) -> dict:
        return {**super().capabilities, "anchor_tokens": 1, "event": "single_mask_token"}

    def score(self, masked_texts: Sequence[str], anchors: Sequence[str]) -> np.ndarray:
        if not masked_texts or not anchors:
            raise ValueError("MLM scoring requires contexts and anchors.")
        self._load("AutoModelForMaskedLM")
        if self._tokenizer.mask_token_id is None:
            raise ValueError("Checkpoint has no mask token.")
        anchor_ids = [self._tokenizer.encode(anchor, add_special_tokens=False) for anchor in anchors]
        if any(len(ids) != 1 for ids in anchor_ids):
            raise ValueError("Single-token MLM baseline cannot score multi-token anchor strings.")
        if any(ids[0] == self._tokenizer.unk_token_id for ids in anchor_ids):
            raise ValueError("An MLM anchor maps to the unknown token.")
        result = []
        for start in range(0, len(masked_texts), self.batch_size):
            batch = self._inputs(masked_texts[start:start + self.batch_size])
            mask = batch["input_ids"] == self._tokenizer.mask_token_id
            if not self._torch.all(mask.sum(dim=1) == 1):
                raise ValueError("Each MLM context must contain exactly one mask token.")
            with self._torch.inference_mode():
                logits = self._model(**batch).logits[mask].float()
                logp = logits.log_softmax(dim=-1)[:, [ids[0] for ids in anchor_ids]]
                result.append(logp.cpu().numpy())
        return np.concatenate(result)


BASELINE_REGISTRY = {"hf_mean_embedding": HFMeanEmbedding, "hf_cross_encoder": HFCrossEncoder,
                     "hf_single_token_mlm": HFSingleTokenMLM}
