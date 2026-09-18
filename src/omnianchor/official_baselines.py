"""Pinned E5 and Qwen retrieval recipes with explicit budgets and no media fallback."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .backends.hf import HFBackend
from .backends.media import freeze_media
from .errors import BudgetExceeded, ResourceUnavailable
from .types import ModelSpec, ResourceProfile, Sample


def require_cuda(revision: str):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Official baselines require an immutable revision.")
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ResourceUnavailable("Baseline requires allocated BF16 CUDA; no fallback.")
    return torch


class E5Embedding:
    def __init__(self, revision: str):
        self.torch = require_cuda(revision)
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large", revision=revision)
        self.model = AutoModel.from_pretrained("intfloat/multilingual-e5-large", revision=revision,
                                               dtype=self.torch.bfloat16).to("cuda").eval()

    def encode(self, texts: Sequence[str], *, role: str = "query") -> np.ndarray:
        if role not in {"query", "passage"} or not texts:
            raise ValueError("E5 needs nonempty texts and explicit query/passage role.")
        rows = []
        for text in texts:
            inputs = self.tokenizer(f"{role}: {text}", return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > 512:
                raise BudgetExceeded("E5 native 512-token limit; no silent truncation.")
            inputs = inputs.to("cuda")
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].bool().unsqueeze(-1)
                vector = hidden.masked_fill(~mask, 0).float().sum(1) / mask.sum(1)
                vector = self.torch.nn.functional.normalize(vector, p=2, dim=-1)
            rows.append(vector.cpu().numpy()[0])
        return np.stack(rows)


class QwenRetrieval:
    """Official last-token embedding or yes/no head; shared strict media preprocessing.

    Reuses the audited upstream model class. The official wrapper's silent NULL-media
    replacement and truncation are deliberately replaced by the existing strict encoder.
    This is recorded as a budget-controlled recipe, not identical upstream preprocessing.
    """
    def __init__(self, model_id: str, revision: str, upstream: Path,
                 resources: ResourceProfile = ResourceProfile(), *, reuse_media_preprocessing: bool = False):
        self.torch = require_cuda(revision)
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if model_id not in {"Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Reranker-2B"}:
            raise ValueError("Unsupported official baseline.")
        self.kind = "embedding" if "Embedding" in model_id else "reranker"
        snapshot = snapshot_download(model_id, revision=revision,
                                     allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        self.processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=False)
        if self.kind == "embedding":
            path = upstream / "qwen3_vl_embedding.py"
            spec = importlib.util.spec_from_file_location("omnianchor_pinned_qwen_embedding", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.model = module.Qwen3VLForEmbedding.from_pretrained(
                snapshot, dtype=self.torch.bfloat16, trust_remote_code=False).to("cuda").eval()
            self.pool = module.Qwen3VLEmbedder._pooling_last
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                snapshot, dtype=self.torch.bfloat16, trust_remote_code=False).to("cuda").eval()
            vocab = self.processor.tokenizer.get_vocab()
            self.yes, self.no = vocab["yes"], vocab["no"]
        self.resources = resources
        self.reuse_media_preprocessing = reuse_media_preprocessing
        self.encoder = HFBackend(ModelSpec(id=model_id, revision=revision), resources,
                                 model=self.model, processor=self.processor)

    def _inputs(self, sample: Sample, instruction: str, query: str | None = None, *, frozen=None):
        frozen = freeze_media(sample, self.resources) if frozen is None else frozen
        text = "\n".join(p.text for p in sample.parts if p.type == "text")
        if len(self.processor.tokenizer.encode(text, add_special_tokens=False)) > self.resources.limits.input_text_tokens:
            raise BudgetExceeded("Baseline text budget exceeded; no truncation.")
        if query is None:
            messages = [{"role": "system", "content": [{"type": "text", "text": instruction}]},
                        {"role": "user", "content": frozen.content}]
        else:
            messages = [{"role": "system", "content": [{"type": "text", "text":
                'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'}]},
                {"role": "user", "content": [{"type": "text", "text": "<Instruct>: " + instruction},
                    {"type": "text", "text": "<Query>:"}, {"type": "text", "text": query},
                    {"type": "text", "text": "\n<Document>:"}, *frozen.content]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.encoder._device_inputs(self.encoder._encode(rendered, frozen))
        for module in self.model.modules():
            if hasattr(module, "rope_deltas"):
                module.rope_deltas = None
        self.last_input_evidence = {"input_token_ids": inputs["input_ids"][0].tolist(),
                                    "media": frozen.records, "instruction": instruction}
        return inputs

    def encode(self, samples: Sequence[Sample], instruction: str = "Represent the user's input.") -> np.ndarray:
        if self.kind != "embedding" or not samples:
            raise ValueError("Embedding model and nonempty samples required.")
        rows = []
        for sample in samples:
            inputs = self._inputs(sample, instruction)
            with self.torch.inference_mode():
                output = self.model(**inputs, use_cache=False)
                vector = self.pool(output.last_hidden_state, inputs["attention_mask"]).float()
                vector = self.torch.nn.functional.normalize(vector, p=2, dim=-1)
            rows.append(vector.cpu().numpy()[0])
        return np.stack(rows)

    def score(self, sample: Sample, queries: Sequence[str], instruction: str) -> np.ndarray:
        if self.kind != "reranker":
            raise ValueError("Reranker model required.")
        rows = []
        frozen = freeze_media(sample, self.resources) if self.reuse_media_preprocessing else None
        for query in queries:
            inputs = self._inputs(sample, instruction, query, frozen=frozen)
            with self.torch.inference_mode():
                output = self.model(**inputs, use_cache=False, logits_to_keep=1)
                logits = output.logits[0, -1, [self.no, self.yes]].float()
                rows.append(logits.softmax(-1)[1].item())
        return np.asarray(rows)
