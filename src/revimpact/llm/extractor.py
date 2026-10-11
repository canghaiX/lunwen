"""Optional local Transformers adapter for structured revision extraction."""

from __future__ import annotations

import json
from typing import Optional

from revimpact.data.schema import Revision
from revimpact.revision.extract import parse_revision_json


class LocalLLMRevisionExtractor:
    """Lazy-load a local causal LM and request schema-constrained JSON."""

    def __init__(self, model_path: str, device: str = "auto", max_new_tokens: int = 384):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
        kwargs = {"trust_remote_code": True, "local_files_only": True}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)

    def extract(self, old_text: str, new_text: str, source_id: Optional[str] = None) -> Revision:
        if self._model is None:
            self._load()
        prompt = (
            "Extract only valid JSON with keys predicate, old_value, new_value, condition, "
            "revision_type, source_id. The condition must contain slots.\n"
            f"OLD:\n{old_text}\nNEW:\n{new_text}\nJSON:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        revision = parse_revision_json(text)
        if source_id and not revision.source_id:
            revision.source_id = source_id
        return revision


def revision_prompt(old_text: str, new_text: str) -> str:
    return json.dumps({"old": old_text, "new": new_text}, ensure_ascii=False)
