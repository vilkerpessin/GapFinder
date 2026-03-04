"""Modal app for Qwen 2.5-7B inference.

Deploy with: modal deploy modal_inference.py
After deploy, set the generated URL as MODAL_INFERENCE_URL in the HF Space secrets.
"""

from pydantic import BaseModel
import modal

app = modal.App("gapfinder-inference")

_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.45.0", "torch", "accelerate", "fastapi[standard]")
)
_MODEL_VOLUME = modal.Volume.from_name("gapfinder-models", create_if_missing=True)

_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
_MODEL_CACHE = "/models"


class _InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 2048


@app.cls(
    gpu="A10G",
    image=_IMAGE,
    volumes={_MODEL_CACHE: _MODEL_VOLUME},
    timeout=300,
    scaledown_window=60,
)
class QwenModel:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, cache_dir=_MODEL_CACHE)
        self._model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=_MODEL_CACHE,
        )

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: _InferenceRequest) -> dict:
        import torch

        messages = [
            {"role": "system", "content": "You are an expert academic research reviewer. Identify distinct research gaps and return them as a JSON array."},
            {"role": "user", "content": request.prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return {"text": self._tokenizer.decode(generated, skip_special_tokens=True)}
