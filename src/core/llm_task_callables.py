from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.core.llm_compression import ActivationCompressionConfig, LLMActivationCompressor
from src.core.mmlu_eval import MMLUEvalConfig, MMLUEvaluator
from src.core.toy_A import grad_oracle


@dataclass(frozen=True)
class LLMBackendConfig:
    model_name: str
    # magnitude | topk_per_token | random | quantization
    activation_strategy: str
    # split mode for choosing hook layer indices: thirds | half
    split_mode: str
    # clamp eta into [eta_min, 1] inside evaluation
    eta_min: float
    # Stein gradient oracle parameters
    grad_sigma: float
    grad_N: int

    # MMLU
    subjects: tuple[str, ...]
    # fast subset (used by accuracy_callable and gradient oracle)
    fast_samples_per_subject: int
    # full subset (used by accuracy_callable_true). For speed you can keep this small.
    full_samples_per_subject: int
    batch_size: int
    eval_seed: int

    # Tokenizer / model
    max_length: Optional[int] = 1024


def _default_subjects() -> tuple[str, ...]:
    # Matches the "selection over topics" idea used in the other repo.
    return (
        "college_computer_science",
        "high_school_mathematics",
        "professional_law",
        "global_facts",
        "miscellaneous",
        "business_ethics",
    )


class LLMInferenceTaskCallables:
    """
    Provides the same interface as `src/core/task_callables.py`:
      - accuracy_callable(eta_np) -> float
      - gradient_callable(eta_np) -> np.ndarray
      - accuracy_callable_true(eta_np) -> float

    Under the hood, it evaluates (compressed-activation) MMLU accuracy.
    """

    def __init__(
        self,
        *,
        config: LLMBackendConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device

        # Lazy imports so unit tests don't require HF deps.
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prefer FP16 on GPU, FP32 on CPU.
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=True,
        )
        # No weight quantization: always load full-precision weights.
        load_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
        self.model.eval()

        # Choose hook layer indices for two cut points (eta has length 2 in this codebase).
        # We keep layer-selection compatible with the other repo's "thirds/half" choice.
        # We rely on activation hooks at those indices.
        from src.core.llm_compression import detect_llm_decoder_layers  # local import

        layers = detect_llm_decoder_layers(self.model)
        n_layers = len(layers)
        if config.split_mode == "thirds":
            cut1 = n_layers // 3
            cut2 = (2 * n_layers) // 3
        elif config.split_mode == "half":
            cut1 = n_layers // 2
            # If we still need two cut points, mirror the first to keep eta length = 2.
            cut2 = min(n_layers - 1, cut1 + 1)
        else:
            raise ValueError(f"Unknown split_mode: {config.split_mode}")

        # Clamp into valid range and avoid duplicates (duplicates are ok but wasteful).
        cut1 = int(np.clip(cut1, 0, n_layers - 1))
        cut2 = int(np.clip(cut2, 0, n_layers - 1))

        activation_cfg = ActivationCompressionConfig(
            strategy=config.activation_strategy,
            layer_indices=[cut1, cut2],
            random_seed=config.eval_seed,
        )
        self.compressor = LLMActivationCompressor(self.model, activation_cfg)

        # Evaluators
        fast_cfg = MMLUEvalConfig(
            subjects=config.subjects,
            samples_per_subject=config.fast_samples_per_subject,
            seed=config.eval_seed,
            max_length=config.max_length,
            batch_size=config.batch_size,
        )
        full_cfg = MMLUEvalConfig(
            subjects=config.subjects,
            samples_per_subject=config.full_samples_per_subject,
            seed=config.eval_seed + 1,
            max_length=config.max_length,
            batch_size=config.batch_size,
        )
        self.evaluator_fast = MMLUEvaluator(tokenizer=self.tokenizer, config=fast_cfg)
        self.evaluator_full = MMLUEvaluator(tokenizer=self.tokenizer, config=full_cfg)

    def _accuracy_for_eta(self, eta_vec: torch.Tensor, *, true: bool) -> float:
        eta_clamped = torch.clamp(eta_vec.to(torch.float32), self.config.eta_min, 1.0)
        self.compressor.set_eta(eta_clamped)

        evaluator = self.evaluator_full if true else self.evaluator_fast
        return evaluator.accuracy(self.model, device=self.device)

    def accuracy_callable(self, eta_vec: np.ndarray) -> float:
        eta = torch.tensor(eta_vec, dtype=torch.float32, device="cpu")
        return float(self._accuracy_for_eta(eta, true=False))

    def gradient_callable(self, eta_vec: np.ndarray) -> np.ndarray:
        eta = torch.tensor(eta_vec, dtype=torch.float32, device="cpu")

        def f(e: torch.Tensor) -> torch.Tensor:
            # Stein oracle expects a torch scalar.
            acc = self._accuracy_for_eta(e, true=False)
            return torch.tensor(acc, dtype=torch.float32, device=e.device)

        g = grad_oracle(
            f,
            eta,
            sigma=float(self.config.grad_sigma),
            N=int(self.config.grad_N),
        )
        return g.detach().cpu().numpy()

    def accuracy_callable_true(self, eta_vec: np.ndarray) -> float:
        eta = torch.tensor(eta_vec, dtype=torch.float32, device="cpu")
        return float(self._accuracy_for_eta(eta, true=True))


def build_llm_backend_from_env(*, eta_min: float) -> LLMInferenceTaskCallables:
    """
    Builds an LLM backend with defaults sourced from environment variables.

    Enable with INFERENCE_OPTIMIZER_BACKEND=llm in main.py / compare scripts.
    See README_LLM.md for env reference.
    """
    model_name = os.environ.get("LLM_MODEL_NAME", "google/gemma-2b")
    activation_strategy = os.environ.get("LLM_ACTIVATION_STRATEGY", "topk_per_token")
    split_mode = os.environ.get("LLM_SPLIT_MODE", "thirds")

    # Stein oracle params: keep small for runtime feasibility.
    grad_sigma = float(os.environ.get("LLM_GRAD_SIGMA", "0.05"))
    grad_N = int(os.environ.get("LLM_GRAD_N", "3"))

    subjects_raw = os.environ.get("MMLU_SUBJECTS", "")
    if subjects_raw.strip():
        subjects = tuple(s.strip() for s in subjects_raw.split(",") if s.strip())
    else:
        subjects = _default_subjects()

    fast_samples_per_subject = int(os.environ.get("MMLU_FAST_SAMPLES_PER_SUBJECT", "5"))
    full_samples_per_subject = int(os.environ.get("MMLU_FULL_SAMPLES_PER_SUBJECT", "10"))

    batch_size = int(os.environ.get("MMLU_BATCH_SIZE", "8"))
    eval_seed = int(os.environ.get("MMLU_SEED", "0"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = LLMBackendConfig(
        model_name=model_name,
        activation_strategy=activation_strategy,
        split_mode=split_mode,
        eta_min=eta_min,
        grad_sigma=grad_sigma,
        grad_N=grad_N,
        subjects=subjects,
        fast_samples_per_subject=fast_samples_per_subject,
        full_samples_per_subject=full_samples_per_subject,
        batch_size=batch_size,
        eval_seed=eval_seed,
    )
    return LLMInferenceTaskCallables(config=cfg, device=device)

