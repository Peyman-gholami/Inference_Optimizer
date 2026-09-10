from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.core.llm_compression import ActivationCompressionConfig, LLMActivationCompressor
from src.core.mmlu_eval import MMLUEvalConfig, MMLUEvaluator
from gradient_oracles import GradientOracle, SteinGradientOracle, \
    AccuracyPredictor, PredictorGradientOracle


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
    # Few-shot evaluation: 0 = zero-shot (default), 5 = standard 5-shot.
    # Examples are drawn from the MMLU dev split (5 per subject).
    n_shot: int = 0

    # Tokenizer / model
    max_length: Optional[int] = 1024
    # Cone projection — applied to gradient before returning
    # project_upward=True: clamp gradient components to >= -cone_margin
    # cone_margin=0.0: hard non-negative projection
    # cone_margin>0.0: soft projection allowing small downward gradients
    project_upward: bool  = False
    cone_margin:    float = 0.0
    # Predictor oracle — if set, use PredictorGradientOracle instead of Stein
    # Path to a saved AccuracyPredictor (.pt file)
    predictor_path: Optional[str] = None


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
            n_shot=config.n_shot,
        )
        full_cfg = MMLUEvalConfig(
            subjects=config.subjects,
            samples_per_subject=config.full_samples_per_subject,
            seed=config.eval_seed + 1,
            max_length=config.max_length,
            batch_size=config.batch_size,
            n_shot=config.n_shot,
        )
        self.evaluator_fast = MMLUEvaluator(tokenizer=self.tokenizer, config=fast_cfg)
        self.evaluator_full = MMLUEvaluator(tokenizer=self.tokenizer, config=full_cfg)

        # Build gradient oracle once — reused every call to gradient_callable
        self._oracle = self._build_gradient_oracle()

    def _accuracy_for_eta(self, eta_vec: torch.Tensor, *, true: bool) -> float:
        eta_clamped = torch.clamp(eta_vec.to(torch.float32), self.config.eta_min, 1.0)
        self.compressor.set_eta(eta_clamped)

        evaluator = self.evaluator_full if true else self.evaluator_fast
        return evaluator.accuracy(self.model, device=self.device)

    def accuracy_callable(self, eta_vec: np.ndarray) -> float:
        eta = torch.tensor(eta_vec, dtype=torch.float32, device="cpu")
        return float(self._accuracy_for_eta(eta, true=False))

    def _build_gradient_oracle(self) -> GradientOracle:
        """
        Build the gradient oracle according to config.

        Priority:
          1. PredictorGradientOracle  — if config.predictor_path is set
          2. SteinGradientOracle      — otherwise (default)

        Both respect project_upward and cone_margin.
        """
        common = dict(
            eta_min        = self.config.eta_min,
            project_upward = self.config.project_upward,
            cone_margin    = self.config.cone_margin,
        )
        if self.config.predictor_path is not None:
            predictor = AccuracyPredictor.load(self.config.predictor_path)
            return PredictorGradientOracle(predictor, **common)
        else:
            return SteinGradientOracle(
                f     = self.accuracy_callable,
                sigma = float(self.config.grad_sigma),
                N     = int(self.config.grad_N),
                **common,
            )

    def gradient_callable(self, eta_vec: np.ndarray) -> np.ndarray:
        return self._oracle(eta_vec)

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

    # Gradient cone projection
    # LLM_GRAD_PROJECT_UPWARD=1  →  project gradient onto upward cone
    # LLM_GRAD_CONE_MARGIN=0.05  →  soft cone margin (0 = hard projection)
    project_upward = os.environ.get("LLM_GRAD_PROJECT_UPWARD", "0") == "1"
    cone_margin    = float(os.environ.get("LLM_GRAD_CONE_MARGIN", "0.0"))

    # Predictor oracle — if set, replaces Stein entirely
    # LLM_GRAD_PREDICTOR_PATH=/path/to/predictor.pt
    predictor_path = os.environ.get("LLM_GRAD_PREDICTOR_PATH", None) or None

    subjects_raw = os.environ.get("MMLU_SUBJECTS", "")
    if subjects_raw.strip():
        subjects = tuple(s.strip() for s in subjects_raw.split(",") if s.strip())
    else:
        subjects = _default_subjects()

    fast_samples_per_subject = int(os.environ.get("MMLU_FAST_SAMPLES_PER_SUBJECT", "5"))
    full_samples_per_subject = int(os.environ.get("MMLU_FULL_SAMPLES_PER_SUBJECT", "10"))

    batch_size = int(os.environ.get("MMLU_BATCH_SIZE", "8"))
    eval_seed  = int(os.environ.get("MMLU_SEED", "0"))
    n_shot     = int(os.environ.get("MMLU_N_SHOT", "5"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = LLMBackendConfig(
        model_name=model_name,
        activation_strategy=activation_strategy,
        split_mode=split_mode,
        eta_min=eta_min,
        grad_sigma=grad_sigma,
        grad_N=grad_N,
        project_upward=project_upward,
        cone_margin=cone_margin,
        predictor_path=predictor_path,
        subjects=subjects,
        fast_samples_per_subject=fast_samples_per_subject,
        full_samples_per_subject=full_samples_per_subject,
        batch_size=batch_size,
        eval_seed=eval_seed,
        n_shot=n_shot,
    )
    return LLMInferenceTaskCallables(config=cfg, device=device)

