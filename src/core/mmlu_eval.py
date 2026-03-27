from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _build_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    choices_txt = "\n".join([f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)])
    return f"Question: {question}\n{choices_txt}\nAnswer:"


@dataclass(frozen=True)
class MMLUEvalConfig:
    subjects: Sequence[str]
    # For fast (used inside gradient oracle) and full (used for plotting).
    samples_per_subject: int
    seed: int = 0
    # Max token length for truncation (kept as safety).
    max_length: Optional[int] = None
    batch_size: int = 8


class MMLUEvaluator:
    """
    Accuracy evaluator for multiple-choice MMLU in HF causal LM format.

    It prebuilds a fixed subset of (prompt, answer_index) pairs so repeated
    evaluations are deterministic for the same eta.
    """

    def __init__(self, *, tokenizer, config: MMLUEvalConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

        # Lazy import so base repo tests don't require HF datasets.
        from datasets import load_dataset  # type: ignore

        rng = np.random.default_rng(config.seed)

        prompts: List[str] = []
        answers: List[int] = []
        for subj in config.subjects:
            ds = load_dataset("cais/mmlu", subj, split="test")
            n = min(len(ds), int(config.samples_per_subject))
            idxs = rng.choice(len(ds), size=n, replace=False).tolist()
            for i in idxs:
                item = ds[int(i)]
                q = item["question"]
                choices = item["choices"]
                answer_letter_idx = int(item["answer"])  # A=0, ..., D=3
                prompts.append(_build_mmlu_prompt(q, choices))
                answers.append(answer_letter_idx)

        if not prompts:
            raise ValueError("MMLUEvaluator got an empty prompt set. Check subjects/samples_per_subject.")

        self.prompts = prompts
        self.answers = torch.tensor(answers, dtype=torch.long)

        # Candidate tokens for A/B/C/D.
        candidates = ["A", "B", "C", "D"]
        cand_ids: List[int] = []
        for c in candidates:
            # Many tokenizers treat answer candidates with a leading space.
            ids = tokenizer.encode(" " + c, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(c, add_special_tokens=False)
            cand_ids.append(int(ids[-1]))
        self.cand_ids = torch.tensor(cand_ids, dtype=torch.long)

        # Tokenize once (dynamic padding), keep on CPU; move per batch.
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_length,
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    @torch.no_grad()
    def accuracy(self, model, *, device: torch.device) -> float:
        """
        Runs the model on all pre-tokenized prompts and computes accuracy.
        """
        model.eval()
        self.cand_ids = self.cand_ids.to(device)
        answers = self.answers.to(device)

        n = int(self.input_ids.size(0))
        correct = 0

        bs = int(self.config.batch_size)
        for start in range(0, n, bs):
            end = min(n, start + bs)
            input_ids = self.input_ids[start:end].to(device)
            attention_mask = self.attention_mask[start:end].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            logits = outputs.logits  # (B, S, V)
            last_positions = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(input_ids.size(0), device=device)
            last_logits = logits[batch_idx, last_positions]  # (B, V)

            cand_logits = last_logits.index_select(dim=1, index=self.cand_ids)  # (B, 4)
            preds = cand_logits.argmax(dim=1)  # (B,)

            correct += int((preds == answers[start:end]).sum().item())

        return float(correct) / float(n)

