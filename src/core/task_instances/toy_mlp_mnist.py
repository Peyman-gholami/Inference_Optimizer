"""
Toy MLP (3 epochs trained) on MNIST with Stein gradient oracle.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ..accuracy_fn import CompressedModel
from ..stein_oracles import MLP, get_flat_params
from ..task import InferenceTask
from ..task_callables import InferenceTaskCallables

# Bytes per compressed activation tensor on each link (float32), for a batch of B
# images. Cuts follow ReLU outputs: (B, 128) then (B, 64). Match ResNet trace
# convention (B = 100).
_MLP_COMM_BATCH = 100
_MLP_A_BYTES = {
    0: float(_MLP_COMM_BATCH * 128 * 4),
    1: float(_MLP_COMM_BATCH * 64 * 4),
}


def setup_model_and_callables():
    """
    Trains a small MLP on MNIST and returns (api, flat_theta) for building tasks.
    api.accuracy_callable(eta_np) -> float, api.gradient_callable(eta_np) -> np.ndarray
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=128,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=tf),
        batch_size=256,
        shuffle=False,
    )

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training MLP (3 epochs)...")
    for _ in range(3):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()

    flat_theta = get_flat_params(model)

    eta_init = torch.ones(2)
    cm = CompressedModel(model, eta_init)
    api = InferenceTaskCallables(
        cm,
        test_loader,
        sigma=0.05,
        N=50,
        n_samples=512,
        flat_theta=None,
        device=device,
    )

    return api, flat_theta


def make_task(
    task_id: int,
    api: InferenceTaskCallables,
    w_k: float = 1.0,
    R_k: float = 10.0,
) -> InferenceTask:
    return InferenceTask(
        task_id=task_id,
        b_k=0,
        L_k=3,
        tau={0: 0.01, 1: 0.01, 2: 0.01},
        a=dict(_MLP_A_BYTES), #10 , 20
        eta_min={0: 0.01, 1: 0.01},
        R_k_callable=lambda t, _R=R_k: float(_R),
        w_k=w_k,
        accuracy_callable=api.accuracy_callable,
        accuracy_callable_true=api.accuracy_callable_true,
        gradient_callable=api.gradient_callable,
    )
