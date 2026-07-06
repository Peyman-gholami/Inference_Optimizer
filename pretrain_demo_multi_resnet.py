"""
Demo: ASYMMETRIC multi-task decoupled compression-aware pretraining on CIFAR-10,
with real per-model activation sizes and a PHYSICAL (task-independent) channel P_c.

Two pretrained CIFAR ResNets of different depth share the 2 cut-point links.
  * Real a_i are computed from each model and cached to activation_sizes.json.
  * Per-task throughput R_k differs (default [1.0, 1.5]) and weights w_k differ.
  * The controller water-fills the joint eta*(c) from PHYSICAL per-link capacities c,
    which each task normalizes by c_i/(R_k a_{k,i}) -- so induced compression differs
    across tasks/links (no scale-free reduction). c is sampled log-uniform or uniform.
  * Each task is fine-tuned (decoupled) at its eta*_k.

Run (both P_c kinds):
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python pretrain_demo_multi_resnet.py --pc loguniform
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python pretrain_demo_multi_resnet.py --pc uniform
"""

import argparse

import numpy as np
import torch

from src.pretraining.resnet_cifar import (
    select_device, get_cifar10_loaders, load_pretrained_cifar_resnet,
    make_compressed_resnet, load_test_tensors,
)
from src.pretraining.accuracy_profile import LiveAccuracyProfile
from src.pretraining.compression_aware import ChannelSampler
from src.pretraining.multitask_live import (
    build_tasks_with_profiles, decoupled_pretrain_live, deployed_accuracy_multi_resnet,
)
from src.optimizers.csi_aware import CSIAwareMultiTaskOptimizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="cifar10_resnet56,cifar10_resnet20")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--refit_every", type=int, default=100)
    p.add_argument("--meas_subset", type=int, default=2000)
    p.add_argument("--n_channels", type=int, default=24)
    p.add_argument("--w", default="1.5,1.0")
    p.add_argument("--R", default="1.0,1.5")
    p.add_argument("--pc", choices=["loguniform", "uniform"], default="loguniform")
    p.add_argument("--eta_min", type=float, default=0.1)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = select_device(args.device)
    model_names = args.models.split(",")
    K = len(model_names)
    w_list = [float(x) for x in args.w.split(",")]
    R_list = [float(x) for x in args.R.split(",")]

    train_loader, test_loader = get_cifar10_loaders()
    x_test, y_test = load_test_tensors(test_loader)
    x_meas, y_meas = x_test[:args.meas_subset], y_test[:args.meas_subset]

    def build_models():
        cms, a_list = [], []
        for m in model_names:
            cm, a = make_compressed_resnet(load_pretrained_cifar_resnet(m).to(device), model_name=m)
            cms.append(cm.to(device)); a_list.append(a)
        return cms, a_list

    cms_ca, a_list = build_models()
    d = cms_ca[0].d
    eta_min = np.full(d, args.eta_min)

    # PHYSICAL, task-independent capacity range: scale chosen from a global reference
    # (K x geometric-mean of R_k a_{k,i}) so eta* lands in a usable band; the SAME range
    # is used for both links, so per-task / per-link eta* differ (honest heterogeneity).
    demands = np.array([R_list[k] * a_list[k][i] for k in range(K) for i in range(d)])
    ref = K * float(np.exp(np.mean(np.log(demands))))
    c_min, c_max = args.eta_min * ref, 1.2 * ref

    def train_sampler():
        return ChannelSampler(d, c_min, c_max, seed=args.seed, kind=args.pc)

    def eval_sampler():   # same channels for both arms (paired on c)
        return ChannelSampler(d, c_min, c_max, seed=args.seed + 123, kind=args.pc)

    print(f"device: {device}   P_c: {args.pc}   w={w_list}   R={R_list}")
    for k, m in enumerate(model_names):
        print(f"  {m}: a={a_list[k].astype(int)}  R={R_list[k]}  w={w_list[k]}")
    print(f"  physical c range [{c_min:.0f}, {c_max:.0f}] (same for both links)")

    def make_controller(cms):
        profs = [LiveAccuracyProfile(cm, x_meas, y_meas, d, device) for cm in cms]
        tasks, M = build_tasks_with_profiles(profs, a_list, eta_min, R_list, w_list)
        return CSIAwareMultiTaskOptimizer(M, tasks), profs

    print("\n[1] Compression-aware decoupled fine-tuning (live A_k) ...")
    opt_ca, profs_ca = make_controller(cms_ca)
    decoupled_pretrain_live(cms_ca, train_loader, opt_ca, profs_ca, sampler=train_sampler(),
                            steps=args.steps, lr=args.lr, device=device, refit_every=args.refit_every)

    print("\n[2] Off-the-shelf baseline ...")
    cms_base, _ = build_models()
    opt_base, profs_base = make_controller(cms_base)

    print("\n[3] Evaluating both under shared-bandwidth P_c (paired channels) ...")
    ev = dict(n_channels=args.n_channels, device=device)
    res_ca = deployed_accuracy_multi_resnet(cms_ca, x_test, y_test, opt_ca, profs_ca,
                                            sampler=eval_sampler(), **ev)
    res_base = deployed_accuracy_multi_resnet(cms_base, x_test, y_test, opt_base, profs_base,
                                              sampler=eval_sampler(), **ev)

    print("\n" + "=" * 82)
    print(f"P_c = {args.pc}    (R={R_list}, w={w_list})")
    print(f"{'arm / task':<28}{'clean':>10}{'deployed E_c[acc]':>22}{'mean eta*':>18}")
    print("-" * 82)
    def rows(tag, r):
        for k, m in enumerate(model_names):
            print(f"{tag+' '+m:<28}{r['clean_per_task'][k]:>10.4f}"
                  f"{r['deployed_mean_per_task'][k]:>13.4f} ± {r['deployed_std_per_task'][k]:.3f}"
                  f"{str(np.round(r['mean_eta_per_task'][k], 2)):>18}")
    rows("off-shelf", res_base)
    rows("comp-aware", res_ca)
    print("-" * 82)
    agg_b = np.mean(res_base["deployed_mean_per_task"]); agg_c = np.mean(res_ca["deployed_mean_per_task"])
    print(f"aggregate deployed: off-shelf={agg_b:.4f}  comp-aware={agg_c:.4f}  gain={agg_c-agg_b:+.4f}")
    print("=" * 82)


if __name__ == "__main__":
    main()
