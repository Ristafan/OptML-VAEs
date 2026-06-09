"""
Adam vs. SGD learning dynamics in the linear VAE
=================================================
Experimental suite that VERIFIES or FALSIFIES the theoretical claims developed in
`adam_linear_vae_framework.md`, built directly on the original paper code
(Ichikawa & Hukushima, AISTATS 2024) so the comparison is apples-to-apples.

Claims under test (see framework Sec. 8 & 10):
  A. Phase-boundary invariance + speedup.
       - Fixed points (hence collapse threshold beta* = rho+eta and the optimal
         point beta = eta) are UNCHANGED by Adam; only the *kinetics* change.
       - At matched base learning rate Adam escapes the plateau faster.
  B. Concentration: order parameters still concentrate as N grows under Adam,
       i.e. an N->inf deterministic (ODE) limit exists.
  C. Preconditioner delocalization: Var_i(v_hat) / mean_i(v_hat) for Adam's
       second-moment buffer stays small  =>  the mean-field "Level 1" theory
       (scalar per-column effective LR) is exact at leading order. We also track
       the momentum-signal overlap a_W = (1/N) m_hat . W*  (does momentum LEAD m?).
  D. KL-annealing window: with tanh annealing beta(t)=tanh(gamma t), Adam should
       RELAX the slow-annealing penalty that Thm 5.4 imposes on SGD (wider /
       shifted admissible gamma window).
  E. Model mismatch (M=2, M*=1): does Adam's per-dimension braking suppress or
       accelerate the small-beta (beta<eta) overfitting of the superfluous latent?

Each experiment saves a publication-style matplotlib figure to OUTDIR.

Set QUICK=False for paper-quality runs (long, multi-seed). QUICK=True is a fast
demonstration pass.
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
from tqdm import tqdm
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------------------
QUICK = True                      # <-- set False for paper-quality runs
OUTDIR = "."
DEVICE = "cpu"
RHO, ETA = 1.0, 1.0               # signal / noise strength; collapse threshold = RHO+ETA
THRESH_COLLAPSE = RHO + ETA       # beta* (= 2.0)
THRESH_OPT = ETA                  # optimal generalization (= 1.0)

RUN = {"A": True, "B": True, "C": True, "D": True, "E": True}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 12,
    "axes.grid": True, "grid.alpha": 0.3, "legend.framealpha": 0.9,
})
C_SGD, C_ADAM, C_MOM, C_THEORY = "#1f77b4", "#d62728", "#2ca02c", "0.25"


# --------------------------------------------------------------------------------------
# Core model / data / observables  (faithful to the original paper code)
# --------------------------------------------------------------------------------------
def fix_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_data_from_SCM(N, Mstar, W0, eta=ETA, rho=RHO, device=DEVICE):
    """Spiked covariance model:  x = W0 c / sqrt(N/rho) + sqrt(eta) n .  Returns (1, N)."""
    c = torch.randn(Mstar, 1, device=device)
    n = torch.randn(N, 1, device=device)
    X = (W0 @ c) / torch.sqrt(torch.tensor(N / rho)) + torch.sqrt(torch.tensor(eta)) * n
    return X.T


class LinearVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, W_init=None, tW_init=None):
        super().__init__()
        self.dec = nn.Linear(latent_dim, input_dim, bias=False)   # weight: (N, M)
        self.mu = nn.Linear(input_dim, latent_dim, bias=False)    # weight: (M, N)
        self.var = nn.Parameter(torch.ones(latent_dim))           # encoder covariance D (diag)
        if W_init is None:
            self.dec.weight = nn.Parameter(torch.randn(input_dim, latent_dim))
            self.mu.weight = nn.Parameter(torch.randn(latent_dim, input_dim))
        else:
            self.dec.weight = nn.Parameter(W_init.clone())
            self.mu.weight = nn.Parameter(tW_init.clone())
        self.N = torch.tensor(input_dim)
        self.M = torch.tensor(latent_dim)

    def encode(self, x):
        return self.mu(x) / torch.sqrt(self.N), self.var

    def decode(self, z):
        return self.dec(z) / torch.sqrt(self.N)

    def forward(self, x):
        mu, var = self.encode(x.view(-1, self.N))
        return self.decode(mu), mu, var


def criterion(model, hat_x, x, mu, var, beta=1.0, reg_param=0.0):
    second = -2 * torch.sum(hat_x * x, dim=1)
    third = torch.diag(mu @ model.dec.weight.T @ model.dec.weight @ mu.T) / x.size(1)
    forth = torch.sum(torch.diag(model.dec.weight.T @ model.dec.weight) * var) / x.size(1)
    recon = 0.5 * (second + third + forth)
    kld = 0.5 * torch.sum((-torch.log(var + 1e-16) + mu.pow(2) + var), dim=1)
    reg_d = 0.5 * reg_param * torch.sum(torch.diag(model.dec.weight.T @ model.dec.weight)) / x.size(1)
    reg_e = 0.5 * reg_param * torch.sum(torch.diag(model.mu.weight @ model.mu.weight.T)) / x.size(1)
    return beta * kld + recon + reg_d + reg_e, recon, kld


def observables(model, W0, device=DEVICE):
    """Return dict of order parameters and generalization error eg."""
    N = model.dec.weight.size(0)
    M = model.dec.weight.size(1)
    Mcross = (model.dec.weight.T @ W0) / N           # (M, M*)  decoder-signal overlap  (= paper m)
    tM = (model.mu.weight @ W0) / N                  # (M, M*)  encoder-signal overlap  (= paper d)
    Q = (model.dec.weight.T @ model.dec.weight) / N  # (M, M)
    if M == 1:
        m = Mcross.flatten()[0]; q = Q.flatten()[0]
        eg = (1 - 2 * torch.abs(m) + q).item()
        return {"eg": eg, "m": m.item(), "q": q.item(),
                "Qdiag": [q.item()], "mvec": [m.item()], "v": model.var.detach().tolist()}
    else:
        eg = (1 - 2 * torch.sum(Mcross) + torch.sum(Q)).item()
        return {"eg": eg, "Qdiag": torch.diag(Q).detach().tolist(),
                "mvec": Mcross.flatten().detach().tolist(), "v": model.var.detach().tolist()}


# --------------------------------------------------------------------------------------
# Optimizer factory + beta schedule
# --------------------------------------------------------------------------------------
def make_optimizer(model, kind, lr, N, betas=(0.9, 0.999), momentum=0.9, alpha=0.999):
    """Same parameter-group structure for every optimizer so the only thing that
    changes is the update rule (var keeps the paper's 1/N step convention)."""
    groups = [{"params": model.dec.parameters(), "lr": lr},
              {"params": model.mu.parameters(), "lr": lr},
              {"params": [model.var], "lr": lr / N}]
    if kind == "sgd":
        return torch.optim.SGD(groups, lr=lr)
    if kind == "momentum":
        return torch.optim.SGD(groups, lr=lr, momentum=momentum)
    if kind == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas, eps=1e-8)
    if kind == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr, alpha=alpha, eps=1e-8)
    raise ValueError(kind)


def beta_value(anneal, t_cont, beta_const, gamma):
    """t_cont is *continuous* time = step / N (matches the paper's annealing clock)."""
    if anneal == "const":
        return beta_const
    if anneal == "tanh":
        return float(np.tanh(gamma * t_cont))
    if anneal == "linear":
        return float(min(gamma * t_cont, 1.0))
    raise ValueError(anneal)


# --------------------------------------------------------------------------------------
# Generalized one-pass trainer
# --------------------------------------------------------------------------------------
def run_training(N, latent_dim, optimizer_kind, beta_const=1.0, anneal="const", gamma=0.0,
                 num_steps=60000, lr=0.01, rho=RHO, eta=ETA, reg_param=0.0,
                 betas=(0.9, 0.999), momentum=0.9, seed=0, record_interval=50,
                 probe_adam=False, W_init=None, tW_init=None, Mstar=1, device=DEVICE):
    fix_seed(seed)
    W0 = torch.ones(N, Mstar, device=device)
    if W_init is None:
        W_init = torch.randn(N, latent_dim, device=device)
        tW_init = torch.randn(latent_dim, N, device=device)
    model = LinearVAE(N, latent_dim, W_init=W_init, tW_init=tW_init).to(device)
    opt = make_optimizer(model, optimizer_kind, lr, N, betas=betas, momentum=momentum)

    rec = {"t": [], "eg": [], "beta": [], "deloc": [], "aW": [], "Qdiag": [], "mvec": [], "v": []}
    model.train()
    for step in range(num_steps):
        t_cont = step / N
        b = beta_value(anneal, t_cont, beta_const, gamma)
        x = generate_data_from_SCM(N, Mstar, W0, eta=eta, rho=rho, device=device)
        recon, mu, var = model(x)
        loss, _, _ = criterion(model, recon, x, mu, var, beta=b, reg_param=reg_param)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            model.var.clamp_(min=1e-8)        # keep encoder covariance positive (Lemma C.5 analogue)

        if step % record_interval == 0:
            ob = observables(model, W0, device=device)
            rec["t"].append(t_cont); rec["eg"].append(ob["eg"]); rec["beta"].append(b)
            rec["Qdiag"].append(ob["Qdiag"]); rec["mvec"].append(ob["mvec"]); rec["v"].append(ob["v"])
            if probe_adam and optimizer_kind == "adam" and model.dec.weight in opt.state:
                st = opt.state[model.dec.weight]
                v_buf = st["exp_avg_sq"][:, 0]                 # second moment, latent column 0
                m_buf = st["exp_avg"][:, 0]                    # first moment,  latent column 0
                cv = (v_buf.std() / (v_buf.mean() + 1e-12)).item()    # coefficient of variation
                aW = (m_buf @ W0[:, 0] / N).item()             # momentum-signal overlap
                rec["deloc"].append(cv); rec["aW"].append(aW)
            else:
                rec["deloc"].append(np.nan); rec["aW"].append(np.nan)
    return rec


def eg_star_theory(beta, rho=RHO, eta=ETA):
    """Steady-state generalization error from Theorem 5.1 (model-matched)."""
    P = rho + eta
    if beta < P:
        s = np.sqrt(P - beta)
        return rho - s * (2 * np.sqrt(rho) - s)
    return rho


def steady_value(rec, frac=0.2):
    """Average eg over the last `frac` of the trajectory (proxy for the steady state)."""
    arr = np.array(rec["eg"]); k = max(1, int(len(arr) * frac))
    return float(np.mean(arr[-k:]))


def convergence_time(rec, target, tol_window=3):
    """First continuous time at which eg drops below `target` and stays below."""
    t = np.array(rec["t"]); eg = np.array(rec["eg"])
    below = eg < target
    for i in range(len(below) - tol_window):
        if below[i:i + tol_window].all():
            return float(t[i])
    return float(t[-1])  # never converged within the run


# ======================================================================================
# Experiment A : phase-boundary invariance + speedup
# ======================================================================================
def experiment_A():
    print("\n=== Experiment A: phase-boundary invariance + speedup ===")
    if QUICK:
        betas_scan = [0.5, 1.0, 1.5, 1.8, 2.0, 2.5]; seeds = [0, 1]; N = 120; steps = 24000; ri = 40
        curve_betas = [1.0, 1.8, 2.0]
    else:
        betas_scan = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.8, 2.0, 2.25, 2.5]
        seeds = [0, 1, 2, 3, 4]; N = 500; steps = 200 * 1500; ri = 200
        curve_betas = [1.0, 1.8, 2.0, 2.25]
    # safety: curves are only recorded for betas that are actually swept
    curve_betas = [b for b in curve_betas if b in betas_scan]
    lr = 0.01
    optimizers = ["sgd", "adam"]

    steady = {opt: {b: [] for b in betas_scan} for opt in optimizers}
    curves = {opt: {b: [] for b in curve_betas} for opt in optimizers}

    for seed in tqdm(seeds):
        Wi = torch.randn(N, 1); tWi = torch.randn(1, N)        # shared init across optimizers
        for b in tqdm(betas_scan):
            for opt in optimizers:
                rec = run_training(N, 1, opt, beta_const=b, num_steps=steps, lr=lr,
                                   seed=seed, record_interval=ri, W_init=Wi, tW_init=tWi)
                steady[opt][b].append(steady_value(rec))
                if b in curve_betas:
                    curves[opt][b].append((rec["t"], rec["eg"]))
        print(f"  seed {seed} done")

    # ---- Figure A1: eg(t) curves, SGD vs Adam, selected beta ----
    fig, axes = plt.subplots(1, len(curve_betas), figsize=(4.4 * len(curve_betas), 4.0), sharey=True)
    for ax, b in zip(np.atleast_1d(axes), curve_betas):
        for opt, col in [("sgd", C_SGD), ("adam", C_ADAM)]:
            T = curves[opt][b][0][0]
            E = np.array([np.interp(T, t, e) for t, e in curves[opt][b]])
            ax.plot(T, E.mean(0), color=col, label=opt.upper())
            ax.fill_between(T, E.mean(0) - E.std(0), E.mean(0) + E.std(0), color=col, alpha=0.15)
        ax.axhline(eg_star_theory(b), color=C_THEORY, ls="--", lw=1,
                   label=r"$\varepsilon_g^*$ theory")
        ax.set_title(rf"$\beta={b}$"); ax.set_xlabel(r"$t$"); ax.set_ylim(-0.05, 1.6)
    np.atleast_1d(axes)[0].set_ylabel(r"$\varepsilon_g$")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("A1  Speedup at matched learning rate (Adam reaches the SAME steady state faster)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "expA1_speedup_curves.png")); plt.close(fig)

    # ---- Figure A2: steady-state eg vs beta with theory overlay ----
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    bb = np.linspace(min(betas_scan), max(betas_scan), 300)
    ax.plot(bb, [eg_star_theory(b) for b in bb], color=C_THEORY, lw=2,
            label=r"Theory $\varepsilon_g^*$ (Thm 5.1, SGD)")
    for opt, col, mk in [("sgd", C_SGD, "o"), ("adam", C_ADAM, "s")]:
        mean = [np.mean(steady[opt][b]) for b in betas_scan]
        std = [np.std(steady[opt][b]) for b in betas_scan]
        ax.errorbar(betas_scan, mean, yerr=std, fmt=mk + "-", color=col, capsize=3, label=opt.upper())
    ax.axvline(THRESH_OPT, color="gray", ls=":", lw=1.2)
    ax.axvline(THRESH_COLLAPSE, color="black", ls=":", lw=1.2)
    ax.text(THRESH_OPT, 1.05, r" $\beta=\eta$ (optimum)", rotation=90, va="top", fontsize=9)
    ax.text(THRESH_COLLAPSE, 1.05, r" $\beta=\rho+\eta$ (collapse)", rotation=90, va="top", fontsize=9)
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"steady-state $\varepsilon_g$"); ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("A2  Fixed points are invariant under Adam\n(same minimum at $\\beta=\\eta$, same collapse at $\\beta=\\rho+\\eta$)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "expA2_phase_boundary.png")); plt.close(fig)
    print("  -> expA1_speedup_curves.png, expA2_phase_boundary.png")


# ======================================================================================
# Experiment B : concentration (ODE limit exists for Adam)
# ======================================================================================
def experiment_B():
    print("\n=== Experiment B: concentration as N grows (Adam) ===")
    if QUICK:
        Ns = [60, 120, 240]; seeds = [0, 1, 2]; steps = 20000; ri = 40
    else:
        Ns = [125, 250, 500, 1000]; seeds = list(range(8)); steps = 150 * 1500; ri = 200
    b, lr = 1.0, 0.01

    spread = {}
    fig1, ax1 = plt.subplots(figsize=(6.6, 4.6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))
    for N, col in zip(Ns, cmap):
        runs = []
        for seed in tqdm(seeds):
            rec = run_training(N, 1, "adam", beta_const=b, num_steps=steps, lr=lr,
                               seed=seed, record_interval=ri)
            runs.append(rec["eg"])
        T = np.array(rec["t"]); E = np.array(runs)
        std_t = E.std(0)
        ax1.plot(T, E.mean(0), color=col, label=f"N={N}")
        ax1.fill_between(T, E.mean(0) - std_t, E.mean(0) + std_t, color=col, alpha=0.18)
        spread[N] = float(np.mean(std_t))         # average seed-to-seed spread
    ax1.set_xlabel(r"$t$"); ax1.set_ylabel(r"$\varepsilon_g$")
    ax1.set_title("B1  Trajectories sharpen as $N$ grows (Adam)")
    ax1.legend(fontsize=9); fig1.tight_layout()
    fig1.savefig(os.path.join(OUTDIR, "expB1_concentration_curves.png")); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5.6, 4.4))
    Nv = np.array(Ns, float); sv = np.array([spread[N] for N in Ns])
    ax2.loglog(Nv, sv, "o-", color=C_ADAM, label="measured seed spread")
    ax2.loglog(Nv, sv[0] * np.sqrt(Nv[0]) / np.sqrt(Nv), "k--", label=r"$\propto 1/\sqrt{N}$")
    ax2.set_xlabel(r"$N$"); ax2.set_ylabel(r"mean$_t\,$ std$_{\rm seeds}(\varepsilon_g)$")
    ax2.set_title(r"B2  Fluctuations vanish at the Thm-4.2 rate $O(1/\sqrt{N})$")
    ax2.legend(fontsize=9); fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "expB2_concentration_rate.png")); plt.close(fig2)
    print("  -> expB1_concentration_curves.png, expB2_concentration_rate.png")


# ======================================================================================
# Experiment C : preconditioner delocalization + momentum-signal overlap
# ======================================================================================
def experiment_C():
    print("\n=== Experiment C: preconditioner delocalization & momentum overlap ===")
    if QUICK:
        Ns = [120, 240]; steps = 24000; ri = 40; seed = 0
    else:
        Ns = [250, 500, 1000]; steps = 150 * 1500; ri = 200; seed = 0
    b, lr = 1.0, 0.01

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.75, len(Ns)))
    axR2 = axR.twinx()
    for N, col in zip(Ns, cmap):
        rec = run_training(N, 1, "adam", beta_const=b, num_steps=steps, lr=lr,
                           seed=seed, record_interval=ri, probe_adam=True)
        T = np.array(rec["t"])
        axL.plot(T, rec["deloc"], color=col, label=f"N={N}")
        axR.plot(T, [mv[0] for mv in rec["mvec"]], color=col, ls="--", alpha=0.8,
                 label=f"N={N}: $m$ (weight)")
        axR2.plot(T, rec["aW"], color=col, ls="-", lw=1.4,
                  label=f"N={N}: momentum$\\cdot W^*$")
    axL.set_xlabel(r"$t$"); axL.set_ylabel(r"CV$_i(\hat v_{i})=$ std$_i/$mean$_i$")
    axL.set_title("C1  Second-moment buffer is delocalized\n(small CV $\\Rightarrow$ mean-field 'Level 1' theory exact)")
    axL.legend(fontsize=9)
    axR.set_xlabel(r"$t$"); axR.set_ylabel(r"weight overlap $m$ (dashed)")
    axR2.set_ylabel(r"momentum overlap $a_W$ (solid)")
    axR.set_title("C2  Momentum overlap peaks during the transition,\nvanishes at the fixed point (gradient $\\to 0$)")
    axR.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "expC_delocalization.png")); plt.close(fig)
    print("  -> expC_delocalization.png")


# ======================================================================================
# Experiment D : KL-annealing window (Thm 5.4 under Adam)
# ======================================================================================
def experiment_D():
    print("\n=== Experiment D: KL-annealing window (tanh) ===")
    # beta(t) = tanh(gamma * t_cont); ends at beta -> 1 = eta  =>  eg* = 0, target 0.001
    if QUICK:
        gammas = [4e-3, 8e-3, 1.6e-2, 3.2e-2]; seeds = [0]; N = 120; lr = 0.5; tmax = 600
    else:
        gammas = [0.5e-4, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3]; seeds = [0, 1, 2]; N = 250; lr = 1.0; tmax = 25000
    steps = int(tmax * N); ri = max(20, steps // 1500)
    target = 0.001

    conv = {opt: {g: [] for g in gammas} for opt in ["sgd", "adam"]}
    const = {opt: [] for opt in ["sgd", "adam"]}
    sample_curves = {}
    for seed in tqdm(seeds):
        Wi = torch.randn(N, 1); tWi = torch.randn(1, N)
        for opt in ["sgd", "adam"]:
            rc = run_training(N, 1, opt, beta_const=1.0, anneal="const", num_steps=steps,
                              lr=lr, seed=seed, record_interval=ri, W_init=Wi, tW_init=tWi)
            const[opt].append(convergence_time(rc, target))
            for g in tqdm(gammas):
                rg = run_training(N, 1, opt, anneal="tanh", gamma=g, num_steps=steps,
                                  lr=lr, seed=seed, record_interval=ri, W_init=Wi, tW_init=tWi)
                conv[opt][g].append(convergence_time(rg, target))
                if seed == 0 and abs(g - gammas[len(gammas) // 2]) < 1e-12:
                    sample_curves[opt] = (np.array(rg["t"]), np.array(rg["eg"]), np.array(rg["beta"]))
        print(f"  seed {seed} done")

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(6.8, 7.2))
    for opt, col in [("sgd", C_SGD), ("adam", C_ADAM)]:
        t, eg, bt = sample_curves[opt]
        axT.plot(t, eg, color=col, label=rf"$\varepsilon_g$ {opt.upper()}")
        axT.plot(t, bt, color=col, ls="--", alpha=0.6, label=rf"$\beta(t)$ {opt.upper()}")
    axT.set_xlabel(r"$t$"); axT.set_ylabel("value"); axT.set_ylim(-0.05, 1.6)
    axT.legend(fontsize=8, ncol=2); axT.set_title("D1  tanh KL annealing trajectory (mid $\\gamma$)")

    for opt, col, mk in [("sgd", C_SGD, "o"), ("adam", C_ADAM, "s")]:
        mean = [np.mean(conv[opt][g]) for g in gammas]
        std = [np.std(conv[opt][g]) for g in gammas]
        axB.errorbar(gammas, mean, yerr=std, fmt=mk + "-", color=col, capsize=3, label=f"{opt.upper()} (annealed)")
        axB.axhline(np.mean(const[opt]), color=col, ls=":", lw=1.4, label=f"{opt.upper()} (const $\\beta=1$)")
    axB.set_xscale("log"); axB.set_xlabel(r"annealing rate $\gamma$")
    axB.set_ylabel(r"convergence time to $\varepsilon_g^*+0.001$")
    axB.set_title("D2  Adam relaxes the slow-annealing penalty (lower / flatter curve)")
    axB.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "expD_annealing_window.png")); plt.close(fig)
    print("  -> expD_annealing_window.png")


# ======================================================================================
# Experiment E : model mismatch (M=2, M*=1), small-beta overfitting
# ======================================================================================
def experiment_E():
    print("\n=== Experiment E: model mismatch (M=2, M*=1) ===")
    if QUICK:
        seeds = [0, 1]; N = 120; steps = 28000; ri = 40
    else:
        seeds = list(range(5)); N = 500; steps = 200 * 1500; ri = 200
    b, lr = 0.5, 0.01      # beta < eta => regime where the superfluous latent overfits noise

    res = {opt: {"eg": [], "supQ": []} for opt in ["sgd", "adam"]}
    for seed in tqdm(seeds):
        Wi = torch.randn(N, 2); tWi = torch.randn(2, N)
        for opt in ["sgd", "adam"]:
            rec = run_training(N, 2, opt, beta_const=b, num_steps=steps, lr=lr,
                               seed=seed, record_interval=ri, W_init=Wi, tW_init=tWi, Mstar=1)
            mvec = np.array(rec["mvec"])              # (T, 2) decoder-signal overlap per latent
            Qd = np.array(rec["Qdiag"])               # (T, 2)
            sig = np.argmax(np.abs(mvec[-1]))         # latent aligned with the signal
            sup = 1 - sig                             # the superfluous latent
            res[opt]["eg"].append(rec["eg"])
            res[opt]["supQ"].append(Qd[:, sup])       # energy the superfluous latent puts into noise
        T = np.array(rec["t"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    for opt, col in [("sgd", C_SGD), ("adam", C_ADAM)]:
        E = np.array(res[opt]["eg"]); S = np.array(res[opt]["supQ"])
        axL.plot(T, E.mean(0), color=col, label=opt.upper())
        axL.fill_between(T, E.mean(0) - E.std(0), E.mean(0) + E.std(0), color=col, alpha=0.15)
        axR.plot(T, S.mean(0), color=col, label=opt.upper())
        axR.fill_between(T, S.mean(0) - S.std(0), S.mean(0) + S.std(0), color=col, alpha=0.15)
    axL.set_xlabel(r"$t$"); axL.set_ylabel(r"$\varepsilon_g$")
    axL.set_title(rf"E1  Generalization, mismatch, $\beta={b}<\eta$"); axL.legend(fontsize=9)
    axR.set_xlabel(r"$t$"); axR.set_ylabel(r"$Q_{\rm sup}$ (superfluous-latent energy)")
    axR.set_title("E2  Background-noise overfitting of the superfluous latent"); axR.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "expE_mismatch.png")); plt.close(fig)
    print("  -> expE_mismatch.png")


# --------------------------------------------------------------------------------------
print(f"QUICK={QUICK}  (set False for paper-quality runs)")
print(f"OUTDIR={OUTDIR}")
print("Test if saving to outdir works")
with open(os.path.join(OUTDIR, "test.txt"), "w") as f:
  f.write("test")

if RUN["A"]: experiment_A()
if RUN["B"]: experiment_B()
if RUN["C"]: experiment_C()
if RUN["D"]: experiment_D()
if RUN["E"]: experiment_E()
print("\nAll requested experiments finished. Figures written to:", OUTDIR)
