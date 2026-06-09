"""
================================================================================
Multi-epoch linear VAE: experimental validation of the Tier-1 / Tier-1.5 theory
================================================================================

Base/reference: OriginalCode.py (Ichikawa & Hukushima, AISTATS 2024), one-pass SGD.
This script does NOT run the original code; it reuses the model/loss structure and
replaces one-pass SGD by **full-batch gradient flow on a FIXED dataset of P samples**
(the multi-epoch limit), to verify or falsify the claims in our working note.

Claims under test (model-matched, M = M* = 1, weight decay lambda = 0):

  Tier-1 closed-form steady state (Result 1 of the note):
      Q11* = nu_+(alpha) - beta                      (decoder self-overlap)
      D*   = beta / nu_+(alpha)                       (encoder variance)
      m11* = sqrt(nu_+ - beta) * o_+                  (overlap with true signal)
      eg   = (sqrt(rho) - m11)^2 + q_perp,   q_perp = (nu_+ - beta)(1 - o_+^2)
  where nu_+(alpha), o_+(alpha) are the Baik-Ben Arous-Pacha (BBP) spike eigenvalue
  and signal overlap of the empirical covariance Sigma_hat = (1/P) sum_mu x_mu x_mu^T,
  at aspect ratio gamma = N/P = 1/alpha:
      theta = rho/eta,  gamma = 1/alpha
      detectability:  alpha_c = eta^2 / rho^2        (theta > sqrt(gamma))
      nu_+   = (rho+eta)(1 + eta/(alpha*rho))         (alpha > alpha_c)
      o_+^2  = (1 - eta^2/(alpha rho^2)) / (1 + eta/(alpha rho))   (alpha > alpha_c), else 0

  Tier-1.5 local stability (this note):
      collapse threshold beta*(alpha) = nu_+(alpha)  -- compare to population rho+eta.

Experiments:
  A. Steady state vs alpha (Q, m, q_perp, eg) at fixed beta  -> fig_expA_steadystate.pdf
  B. Detectability (BBP) transition: m, o_+^2 vs alpha        -> fig_expB_detectability.pdf
  C. Collapse threshold beta*(alpha): Q vs beta per alpha     -> fig_expC_collapse.pdf
  D. Tier-1.5 stability: max Re eig vs beta, both branches    -> fig_expD_stability.pdf
  E. (theory + measured) phase diagram in the (alpha, beta) plane -> fig_expE_phase.pdf

Switch CONFIG = "PAPER" for publication-scale runs (larger N, more seeds/steps).
================================================================================
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


OUTDIR = "."
os.makedirs(OUTDIR, exist_ok=True)
DEVICE = "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))

# ----------------------------------------------------------------------------- #
#  Run configuration (QUICK = fast demo figures; PAPER = publication scale)
# ----------------------------------------------------------------------------- #
CONFIG = "QUICK"
if CONFIG == "QUICK":
    N_DEFAULT   = 200
    LR          = 0.03
    MAX_STEPS   = 5000
    SEEDS_A     = 2
    SEEDS_B     = 3
    SEEDS_C     = 1
else:  # "PAPER"
    N_DEFAULT   = 800
    LR          = 0.01
    MAX_STEPS   = 20000
    SEEDS_A     = 8
    SEEDS_B     = 8
    SEEDS_C     = 5

# Steady-state experiments (A/B/C) only need the gradient-flow FIXED POINT, so we
# use Adam as an efficient solver for the minimizer of the empirical risk. The GF
# *transient* is characterized analytically by the reduced system (Experiment D).
OPTIMIZER = "adam"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "legend.framealpha": 0.9,
})

# ============================================================================= #
#  Model  (reused from OriginalCode.py, M = M* = 1)
# ============================================================================= #
class LinearVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=1, init_scale_W=1.0, init_scale_V=0.3, gen=None):
        super().__init__()
        self.dec = nn.Linear(latent_dim, input_dim, bias=False)   # W  (N, M)
        self.mu  = nn.Linear(input_dim, latent_dim, bias=False)   # V^T (M, N)
        self.var = nn.Parameter(torch.ones(latent_dim))           # D  (diag)
        self.dec.weight = nn.Parameter(init_scale_W * torch.randn(input_dim, latent_dim, generator=gen))
        self.mu.weight  = nn.Parameter(init_scale_V * torch.randn(latent_dim, input_dim, generator=gen))
        self.N = torch.tensor(float(input_dim))
        self.M = latent_dim

    def encode(self, x):
        return (self.mu(x) / torch.sqrt(self.N))          # (P, M)

    def forward(self, x):
        mu = self.encode(x)
        xhat = self.dec(mu) / torch.sqrt(self.N)          # (P, N)
        return xhat, mu, self.var


def empirical_risk(model, X, beta, reg_param=0.0):
    """Full-batch beta-VAE empirical risk (mean over the P samples), matching
    OriginalCode.criterion up to the parameter-independent ||x||^2 constant."""
    N = model.N
    mu = model.encode(X)                                  # (P, M)
    xhat = model.dec(mu) / torch.sqrt(N)                  # (P, N)
    WtW = model.dec.weight.T @ model.dec.weight           # (M, M)
    recon = 0.5 * ((X - xhat) ** 2).sum(dim=1).mean()
    var_term = 0.5 * (torch.diagonal(WtW) * model.var).sum() / N          # E_q variance part
    kl = 0.5 * ((mu ** 2).sum(dim=1).mean()
                + (model.var - torch.log(model.var.clamp_min(1e-12))).sum())
    reg = 0.5 * reg_param * (torch.diagonal(WtW).sum()
                             + torch.diagonal(model.mu.weight @ model.mu.weight.T).sum()) / N
    loss = recon + var_term + beta * kl + reg
    return loss


@torch.no_grad()
def observables(model, W0, rho):
    """Decoder-side order parameters for M = M* = 1."""
    N = float(model.N)
    W = model.dec.weight.detach().reshape(-1)             # (N,)
    m = (W @ W0).item() / N
    Q = (W @ W).item() / N
    D = model.var.detach().item()
    am = abs(m)
    eg = rho - 2.0 * np.sqrt(rho) * am + Q
    q_perp = max(Q - m * m, 0.0)                          # = Q (1 - o_+^2)
    o2 = (m * m / Q) if Q > 1e-12 else 0.0                # measured eigenvector overlap^2
    return dict(m=am, Q=Q, D=D, eg=eg, q_perp=q_perp, o2=o2)


# ============================================================================= #
#  Data: spiked covariance model, fixed dataset of P samples
# ============================================================================= #
def generate_dataset_SCM(N, P, W0, eta, rho, gen):
    c = torch.randn(P, 1, generator=gen)                  # latent factor per sample
    noise = torch.randn(P, N, generator=gen)
    X = (c @ W0.reshape(1, N)) * np.sqrt(rho / N) + np.sqrt(eta) * noise
    return X                                              # (P, N)


@torch.no_grad()
def empirical_spike(X, W0):
    """Top eigenvalue nu_+ and squared overlap o_+^2 of Sigma_hat with the true direction."""
    P, N = X.shape
    Sigma = (X.T @ X) / P                                 # (N, N)
    evals, evecs = torch.linalg.eigh(Sigma)               # ascending
    nu_plus = evals[-1].item()
    u = evecs[:, -1]
    What = W0 / torch.linalg.norm(W0)
    o2 = (u @ What).item() ** 2
    return nu_plus, o2


# ============================================================================= #
#  Theory (BBP / Marchenko-Pastur)
# ============================================================================= #
def bbp_nu_plus(alpha, rho, eta):
    theta, gamma = rho / eta, 1.0 / alpha
    if theta > np.sqrt(gamma):                            # detectable: spike detaches
        return (rho + eta) * (1.0 + eta / (alpha * rho))
    return eta * (1.0 + np.sqrt(gamma)) ** 2              # else: MP bulk edge

def bbp_o2(alpha, rho, eta):
    theta, gamma = rho / eta, 1.0 / alpha
    if theta > np.sqrt(gamma):
        return (1.0 - gamma / theta ** 2) / (1.0 + gamma / theta)
    return 0.0

def alpha_c(rho, eta):
    return (eta / rho) ** 2

def tier1_predictions(nu_plus, o2, beta, rho):
    Q = max(nu_plus - beta, 0.0)
    m = np.sqrt(Q) * np.sqrt(o2)
    return dict(Q=Q, m=m, D=(beta / nu_plus if nu_plus > 0 else 1.0),
                q_perp=Q * (1.0 - o2), eg=rho - 2.0 * np.sqrt(rho) * m + Q)


# ============================================================================= #
#  Tier-1.5 reduced signal-mode dynamics and its Jacobian
# ============================================================================= #
def reduced_jacobian(a, b, D, nu, beta):
    """3x3 Jacobian of (a_dot, b_dot, D_dot) in the signal eigenmode (tau = 1)."""
    return np.array([
        [-nu * b * b - D,       nu * (1 - 2 * a * b),   -a          ],
        [ nu * (1 - 2 * a * b), -nu * (a * a + beta),    0.0         ],
        [-2 * a,                 0.0,                    -beta / D / D],
    ])

def stability_maxeig(nu, beta, branch):
    if branch == "collapse":
        a, b, D = 0.0, 0.0, 1.0
    elif branch == "learnable":
        if beta >= nu:
            return np.nan                                 # learnable FP does not exist
        a = np.sqrt(nu - beta); b = a / nu; D = beta / nu
    J = reduced_jacobian(a, b, D, nu, beta)
    return np.max(np.real(np.linalg.eigvals(J)))


# ============================================================================= #
#  Full-batch gradient-flow trainer
# ============================================================================= #
def run_gf(N, alpha, beta, rho=1.0, eta=1.0, reg_param=0.0,
           lr=LR, max_steps=MAX_STEPS, seed=0, tol=1e-4, check=100, patience=8):
    P = max(int(round(alpha * N)), 1)
    gen = torch.Generator().manual_seed(1000 * seed + 7)
    W0 = torch.ones(N)
    X = generate_dataset_SCM(N, P, W0, eta, rho, gen)
    model = LinearVAE(N, 1, gen=gen).to(DEVICE)
    if OPTIMIZER == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)  # true gradient flow (slow)

    prev, stable = None, 0
    for step in range(max_steps):
        opt.zero_grad()
        loss = empirical_risk(model, X, beta, reg_param)
        loss.backward()
        opt.step()
        model.var.data.clamp_(min=1e-6)
        if step % check == 0:
            ob = observables(model, W0, rho)
            cur = (ob["Q"], ob["m"], ob["D"])
            if prev is not None and sum(abs(c - p) for c, p in zip(cur, prev)) < tol:
                stable += 1
                if stable >= patience:
                    break
            else:
                stable = 0
            prev = cur

    ob = observables(model, W0, rho)
    nu_emp, o2_emp = empirical_spike(X, W0)
    ob.update(nu_emp=nu_emp, o2_emp=o2_emp, steps=step + 1)
    return ob


def aggregate(N, alpha, beta, seeds, **kw):
    runs = [run_gf(N, alpha, beta, seed=s, **kw) for s in range(seeds)]
    keys = ["m", "Q", "D", "eg", "q_perp", "o2", "nu_emp", "o2_emp"]
    out = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs])) for k in keys}
    out["steps"] = np.mean([r["steps"] for r in runs])
    return out


# ============================================================================= #
#  EXPERIMENT A : steady state vs alpha
# ============================================================================= #
def experiment_A(rho=1.0, eta=1.0, beta=1.0, N=N_DEFAULT, seeds=SEEDS_A):
    alphas = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    meas = {k: [] for k in ["m", "Q", "q_perp", "eg"]}
    err  = {k: [] for k in ["m", "Q", "q_perp", "eg"]}
    nu_emp_list, o2_emp_list = [], []
    for a in tqdm(alphas):
        agg = aggregate(N, a, beta, seeds, rho=rho, eta=eta)
        for k in tqdm(meas):
            meas[k].append(agg[k][0]); err[k].append(agg[k][1])
        nu_emp_list.append(agg["nu_emp"][0]); o2_emp_list.append(agg["o2_emp"][0])
        print(f"[A] alpha={a:>4}  Q={agg['Q'][0]:.3f} (th {bbp_nu_plus(a,rho,eta)-beta:.3f})"
              f"  m={agg['m'][0]:.3f}  qperp={agg['q_perp'][0]:.3f}  steps={agg['steps']:.0f}")

    af = np.linspace(alphas.min(), alphas.max(), 400)
    th = {k: [] for k in ["m", "Q", "q_perp", "eg"]}
    for a in af:
        p = tier1_predictions(bbp_nu_plus(a, rho, eta), bbp_o2(a, rho, eta), beta, rho)
        for k in th: th[k].append(p[k])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    panels = [("Q", r"$Q_{11}$  (decoder mass)"),
              ("m", r"$m_{11}$  (signal overlap)"),
              ("q_perp", r"$q_\perp = Q_{11}-m_{11}^2$  (overfitting)"),
              ("eg", r"$\varepsilon_g$  (generalization)")]
    ac = alpha_c(rho, eta)
    for ax, (k, lab) in zip(axes.ravel(), panels):
        ax.plot(af, th[k], "-", color="crimson", lw=2, label="Tier-1 (BBP)")
        ax.errorbar(alphas, meas[k], yerr=err[k], fmt="o", ms=5, color="navy",
                    capsize=2, label="full-batch GF")
        if ac >= alphas.min():
            ax.axvline(ac, ls=":", color="gray", label=r"$\alpha_c=\eta^2/\rho^2$")
        ax.set_xlabel(r"$\alpha = P/N$"); ax.set_ylabel(lab)
    axes[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle(rf"Experiment A: steady state vs $\alpha$  "
                 rf"($\beta={beta},\ \rho={rho},\ \eta={eta},\ \lambda=0,\ N={N}$)")
    path = os.path.join(OUTDIR, "fig_expA_steadystate.pdf")
    fig.savefig(path); plt.close(fig); print("saved", path)


# ============================================================================= #
#  EXPERIMENT B : detectability (BBP) transition
# ============================================================================= #
def experiment_B(rho=1.0, eta=1.0, beta=0.5, N=300, seeds=SEEDS_B):
    alphas = np.array([0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    m_mean, m_err, o2_mean, o2_err = [], [], [], []
    for a in tqdm(alphas):
        agg = aggregate(N, a, beta, seeds, rho=rho, eta=eta)
        m_mean.append(agg["m"][0]); m_err.append(agg["m"][1])
        o2_mean.append(agg["o2"][0]); o2_err.append(agg["o2"][1])
        print(f"[B] alpha={a:>4}  m={agg['m'][0]:.3f}  o2={agg['o2'][0]:.3f}"
              f"  (BBP o2 {bbp_o2(a,rho,eta):.3f})")

    af = np.linspace(alphas.min(), alphas.max(), 500)
    m_th  = [tier1_predictions(bbp_nu_plus(a, rho, eta), bbp_o2(a, rho, eta), beta, rho)["m"] for a in af]
    o2_th = [bbp_o2(a, rho, eta) for a in af]
    ac = alpha_c(rho, eta)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].plot(af, m_th, "-", color="crimson", lw=2, label="Tier-1 (BBP)")
    axes[0].errorbar(alphas, m_mean, yerr=m_err, fmt="o", color="navy", ms=5, capsize=2, label="GF")
    axes[0].set_ylabel(r"$m_{11}$  (signal recovery)")
    axes[1].plot(af, o2_th, "-", color="crimson", lw=2, label=r"BBP $o_+^2$")
    axes[1].errorbar(alphas, o2_mean, yerr=o2_err, fmt="o", color="navy", ms=5, capsize=2, label="GF")
    axes[1].set_ylabel(r"$o_+^2 = m_{11}^2/Q_{11}$  (eigenvector overlap)")
    for ax in axes:
        ax.axvline(ac, ls=":", color="gray", label=r"$\alpha_c=\eta^2/\rho^2$")
        ax.set_xlabel(r"$\alpha = P/N$"); ax.legend(loc="best", fontsize=9)
    fig.suptitle(rf"Experiment B: detectability transition  "
                 rf"($\beta={beta},\ \rho={rho},\ \eta={eta},\ N={N}$)")
    path = os.path.join(OUTDIR, "fig_expB_detectability.pdf")
    fig.savefig(path); plt.close(fig); print("saved", path)


# ============================================================================= #
#  EXPERIMENT C : collapse threshold beta*(alpha)
# ============================================================================= #
def experiment_C(rho=1.0, eta=1.0, N=140, seeds=SEEDS_C):
    alphas = [0.5, 1.0, 2.0, 6.0]
    betas = np.linspace(0.4, 6.5, 16)
    measured_thresh = {}
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(alphas)))
    for a, col in zip(alphas, colors):
        Qm, Qe = [], []
        for b in tqdm(betas):
            agg = aggregate(N, a, b, seeds, rho=rho, eta=eta, max_steps=3000)
            Qm.append(agg["Q"][0]); Qe.append(agg["Q"][1])
        Qm = np.array(Qm)
        # measured threshold: last beta with Q above a small floor
        above = betas[Qm > 0.05]
        bstar_meas = float(above.max()) if above.size else float(betas.min())
        measured_thresh[a] = bstar_meas
        nu = bbp_nu_plus(a, rho, eta)
        ax.errorbar(betas, Qm, yerr=Qe, fmt="o-", ms=4, color=col, capsize=2,
                    label=rf"$\alpha={a}$ (meas $\beta^*\!\approx${bstar_meas:.2f}, $\nu_+={nu:.2f}$)")
        ax.axvline(nu, ls="--", color=col, alpha=0.7)
        print(f"[C] alpha={a}: measured beta*~{bstar_meas:.2f}, nu_+={nu:.2f}, rho+eta={rho+eta:.2f}")
    ax.axvline(rho + eta, ls=":", color="black", lw=2, label=r"population $\rho+\eta$")
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$Q_{11}$  (steady state)")
    ax.set_title(rf"Experiment C: collapse threshold  ($\rho={rho},\ \eta={eta},\ N={N}$)"
                 "\n dashed = predicted $\\beta^*(\\alpha)=\\nu_+(\\alpha)$, dotted = population $\\rho+\\eta$")
    ax.legend(loc="upper right", fontsize=8)
    path = os.path.join(OUTDIR, "fig_expC_collapse.pdf")
    fig.savefig(path); plt.close(fig); print("saved", path)
    return measured_thresh


# ============================================================================= #
#  EXPERIMENT D : Tier-1.5 local stability (Figure-5 analogue)
# ============================================================================= #
def experiment_D(rho=1.0, eta=1.0):
    alphas = [1.0, 0.5]
    fig, axes = plt.subplots(1, len(alphas), figsize=(11, 4.4), constrained_layout=True, sharey=True)
    for ax, a in zip(axes, alphas):
        nu = bbp_nu_plus(a, rho, eta)
        betas = np.linspace(0.2, 1.5 * nu, 400)
        learn = [stability_maxeig(nu, b, "learnable") for b in betas]
        coll  = [stability_maxeig(nu, b, "collapse")  for b in betas]
        ax.plot(betas, learn, "-", color="navy",    lw=2, label="learnable FP")
        ax.plot(betas, coll,  "-", color="darkorange", lw=2, label="collapsed FP")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(nu, ls="--", color="crimson", label=rf"$\nu_+={nu:.2f}$")
        ax.axvline(rho + eta, ls=":", color="gray", label=rf"$\rho+\eta={rho+eta:.2f}$")
        ax.set_xlabel(r"$\beta$"); ax.set_title(rf"$\alpha={a}$  ($\nu_+={nu:.3f}$)")
        ax.set_ylim(-1.5, 0.8)
    axes[0].set_ylabel(r"$\max_i \mathrm{Re}\,\lambda_i$  of reduced Jacobian")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Experiment D (Tier-1.5): local stability of fixed points; "
                 "exchange of stability at $\\beta=\\nu_+(\\alpha)$")
    path = os.path.join(OUTDIR, "fig_expD_stability.pdf")
    fig.savefig(path); plt.close(fig); print("saved", path)


# ============================================================================= #
#  EXPERIMENT E : (alpha, beta) phase diagram (theory + measured collapse points)
# ============================================================================= #
def experiment_E(rho=1.0, eta=1.0, measured_thresh=None):
    av = np.linspace(0.2, 8.0, 500)
    nu_line = [bbp_nu_plus(a, rho, eta) for a in av]
    ac = alpha_c(rho, eta)
    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    ax.plot(av, nu_line, "-", color="crimson", lw=2.2, label=r"collapse $\beta^*(\alpha)=\nu_+(\alpha)$")
    ax.axhline(rho + eta, ls=":", color="black", lw=1.6, label=r"population $\rho+\eta$")
    ax.axvline(ac, ls="--", color="teal", lw=1.6, label=r"detectability $\alpha_c=\eta^2/\rho^2$")
    ax.fill_between(av, nu_line, 6.5, color="crimson", alpha=0.07)
    ax.text(6.2, 4.6, "posterior\ncollapse", color="crimson", ha="center", fontsize=11)
    ax.text(3.2, 0.8, "signal recovery", color="navy", ha="center", fontsize=11)
    if ac > 0.25:
        ax.text(max(ac * 0.55, 0.3), 2.6, "no\ndetect.", color="teal", ha="center", fontsize=9)
    if measured_thresh:
        xs = sorted(measured_thresh); ys = [measured_thresh[x] for x in xs]
        ax.plot(xs, ys, "s", color="navy", ms=8, label="measured collapse (Exp C)")
    ax.set_xlim(0.2, 8.0); ax.set_ylim(0.0, 6.3)
    ax.set_xlabel(r"$\alpha = P/N$"); ax.set_ylabel(r"$\beta$")
    ax.set_title(rf"Experiment E: phase diagram  ($\rho={rho},\ \eta={eta}$)")
    ax.legend(loc="lower right", fontsize=9)
    path = os.path.join(OUTDIR, "fig_expE_phase.pdf")
    fig.savefig(path); plt.close(fig); print("saved", path)


# ============================================================================= #

# Note: the full QUICK suite takes a few minutes on CPU. To stay within tight
# time limits you can run the experiments individually, e.g.
#   python -c "import multiepoch_vae_experiments as E; E.experiment_C()"
print(f"=== CONFIG={CONFIG}, N={N_DEFAULT}, lr={LR}, max_steps={MAX_STEPS} ===")
experiment_D()                         # pure theory, instant
experiment_A()
experiment_B()
thresh = experiment_C()
experiment_E(measured_thresh=thresh)
print("All figures written to", OUTDIR)
