
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

# ----- constants you can tune -----
OMEGA = 0.0     # omega
LAM   = 1.0     # lambda
ALPHA = -1.0    # alpha (note: f uses 1/(2*ALPHA))
SIGMA = 1.0     # unused here, kept for continuity

R_MIN = 1e-3    # left boundary (>0)
R_MAX = 80.0    # "infinity"
SAMPLE_PTS = 2000

KAPPA_INF = 0.1 # only used for initial guess of profiles; not a BC

# small epsilons for robust divisions
EPS_B     = 1e-12   # for divides by B or r^2 B
EPS_NONAN = 1e-30
EPS_A     = 1e-10   # for safe_ratio thresholds

# ---------------- helpers ----------------

def safe_ratio(num, den, eps=EPS_A):
    """Vectorized safe division with mask; returns NaN where |den|<=eps."""
    den = np.asarray(den, dtype=float)
    out = np.full_like(den, np.nan, dtype=float)
    mask = np.abs(den) > eps
    # only perform division where mask is True
    np.divide(num, den, out=out, where=mask)
    return out

def initial_guess(r):
    """Reasonable initial profiles that roughly satisfy far/near behavior (for visualization & guesses)."""
    S0 = np.exp(-r / 5.0)
    w0 = np.ones_like(r)
    m0 = np.zeros_like(r)
    g0 = np.zeros_like(r)     # gamma
    k0 = KAPPA_INF * np.ones_like(r)  # just a guess; not enforced as BC

    B0  = w0 - 2.0*m0/np.maximum(r, EPS_B) + g0*r + k0*r**2
    S0p = -(1.0/5.0) * np.exp(-r / 5.0)
    y0  = r**2 * B0 * S0p     # y = r^2 B S'

    return np.vstack([S0, y0, w0, m0, g0, k0])

def _compute_B_and_Bp(r, S, y, w, m, gamma, kappa):
    """Vectorized helper to compute B, B' and S' on any sampling of r (arrays)."""
    r = np.asarray(r, dtype=float)
    B  = w - 2.0*m/np.maximum(r, EPS_B) + gamma*r + kappa*r**2  # consistent +kappa
    g  = r**2 * B
    g  = np.where(np.abs(g) < EPS_B, np.sign(g + EPS_NONAN) * EPS_B, g)
    Sp = y / g
    # forced dB
    Bp = 2.0*m/np.maximum(r**2, EPS_B) + gamma - 2.0*kappa*r
    return B, Bp, Sp

# -------------- IVP RHS (scalar state) --------------

def rhs_ivp(r, y_vec, omega=OMEGA, lam=LAM, alpha=ALPHA):
    """
    Scalar (single-point) RHS for solve_ivp Radau.
    y_vec = [S, y, w, m, gamma, kappa]
    """
    S, y, w, m, gamma, kappa = y_vec

    # avoid singularities
    r_safe  = max(r, EPS_B)
    r2_safe = max(r*r, EPS_B)

    # metric functions
    B      = w - 2.0*m/r_safe + gamma*r + kappa*r*r
    B_safe = B if abs(B) >= EPS_B else np.sign(B + EPS_NONAN) * EPS_B

    g      = r*r*B_safe
    g_safe = g if abs(g) >= EPS_B else np.sign(g + EPS_NONAN) * EPS_B

    # S' and curvature-like R
    dS = y / g_safe
    R  = 2.0*(w - 1.0)/r2_safe - 12.0*kappa

    # y'
    dy = r*r*(lam*S*S - (omega*omega/B_safe - R/6.0)) * S

    # forced dB
    dB = 2.0*m/r2_safe + gamma - 2.0*kappa*r

    # S'' via chain rule (g' = 2 r B + r^2 dB)
    gprime = 2.0*r*B_safe + r*r*dB
    ddS    = (dy / g_safe) - dS*(gprime / g_safe)

    # f and metric ODEs
    f      = -(1.0/(2.0*alpha)) * (dS*dS + S*ddS)
    dw     =  0.5       * r**3 * f
    dm     = (1.0/12.0) * r**4 * f
    dgamma = -0.5       * r**2 * f
    dkappa = -(1.0/6.0) * r    * f

    return np.array([dS, dy, dw, dm, dgamma, dkappa], dtype=float)

# -------------- Shooting setup --------------

def left_state_from_guess(S0, kappa0):
    """Assemble left boundary state at r=R_MIN given S0 and kappa0."""
    y0     = 0.0   # y(0)=0
    w0     = 1.0   # w(0)=1
    m0     = 0.0   # m(0)=0
    gamma0 = 0.0   # gamma(0)=0
    return np.array([S0, y0, w0, m0, gamma0, kappa0], dtype=float)

def initial_guess_params():
    """Use your profile guess to seed (S0, kappa0) at r=R_MIN."""
    rgrid = np.linspace(R_MIN, R_MAX, 200)
    Yg = initial_guess(rgrid)
    S0_guess     = float(Yg[0, 0])
    kappa0_guess = float(Yg[5, 0])
    return np.array([S0_guess, kappa0_guess], dtype=float)

def shoot_residuals(x):
    S0, kappa0 = x
    y0 = left_state_from_guess(S0, kappa0)
    sol = solve_ivp(
        lambda r, y: rhs_ivp(r, y, OMEGA, LAM, ALPHA),
        (R_MIN, R_MAX), y0,
        method="Radau",
        dense_output=False,    # <- off
        rtol=1e-6, atol=1e-8   # <- looser
    )
    if not sol.success:
        return np.array([1e3, 1e3])
    S_R, kappa_R = sol.y[0, -1], sol.y[5, -1]
    return np.array([S_R, 2.0*kappa_R])


def solve_with_radau_shooting(R_max, x0=None, eval_dense=False, tol=(1e-6, 1e-8)):
    """Solve for given R_max using Radau shooting.
       x0: optional initial guess [S0, kappa0]
       eval_dense: if True, build dense_output for plots (final pass)
       tol: (rtol, atol) for Radau
    """
    rtol, atol = tol

    def initial_guess_params(R_max_local):
        rgrid = np.linspace(R_MIN, R_max_local, 200)
        Yg = initial_guess(rgrid)
        return np.array([float(Yg[0, 0]), float(Yg[5, 0])], dtype=float)

    # residuals closure that uses this stage's R_max
    def shoot_residuals_stage(x):
        S0, kappa0 = x
        y0 = left_state_from_guess(S0, kappa0)
        sol_ivp = solve_ivp(
            lambda r, y: rhs_ivp(r, y, OMEGA, LAM, ALPHA),
            (R_MIN, R_max), y0,
            method="Radau",
            dense_output=False,           # cheap during root finding
            rtol=rtol, atol=atol
        )
        if not sol_ivp.success:
            return np.array([1e3, 1e3], dtype=float)
        S_R, kappa_R = sol_ivp.y[0, -1], sol_ivp.y[5, -1]
        return np.array([S_R, 2.0*kappa_R], dtype=float)

    if x0 is None:
        x0 = initial_guess_params(R_max)

    root_sol = root(shoot_residuals_stage, x0, tol=1e-10, method="hybr")
    if not root_sol.success:
        print("Shooting nonlinear solver did not converge:", root_sol.message)

    # final integration for this stage
    y0_opt = left_state_from_guess(*root_sol.x)
    sol_ivp = solve_ivp(
        lambda r, y: rhs_ivp(r, y, OMEGA, LAM, ALPHA),
        (R_MIN, R_max), y0_opt,
        method="Radau",
        dense_output=eval_dense,         # True only on the last stage
        rtol=rtol, atol=atol
    )
    return root_sol, sol_ivp


# ---------------- main & plotting ----------------

def main():
    stages = [20.0, 40.0, 80.0]   # your continuation ladder
    x0 = None
    last_root, last_ivp = None, None

    # fast, loose solves for intermediate stages
    for R_max in stages[:-1]:
        print(f"\n=== Continuation stage: R_MAX={R_max} ===")
        last_root, last_ivp = solve_with_radau_shooting(
            R_max, x0=x0, eval_dense=False, tol=(1e-6, 1e-8)
        )
        x0 = last_root.x  # carry (S0, kappa0) forward

    # final, accurate pass with dense_output for plotting
    R_max_final = stages[-1]
    print(f"\n=== Final stage (plots): R_MAX={R_max_final} ===")
    last_root, last_ivp = solve_with_radau_shooting(
        R_max_final, x0=x0, eval_dense=True, tol=(1e-8, 1e-10)
    )

    print("\n=== Shooting status ===")
    print(f"Nonlinear solver success: {last_root.success} | message: {last_root.message}")

    # sample & plot using the final stage's solution/extent
    rr = np.linspace(R_MIN, R_max_final, SAMPLE_PTS)
    Y  = last_ivp.sol(rr)
    S, y, w, m, gamma, kappa = Y


    # derived quantities
    # derived quantities
    B, Bp, Sp = _compute_B_and_Bp(rr, S, y, w, m, gamma, kappa)
    # Note: for plotting R we keep your earlier "augmented" expression:
    Rcurv = 2.0*(w - 1.0)/rr**2 + 6.0*gamma/rr - 12.0*kappa

    # ---- quick sign check (vectorized) ----
    mask = np.abs(gamma) > EPS_B
    if np.any(mask) and np.any((w[mask] + 1.0) / gamma[mask] < 0):
        print("NEG")
    else:
        print("A POSITIVE!")

    # ---------- plots (and SAVE) ----------
    plt.figure()
    plt.plot(rr, B,  label="B(r)")
    plt.plot(rr, Bp, label="B'(r)", linestyle="--")
    plt.xlabel("r"); plt.ylabel("value"); plt.title("B and B'")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("B_Bp_ver2.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(rr, S,  label="S(r)")
    plt.plot(rr, Sp, label="S'(r)", linestyle="--")
    plt.xlabel("r"); plt.ylabel("value"); plt.title("S and S'")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("S_Sp_ver2.png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(rr, w,     label="w(r)")
    plt.plot(rr, m,     label="m(r)")
    plt.plot(rr, gamma, label=r"$\gamma(r)$")
    plt.plot(rr, kappa, label=r"$\kappa(r)$")
    plt.xlabel("r"); plt.ylabel("value"); plt.title("w, m, gamma, kappa")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("MK_param_ver2.png", dpi=300, bbox_inches="tight")

    # log–log positive segments
    B_pos  = np.where(B  > 0, B,  np.nan)
    Bp_pos = np.where(Bp > 0, Bp, np.nan)
    plt.figure()
    plt.loglog(rr, B_pos,  label="B(r) > 0")
    plt.loglog(rr, Bp_pos, label="B'(r) > 0", linestyle="--")
    plt.xlabel("r"); plt.ylabel("value")
    plt.title("Log–log: B and B' (positive segments)")
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("B_Bp_loglog_ver2.png", dpi=300, bbox_inches="tight")

    # R(r)
    plt.figure()
    plt.plot(rr, Rcurv, label="R(r)")
    plt.xlabel("r"); plt.ylabel("R")
    plt.title("R(r)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("R_ver2.png", dpi=300, bbox_inches="tight")

    # ===== Compare S1 (numerical) with S2(r) = S0 * a / (r + a) =====
    S0_val = S[0]
    a1_arr = safe_ratio(w + 1.0, gamma, EPS_A)
    a2_arr = safe_ratio(6.0 * m, 1.0 - w, EPS_A)

    def nanmedian_or_nan(x):
        x = x[np.isfinite(x)]
        return np.nan if x.size == 0 else np.nanmedian(x)

    a1_const = nanmedian_or_nan(a1_arr)
    a2_const = nanmedian_or_nan(a2_arr)

    S2_a1 = None if not np.isfinite(a1_const) or np.abs(a1_const) < EPS_A else S0_val * a1_const / (rr + a1_const)
    S2_a2 = None if not np.isfinite(a2_const) or np.abs(a2_const) < EPS_A else S0_val * a2_const / (rr + a2_const)

    plt.figure()
    plt.plot(rr, S, label="S1 (numerical)")
    if S2_a1 is not None:
        plt.plot(rr, S2_a1, "--", label=f"S2, a=(w+1)/γ ≈ {a1_const:.3g}")
    if S2_a2 is not None:
        plt.plot(rr, S2_a2, ":", label=f"S2, a=6m/(1−w) ≈ {a2_const:.3g}")
    plt.xlabel("r"); plt.ylabel("S")
    plt.title("S1 vs S2(r) = S0·a/(r+a)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("S_compare.png", dpi=300, bbox_inches="tight")

    # --- S(y) with r as parameter (color = r) ---
    valid = np.isfinite(y) & np.isfinite(S)
    plt.figure()
    sc = plt.scatter(y[valid], S[valid], c=rr[valid], s=12)
    plt.xlabel("y"); plt.ylabel("S")
    plt.title("S(y) (points colored by r)")
    plt.colorbar(sc, label="r")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("S_vs_y_param_r_num.png", dpi=300, bbox_inches="tight")

    # --- f(r) from S, S'', alpha (vectorized post-processing) ---
    B_safe   = np.where(np.abs(B) < EPS_B, np.sign(B + EPS_NONAN) * EPS_B, B)
    dy_expr  = rr**2 * (LAM * S**2 - (OMEGA**2 / B_safe - (2.0*(w - 1.0)/rr**2 - 12.0*kappa)/6.0)) * S
    dB_forced = 2.0 * m / np.maximum(rr**2, EPS_B) + gamma - 2.0 * kappa * rr
    g        = rr**2 * B_safe
    g_safe   = np.where(np.abs(g) < EPS_B, np.sign(g + EPS_NONAN) * EPS_B, g)
    gprime   = 2.0 * rr * B_safe + rr**2 * dB_forced
    ddS      = (dy_expr / g_safe) - Sp * (gprime / g_safe)
    f_rr     = -(1.0 / (2.0 * ALPHA)) * (Sp**2 + S * ddS)

    # --- find maximum of f(r) (left-to-right scan over rr) ---
    valid_f = np.isfinite(f_rr)
    idx_max = None
    if np.any(valid_f):
        idx_max = np.nanargmax(np.where(valid_f, f_rr, -np.inf))
        r_max = rr[idx_max]
        f_max = f_rr[idx_max]
        print(f"f_max = {f_max:.6g} at r = {r_max:.6g}")

    # --- plot and save with max marker + radius in legend ---
    plt.figure()
    plt.plot(rr, f_rr, label="f(r)")
    if idx_max is not None:
        plt.scatter([r_max], [f_max], s=40, zorder=5, label=f"max at r = {r_max:.4g}")
    plt.xlabel("r"); plt.ylabel("f")
    plt.title("f(r)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig("f_ver2.png", dpi=300, bbox_inches="tight")

    plt.show()

    print("\nSaved figures:")
    print("  - B_Bp_ver2.png")
    print("  - S_Sp_ver2.png")
    print("  - MK_param_ver2.png")
    print("  - B_Bp_loglog_ver2.png")
    print("  - R_ver2.png")
    print("  - S_compare.png")
    print("  - S_vs_y_param_r_num.png")
    print("  - f_ver2.png")

if __name__ == "__main__":
    main()
