#!/usr/bin/env python3
# ex2.py — Python port of ex2.m (Vector Fitting example with two elements)
# Original MATLAB example by Bjorn Gustavsen (VFIT3.zip, 08.08.2008)
# This Python version generates the same synthetic frequency responses.
# If scikit-rf is installed, it will also perform a vector fit and plot results.

import numpy as np
import sys

try:
    import matplotlib.pyplot as plt  # optional plotting
except Exception:  # pragma: no cover
    plt = None

# Try to use scikit-rf's VectorFitting if available
try:
    from skrf.vectorfitting import VectorFitting  # type: ignore
except Exception:
    VectorFitting = None  # type: ignore


def partial_fraction_response(r: np.ndarray, p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Compute sum_n r[n] / (s - p[n]) for all s (vectorized)."""
    # r: (N_poles,), p: (N_poles,), s: (N_freq,)
    return (r[None, :] / (s[:, None] - p[None, :])).sum(axis=1)


def main() -> int:
    # Parameters
    D = 0.2
    E = 2e-5

    # Define poles (p) and residues (r) in Hz-domain (before 2*pi scaling)
    p = np.array([
        -4500,
        -41000,
        (-100 + 1j * 5e3), (-100 - 1j * 5e3),
        (-120 + 1j * 15e3), (-120 - 1j * 15e3),
        (-3e3 + 1j * 35e3), (-3e3 - 1j * 35e3),
        (-200 + 1j * 45e3), (-200 - 1j * 45e3),
        (-1500 + 1j * 45e3), (-1500 - 1j * 45e3),
        (-5e2 + 1j * 70e3), (-5e2 - 1j * 70e3),
        (-1e3 + 1j * 73e3), (-1e3 - 1j * 73e3),
        (-2e3 + 1j * 90e3), (-2e3 - 1j * 90e3),
    ], dtype=complex)

    r = np.array([
        -3000,
        -83000,
        (-5 + 1j * 7e3), (-5 - 1j * 7e3),
        (-20 + 1j * 18e3), (-20 - 1j * 18e3),
        (6e3 + 1j * 45e3), (6e3 - 1j * 45e3),
        (40 + 1j * 60e3), (40 - 1j * 60e3),
        (90 + 1j * 10e3), (90 - 1j * 10e3),
        (5e4 + 1j * 80e3), (5e4 - 1j * 80e3),
        (1e3 + 1j * 45e3), (1e3 - 1j * 45e3),
        (-5e3 + 1j * 92e3), (-5e3 - 1j * 92e3),
    ], dtype=complex)

    # Frequency vector (rad/s) and s-plane points
    w = 2 * np.pi * np.linspace(1.0, 1e5, 100)
    s = 1j * w
    Ns = s.size

    # Scale poles and residues to rad/s domain
    p = 2 * np.pi * p
    r = 2 * np.pi * r

    # Split into two groups as in ex2.m
    p1, r1 = p[:10], r[:10]
    p2, r2 = p[8:18], r[8:18]

    # Build responses f1 and f2
    f1 = partial_fraction_response(r1, p1, s) + s * E
    f1 = f1 + D

    f2 = partial_fraction_response(r2, p2, s) + s * 3 * E
    # NOTE: The original ex2.m adds 2*D to f(1,:) again on line 46.
    # This Python port reproduces that behavior verbatim:
    f1 = f1 + 2 * D
    # If you intended the offset for the second element instead, use:
    # f2 = f2 + 2 * D

    f = np.vstack([f1, f2])  # shape (2, Ns)

    # =====================================
    # Rational function approximation of f(s)
    # =====================================
    N = 18  # Order of approximation

    # Complex starting poles (rad/s)
    bet = np.linspace(w[0], w[-1], N // 2)
    poles = []
    for b in bet:
        alf = -b * 1e-2
        poles.extend([alf - 1j * b, alf + 1j * b])
    poles = np.array(poles, dtype=complex)

    # Weights and options (for reference; not all are used below)
    weight = np.ones(Ns)
    opts = {
        'relax': 1,
        'stable': 1,
        'asymp': 3,        # include both D and E
        'skip_pole': 0,
        'skip_res': 1,     # set to 0 in the last iteration in the MATLAB script
        'cmplx_ss': 0,
        # plotting options from the MATLAB script
        'spy1': 0,
        'spy2': 1,
        'logx': 0,
        'logy': 1,
        'errplot': 1,
        'phaseplot': 0,
        'legend': 1,
    }

    print('vector fitting...')

    # Optional: perform vector fitting using scikit-rf, if available
    if VectorFitting is not None:
        rms = []
        fits = []
        for idx, fi in enumerate([f1, f2], start=1):
            # Use 9 complex-conjugate pole pairs (no real poles)
            vf = VectorFitting(fi, s, n_poles_real=0, n_poles_cmplx=N // 2)
            # Attempt a few iterations; API may ignore if unsupported
            try:
                vf.vector_fit()  # default iterations
            except TypeError:
                # Some versions may accept parameters, try a common variant
                vf.vector_fit(n_iter=3)

            # Evaluate fitted model
            try:
                H = vf.get_model()
                fi_fit = H(s)
            except Exception:
                # Fallback if API differs
                try:
                    fi_fit = vf.evaluate(s)  # type: ignore[attr-defined]
                except Exception:
                    fi_fit = None

            if fi_fit is not None:
                rmse = np.sqrt(np.mean(np.abs(fi - fi_fit) ** 2))
                rms.append(rmse)
                fits.append(fi_fit)
                print(f'  Element {idx}: RMSE = {rmse:.4e}')
            else:
                print(f'  Element {idx}: fitted response not available with this VectorFitting API')

        # Plot magnitude if matplotlib is available and fits were computed
        if plt is not None and len(fits) == 2:
            freq_hz = w / (2 * np.pi)
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            for i, ax in enumerate(axes):
                ax.plot(freq_hz, np.abs(f[i]), label='|f| (data)')
                ax.plot(freq_hz, np.abs(fits[i]), '--', label='|f_fit|')
                ax.set_yscale('log')
                ax.grid(True, which='both', ls=':')
                ax.legend()
                ax.set_ylabel(f'Element {i+1}')
            axes[-1].set_xlabel('Frequency [Hz]')
            fig.suptitle('Vector Fitting of f(s) (magnitude)')
            plt.tight_layout()
            plt.show()
    else:
        print('  scikit-rf not found. To run the vector fitting step, install it via:\n'
              '    pip install scikit-rf matplotlib')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())