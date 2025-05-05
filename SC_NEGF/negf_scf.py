# NEGF Script for 14-Atom Li Chain with SCF Hamiltonian Update
# Based on negf7 (1).ipynb and previous modifications
# MODIFIED: Post-processing uses INITIAL Hamiltonian
# MODIFIED: Plotting uses Matplotlib (from original negf7.py)

import numpy as np
import scipy.linalg
import scipy.integrate
from pyscf import gto, scf, dft, lib
import matplotlib.pyplot as plt # Changed from plotly
import warnings
import time
import traceback # For detailed error printing
# Removed plotly imports: from plotly.subplots import make_subplots, import plotly.graph_objects as go

# --- Constants ---
HARTREE_TO_EV = 27.2114
KB_AU = 3.1668e-6 # Boltzmann constant in au (Hartree/Kelvin)
LANDAUER_PREFACTOR_AU = 1.0 / np.pi # 2e^2/h = 1/pi in atomic units (assuming spin degeneracy included in trace)

# --- NEGF/SCF Parameters ---
ETA = 1e-4 # Small imaginary part for Green's functions (Adjusted slightly)
ETA_COMPLEX = 1j * ETA
print(f"Using ETA = {ETA:.1e}")

E_FERMI_GUESS_AU = 0.0 # Will be updated after DFT
E_WINDOW_AU = 1.5     # Energy window width in Hartree (Slightly wider)
N_ENERGY_POINTS = 400 # Number of points for real-axis integration/plotting

APPLIED_BIAS_VOLTS = 0.0 # Apply 0.1V bias
print(f"Applied Bias = {APPLIED_BIAS_VOLTS} V")
CHEMICAL_POTENTIAL_L = E_FERMI_GUESS_AU # Will be updated
CHEMICAL_POTENTIAL_R = E_FERMI_GUESS_AU # Will be updated
TEMPERATURE = 300.0 # Kelvin

SCF_TOLERANCE = 1e-2  # Convergence threshold for density matrix norm change
SCF_MAX_ITER = 500     # Maximum SCF iterations (reduced for faster example)
SCF_MIXING_ALPHA = 0.2 # Linear mixing parameter for D_AO
POTENTIAL_SHIFT_DAMPING = 0.1 # Damping for potential alignment step
print(f"Using POTENTIAL_SHIFT_DAMPING = {POTENTIAL_SHIFT_DAMPING}")

# --- SRL Algorithm Parameters ---
SRL_TOLERANCE = 1e-8
SRL_MAX_ITER = 300

# --- Contour Integration Parameters ---
N_LINE = 50 # Number of points on the line segment of the contour
N_ARC = 50  # Number of points on the arc segment of the contour
N_POLES = 20 # Number of Matsubara poles to include

# --- Utility Functions ---

def fermi_dirac(E, mu, T):
    """Calculates the Fermi-Dirac distribution for REAL energy E."""
    if T < 1e-6: return np.where(E <= mu, 1.0, 0.0)
    kT_au = KB_AU * T; arg = (E - mu) / kT_au
    # Use np.where for numerical stability with large arguments
    return np.where(arg > 100, 0.0, np.where(arg < -100, 1.0, 1.0 / (1.0 + np.exp(arg))))

def fermi_dirac_complex(z, mu, T):
    """Calculates the Fermi-Dirac distribution for COMPLEX energy z."""
    if T < 1e-6:
        warnings.warn("FD at T=0 complex E", RuntimeWarning)
        return np.where(np.real(z) <= mu, 1.0, 0.0)
    kT_au = KB_AU * T
    arg = (z - mu) / kT_au
    with np.errstate(over='ignore', invalid='ignore'):
        exp_arg = np.exp(arg)
    # Check for NaN/Inf results from complex exp
    if np.isnan(exp_arg).any() or np.isinf(exp_arg).any():
         warnings.warn(f"Complex exp NaN/Inf arg={arg}. Heuristic.", RuntimeWarning)
         # Heuristic: return based on real part if overflow/underflow likely
         return np.where(np.real(arg)>100, 0.0, np.where(np.real(arg)<-100, 1.0, 0.5))
    else:
        denominator = 1.0 + exp_arg
        # Avoid division by very small numbers
        if np.abs(denominator) < 1e-150:
            # Likely corresponds to f -> 1
            return 1.0
        else:
            return 1.0 / denominator

def define_contour(E_min, E_max):
    """Defines a semi-circular contour in the upper complex plane."""
    delta = ETA
    if delta < 1e-12: delta = 1e-12 # Ensure numerical significance
    line = np.linspace(E_min + 1j*delta, E_max + 1j*delta, N_LINE, dtype=complex)
    center_re = (E_min + E_max) / 2.0
    radius = (E_max - E_min) / 2.0
    if radius <= 1e-12:
        warnings.warn("Contour radius near zero", RuntimeWarning)
        return line # Return only line if bounds are too close
    center_im = delta + radius
    center = center_re + 1j * center_im
    theta = np.linspace(np.pi, 0, N_ARC, dtype=float) # Arc from pi to 0 (upper plane)
    arc_points = center + radius * np.exp(1j * theta)
    # Concatenate line (excluding last point) and arc
    return np.concatenate([line[:-1], arc_points])

# --- Core NEGF Functions ---

def calculate_surface_gf_srl(E, H0, S0, H1, S1, tol=SRL_TOLERANCE):
    """Calculates surface Green's function using SRL. Accepts complex E."""
    n0 = H0.shape[0]
    # Add dimension checks
    if H0.shape != (n0, n0) or S0.shape != (n0, n0) or H1.shape != (n0, n0) or S1.shape != (n0, n0):
        return None

    if not isinstance(E, complex): Es = E + ETA_COMPLEX
    else: Es = E
    H0, S0, H1, S1 = map(lambda x: x.astype(complex), [H0, S0, H1, S1])
    eps, eps_S = H1.copy(), S1.copy()
    tau, tau_S = H1.conj().T, S1.conj().T
    try:
        g0_inv = Es * S0 - H0
        # Check for NaN/Inf before inversion
        if np.isnan(g0_inv).any() or np.isinf(g0_inv).any(): raise np.linalg.LinAlgError("g0_inv NaN/Inf before SRL start")
        gR = np.linalg.inv(g0_inv)
    except np.linalg.LinAlgError:
        # warnings.warn(f"SRL initial inversion failed E={E:.4f}{E.imag:+.4f}j", RuntimeWarning)
        return None # Failed initial inversion
    except Exception as e:
        warnings.warn(f"SRL initial setup unexpected error E={E}: {e}", RuntimeWarning)
        return None

    for i in range(SRL_MAX_ITER):
        gR_old = gR
        try:
            # Check for NaN/Inf in gR from previous iteration
            if np.isnan(gR).any() or np.isinf(gR).any(): raise ValueError("gR is NaN/Inf")

            coupling_eps = Es * eps_S - eps
            coupling_tau = Es * tau_S - tau
            # Check coupling terms
            if np.isnan(coupling_eps).any() or np.isinf(coupling_eps).any() or \
               np.isnan(coupling_tau).any() or np.isinf(coupling_tau).any(): raise ValueError("Coupling term NaN/Inf")

            g_eps = gR @ coupling_eps
            g_tau = gR @ coupling_tau

            new_eps = eps @ g_eps
            new_eps_S = eps_S @ g_eps
            new_tau = tau @ g_tau
            new_tau_S = tau_S @ g_tau
            # Check new transfer matrices
            if np.isnan(new_eps).any() or np.isinf(new_eps).any() or \
               np.isnan(new_eps_S).any() or np.isinf(new_eps_S).any() or \
               np.isnan(new_tau).any() or np.isinf(new_tau).any() or \
               np.isnan(new_tau_S).any() or np.isinf(new_tau_S).any(): raise ValueError("New Transfer matrix NaN/Inf")

            delta_inv = (Es * new_eps_S - new_eps) + (Es * new_tau_S - new_tau)
            g0_inv = g0_inv - delta_inv
            # Check g0_inv before inversion
            if np.isnan(g0_inv).any() or np.isinf(g0_inv).any(): raise np.linalg.LinAlgError("g0_inv update NaN/Inf")

            gR = np.linalg.inv(g0_inv)

            eps, eps_S = new_eps, new_eps_S
            tau, tau_S = new_tau, new_tau_S

            # Convergence check
            diff_norm = np.linalg.norm(gR - gR_old) / (np.linalg.norm(gR) + 1e-10) # Add small value to denominator
            if diff_norm < tol:
                return gR # Converged

            # Check if transfer matrices become very small (alternative convergence)
            norm_eps = np.linalg.norm(eps)
            norm_tau = np.linalg.norm(tau)
            if norm_eps < tol * 1e-3 and norm_tau < tol * 1e-3:
                return gR # Converged (transfer matrices vanished)

        except (ValueError, np.linalg.LinAlgError):
             # warnings.warn(f"SRL iter {i} LinAlg/Value error E={E:.4f}{E.imag:+.4f}j", RuntimeWarning)
             return None # Numerical error during iteration
        except Exception as e:
             warnings.warn(f"SRL unexpected iter {i} error E={E}: {e}", RuntimeWarning)
             return None # Other unexpected error

    warnings.warn(f"SRL failed to converge within {SRL_MAX_ITER} iterations at E={E:.4f}{E.imag:+.4f}j (diff_norm={diff_norm:.2e})", RuntimeWarning)
    return None # Failed to converge

def calculate_sigma(E, H_L_0, S_L_0, H_L_01, S_L_01, H_LD, S_LD,
                      H_R_0, S_R_0, H_R_01, S_R_01, H_RD, S_RD):
    """Calculates Sigma^R_L and Sigma^R_R using SRL. Accepts complex E."""
    # Note: Right lead requires conjugate transpose of coupling matrices for SRL
    gR_L = calculate_surface_gf_srl(E, H_L_0, S_L_0, H_L_01, S_L_01)
    gR_R = calculate_surface_gf_srl(E, H_R_0, S_R_0, H_R_01.conj().T, S_R_01.conj().T) # Use H_dagger for right lead recursion

    if gR_L is None or gR_R is None:
        # warnings.warn(f"Sigma calculation failed: Surface GF is None at E={E:.4f}{E.imag:+.4f}j", RuntimeWarning)
        return None, None

    if not isinstance(E, complex): Es = E + ETA_COMPLEX
    else: Es = E

    # Ensure complex type for operations
    H_LD, S_LD, H_RD, S_RD = map(lambda x: x.astype(complex), [H_LD, S_LD, H_RD, S_RD])
    gR_L, gR_R = gR_L.astype(complex), gR_R.astype(complex)

    try:
        # Left Self-Energy
        V_LD_op = H_LD - Es * S_LD
        V_DL_op = V_LD_op.conj().T # Correct hermitian conjugate
        # Check for NaN/Inf before matrix multiplication
        if np.isnan(V_LD_op).any() or np.isinf(V_LD_op).any() or \
           np.isnan(V_DL_op).any() or np.isinf(V_DL_op).any(): raise ValueError("V_LD/V_DL NaN/Inf")
        if np.isnan(gR_L).any() or np.isinf(gR_L).any(): raise ValueError("gR_L NaN/Inf")
        Sigma_R_L = V_DL_op @ gR_L @ V_LD_op
        if np.isnan(Sigma_R_L).any() or np.isinf(Sigma_R_L).any(): raise ValueError("Sigma_R_L NaN/Inf")

        # Right Self-Energy
        V_RD_op = H_RD - Es * S_RD
        V_DR_op = V_RD_op.conj().T # Correct hermitian conjugate
        # Check for NaN/Inf before matrix multiplication
        if np.isnan(V_RD_op).any() or np.isinf(V_RD_op).any() or \
           np.isnan(V_DR_op).any() or np.isinf(V_DR_op).any(): raise ValueError("V_RD/V_DR NaN/Inf")
        if np.isnan(gR_R).any() or np.isinf(gR_R).any(): raise ValueError("gR_R NaN/Inf")
        Sigma_R_R = V_DR_op @ gR_R @ V_RD_op
        if np.isnan(Sigma_R_R).any() or np.isinf(Sigma_R_R).any(): raise ValueError("Sigma_R_R NaN/Inf")

    except (ValueError, np.linalg.LinAlgError):
        # warnings.warn(f"Sigma calculation LinAlg/Value error E={E:.4f}{E.imag:+.4f}j", RuntimeWarning)
        return None, None
    except Exception as e:
        warnings.warn(f"Sigma unexpected calc error E={E}: {e}", RuntimeWarning)
        return None, None

    return Sigma_R_L, Sigma_R_R

def calculate_greens_functions_rgf(E, H_DD, S_DD, Sigma_R_L, Sigma_R_R):
    """Calculates G^R, G^A, Gamma_L, Gamma_R. Accepts complex E."""
    if Sigma_R_L is None or Sigma_R_R is None:
        # warnings.warn(f"GF calculation skipped: Sigma is None at E={E:.4f}{E.imag:+.4f}j", RuntimeWarning)
        return None # Propagate None if sigma failed

    n_D = H_DD.shape[0]
    # Check dimension consistency
    if Sigma_R_L.shape != (n_D, n_D) or Sigma_R_R.shape != (n_D, n_D):
         # warnings.warn(f"GF Sigma dimension mismatch at E={E:.4f}{E.imag:+.4f}j. Skipping.", RuntimeWarning)
         return None

    if not isinstance(E, complex): Es = E + ETA_COMPLEX
    else: Es = E

    # Ensure complex types
    H_DD, S_DD, Sigma_R_L, Sigma_R_R = map(lambda x: x.astype(complex), [H_DD, S_DD, Sigma_R_L, Sigma_R_R])

    try:
        H_eff = H_DD + Sigma_R_L + Sigma_R_R
        # Check H_eff before inversion
        if np.isnan(H_eff).any() or np.isinf(H_eff).any(): raise np.linalg.LinAlgError("H_eff NaN/Inf")

        inv_term = Es * S_DD - H_eff
        # Check term before inversion
        if np.isnan(inv_term).any() or np.isinf(inv_term).any(): raise np.linalg.LinAlgError("GF Inversion term NaN/Inf")

        G_R = np.linalg.inv(inv_term)
        # Check result of inversion
        if np.isnan(G_R).any() or np.isinf(G_R).any(): raise ValueError("G_R NaN/Inf after inversion")

        G_A = G_R.conj().T # Advanced Green's function

        # Calculate Gamma matrices (broadening)
        Sigma_A_L = Sigma_R_L.conj().T
        Sigma_A_R = Sigma_R_R.conj().T

        Gamma_L = 1j * (Sigma_R_L - Sigma_A_L)
        Gamma_R = 1j * (Sigma_R_R - Sigma_A_R)

        # Check Gamma matrices
        if np.isnan(Gamma_L).any() or np.isinf(Gamma_L).any() or \
           np.isnan(Gamma_R).any() or np.isinf(Gamma_R).any(): raise ValueError("Gamma NaN/Inf")

        # Ensure Gammas are Hermitian (numerically)
        Gamma_L = 0.5 * (Gamma_L + Gamma_L.conj().T)
        Gamma_R = 0.5 * (Gamma_R + Gamma_R.conj().T)

    except (np.linalg.LinAlgError, ValueError):
        # warnings.warn(f"GF calculation LinAlg/Value error E={E:.4f}{E.imag:+.4f}j", RuntimeWarning)
        return None
    except Exception as e:
        warnings.warn(f"GF unexpected calculation error E={E}: {e}", RuntimeWarning)
        return None

    return G_R, G_A, Gamma_L, Gamma_R

def calculate_negf_density_matrix_contour_neq(
    H_DD, S_DD, H_L0, S_L0, H_L01, S_L01, H_LD, S_LD,
    H_R0, S_R0, H_R01, S_R01, H_RD, S_RD,
    mu_L, mu_R, T, energy_grid_real):
    """Calculates NEGF density matrix using contour+poles for equilibrium part (ref mu_L)
       and real-axis integration for non-equilibrium correction."""
    neq_calc_start_time = time.time()
    n = H_DD.shape[0]
    D_DD_eq = np.zeros((n,n), dtype=complex) # Initialize complex density matrix

    # --- 1. Equilibrium Part (Contour Integration + Matsubara Poles, reference mu_L) ---
    print("  Calculating Equilibrium DM part (Contour, ref mu_L)...")
    try:
        # Define Contour based on real energy grid
        E_min, E_max = energy_grid_real.min(), energy_grid_real.max()
        # Add buffer to contour bounds
        buffer = 0.1 * abs(E_max - E_min) if E_max != E_min else 0.1
        contour_E_min = E_min - buffer
        contour_E_max = E_max + buffer
        contour = define_contour(contour_E_min, contour_E_max)
        if len(contour) < 2: raise ValueError("Contour definition failed (too few points).")

        # Prepare for trapezoidal rule on the contour
        contour_closed = np.append(contour, contour[0]) # Close the contour for integration
        dz = np.diff(contour_closed) # Complex differentials

        # Integrate G^R(z) * f(z, mu_L) along the contour
        integral_eq = np.zeros_like(D_DD_eq)
        valid_contour_points = 0
        print(f"    Integrating over {len(contour)} contour points: [", end="")
        progress_step = max(1, len(contour) // 20)
        for i, z_start in enumerate(contour):
            if i > 0 and i % progress_step == 0: print("=", end="")
            z_mid = z_start + dz[i]/2.0 # Midpoint for better accuracy

            # Calculate Sigma at the midpoint z_mid
            Sigma_L_z, Sigma_R_z = calculate_sigma(z_mid, H_L0,S_L0,H_L01,S_L01,H_LD,S_LD, H_R0,S_R0,H_R01,S_R01,H_RD,S_RD)
            if Sigma_L_z is None or Sigma_R_z is None: continue # Skip if sigma calc failed

            # Calculate Green's function G^R at z_mid
            result_z = calculate_greens_functions_rgf(z_mid, H_DD, S_DD, Sigma_L_z, Sigma_R_z)
            if result_z is None: continue # Skip if GF calc failed
            G_R_z, _, _, _ = result_z

            try:
                # Calculate Fermi-Dirac at complex energy z_mid (referenced to mu_L)
                fd_val = fermi_dirac_complex(z_mid, mu_L, T)
                if np.isnan(fd_val).any() or np.isinf(fd_val).any(): raise ValueError("FD Complex NaN/Inf")

                # Calculate integrand G^R(z) * f(z)
                integrand_segment = G_R_z * fd_val
                if np.isnan(integrand_segment).any() or np.isinf(integrand_segment).any(): raise ValueError("Integrand NaN/Inf")

                # Add segment contribution to integral
                integral_eq += integrand_segment * dz[i]
                valid_contour_points += 1
            except (ValueError, np.linalg.LinAlgError):
                # warnings.warn(f"Contour segment {i} LinAlg/Value error", RuntimeWarning)
                continue # Skip this segment
            except Exception as e:
                warnings.warn(f"Unexpected Contour segment {i} error: {e}", RuntimeWarning)
                continue # Skip this segment
        print("]")

        # Check if enough contour points were successful
        if valid_contour_points < len(contour) * 0.8:
            warnings.warn(f"Low success rate ({valid_contour_points}/{len(contour)}) for contour integration.", RuntimeWarning)
        if valid_contour_points == 0:
            raise RuntimeError("Equilibrium contour integral failed completely (0 valid points).")

        # Apply prefactor for contour integral part of DM
        D_DD_eq += (-1.0 / (np.pi * 1j)) * integral_eq

        # Add Matsubara Pole Summation (referenced to mu_L)
        kBT = KB_AU * T
        psum_eq = np.zeros_like(D_DD_eq) # Pole sum contribution
        if kBT > 1e-10 and N_POLES > 0: # Check if T is high enough and poles requested
            poles = [mu_L + 1j*(2*m+1)*np.pi*kBT for m in range(N_POLES)] # Matsubara frequencies
            print(f"    Summing over {N_POLES} Matsubara poles (ref mu_L)...")
            valid_pole_points = 0
            for zp in poles:
                 # Calculate Sigma at the pole zp
                 Sigma_L_p, Sigma_R_p = calculate_sigma(zp, H_L0,S_L0,H_L01,S_L01,H_LD,S_LD, H_R0,S_R0,H_R01,S_R01,H_RD,S_RD)
                 if Sigma_L_p is None or Sigma_R_p is None: continue # Skip if sigma fails

                 # Calculate Green's function G^R at zp
                 result_p = calculate_greens_functions_rgf(zp, H_DD, S_DD, Sigma_L_p, Sigma_R_p)
                 if result_p is None: continue # Skip if GF fails
                 Gp, _, _, _ = result_p

                 try:
                     # Check Gp before adding
                     if np.isnan(Gp).any() or np.isinf(Gp).any(): raise ValueError("Gp NaN/Inf")
                     psum_eq += Gp # Sum G^R(z_pole)
                     valid_pole_points += 1
                 except (ValueError, np.linalg.LinAlgError):
                     # warnings.warn(f"Pole calculation LinAlg/Value error", RuntimeWarning)
                     continue
                 except Exception as e:
                     warnings.warn(f"Unexpected Pole error: {e}", RuntimeWarning)
                     continue

            # Check pole summation success rate
            if valid_pole_points < N_POLES * 0.8:
                warnings.warn(f"Low success rate ({valid_pole_points}/{N_POLES}) for pole summation.", RuntimeWarning)

            # Add pole contribution to equilibrium DM (factor 2*pi*i * (kBT / (2*pi*i)) = kBT)
            D_DD_eq += kBT * psum_eq
        else:
            print("    Skipping pole summation (T=0 or N_POLES=0).")

    except Exception as e:
        warnings.warn(f"Error during equilibrium DM calculation (contour/poles): {e}. Setting EQ DM to zero.", RuntimeWarning)
        traceback.print_exc()
        D_DD_eq = np.zeros((n,n), dtype=complex) # Fallback to zero DM if calculation fails


    # --- 2. Non-Equilibrium Correction (Real Axis Integration) ---
    # Integral of (f_L - f_R) * G^R * Gamma_R * G^A / (2*pi)
    print("  Calculating Non-Equilibrium DM Correction (Real Axis)...")
    D_DD_neq = np.zeros_like(D_DD_eq) # Initialize complex NEQ correction
    n_energy = len(energy_grid_real)
    integrand_neq = np.zeros((n_energy, n, n), dtype=complex) # Store integrand values
    valid_points_neq = 0

    # Pre-calculate Fermi functions on the real energy grid
    f_L_grid = fermi_dirac(energy_grid_real, mu_L, T)
    f_R_grid = fermi_dirac(energy_grid_real, mu_R, T)
    delta_f_grid = f_L_grid - f_R_grid # Difference drives the NEQ part

    print(f"    Integrating NEQ correction over {n_energy} real points: [", end="")
    progress_step = max(1, n_energy // 20)
    for i, E_real in enumerate(energy_grid_real):
        if i > 0 and i % progress_step == 0: print("=", end="")

        # Only calculate if Fermi function difference is significant
        if abs(delta_f_grid[i]) < 1e-12: # Threshold to avoid unnecessary computation
            integrand_neq[i, :, :] = 0.0
            valid_points_neq += 1
            continue

        # Calculate Sigma at real energy E_real
        Sigma_R_L_real, Sigma_R_R_real = calculate_sigma(E_real, H_L0, S_L0, H_L01, S_L01, H_LD, S_LD, H_R0, S_R0, H_R01, S_R01, H_RD, S_RD)
        if Sigma_R_L_real is None or Sigma_R_R_real is None:
            integrand_neq[i, :, :] = np.nan # Mark as invalid if sigma failed
            continue

        # Calculate GFs and Gammas at real energy E_real
        result_real = calculate_greens_functions_rgf(E_real, H_DD, S_DD, Sigma_R_L_real, Sigma_R_R_real)
        if result_real is None:
            integrand_neq[i, :, :] = np.nan # Mark as invalid if GF failed
            continue
        G_R_real, G_A_real, _, Gamma_R_real = result_real # Need G_R, G_A, Gamma_R

        try:
            # Check for NaN/Inf in components
            if np.isnan(G_R_real).any() or np.isinf(G_R_real).any() or \
               np.isnan(Gamma_R_real).any() or np.isinf(Gamma_R_real).any() or \
               np.isnan(G_A_real).any() or np.isinf(G_A_real).any(): raise ValueError("NaN/Inf in NEQ Green's function components")

            # Calculate the core term G^R * Gamma_R * G^A
            term = G_R_real @ Gamma_R_real @ G_A_real
            if np.isnan(term).any() or np.isinf(term).any(): raise ValueError("NaN/Inf in NEQ term G*Gamma*G")

            # Full integrand segment
            integrand_neq[i, :, :] = term * delta_f_grid[i]
            valid_points_neq += 1
        except Exception as e:
            # warnings.warn(f"NEQ real axis point {i} (E={E_real:.4f}) error: {e}", RuntimeWarning)
            integrand_neq[i, :, :] = np.nan # Mark as invalid on error

    print("]")
    # Check success rate for real axis points
    if valid_points_neq < n_energy * 0.8:
        warnings.warn(f"Low success rate ({valid_points_neq}/{n_energy}) for NEQ real axis integration.", RuntimeWarning)

    # Integrate the valid points using trapezoidal rule
    mask_neq = ~np.isnan(integrand_neq[:, 0, 0]) & ~np.isnan(energy_grid_real) # Mask for valid energy points and successful calculations
    if np.sum(mask_neq) >= 2: # Need at least two points to integrate
        # Sort by energy for correct integration order
        sorted_indices = np.argsort(energy_grid_real[mask_neq])
        x_int = energy_grid_real[mask_neq][sorted_indices]
        y_int = integrand_neq[mask_neq][sorted_indices]

        # Integrate real and imaginary parts separately
        D_DD_neq_real = scipy.integrate.trapezoid(np.real(y_int), x=x_int, axis=0)
        D_DD_neq_imag = scipy.integrate.trapezoid(np.imag(y_int), x=x_int, axis=0)

        # Combine and apply 1/(2*pi) prefactor
        D_DD_neq = (D_DD_neq_real + 1j * D_DD_neq_imag) / (2 * np.pi)
    else:
        warnings.warn("Integration failed for NEQ correction (less than 2 valid points). Setting NEQ DM to zero.", RuntimeWarning)
        D_DD_neq = np.zeros_like(D_DD_eq) # Fallback

    # --- 3. Combine Equilibrium and Non-Equilibrium Parts ---
    D_DD_total = D_DD_eq + D_DD_neq # Total complex density matrix

    # Ensure hermiticity (numerically) and take real part for PySCF
    D_DD_final = 0.5 * (D_DD_total + D_DD_total.conj().T)

    # Check how large the imaginary part is after symmetrization (should be small)
    imag_norm = np.linalg.norm(np.imag(D_DD_final))
    real_norm = np.linalg.norm(np.real(D_DD_final))
    imag_norm_ratio = imag_norm / (real_norm + 1e-12) # Avoid division by zero
    if imag_norm_ratio > 1e-3: # Tolerance for imaginary part check
        warnings.warn(f"Final DM imaginary norm ratio is significant: {imag_norm_ratio:.2e}.", RuntimeWarning)

    print(f"  Total DM calculation took {time.time()-neq_calc_start_time:.2f} s")
    # Return the real part as expected by PySCF density matrix updates
    return D_DD_final.real

# --- Utility Function for Potential Alignment ---
def calculate_avg_potential(H_matrix, indices):
    """Calculates the average of the diagonal elements of a specified block."""
    if len(indices) == 0: return 0.0
    try:
        # Extract the diagonal block
        block = H_matrix[np.ix_(indices, indices)]
        # Get diagonal elements
        diag_elements = np.diag(block)
        # Return the real part of the mean
        return np.mean(diag_elements).real
    except IndexError:
        warnings.warn(f"IndexError calculating avg potential. Indices: {indices[:5]}... Max H index: {H_matrix.shape[0]-1}", RuntimeWarning)
        return 0.0
    except Exception as e:
        warnings.warn(f"Error calculating avg potential: {e}", RuntimeWarning)
        return 0.0

# --- Hamiltonian Update Function (Self-Consistent + Potential Alignment) ---
def update_hamiltonian_scf(pyscf_mf, D_DD_new, D_AO_old, device_indices, lead_L0_indices, lead_R0_indices, S_ao, scf_mixing_alpha):
    """Updates the AO Hamiltonian using PySCF's get_veff based on a mixed density, and applies potential shift."""
    print("      Running Self-Consistent Hamiltonian update (with Potential Alignment)...")
    n_ao = D_AO_old.shape[0]
    n_D = D_DD_new.shape[0]

    # Ensure D_DD_new has the correct dimensions matching device_indices
    if len(device_indices) != n_D:
        raise ValueError(f"Dimension mismatch: D_DD_new ({n_D}x{n_D}) vs device_indices ({len(device_indices)}).")

    # Create a full AO-basis density matrix, inserting the NEGF device block
    D_AO_new = D_AO_old.copy() # Start with the previous iteration's full DM
    if len(device_indices) > 0:
        # Ensure D_DD_new is real before insertion
        D_AO_new[np.ix_(device_indices, device_indices)] = D_DD_new.real
    else:
        warnings.warn("Device indices are empty, D_AO not updated with NEGF result.", RuntimeWarning)

    # Simple linear mixing of the full AO density matrix
    D_AO_mixed = scf_mixing_alpha * D_AO_new + (1.0 - scf_mixing_alpha) * D_AO_old
    # Ensure hermiticity of the mixed density matrix
    D_AO_mixed = 0.5 * (D_AO_mixed + D_AO_mixed.T)

    # Calculate the effective potential V_eff using the mixed density matrix
    try:
        # Ensure dm is float64 for PySCF
        dm_for_pyscf = D_AO_mixed.astype(np.float64, copy=False)
        # Call PySCF to get the potential (includes Hartree, XC)
        V_eff_mixed = pyscf_mf.get_veff(pyscf_mf.mol, dm=dm_for_pyscf)
    except Exception as e:
        warnings.warn(f"PySCF get_veff failed: {e}", RuntimeWarning)
        traceback.print_exc()
        print("       WARNING: V_eff calculation failed. Hamiltonian NOT updated. Returning previous H.")
        # Attempt to return the Hamiltonian from the *start* of this iteration
        H_core = pyscf_mf.get_hcore()
        try:
            # Try to recalculate V_eff with the *old* density to get the H before update attempt
            V_eff_old = pyscf_mf.get_veff(pyscf_mf.mol, dm=D_AO_old.astype(np.float64, copy=False))
            H_old = H_core + V_eff_old
            warnings.warn("       Reverted H update attempt.", RuntimeWarning)
            return H_old, D_AO_old.copy() # Return H before update and the old DM
        except Exception as e_old:
            warnings.warn(f"PySCF get_veff also failed for OLD density: {e_old}", RuntimeWarning)
            print("       SEVERE WARNING: Cannot calculate any V_eff. Returning core H.")
            return H_core, D_AO_old.copy() # Fallback: return core H and old DM

    # Construct the new Fock matrix (unshifted)
    H_core = pyscf_mf.get_hcore()
    H_ao_unshifted = H_core + V_eff_mixed

    # --- Potential Alignment Step ---
    # Calculate average potential in device and lead regions
    V_avg_D = calculate_avg_potential(H_ao_unshifted, device_indices)
    V_avg_L0 = calculate_avg_potential(H_ao_unshifted, lead_L0_indices)
    V_avg_R0 = calculate_avg_potential(H_ao_unshifted, lead_R0_indices)

    # Determine target potential (average of lead potentials if available)
    n_L0 = len(lead_L0_indices)
    n_R0 = len(lead_R0_indices)
    if n_L0 > 0 and n_R0 > 0:
        V_target = (V_avg_L0 + V_avg_R0) / 2.0
    elif n_L0 > 0: # Only left lead defined
        V_target = V_avg_L0
    elif n_R0 > 0: # Only right lead defined
        V_target = V_avg_R0
    else: # No leads defined, use device average (no shift applied)
        V_target = V_avg_D
        warnings.warn("Potential alignment target based on device itself (no lead indices?).", RuntimeWarning)

    # Calculate the raw shift needed
    potential_shift_raw = V_target - V_avg_D
    # Apply damping
    potential_shift_damped = POTENTIAL_SHIFT_DAMPING * potential_shift_raw
    print(f"      Avg Potentials: D={V_avg_D:.4f}, L0={V_avg_L0:.4f}, R0={V_avg_R0:.4f}, Target={V_target:.4f}, Shift={potential_shift_damped:.4f}")

    # Apply the damped shift only to the device block diagonal
    H_ao_shifted = H_ao_unshifted.copy()
    if len(device_indices) > 0 and abs(potential_shift_damped) > 1e-9: # Only shift if non-negligible
        # Create identity matrix matching device block size
        identity_D = np.identity(len(device_indices))
        # Add shift * Identity to the device diagonal block
        H_ao_shifted[np.ix_(device_indices, device_indices)] += potential_shift_damped * identity_D

    print("      Hamiltonian update complete (with potential shift).")
    # Return the potentially shifted Hamiltonian and the mixed density matrix used
    return H_ao_shifted, D_AO_mixed


# --- Main Workflow ---
# 1. System Setup with PySCF
print("Setting up PySCF molecule (14 Li atoms)...")
atom_list = [ ['Li', [i * 2.8, 0.0, 0.0]] for i in range(14) ]
mol = gto.M(atom=atom_list, basis='sto-3g', unit='angstrom')
mol.build()
print(f"Total number of atoms: {mol.natm}")
print(f"Total number of basis functions: {mol.nao}")

# 2. Perform Initial DFT calculation
print("Running initial PySCF DFT calculation...")
mf = dft.RKS(mol); mf.xc = 'lda'; mf.kernel()
print(f"Initial PySCF Converged: {mf.converged}")
if not mf.converged: print("Warning: Initial PySCF calculation did not converge!")

# 3. Estimate Fermi level & Setup NEGF Parameters
homo_idx = mol.nelectron // 2 - 1; lumo_idx = homo_idx + 1
if homo_idx < 0 or lumo_idx >= len(mf.mo_energy): fermi_level_estimate = 0.0
else: fermi_level_estimate = (mf.mo_energy[homo_idx] + mf.mo_energy[lumo_idx]) / 2.0
print(f"Estimated Fermi level: {fermi_level_estimate * HARTREE_TO_EV:.4f} eV ({fermi_level_estimate:.4f} Hartree)")
E_FERMI_GUESS_AU = fermi_level_estimate; bias_au = APPLIED_BIAS_VOLTS / HARTREE_TO_EV
CHEMICAL_POTENTIAL_L = E_FERMI_GUESS_AU + bias_au / 2.0; CHEMICAL_POTENTIAL_R = E_FERMI_GUESS_AU - bias_au / 2.0
print(f"Using mu_L = {CHEMICAL_POTENTIAL_L:.4f} Ha, mu_R = {CHEMICAL_POTENTIAL_R:.4f} Ha")
ENERGY_RANGE_AU = np.linspace(E_FERMI_GUESS_AU - E_WINDOW_AU / 2, E_FERMI_GUESS_AU + E_WINDOW_AU / 2, N_ENERGY_POINTS)

# 4. Extract Initial Matrices
print("Extracting initial Hamiltonian, Overlap, and Density matrices...")
D_AO_initial = mf.make_rdm1(); H_ao_initial = mf.get_fock(dm=D_AO_initial); S_ao = mf.get_ovlp()

# 5. Define Partitioning Scheme
print("Defining partitioning scheme (L1, L0, D, R0, R1)...")
# Use the same partitioning as the SCF code provided
atom_indices_L1 = [1]; atom_indices_L0 = [2]
atom_indices_D = list(range(3, 11))
atom_indices_R0 = [11]; atom_indices_R1 = [12]
ao_slices = mol.aoslice_by_atom()[:, 2:4]
def get_ao_indices(atom_idx_list):
    indices = [];
    for atom_idx in atom_idx_list:
        if atom_idx >= mol.natm: raise IndexError(f"Atom index {atom_idx} out of bounds.")
        start, stop = ao_slices[atom_idx]; indices.extend(list(range(start, stop)))
    return np.array(sorted(list(set(indices))), dtype=int)
idx_L1 = get_ao_indices(atom_indices_L1); idx_L0 = get_ao_indices(atom_indices_L0)
idx_D  = get_ao_indices(atom_indices_D); idx_R0 = get_ao_indices(atom_indices_R0)
idx_R1 = get_ao_indices(atom_indices_R1)
n_L1, n_L0, n_D, n_R0, n_R1 = map(len, [idx_L1, idx_L0, idx_D, idx_R0, idx_R1])
print(f"Partitioning dims: L1({n_L1}), L0({n_L0}), D({n_D}), R0({n_R0}), R1({n_R1})")
if n_D == 0 or n_L0 == 0 or n_R0 == 0 or n_L1 == 0 or n_R1 == 0: raise ValueError("Partitioning error.")
if n_L0 != n_L1: warnings.warn(f"Warning: Left lead L0/L1 dims differ ({n_L0} vs {n_L1}). Check partitioning.", RuntimeWarning)
if n_R0 != n_R1: warnings.warn(f"Warning: Right lead R0/R1 dims differ ({n_R0} vs {n_R1}). Check partitioning.", RuntimeWarning)

# 6. Self-Consistency Loop
print("\nStarting NEGF-SCF loop (Combined DM + SCF H UPDATE)...")
H_ao_current = H_ao_initial.copy(); D_AO_current = D_AO_initial.copy()
D_DD_old = np.zeros((n_D, n_D), dtype=float) # Initial guess for device density block
H_ao_previous = H_ao_initial.copy() # Store for convergence check if needed
converged = False; start_time = time.time()
H_ao_converged = None # Store the converged Hamiltonian
D_AO_converged = None # Store the converged AO density
D_DD_converged = None # Store the converged Device density block

for scf_iter in range(SCF_MAX_ITER):
    iter_start_time = time.time(); print(f"\n--- SCF Iteration {scf_iter + 1} ---")

    # Partition CURRENT Hamiltonian and Overlap for NEGF calculation
    H_ao_current_f64 = H_ao_current.astype(np.float64); S_ao_f64 = S_ao.astype(np.float64)
    S_DD = S_ao_f64[np.ix_(idx_D, idx_D)]; H_DD = H_ao_current_f64[np.ix_(idx_D, idx_D)]
    H_L_0 = H_ao_current_f64[np.ix_(idx_L0, idx_L0)]; S_L_0 = S_ao_f64[np.ix_(idx_L0, idx_L0)]
    H_L_01= H_ao_current_f64[np.ix_(idx_L0, idx_L1)]; S_L_01= S_ao_f64[np.ix_(idx_L0, idx_L1)]
    H_LD = H_ao_current_f64[np.ix_(idx_L0, idx_D)];  S_LD = S_ao_f64[np.ix_(idx_L0, idx_D)]
    H_R_0 = H_ao_current_f64[np.ix_(idx_R0, idx_R0)]; S_R_0 = S_ao_f64[np.ix_(idx_R0, idx_R0)]
    H_R_01= H_ao_current_f64[np.ix_(idx_R0, idx_R1)]; S_R_01= S_ao_f64[np.ix_(idx_R0, idx_R1)]
    H_RD = H_ao_current_f64[np.ix_(idx_R0, idx_D)];  S_RD = S_ao_f64[np.ix_(idx_R0, idx_D)]

    # Calculate NEGF density matrix for the device block
    try:
        D_DD_new = calculate_negf_density_matrix_contour_neq(
            H_DD, S_DD, H_L_0, S_L_0, H_L_01, S_L_01, H_LD, S_LD,
            H_R_0, S_R_0, H_R_01, S_R_01, H_RD, S_RD,
            CHEMICAL_POTENTIAL_L, CHEMICAL_POTENTIAL_R, TEMPERATURE, ENERGY_RANGE_AU)
        if D_DD_new is None: raise RuntimeError("Density matrix calculation returned None.")
        num_electrons_D = np.trace(D_DD_new @ S_DD); print(f"  Number of electrons in Device (Tr[D_DD*S_DD]): {num_electrons_D:.4f}")
    except Exception as e:
        print(f"\n***FATAL ERROR*** during density calculation in iter {scf_iter+1}: {e}")
        traceback.print_exc()
        converged = False # Mark as not converged on error
        break # Exit SCF loop

    # Check convergence based on the change in the device density block
    diff_norm = np.linalg.norm(D_DD_new - D_DD_old)
    print(f"  Density matrix change (norm D_DD): {diff_norm:.6e}")
    if diff_norm < SCF_TOLERANCE and scf_iter > 0: # Require at least one update
        converged = True
        H_ao_converged = H_ao_current.copy() # Save the H that yielded this density
        D_AO_converged = D_AO_current.copy() # Save the AO density used for this H
        D_DD_converged = D_DD_new.copy()     # Save the converged device density
        print(f"\nNEGF-SCF Converged after {scf_iter + 1} iterations!")
        break # Exit SCF loop

    # Store the new density for the next iteration's comparison
    D_DD_old = D_DD_new.copy() # No mixing needed here, update_hamiltonian_scf does mixing

    # Update the Hamiltonian using the new device density and the previous full AO density
    try:
        H_ao_next, D_AO_mixed = update_hamiltonian_scf(
            mf, D_DD_new, D_AO_current, idx_D, idx_L0, idx_R0, S_ao, SCF_MIXING_ALPHA)
        # Check change in Hamiltonian (optional convergence criterion)
        delta_H_norm = np.linalg.norm(H_ao_next - H_ao_current)
        print(f"  Hamiltonian change (norm H_new - H_old): {delta_H_norm:.6e}")

        # Update current H and AO density for the next iteration
        H_ao_current = H_ao_next
        D_AO_current = D_AO_mixed # Use the mixed density from the update function

    except Exception as e:
        print(f"\n***FATAL ERROR*** during Hamiltonian update in iter {scf_iter+1}: {e}")
        traceback.print_exc()
        converged = False # Mark as not converged on error
        break # Exit SCF loop

    print(f"  Iteration {scf_iter + 1} time: {time.time() - iter_start_time:.2f} s")

# Handle case where loop finishes without converging
if not converged:
    warnings.warn(f"NEGF-SCF did not converge within {SCF_MAX_ITER} iterations.", RuntimeWarning)
    # Use the last calculated values if not converged
    H_ao_converged = H_ao_current.copy()
    D_AO_converged = D_AO_current.copy()
    D_DD_converged = D_DD_old.copy() # Use the density from the start of the last iteration

total_scf_time = time.time() - start_time; print(f"\nTotal SCF loop time: {total_scf_time:.2f} s")

# 7. Post-Processing: Calculate Transmission and DOS using INITIAL Hamiltonian
print("\nPerforming post-processing (Transmission, DOS using INITIAL H)...") # <<< MODIFIED PRINT
transmission = np.zeros(len(ENERGY_RANGE_AU)); dos_trace = np.zeros(len(ENERGY_RANGE_AU))
valid_post_points = 0; post_start_time = time.time()

# --- Partition INITIAL matrices for post-processing ---  # <<< MODIFIED BLOCK START >>>
H_ao_initial_f64 = H_ao_initial.astype(np.float64) # Use INITIAL Hamiltonian
S_ao_f64 = S_ao.astype(np.float64) # Overlap is constant

H_DD_i = H_ao_initial_f64[np.ix_(idx_D, idx_D)]; S_DD_i = S_ao_f64[np.ix_(idx_D, idx_D)]
H_L_0_i = H_ao_initial_f64[np.ix_(idx_L0, idx_L0)]; S_L_0_i = S_ao_f64[np.ix_(idx_L0, idx_L0)]
H_L_01_i= H_ao_initial_f64[np.ix_(idx_L0, idx_L1)]; S_L_01_i= S_ao_f64[np.ix_(idx_L0, idx_L1)]
H_LD_i = H_ao_initial_f64[np.ix_(idx_L0, idx_D)];  S_LD_i = S_ao_f64[np.ix_(idx_L0, idx_D)]
H_R_0_i = H_ao_initial_f64[np.ix_(idx_R0, idx_R0)]; S_R_0_i = S_ao_f64[np.ix_(idx_R0, idx_R0)]
H_R_01_i= H_ao_initial_f64[np.ix_(idx_R0, idx_R1)]; S_R_01_i= S_ao_f64[np.ix_(idx_R0, idx_R1)]
H_RD_i = H_ao_initial_f64[np.ix_(idx_R0, idx_D)];  S_RD_i = S_ao_f64[np.ix_(idx_R0, idx_D)]
# <<< MODIFIED BLOCK END >>>

print("  Calculating Transmission/DOS (using Initial H): [", end="")
progress_step = max(1, len(ENERGY_RANGE_AU) // 20)
for i, E in enumerate(ENERGY_RANGE_AU):
    if i > 0 and i % progress_step == 0: print("=", end="")
    # Use INITIAL matrices for sigma and GF calculation
    Sigma_R_L_i, Sigma_R_R_i = calculate_sigma(E, H_L_0_i, S_L_0_i, H_L_01_i, S_L_01_i, H_LD_i, S_LD_i, H_R_0_i, S_R_0_i, H_R_01_i, S_R_01_i, H_RD_i, S_RD_i)
    result_i = calculate_greens_functions_rgf(E, H_DD_i, S_DD_i, Sigma_R_L_i, Sigma_R_R_i)

    if result_i is None:
        transmission[i], dos_trace[i] = np.nan, np.nan
        continue

    G_R_i, G_A_i, Gamma_L_i, Gamma_R_i = result_i

    # Calculate Transmission T(E) = Tr[Gamma_L * G_R * Gamma_R * G_A]
    try:
        if np.isnan(Gamma_L_i).any() or np.isinf(Gamma_L_i).any() or np.isnan(G_R_i).any() or np.isinf(G_R_i).any() or \
           np.isnan(Gamma_R_i).any() or np.isinf(Gamma_R_i).any() or np.isnan(G_A_i).any() or np.isinf(G_A_i).any(): raise ValueError("NaN/Inf in T components")
        T_matrix = Gamma_L_i @ G_R_i @ Gamma_R_i @ G_A_i
        if np.isnan(T_matrix).any() or np.isinf(T_matrix).any(): raise ValueError("NaN/Inf in T_matrix")
        transmission[i] = np.trace(T_matrix).real
        # Ensure transmission is non-negative (can be slightly negative due to numerical noise)
        transmission[i] = max(0.0, transmission[i])
    except Exception as e:
        # warnings.warn(f"Transmission calc failed at E={E:.4f}: {e}", RuntimeWarning)
        transmission[i] = np.nan

    # Calculate Projected DOS = Tr[A(E) * S_DD] / (2*pi), where A(E) = i(G^R - G^A)
    try:
        A_E = 1j * (G_R_i - G_A_i)
        A_E = 0.5 * (A_E + A_E.conj().T) # Ensure Hermitian
        if np.isnan(A_E).any() or np.isinf(A_E).any(): raise ValueError("NaN/Inf in A(E)")
        dos_trace[i] = np.trace(A_E @ S_DD_i).real / (2 * np.pi) # Use S_DD_i
        # Ensure DOS is non-negative
        dos_trace[i] = max(0.0, dos_trace[i])
    except Exception as e:
        # warnings.warn(f"DOS calc failed at E={E:.4f}: {e}", RuntimeWarning)
        dos_trace[i] = np.nan

    # Count successful points
    if not np.isnan(transmission[i]) and not np.isnan(dos_trace[i]):
        valid_post_points +=1

print("]")
total_post_time = time.time() - post_start_time; print(f"Post-processing time: {total_post_time:.2f} s")
if valid_post_points < len(ENERGY_RANGE_AU) * 0.8:
    warnings.warn(f"Post-processing failed for many points ({len(ENERGY_RANGE_AU)-valid_post_points}/{len(ENERGY_RANGE_AU)}).", RuntimeWarning)

# 8. Calculate Current (using results from Initial H)
current = 0.0
if abs(CHEMICAL_POTENTIAL_L - CHEMICAL_POTENTIAL_R) > 1e-9 and valid_post_points > 1:
    print("Calculating current...")
    fL = fermi_dirac(ENERGY_RANGE_AU, CHEMICAL_POTENTIAL_L, TEMPERATURE)
    fR = fermi_dirac(ENERGY_RANGE_AU, CHEMICAL_POTENTIAL_R, TEMPERATURE)
    integrand = transmission * (fL - fR) # Use transmission calculated with Initial H
    mask = ~np.isnan(integrand) & ~np.isnan(ENERGY_RANGE_AU)
    if np.sum(mask) > 1:
        sorted_indices = np.argsort(ENERGY_RANGE_AU[mask])
        x_int = ENERGY_RANGE_AU[mask][sorted_indices]
        y_int = integrand[mask][sorted_indices]
        current = LANDAUER_PREFACTOR_AU * scipy.integrate.trapezoid(y_int, x=x_int)
        print(f"Calculated Current (from Initial H): {current:.6e} atomic units")
    else:
        print("Current calculation failed: Not enough valid points in integrand.")
elif valid_post_points <= 1:
    print("Skipping current calculation: Not enough valid post-processing points.")
else:
    print("Skipping current calculation: Zero bias.")

# 9. Plotting Results (using Matplotlib from original negf7.py)
print("Plotting results using Matplotlib...")
energy_ev = ENERGY_RANGE_AU * HARTREE_TO_EV
mu_L_ev = CHEMICAL_POTENTIAL_L * HARTREE_TO_EV
mu_R_ev = CHEMICAL_POTENTIAL_R * HARTREE_TO_EV
fermi_ev = E_FERMI_GUESS_AU * HARTREE_TO_EV

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Transmission
color = 'tab:red' # Transmission color
ax1.set_xlabel(f'Energy (eV)')
ax1.set_ylabel('Transmission T(E)', color=color)
valid_mask_t = ~np.isnan(transmission) & ~np.isnan(energy_ev)
if np.any(valid_mask_t):
    ax1.plot(energy_ev[valid_mask_t], transmission[valid_mask_t], color=color, label='T(E)', linewidth=1.5)
    max_T = np.nanmax(transmission[valid_mask_t]) if np.sum(valid_mask_t)>0 else 1.0
    ax1.set_ylim(bottom=-0.05, top=max(max_T * 1.1, 0.5)) # Adjust ylim minimum
else:
    ax1.set_ylim(bottom=-0.05, top=0.5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

# Plot vertical lines for Fermi level and chemical potentials
ax1.axvline(fermi_ev, color='gray', linestyle='-', linewidth=1.0, label=f'E_Fermi={fermi_ev:.2f} eV')
ax1.axvline(mu_L_ev, color='blue', linestyle=':', linewidth=1.5, label=f'μ_L={mu_L_ev:.2f} eV')
if abs(mu_L_ev - mu_R_ev) > 1e-6: # Only plot mu_R if bias is applied
    ax1.axvline(mu_R_ev, color='red', linestyle='--', linewidth=1.5, label=f'μ_R={mu_R_ev:.2f} eV')

# Plot Projected DOS on secondary y-axis
ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
color = 'tab:blue' # DOS color
ax2.set_ylabel('Projected DOS (a.u.)', color=color)
valid_mask_dos = ~np.isnan(dos_trace) & ~np.isnan(energy_ev)
if np.any(valid_mask_dos):
    ax2.plot(energy_ev[valid_mask_dos], dos_trace[valid_mask_dos], color=color, linestyle='--', label='DOS', linewidth=1.5)
    max_dos = np.nanmax(dos_trace[valid_mask_dos]) if np.sum(valid_mask_dos)>0 else 0.1
    ax2.set_ylim(bottom=0, top=max(max_dos * 1.1, 0.1)) # Ensure y-axis starts at 0 for DOS
else:
    ax2.set_ylim(bottom=0, top=0.1)
ax2.tick_params(axis='y', labelcolor=color)

# Title and Legend
scf_status = "Converged" if converged else f"Not Converged ({SCF_MAX_ITER} iter)"
plt.title(f'NEGF Results: 14 Li Chain ({scf_status}, V={APPLIED_BIAS_VOLTS}V, Post-Proc: Initial H)')

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
unique_labels = {} # Use dict to handle potential duplicate labels (like Fermi line if plotted on both)
for line, label in zip(lines + lines2, labels + labels2):
    if label not in unique_labels:
        unique_labels[label] = line
ax1.legend(unique_labels.values(), unique_labels.keys(), loc='best')

fig.tight_layout() # otherwise the right y-label is slightly clipped
plt.show()

print("\nNEGF script finished.")