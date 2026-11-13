#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

#рівномірна сітка N>=3
def create_grid(xi_min, xi_max, N):
    if N < 3:
        raise ValueError("N must be >= 3.")
    xi = np.linspace(xi_min, xi_max, N)
    h = (xi_max - xi_min) / (N - 1)
    xi_int = xi[1:-1]  # u_0 = u_N-1 = 0
    return xi, xi_int, h

# тридіагональна матриця H0 (ненульові елементи)
def build_tridiagonal_H0(U_int, h):
    M = len(U_int)
    d = (2.0 / (h ** 2)) + U_int
    e = np.full(M - 1, -1.0 / (h ** 2)) # побічні діагоналі
    return d, e

# розв'язок лінійної задачі
def solve_linear_eigenproblem(xi, U_int, h, num_states=3):
    d, e = build_tridiagonal_H0(U_int, h)
    eigenvalues, eigenvectors = eigh_tridiagonal(
        d, e, select='i', select_range=(0, num_states - 1)
    )

    N = len(xi)
    states = []

    for idx in range(num_states):
        lam = eigenvalues[idx]
        u_int = eigenvectors[:, idx]

        # дискретне нормування
        u_int = u_int / np.sqrt(h * np.sum(u_int ** 2))

        # узгодження знаку
        first_nonzero = np.argmax(np.abs(u_int) > 1e-6)
        if u_int[first_nonzero] < 0:
            u_int = -u_int

        u_full = np.zeros(N)
        u_full[1:-1] = u_int
        states.append((lam, u_full))

    return states

# характеристики 
def compute_diagnostics(xi, u_full, lam, U_int, h, gamma=0.0):
    u_int = u_full[1:-1]
    d, e = build_tridiagonal_H0(U_int, h)

    H0_u = d * u_int
    H0_u[:-1] += e * u_int[1:]
    H0_u[1:] += e * u_int[:-1]

    # перевірка дискретного нормування
    delta_N = np.abs(h * np.sum(u_int ** 2) - 1.0)

    # 
    u_H0_u = h * np.dot(u_int, H0_u)
    sum_u4 = h * np.sum(u_int ** 4)
    
    E_h = u_H0_u + (gamma / 2.0) * sum_u4
    lambda_h = u_H0_u + gamma * sum_u4 # співвідношення Релея
    identity_diff = np.abs((lambda_h - E_h) - (gamma / 2.0) * sum_u4)

    # масштабована нев'язка 
    r = H0_u + gamma * (u_int ** 3) - lam * u_int
    norm_r = np.sqrt(h * np.sum(r ** 2))
    norm_H0_u = np.sqrt(h * np.sum(H0_u ** 2))
    R_rel = norm_r / (1.0 + np.abs(lam) + norm_H0_u)

    return {
        "lambda": lam,
        "E_h": E_h,
        "delta_N": delta_N,
        "identity_diff": identity_diff,
        "R_rel": R_rel,
    }


# ітераційний процес
def solve_scf_branch(
    xi, U_int, h, gamma, branch_n=1,
    alpha=0.5, max_iter=500, tol_rho=1e-5, tol_res=1e-5, tol_norm=1e-7
):
    N = len(xi)
    d0, e = build_tridiagonal_H0(U_int, h)

    # Початкове наближення
    states_lin = solve_linear_eigenproblem(xi, U_int, h, num_states=branch_n)
    _, u_lin_full = states_lin[branch_n - 1]
    u_prev = u_lin_full[1:-1].copy()
    rho_k = u_prev ** 2

    rho_k = rho_k / (h * np.sum(rho_k))

    converged = False
    iter_count = 0
    lam_v = 0.0
    u_curr = u_prev.copy()

    for k in range(1, max_iter + 1):
        iter_count = k

        # ефективна головна діагональ
        d_k = d0 + gamma * rho_k

        # Крок 2. Знаходження n-ї власної пари поточної лінійної матриці
        _, evecs = eigh_tridiagonal(
            d_k, e, select='i', select_range=(0, branch_n - 1)
        )
        v_next = evecs[:, branch_n - 1]

        # нормування та фіксація знака
        v_next = v_next / np.sqrt(h * np.sum(v_next ** 2))
        if h * np.dot(v_next, u_prev) < 0:
            v_next = -v_next

        # нове значення густини
        rho_tilde = v_next ** 2

        # релаксація
        rho_next = (1.0 - alpha) * rho_k + alpha * rho_tilde
        rho_next = rho_next / (h * np.sum(rho_next))

        # Контроль самоузгодженості
        delta_rho = np.sqrt(h * np.sum((rho_next - rho_k) ** 2)) # зміна густини
        delta_N = np.abs(h * np.sum(v_next ** 2) - 1.0) # дискретне нормування
        # нев'язка
        H0_v = d0 * v_next
        H0_v[:-1] += e * v_next[1:]
        H0_v[1:] += e * v_next[:-1]

        lam_v = h * np.dot(v_next, H0_v) + gamma * h * np.sum(v_next ** 4)
        r = H0_v + gamma * (v_next ** 3) - lam_v * v_next
        R_rel = np.sqrt(h * np.sum(r ** 2)) / (1.0 + np.abs(lam_v) + np.sqrt(h * np.sum(H0_v ** 2)))

        u_curr = v_next
        u_prev = v_next
        rho_k = rho_next

        if delta_rho <= tol_rho and R_rel <= tol_res and delta_N <= tol_norm:
            converged = True
            break

    u_full = np.zeros(N)
    u_full[1:-1] = u_curr

    diag = compute_diagnostics(xi, u_full, lam_v, U_int, h, gamma=gamma)
    diag["converged"] = converged
    diag["iterations"] = iter_count

    return u_full, diag


#%% ПОТЕНЦІАЛИ ДЛЯ ЛІНІЙНОЇ ЗАДАЧІ
# нескінченно глибока прямокутна яма
def potential_infinite_well(xi):
    return np.zeros_like(xi)

# скінченна прямокутна яма
def potential_finite_well(xi, a, U_b):
    return np.where(np.abs(xi) <= a, 0.0, U_b)

# симетрична подвійна прямокутна яма
def potential_double_well(xi, b, a, U_c, U_out):
    U = np.full_like(xi, U_out, dtype=float)
    U[np.abs(xi) <= a] = 0.0
    U[np.abs(xi) <= b] = U_c
    return U
#%% ТЕСТИ


def run_test_A(L=4.0):

    print(f"Test A: linear infinite well (L = {L})")
    

    N_list = [201, 401, 801, 1601]
    errors_cont_n1 = []

    for N in N_list:
        xi, xi_int, h = create_grid(-L / 2, L / 2, N)
        U_int = np.zeros_like(xi_int)  
        states = solve_linear_eigenproblem(xi, U_int, h, num_states=3)

       
        print(f"\n N = {N} (h = {h:.6f})")
        print(f"{'n':<3} | {'lambda_num':<12} | {'error disc':<15} | {'error cont':<15}")
        print("-" * 55)

        for n in range(1, 4):
            lam_num, _ = states[n - 1]
            lam_disc_exact = (4.0 / (h ** 2)) * (np.sin(n * np.pi / (2.0 * (N - 1))) ** 2)
            lam_cont_exact = (n * np.pi / L) ** 2

            err_disc = np.abs(lam_num - lam_disc_exact)
            err_cont = np.abs(lam_num - lam_cont_exact)

            if n == 1:
                errors_cont_n1.append(err_cont)

            print(f"{n:<3} | {lam_num:<12.8f} | {err_disc:<15.2e} | {err_cont:<15.2e}")

    print("\nOrder of convergence p = log2(err(2h) / err(h)) for n = 1:")
    for i in range(1, len(N_list)):
        order = np.log2(errors_cont_n1[i - 1] / errors_cont_n1[i])
        print(f"N: {N_list[i-1]} -> {N_list[i]} | p = {order:.4f}")


def run_test_B(L=4.0, N=401):
    print(f"\nTest B: gamma = 0 in SCF (L = {L}, N = {N})")
    

    xi, xi_int, h = create_grid(-L / 2, L / 2, N)
    U_int = np.zeros_like(xi_int)

    print(f"{'n':<3} | {'Convergence':<10} | {'Iter.':<9} | {'lambda':<12} | {'delta_N':<10} | {'R_rel':<10}")
    print("-" * 70)

    for n in [1, 2, 3]:
        _, diag = solve_scf_branch(xi, U_int, h, gamma=0.0, branch_n=n, alpha=1.0)
        print(
            f"{n:<3} | {str(diag['converged']):<10} | {diag['iterations']:<9d} | "
            f"{diag['lambda']:<12.8f} | {diag['delta_N']:<10.2e} | {diag['R_rel']:<10.2e}"
        )


if __name__ == "__main__":
    run_test_A(L=4.0)
    run_test_B(L=4.0, N=401)

# %%
