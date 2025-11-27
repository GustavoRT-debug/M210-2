import numpy as np
import pandas as pd
import streamlit as st


# ---------- SIMPLEX + SENSIBILIDADE ----------
def simplex_with_sensitivity(A, b, c, tol=1e-9, max_iter=1000, return_basis=False):
    A, b, c = np.array(A, float), np.array(b, float), np.array(c, float)
    m, n = A.shape

    if np.any(b < -tol):
        raise ValueError("Modelo fora da forma padrão (b < 0 após normalização).")

    tab = np.zeros((m + 1, n + m + 1))
    tab[:m, :n] = A
    tab[:m, n:n + m] = np.eye(m)
    tab[:m, -1] = b
    tab[m, :n] = -c
    basis = list(range(n, n + m))

    for _ in range(max_iter):
        j = tab[m, :-1].argmin()
        if tab[m, j] >= -tol:
            break

        col = tab[:m, j]
        if np.all(col <= tol):
            raise ValueError("Problema ilimitado.")

        ratios = np.array([tab[i, -1] / col[i] if col[i] > tol else np.inf for i in range(m)])
        i = ratios.argmin()

        tab[i] /= tab[i, j]
        for r in range(m + 1):
            if r != i:
                tab[r] -= tab[r, j] * tab[i]
        basis[i] = j

    x_full = np.zeros(n + m)
    for r, v in enumerate(basis):
        x_full[v] = tab[r, -1]
    x, z = x_full[:n], float(tab[-1, -1])

    if np.any(x < -tol):
        raise ValueError("Solução viola x ≥ 0.")

    A_ext = np.hstack([A, np.eye(m)])
    B = A_ext[:, basis]
    B_inv = np.linalg.inv(B)
    c_ext = np.concatenate([c, np.zeros(m)])
    pi = B_inv.T @ c_ext[basis]
    x_B = tab[:m, -1]

    info = []
    for i in range(m):
        d = B_inv[:, i]
        dmin, dmax = -np.inf, np.inf
        for k in range(m):
            if abs(d[k]) > tol:
                bound = -x_B[k] / d[k]
                dmin, dmax = (max(dmin, bound), dmax) if d[k] > 0 else (dmin, min(dmax, bound))
        dmin, dmax = min(0, dmin), max(0, dmax)

        info.append({
            "constraint_index": i,
            "shadow_price": round(pi[i], 2),
            "delta_min": round(dmin, 2),
            "delta_max": round(dmax, 2),
            "b_interval": (round(b[i] + dmin, 2), round(b[i] + dmax, 2)),
        })

    return (x, z, info, basis) if return_basis else (x, z, info)


def convert_info_to_original(info_norm, b, ineq_signs):
    sign = np.array([1.0 if s == "≤" else -1.0 for s in ineq_signs])
    out = []
    for i, r in enumerate(info_norm):
        if sign[i] == 1:
            pi_o, dmin_o, dmax_o = r["shadow_price"], r["delta_min"], r["delta_max"]
        else:
            pi_o, dmin_o, dmax_o = -r["shadow_price"], -r["delta_max"], -r["delta_min"]

        out.append({
            "constraint_index": i,
            "shadow_price": round(pi_o, 2),
            "delta_min": round(dmin_o, 2),
            "delta_max": round(dmax_o, 2),
            "b_interval": (round(b[i] + dmin_o, 2), round(b[i] + dmax_o, 2)),
            "ineq": ineq_signs[i],
        })
    return out


def testar_variacao_simultanea(delta_norm, b_norm, A_norm, basis, c):
    m = len(b_norm)
    A_ext = np.hstack([A_norm, np.eye(m)])
    B_inv = np.linalg.inv(A_ext[:, basis])

    novo_xB = B_inv @ (b_norm + delta_norm)
    viavel = np.all(novo_xB >= -1e-9)

    c_ext = np.concatenate([c, np.zeros(m)])
    pi = B_inv.T @ c_ext[basis]

    return viavel, pi


# ---------- APP ----------
def main():
    st.set_page_config(page_title="Simplex com Sensibilidade", layout="wide")
    st.title("Simplex com Análise de Sensibilidade")

    m = st.sidebar.number_input("Número de restrições", 1, 6, 2, 1)
    n = st.sidebar.number_input("Número de variáveis", 2, 4, 2, 1)

    st.subheader("Função objetivo (max z)")
    c = [st.number_input(f"c{j+1}", value=1.0) for j in range(n)]

    st.subheader("Restrições")
    A, b, ineq = np.zeros((m, n)), np.zeros(m), []
    for i in range(m):
        cols = st.columns(n + 2)
        for j in range(n):
            A[i, j] = cols[j].number_input(f"a{i+1}{j+1}", value=1.0)
        ineq.append(cols[n].selectbox("Tipo", ["≤", "≥"], key=i))
        b[i] = cols[n + 1].number_input(f"b{i+1}", value=10.0)

    if st.button("Resolver"):
        try:
            sign = np.array([1 if s == "≤" else -1 for s in ineq])
            A_n, b_n = A * sign[:, None], b * sign
            x, z, info, basis = simplex_with_sensitivity(A_n, b_n, c, return_basis=True)

            st.session_state.sol = {
                "x": x, "z": z, "info": convert_info_to_original(info, b, ineq),
                "A": A, "b": b, "ineq": ineq, "c": c,
                "basis": basis
            }
            st.success("Solução encontrada com sucesso.")
        except Exception as e:
            st.error(e)

    sol = st.session_state.get("sol")
    if not sol:
        return

    st.subheader("Solução ótima")
    for i, val in enumerate(sol["x"]):
        st.metric(f"x{i+1}", f"{val:.2f}")
    st.metric("z*", f"{sol['z']:.2f}")

    st.subheader("Preços-sombra")
    df = pd.DataFrame(sol["info"])
    st.dataframe(df.style.format(precision=2), use_container_width=True)

    # -------- VARIAÇÃO SIMULTÂNEA Δ --------
    st.subheader("Variação simultânea dos recursos (Δ)")
    delta = [st.number_input(f"Δ{i+1}", value=0.0) for i in range(m)]

    if st.button("Testar variação"):
        sign = np.array([1 if s == "≤" else -1 for s in ineq])
        A_n, b_n = A * sign[:, None], b * sign
        delta_n = np.array(delta) * sign

        viavel, pi = testar_variacao_simultanea(
            delta_n, b_n, A_n, sol["basis"], np.array(c)
        )

        if viavel:
            z_novo = sol["z"] + pi @ delta_n
            st.success(f"✅ Viável | Novo lucro: z' = {z_novo:.2f}")
        else:
            st.error("❌ Variação simultânea NÃO viável.")


if __name__ == "__main__":
    main()

