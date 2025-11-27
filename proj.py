import numpy as np
import pandas as pd
import streamlit as st


# ---------- SIMPLEX + SENSIBILIDADE ----------
def simplex_with_sensitivity(A, b, c, tol=1e-9, max_iter=1000):
    A, b, c = np.array(A, float), np.array(b, float), np.array(c, float)
    m, n = A.shape

    if np.any(b < -tol):
        raise ValueError(
            "Modelo fora da forma padrão (algum b < 0 após normalização). "
            "Esse caso exigiria Fase I do Simplex."
        )

    tab = np.zeros((m + 1, n + m + 1))
    tab[:m, :n] = A
    tab[:m, n:n + m] = np.eye(m)
    tab[:m, -1] = b
    tab[m, :n] = -c
    basis = list(range(n, n + m))

    for _ in range(max_iter):
        obj = tab[m, :-1]
        j = obj.argmin()
        if obj[j] >= -tol:
            break
        col = tab[:m, j]
        if np.all(col <= tol):
            raise ValueError("Problema ilimitado.")
        ratios = np.array([tab[i, -1] / col[i] if col[i] > tol else np.inf for i in range(m)])
        i = ratios.argmin()
        tab[i, :] /= tab[i, j]
        for r in range(m + 1):
            if r != i:
                tab[r, :] -= tab[r, j] * tab[i, :]
        basis[i] = j

    x_full = np.zeros(n + m)
    for r, v in enumerate(basis):
        x_full[v] = tab[r, -1]
    x = x_full[:n]
    z = float(tab[-1, -1])

    if np.any(x < -tol):
        raise ValueError("Solução viola x >= 0. Provável inviabilidade.")

    A_ext = np.hstack([A, np.eye(m)])
    B = A_ext[:, basis]
    B_inv = np.linalg.inv(B)
    c_ext = np.concatenate([c, np.zeros(m)])
    pi = B_inv.T @ c_ext[basis]
    x_B = np.array([tab[r, -1] for r in range(m)])

    info = []
    for i in range(m):
        e_i = np.zeros(m); e_i[i] = 1
        d = B_inv @ e_i
        dmin, dmax = -np.inf, np.inf
        for k in range(m):
            if abs(d[k]) <= tol:
                continue
            bound = -x_B[k] / d[k]
            if d[k] > 0:
                dmin = max(dmin, bound)
            else:
                dmax = min(dmax, bound)
        if dmin > 0: dmin = 0
        if dmax < 0: dmax = 0
        info.append({
            "constraint_index": i,
            "shadow_price": float(pi[i]),
            "delta_min": float(dmin),
            "delta_max": float(dmax),
            "b_interval": (float(b[i] + dmin), float(b[i] + dmax)),
        })
    return x, z, info


def convert_info_to_original(info_norm, b, ineq_signs):
    sign = np.array([1.0 if s == "≤" else -1.0 for s in ineq_signs])
    out = []
    for i, r in enumerate(info_norm):
        pi_n, dmin_n, dmax_n = r["shadow_price"], r["delta_min"], r["delta_max"]
        if sign[i] == 1:
            pi_o, dmin_o, dmax_o = pi_n, dmin_n, dmax_n
        else:
            pi_o, dmin_o, dmax_o = -pi_n, -dmax_n, -dmin_n
        out.append({
            "constraint_index": r["constraint_index"],
            "shadow_price": float(pi_o),
            "delta_min": float(dmin_o),
            "delta_max": float(dmax_o),
            "b_interval": (float(b[i] + dmin_o), float(b[i] + dmax_o)),
            "ineq": ineq_signs[i],
        })
    return out


# ---------- APP ----------
def main():
    st.set_page_config(page_title="Simplex com Sensibilidade", layout="wide")
    st.title("Simplex com Análise de Sensibilidade")

    m = st.sidebar.number_input("Número de restrições (m)", 1, 6, 2, 1)
    n = st.sidebar.number_input("Número de variáveis (n)", 2, 4, 2, 1)

    st.subheader("Função objetivo (max z)")
    c = [st.number_input(f"c{j+1}", value=1.0, key=f"c_{j}") for j in range(n)]

    st.subheader("Restrições originais")
    A = np.zeros((m, n)); b = np.zeros(m); ineq_signs = []
    for i in range(m):
        cols = st.columns(n + 2)
        for j in range(n):
            A[i, j] = cols[j].number_input(
                f"a{i+1}{j+1}",
                value=1.0 if (i == 0 and j == 0) else 0.0,
                key=f"a_{i}_{j}",
            )
        ineq_signs.append(cols[n].selectbox(f"Tipo {i+1}", ["≤", "≥"], key=f"ineq_{i}"))
        b[i] = cols[n + 1].number_input(f"b{i+1}", value=10.0, key=f"b_{i}")

    if st.button("Calcular solução ótima"):
        try:
            sign = np.array([1.0 if s == "≤" else -1.0 for s in ineq_signs])
            A_norm, b_norm = A * sign[:, None], b * sign
            x_opt, z_opt, info_norm = simplex_with_sensitivity(A_norm, b_norm, c)
            st.session_state["sol"] = {
                "x_opt": x_opt.tolist(),
                "z_opt": float(z_opt),
                "info": convert_info_to_original(info_norm, b, ineq_signs),
                "A": A.tolist(),
                "b": b.tolist(),
                "ineq": ineq_signs,
                "c": c,
            }
            st.success("Solução ótima calculada.")
        except Exception as e:
            st.error(f"Erro ao resolver: {e}")

    sol = st.session_state.get("sol")
    if not sol:
        return

    x_opt = np.array(sol["x_opt"])
    z_opt = sol["z_opt"]
    info = sol["info"]
    A_sol = np.array(sol["A"])
    b_sol = np.array(sol["b"])
    ineq_signs = sol["ineq"]; c_sol = sol["c"]
    m, n = A_sol.shape

    st.subheader("Solução ótima (original)")
    cols = st.columns(n + 1)
    for j in range(n):
        cols[j].metric(f"x{j+1}*", f"{x_opt[j]:.4f}")
    cols[-1].metric("z*", f"{z_opt:.4f}")

    st.subheader("Preços-sombra (originais)")
    df = []
    for r in info:
        i = r["constraint_index"]
        bmin, bmax = r["b_interval"]
        df.append({
            "Restrição": f"R{i+1}",
            "Tipo": ineq_signs[i],
            "π": r["shadow_price"],
            "Δb_min": r["delta_min"],
            "Δb_max": r["delta_max"],
            "b_min": bmin,
            "b_max": bmax,
        })
    st.dataframe(pd.DataFrame(df).style.format(precision=4), use_container_width=True)

    st.subheader("Testar alteração em uma restrição inteira")
    idx = st.selectbox("Restrição a alterar", range(m),
                       format_func=lambda i: f"Restrição {i+1}")

    st.text(f"Nova forma da restrição {idx+1}:")
    cols2 = st.columns(n + 2)
    new_row = [cols2[j].number_input(
        f"a{idx+1}{j+1} (novo)", value=float(A_sol[idx, j]),
        key=f"test_a_{idx}_{j}") for j in range(n)]
    new_ineq = cols2[n].selectbox(
        "Tipo (novo)", ["≤", "≥"],
        index=0 if ineq_signs[idx] == "≤" else 1,
        key=f"test_ineq_{idx}",
    )
    new_b = cols2[n + 1].number_input(
        f"b{idx+1} (novo)", value=float(b_sol[idx]), key=f"test_b_{idx}"
    )

    if st.button("Verificar viabilidade da alteração"):
        A_t, b_t, ineq_t = A_sol.copy(), b_sol.copy(), ineq_signs[:]
        A_t[idx, :] = new_row
        b_t[idx] = new_b
        ineq_t[idx] = new_ineq
        try:
            sign2 = np.array([1.0 if s == "≤" else -1.0 for s in ineq_t])
            A_norm2, b_norm2 = A_t * sign2[:, None], b_t * sign2
            x_new, z_new, info_norm2 = simplex_with_sensitivity(A_norm2, b_norm2, c_sol)
            info_new = convert_info_to_original(info_norm2, b_t, ineq_t)
            pi_new = info_new[idx]["shadow_price"]
            st.write("✅ Alteração viável.")
            st.write("Nova solução ótima:")
            st.write(", ".join(f"x{j+1} = {x_new[j]:.4f}" for j in range(len(x_new))))
            st.write(f"Novo preço-sombra da restrição {idx+1} ({new_ineq}): π = {pi_new:.4f}")
            st.success(f"Novo lucro ótimo z' = {z_new:.4f} .")
        except Exception as e:
            st.error(f"A alteração não é viável: {e}")


if __name__ == "__main__":
    main()



