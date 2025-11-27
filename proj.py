import numpy as np
import pandas as pd
import streamlit as st


# =========================
#   ALGORITMO SIMPLEX
# =========================
def simplex_with_sensitivity(A, b, c, tol: float = 1e-9, max_iter: int = 1000):
    """
    Resolve um PPL de maximização:
        max c^T x
        s.a. A x <= b, x >= 0

    usando método Simplex (tableau com variáveis de folga) e
    calcula preços-sombra e intervalos de validade de b.
    Aqui A x <= b já deve estar normalizado.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    m, n = A.shape

    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[m, :n] = -c  # max

    basis = list(range(n, n + m))

    for _ in range(max_iter):
        obj_row = tableau[m, :-1]
        min_val = obj_row.min()
        if min_val >= -tol:
            break

        j = obj_row.argmin()
        col = tableau[:m, j]
        if np.all(col <= tol):
            raise ValueError("Problema ilimitado (unbounded).")

        ratios = np.array([
            tableau[i, -1] / col[i] if col[i] > tol else np.inf
            for i in range(m)
        ])
        i = ratios.argmin()

        pivot = tableau[i, j]
        tableau[i, :] /= pivot
        for r in range(m + 1):
            if r != i:
                tableau[r, :] -= tableau[r, j] * tableau[i, :]

        basis[i] = j

    x_full = np.zeros(n + m)
    for row, var_idx in enumerate(basis):
        x_full[var_idx] = tableau[row, -1]

    z_opt = float(tableau[-1, -1])
    x_decision = x_full[:n]

    A_ext = np.hstack([A, np.eye(m)])   # [A | I]
    B = A_ext[:, basis]
    B_inv = np.linalg.inv(B)

    c_ext = np.concatenate([c, np.zeros(m)])
    c_B = c_ext[basis]

    # π^T = c_B^T B^{-1}
    pi = B_inv.T @ c_B

    x_B = np.array([tableau[row, -1] for row in range(m)])

    info = []
    for i in range(m):
        e_i = np.zeros(m)
        e_i[i] = 1.0
        d = B_inv @ e_i

        delta_min = -np.inf
        delta_max = np.inf

        for k in range(m):
            if abs(d[k]) <= tol:
                continue

            bound = -x_B[k] / d[k]
            if d[k] > 0:
                if bound > delta_min:
                    delta_min = bound
            else:
                if bound < delta_max:
                    delta_max = bound

        if delta_min > 0:
            delta_min = 0.0
        if delta_max < 0:
            delta_max = 0.0

        b_min = b[i] + delta_min
        b_max = b[i] + delta_max

        info.append({
            "constraint_index": i,
            "shadow_price": float(pi[i]),
            "delta_min": float(delta_min),
            "delta_max": float(delta_max),
            "b_interval": (float(b_min), float(b_max)),
        })

    return x_decision, z_opt, info


# =========================
#      FUNÇÕES AUXILIARES
# =========================
def parse_float(value_str: str, default: float = 0.0) -> float:
    try:
        return float(value_str.replace(",", "."))
    except Exception:
        return default


def convert_info_to_original(info_norm, b, ineq_signs):
    """
    Converte preços-sombra e intervalos de Δb do sistema normalizado (<=)
    para o sistema original (≤ ou ≥).
    """
    sign_rows = np.array([1.0 if s == "≤" else -1.0 for s in ineq_signs])
    info_orig = []

    for i, resn in enumerate(info_norm):
        sign = sign_rows[i]
        b_i = b[i]
        pi_n = resn["shadow_price"]
        dmin_n = resn["delta_min"]
        dmax_n = resn["delta_max"]

        if sign == 1.0:
            pi_o = pi_n
            dmin_o = dmin_n
            dmax_o = dmax_n
        else:
            pi_o = -pi_n
            dmin_o = -dmax_n
            dmax_o = -dmin_n

        info_orig.append({
            "constraint_index": resn["constraint_index"],
            "shadow_price": float(pi_o),
            "delta_min": float(dmin_o),
            "delta_max": float(dmax_o),
            "b_interval": (float(b_i + dmin_o), float(b_i + dmax_o)),
            "ineq": ineq_signs[i],
        })

    return info_orig


# =========================
#   INTERFACE: ENTRADAS
# =========================
def build_lp_inputs():
    st.sidebar.subheader("Dimensões do problema")
    m = st.sidebar.number_input(
        "Número de restrições (m)",
        min_value=1, max_value=6, value=3, step=1,
        key="m"
    )
    n = st.sidebar.number_input(
        "Número de variáveis (n)",
        min_value=2, max_value=4, value=4, step=1,
        key="n"
    )

    st.subheader("Função objetivo")
    st.markdown("Maximizar:  **z = c₁·x₁ + c₂·x₂ + ... + cₙ·xₙ**")

    cols = st.columns(n)
    c = []
    for j in range(n):
        with cols[j]:
            c_val = st.number_input(
                f"c{j+1}",
                value=1.0,
                key=f"c_{j}",
                help=f"Coeficiente da variável x{j+1} na função objetivo."
            )
        c.append(c_val)

    st.divider()
    st.subheader("Restrições")

    A = []
    b = []
    ineq_signs = []

    for i in range(m):
        with st.container():
            st.markdown(
                "<div class='constraint-card'>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<span class='constraint-title'>Restrição {i+1}</span>",
                unsafe_allow_html=True
            )

            cols_row = st.columns(2 * n + 2)
            row = []
            col_idx = 0

            for j in range(n):
                with cols_row[col_idx]:
                    coef_str = st.text_input(
                        f"a{i+1}{j+1}",
                        value="1" if (i == 0 and j == 0) else "0",
                        key=f"a_{i}_{j}_coef",
                        help=f"Coeficiente associado à variável x{j+1} na restrição {i+1}."
                    )
                    st.markdown(
                        f"<div class='var-label'>x{j+1}</div>",
                        unsafe_allow_html=True
                    )
                col_idx += 1

                with cols_row[col_idx]:
                    sign = st.selectbox(
                        f"sinal_{i+1}_{j+1}",
                        ["+", "-"],
                        index=0,
                        key=f"sign_{i}_{j}",
                        label_visibility="collapsed",
                        help="Escolha o sinal do próximo termo."
                    )
                    st.markdown(
                        "<div class='plusminus-label'>(+ / -)</div>",
                        unsafe_allow_html=True
                    )
                col_idx += 1

                coef_val = parse_float(coef_str, default=0.0)
                if sign == "-":
                    coef_val *= -1
                row.append(coef_val)

            with cols_row[col_idx]:
                ineq = st.selectbox(
                    f"ineq_{i+1}",
                    ["≤", "≥"],
                    index=0,
                    key=f"ineq_{i}",
                    help="Escolha o tipo de desigualdade desta restrição."
                )
                st.markdown(
                    "<div class='ineq-label'>(≤ / ≥)</div>",
                    unsafe_allow_html=True
                )
            ineq_signs.append(ineq)
            col_idx += 1

            with cols_row[col_idx]:
                b_str = st.text_input(
                    f"b{i+1}",
                    value="10",
                    key=f"b_{i}_value",
                    help="Valor do lado direito (b) desta restrição."
                )
                st.markdown(
                    f"<div class='var-label'>b{i+1}</div>",
                    unsafe_allow_html=True
                )

            b_val = parse_float(b_str, default=0.0)
            b.append(b_val)
            A.append(row)

            symbolic = " + ".join([f"a{i+1}{j+1}·x{j+1}" for j in range(n)])
            st.markdown(
                f"<div class='constraint-footer'>Forma simbólica: "
                f"{symbolic} {ineq} b{i+1}</div>",
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)

    return A, b, c, m, n, ineq_signs


# =========================
#        APLICAÇÃO
# =========================
def main():
    st.set_page_config(
        page_title="Simplex com Análise de Sensibilidade",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .constraint-card {
            background-color: #f8f9fc;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 12px;
            border: 1px solid #e0e3f0;
        }
        .constraint-title {
            font-weight: 600;
            font-size: 0.95rem;
            color: #1f2933;
            margin-bottom: 6px;
            display: block;
        }
        .constraint-footer {
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 4px;
        }
        .var-label {
            text-align: center;
            font-size: 0.75rem;
            color: #4b5563;
            margin-top: 2px;
        }
        .ineq-label, .plusminus-label {
            text-align: center;
            font-size: 0.70rem;
            color: #9ca3af;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Simplex (Tableau) com Análise de Sensibilidade")

    if "viability_logs" not in st.session_state:
        st.session_state["viability_logs"] = []
    if "solution" not in st.session_state:
        st.session_state["solution"] = None

    with st.sidebar:
        st.header("Passo a passo")
        st.markdown(
            "1. Defina **n** variáveis (2 a 4) e **m** restrições.\n"
            "2. Preencha a função objetivo e as restrições (≤ ou ≥).\n"
            "3. Clique em **Calcular solução ótima**.\n"
            "4. Depois, use a área de **viabilidade** para alterar uma restrição"
            " inteira (coeficientes, sinal e b) e ver o novo ótimo."
        )

    A_orig, b_orig, c, m, n, ineq_signs = build_lp_inputs()

    # ----------------- SOLUÇÃO ORIGINAL ------------------
    if st.button("Calcular solução ótima", type="primary"):
        try:
            sign_rows = np.array([1.0 if s == "≤" else -1.0 for s in ineq_signs])
            A_norm = []
            b_norm = []
            for i in range(m):
                A_norm.append([coef * sign_rows[i] for coef in A_orig[i]])
                b_norm.append(b_orig[i] * sign_rows[i])

            x_opt, z_opt, info_norm = simplex_with_sensitivity(A_norm, b_norm, c)
            info_orig = convert_info_to_original(info_norm, b_orig, ineq_signs)

            st.session_state["solution"] = {
                "x_opt": x_opt.tolist(),
                "z_opt": float(z_opt),
                "info": info_orig,
                "b": list(b_orig),
                "A_orig": A_orig,
                "c": c,
                "m": m,
                "n": n,
                "ineq_signs": ineq_signs,
            }
            st.session_state["viability_logs"] = []
            st.success("Solução ótima encontrada com sucesso!")
        except Exception as e:
            st.error(f"Ocorreu um erro ao resolver o problema: {e}")

    # -------------- MOSTRA RESULTADOS -----------------
    sol = st.session_state["solution"]
    if sol is not None:
        x_opt = np.array(sol["x_opt"])
        z_opt = sol["z_opt"]
        info = sol["info"]
        b_vec = np.array(sol["b"], dtype=float)
        A_saved = sol["A_orig"]
        c_saved = sol["c"]
        m = sol["m"]
        n = sol["n"]
        ineq_signs_sol = sol["ineq_signs"]

        st.subheader("Solução ótima (problema original)")
        cols_sol = st.columns(n + 1)
        for j in range(n):
            cols_sol[j].metric(f"x{j+1}*", f"{x_opt[j]:.4f}")
        cols_sol[-1].metric("z*", f"{z_opt:.4f}")

        st.subheader("Preços-sombra e intervalos de validade (problema original)")
        data = []
        for res in info:
            i = res["constraint_index"]
            bmin, bmax = res["b_interval"]
            data.append({
                "Restrição": f"R{i+1}",
                "Tipo": ineq_signs_sol[i],
                "Preço-sombra π": res["shadow_price"],
                "Δb_min": res["delta_min"],
                "Δb_max": res["delta_max"],
                "b_min": bmin,
                "b_max": bmax,
            })
        df = pd.DataFrame(data)
        st.dataframe(df.style.format(precision=4), use_container_width=True)

        # ========= TESTE: ALTERAR RESTRIÇÃO INTEIRA =========
        st.subheader("Testar alteração em uma restrição inteira")

        idx = st.selectbox(
            "Escolha a restrição para alterar",
            options=list(range(m)),
            format_func=lambda i: f"Restrição {i+1}",
            key="select_restriction_viab"
        )

        with st.container():
            st.markdown(
                "<div class='constraint-card'>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<span class='constraint-title'>Nova forma da restrição {idx+1}</span>",
                unsafe_allow_html=True
            )

            cols_edit = st.columns(2 * n + 2)
            col_idx = 0
            new_row = []

            for j in range(n):
                with cols_edit[col_idx]:
                    coef_str = st.text_input(
                        f"test_a{idx+1}{j+1}",
                        value=str(A_saved[idx][j]),
                        key=f"test_coef_{idx}_{j}",
                        help=f"Novo coeficiente (com sinal) de x{j+1}."
                    )
                    st.markdown(
                        f"<div class='var-label'>x{j+1}</div>",
                        unsafe_allow_html=True
                    )
                col_idx += 1

                with cols_edit[col_idx]:
                    st.markdown(
                        "<div class='plusminus-label'>(coef com sinal)</div>",
                        unsafe_allow_html=True
                    )
                col_idx += 1

                new_row.append(parse_float(coef_str, default=A_saved[idx][j]))

            with cols_edit[col_idx]:
                new_ineq = st.selectbox(
                    f"test_ineq_{idx+1}",
                    ["≤", "≥"],
                    index=0 if ineq_signs_sol[idx] == "≤" else 1,
                    key=f"test_ineq_key_{idx}",
                    help="Novo tipo de desigualdade para esta restrição."
                )
                st.markdown(
                    "<div class='ineq-label'>(≤ / ≥)</div>",
                    unsafe_allow_html=True
                )
            col_idx += 1

            with cols_edit[col_idx]:
                new_b_str = st.text_input(
                    f"test_b{idx+1}",
                    value=str(b_vec[idx]),
                    key=f"test_b_val_{idx}",
                    help="Novo valor de b para esta restrição."
                )
                st.markdown(
                    f"<div class='var-label'>b{idx+1}</div>",
                    unsafe_allow_html=True
                )

            new_b_val = parse_float(new_b_str, default=b_vec[idx])

            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Verificar viabilidade da alteração"):
            # monta novo problema com a restrição alterada
            A_test = [list(row) for row in A_saved]
            b_test = list(b_vec)
            ineq_test = list(ineq_signs_sol)

            A_test[idx] = new_row
            b_test[idx] = new_b_val
            ineq_test[idx] = new_ineq

            try:
                sign_rows2 = np.array([1.0 if s == "≤" else -1.0 for s in ineq_test])
                A_norm2 = []
                b_norm2 = []
                for i in range(m):
                    A_norm2.append([coef * sign_rows2[i] for coef in A_test[i]])
                    b_norm2.append(b_test[i] * sign_rows2[i])

                x_new, z_new, info_norm2 = simplex_with_sensitivity(
                    A_norm2, b_norm2, c_saved
                )
                info_new = convert_info_to_original(info_norm2, b_test, ineq_test)

                # preço-sombra da restrição alterada no NOVO problema
                pi_new = info_new[idx]["shadow_price"]

                log = {
                    "idx": idx,
                    "old_b": b_vec[idx],
                    "new_b": new_b_val,
                    "feasible": True,
                    "new_z": float(z_new),
                    "new_x": x_new.tolist(),
                    "new_pi": float(pi_new),
                    "new_ineq": new_ineq,
                }
                st.session_state["viability_logs"].append(log)

            except Exception:
                log = {
                    "idx": idx,
                    "old_b": b_vec[idx],
                    "new_b": new_b_val,
                    "feasible": False,
                    "new_z": None,
                    "new_x": None,
                    "new_pi": None,
                    "new_ineq": new_ineq,
                }
                st.session_state["viability_logs"].append(log)

        # ------------- HISTÓRICO DE TESTES -------------
        if st.session_state["viability_logs"]:
            st.subheader("Histórico de testes de viabilidade")

            for i, log in enumerate(st.session_state["viability_logs"], start=1):
                idx = log["idx"]
                old_b = log["old_b"]
                new_b = log["new_b"]
                feasible = log["feasible"]
                new_z = log["new_z"]
                new_x = log["new_x"]
                new_pi = log["new_pi"]
                new_ineq = log["new_ineq"]

                st.markdown(f"**Teste {i} — Restrição {idx+1}**")
                st.write(
                    f"- b{idx+1} original = {old_b:.4f}  "
                    f"→ novo b{idx+1} = {new_b:.4f}"
                )

                if feasible:
                    st.write(
                        "- Nova solução ótima: " +
                        ", ".join([f"x{j+1} = {new_x[j]:.4f}"
                                   for j in range(len(new_x))])
                    )
                    st.write(
                        f"- Novo preço-sombra da restrição {idx+1} "
                        f"({new_ineq}): π = {new_pi:.4f}"
                    )
                    st.success(f"Novo lucro ótimo z' = {new_z:.4f} .")
                else:
                    st.error("A alteração não é viável.")


if __name__ == "__main__":
    main()
