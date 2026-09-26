import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="DHS-Predict", page_icon="🫀", layout="wide")

# -----------------------------------------------------------------------
# Fixed reference / model parameters — NOT collected from the user.
# Edit these constants directly in the code if you need to tune the model.
# -----------------------------------------------------------------------
P0 = 125.0       # Reference systolic BP (mmHg)
S0 = 20.0        # Baseline sensitivity
S_INIT = 25.0    # Initial sensitivity S(0)
A = 0.20         # Recovery rate
ETA = 0.08       # Adaptation rate
GAMMA = 0.04     # Coupling between pressure deviation and sensitivity
D = 0.05         # Effective physiological disturbance
HOURS = 9        # Prediction horizon (fixed at ~8-10 hours)
DT = 0.02        # Integration step size


def simulate_bp(P_init, hours=HOURS, dt=DT):
    n = int(hours / dt) + 1
    t = np.linspace(0, hours, n)
    x = np.zeros(n)
    S = np.zeros(n)
    x[0] = P_init - P0
    S[0] = S_INIT

    def f(xv, Sv):
        return -A * xv + Sv * D, ETA * (S0 - Sv) - GAMMA * xv

    for i in range(n - 1):
        h = dt
        k1x, k1s = f(x[i], S[i])
        k2x, k2s = f(x[i] + h * k1x / 2, S[i] + h * k1s / 2)
        k3x, k3s = f(x[i] + h * k2x / 2, S[i] + h * k2s / 2)
        k4x, k4s = f(x[i] + h * k3x, S[i] + h * k3s)
        x[i + 1] = x[i] + h * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        S[i + 1] = S[i] + h * (k1s + 2 * k2s + 2 * k3s + k4s) / 6

    return t, P0 + x, S


st.title("🫀 DHS-Predict")
st.subheader("Personalized Dynamic Blood-Pressure Trajectory")
st.markdown(
    "A demonstration of the proposed **Dynamic Hypertension Sensitivity (DHS)** "
    "model, where blood pressure and responsiveness evolve together."
)
st.info(
    "Research demonstration only — not a medical diagnostic tool and not for "
    "treatment or medication decisions."
)

# -----------------------------------------------------------------------
# Only patient-specific inputs are collected. Everything else uses the
# fixed reference values defined above.
# -----------------------------------------------------------------------
st.sidebar.header("Patient Information")
name = st.sidebar.text_input("Patient name", "Demo Patient")
age = st.sidebar.number_input("Age", 1, 120, 35)

st.sidebar.header("Current Blood Pressure")
P_init = st.sidebar.number_input("Current systolic BP (mmHg)", 60.0, 250.0, 145.0)

st.sidebar.caption(
    f"Reference BP fixed at {P0:.0f} mmHg · prediction horizon fixed at {HOURS} hours."
)

t, P, S = simulate_bp(P_init)
min_P, max_P, final_P = float(P.min()), float(P.max()), float(P[-1])

st.markdown(f"### Personalized projection for {name}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Age", f"{age} years")
c2.metric("Current SBP", f"{P_init:.1f} mmHg")
c3.metric(f"Predicted SBP ({HOURS} h)", f"{final_P:.1f} mmHg")
c4.metric("Minimum predicted SBP", f"{min_P:.1f} mmHg")

st.markdown("### Pressure-fall screening")
drop = P_init - min_P
if drop >= 10:
    st.warning(f"The model predicts a fall of approximately {drop:.1f} mmHg from the starting pressure.")
else:
    st.success(f"No fall of 10 mmHg or more is predicted. Maximum modeled fall: {max(0, drop):.1f} mmHg.")
st.caption("This is a mathematical screening output, not a clinical prediction of hypotension.")

st.markdown("### 1. Predicted blood-pressure trajectory")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(t, P, linewidth=2, label="Predicted SBP")
ax.axhline(P0, linestyle="--", linewidth=1.5, label="Reference BP")
ax.scatter([0], [P_init], s=45, label="Current BP")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Systolic BP (mmHg)")
ax.set_title("DHS model: personalized BP trajectory")
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig)

st.markdown("### 2. Dynamic hypertension sensitivity")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(t, S, linewidth=2, label="S(t)")
ax.axhline(S0, linestyle="--", linewidth=1.5, label="Baseline S₀")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Sensitivity S(t)")
ax.set_title("Evolution of the responsiveness state")
ax.grid(alpha=0.25)
ax.legend()
st.pyplot(fig)

st.markdown("### Model interpretation")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"**Initial state**\n\n"
        f"- BP deviation: **{P_init - P0:.1f} mmHg**\n"
        f"- Initial sensitivity: **{S_INIT:.2f}**\n"
        f"- Reference BP: **{P0:.1f} mmHg**"
    )
with c2:
    st.markdown(
        f"**Predicted evolution**\n\n"
        f"- Minimum SBP: **{min_P:.1f} mmHg**\n"
        f"- Maximum SBP: **{max_P:.1f} mmHg**\n"
        f"- Final change: **{final_P - P_init:+.1f} mmHg**"
    )

with st.expander("View mathematical model"):
    st.latex(r"\frac{dx}{dt}=-ax+S(t)D")
    st.latex(r"\frac{dS}{dt}=\eta(S_0-S)-\gamma x")
    st.latex(r"P(t)=P_0+x(t)")
    st.caption(
        f"Fixed reference values used: P₀={P0}, S₀={S0}, S(0)={S_INIT}, "
        f"a={A}, η={ETA}, γ={GAMMA}, D={D}, horizon={HOURS} h."
    )

st.markdown("---")
st.caption("DHS-Predict | Mathematical modelling demonstration | DHS is a proposed reduced-order construct.")
