import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="NYC Airbnb Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; font-weight: 400 !important; }

.hero-tag {
    display: inline-block; background: #e1f5ee; color: #0f6e56;
    font-size: 12px; font-weight: 500; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 5px 12px; border-radius: 4px;
    margin-bottom: 0.75rem;
}
.metric-card {
    background: #f8f9fa; border-radius: 10px; padding: 1rem 1.25rem;
    border: 1px solid #e9ecef;
}
.finding-item {
    padding: 14px 0; border-bottom: 1px solid #f0f0f0;
    font-size: 14px; line-height: 1.7; color: #555;
}
.model-best-badge {
    background: #e6f1fb; color: #185fa5; font-size: 11px;
    font-weight: 500; padding: 3px 9px; border-radius: 4px;
    display: inline-block; margin-bottom: 6px;
}
.footer-note {
    font-size: 12px; color: #aaa; margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# Data from what we already get
MODEL_RESULTS = {
    "OLS (Manual)": {
        "mae_log": 0.3968, "rmse_log": 0.4951, "r2": 0.5045,
        "mae_usd": 72.40, "rmse_usd": 110.31,
        "ci_coverage": None, "ci_width": None,
        "color": "#1d9e75",
        "description": "Closed-form normal equations in NumPy — interpretable coefficients grounded in hedonic pricing theory.",
        "strength": "Interpretability"
    },
    "Neural Network": {
        "mae_log": 0.3009, "rmse_log": 0.4054, "r2": 0.7261,
        "mae_usd": 71.94, "rmse_usd": 155.88,
        "ci_coverage": None, "ci_width": None,
        "color": "#185fa5",
        "description": "Two-hidden-layer MLP in PyTorch with ReLU, dropout, Adam optimizer, and early stopping.",
        "strength": "Predictive accuracy"
    },
    "Bayesian Regression": {
        "mae_log": 0.3969, "rmse_log": 0.4953, "r2": 0.5042,
        "mae_usd": 72.37, "rmse_usd": 110.00,
        "ci_coverage": 0.960, "ci_width": 1.968,
        "color": "#ba7517",
        "description": "Analytical posterior in R from scratch (no rstan/brms) — produces calibrated 95% credible intervals.",
        "strength": "Uncertainty quantification"
    }
}

BAYESIAN_COEFS = pd.DataFrame([
    {"feature": "accommodates", "mean": 0.197, "lower": 0.186, "upper": 0.208},
    {"feature": "review_scores_rating", "mean": 0.055, "lower": 0.047, "upper": 0.064},
    {"feature": "bathrooms", "mean": 0.037, "lower": 0.027, "upper": 0.046},
    {"feature": "number_of_reviews", "mean": 0.016, "lower": 0.008, "upper": 0.025},
    {"feature": "bedrooms", "mean": 0.003, "lower": -0.008, "upper": 0.013},
    {"feature": "host_is_superhost", "mean": -0.058, "lower": -0.077, "upper": -0.039},
    {"feature": "minimum_nights", "mean": -0.068, "lower": -0.077, "upper": -0.060},
    {"feature": "room: hotel room", "mean": -0.247, "lower": -0.699, "upper": 0.206},
    {"feature": "room: private room", "mean": -0.177, "lower": -0.578, "upper": 0.225},
    {"feature": "room: shared room", "mean": -0.333, "lower": -1.020, "upper": 0.354},
]).sort_values("mean")

# Sidebar plot
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Navigation", ["Overview", "Model Explorer"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**DS4420 Final Project · Spring 2026**")
st.sidebar.markdown("Meixu Chen & Yuetong Yin")
st.sidebar.markdown("Professor: Dr. Eric Gerber")
st.sidebar.markdown("Northeastern University")
st.sidebar.markdown("---")
st.sidebar.markdown("**Data:** Inside Airbnb · NYC · Nov 7, 2025")
st.sidebar.markdown("**Listings:** 20,772 (after cleaning)")
st.sidebar.markdown("**Price range:** $$$9 –$$$815/night")


# The first page: OVERVIEW

if page == "Overview":

    st.markdown('<div class="hero-tag">DS4420 Machine Learning · Spring 2026</div>', unsafe_allow_html=True)
    st.title("Predicting NYC Airbnb Prices\nwith Three Models")
    st.markdown(
        "Comparing **manual OLS**, a **neural network**, and **Bayesian regression** on 20,772 "
        "New York City listings — evaluating accuracy, interpretability, and uncertainty quantification.",
        unsafe_allow_html=True
    )

    # Summary metrics
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings analyzed", "20,772", help="After cleaning & 97th-percentile outlier removal")
    c2.metric("Median price / night", "$150", help="Original dollar scale after log-transformation target")
    c3.metric("Best test R²", "0.726", help="Neural network on log-price scale")
    c4.metric("Bayesian CI coverage", "96.0%", help="95% credible interval empirical coverage on test set")

    # Model cards
    st.markdown("### The three approaches")
    cols = st.columns(3)
    for col, (name, info) in zip(cols, MODEL_RESULTS.items()):
        with col:
            if name == "Neural Network":
                st.markdown('<div class="model-best-badge">Best accuracy</div>', unsafe_allow_html=True)
            st.markdown(f"**{name}**")
            st.markdown(f"<p style='font-size:13px;color:#666;line-height:1.5'>{info['description']}</p>",
                        unsafe_allow_html=True)
            st.markdown(f"**Strength:** {info['strength']}")
            sub_cols = st.columns(2)
            sub_cols[0].metric("Test R²", f"{info['r2']:.3f}")
            sub_cols[1].metric("MAE ($)", f"${info['mae_usd']:.2f}")
            if info["ci_coverage"]:
                st.metric("CI coverage", f"{info['ci_coverage']:.1%}")

    # Key findings
    st.markdown("---")
    st.markdown("### Key findings")

    findings = [
        ("1",
         "Nonlinearity matters: The neural network's R² of 0.726 substantially exceeds the linear models' ~0.504, confirming that NYC Airbnb pricing contains meaningful nonlinear structure — interactions across listing type, size, and host signals — that OLS cannot capture."),
        ("2",
         "Interpretability has real value: OLS coefficients directly quantify each predictor's association with log-price. Accommodation capacity shows the strongest positive effect; shared-room listings are priced ~33% below the reference category on the log scale."),
        ("3",
         "Bayesian regression uniquely quantifies uncertainty: With 95% credible interval coverage of 96.0% and an average width of 1.97 on the log-price scale, the Bayesian model provides calibrated uncertainty estimates that neither OLS nor the neural network can offer."),
        ("4",
         "The neural network struggles at the upper tail: Despite better log-scale RMSE (0.405 vs 0.495), the NN's dollar-scale RMSE rises to $155.88 vs OLS's $110.31 — large errors on high-priced listings amplify on the original scale after exponentiation."),
    ]
    for icon, text in findings:
        st.markdown(f"<div class='finding-item'>{icon} &nbsp; {text}</div>", unsafe_allow_html=True)

    # Full comparison table
    st.markdown("---")
    st.markdown("### Full model comparison — test set")

    df_table = pd.DataFrame({
        "Metric": ["MAE (log)", "RMSE (log)", "R²", "MAE (USD)", "RMSE (USD)", "95% CI coverage", "Avg CI width (log)"],
        "OLS": ["0.3968", "0.4951", "0.5045", "72.40", "110.31", "—", "—"],
        "Neural Network": ["0.3009", "0.4054", "0.7261", "71.94", "155.88", "—", "—"],
        "Bayesian": ["0.3969", "0.4953", "0.5042", "72.37", "110.00", "0.960", "1.968"],
    })

    st.dataframe(df_table.set_index("Metric"), width="stretch")

    st.markdown(
        "<div class='footer-note'>Data: Inside Airbnb, NYC snapshot Nov 7 2025. "
        "All models trained/validated/tested on an identical 70/15/15 stratified split (fixed random seed). "
        "Target variable: log_price. Metrics on original dollar scale computed by exponentiating predictions.</div>",
        unsafe_allow_html=True
    )



# The second page : MODEL EXPLORER

else:
    st.title("Model Performance Explorer")
    st.markdown("Interactively compare the three models and inspect Bayesian coefficient estimates.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Graph 1: Metric comparison", "Graph 2: Actual vs predicted", "Graph 3: Bayesian coefficients"])

    # First Tab: Metric bar chart
    with tab1:
        st.markdown("#### Compare models by metric")

        metric_options = {
            "R² (higher is better)": ("r2", True),
            "MAE — log scale": ("mae_log", False),
            "RMSE — log scale": ("rmse_log", False),
            "MAE — USD scale": ("mae_usd", False),
            "RMSE — USD scale": ("rmse_usd", False),
        }

        selected_label = st.selectbox("Select metric", list(metric_options.keys()))
        metric_key, higher_better = metric_options[selected_label]

        names = list(MODEL_RESULTS.keys())
        values = [MODEL_RESULTS[m][metric_key] for m in names]
        colors_list = [MODEL_RESULTS[m]["color"] for m in names]


        def hex_to_rgba(hex_color, alpha=1.0):
            hex_color = hex_color.lstrip("#")
            r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"


        best_val = max(values) if higher_better else min(values)
        bar_colors = [
            hex_to_rgba(c, 1.0) if v == best_val else hex_to_rgba(c, 0.45)
            for c, v in zip(colors_list, values)
        ]

        fig = go.Figure(go.Bar(
            x=names, y=values,
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            height=380, margin=dict(t=30, b=20, l=20, r=20),
            yaxis_title=selected_label,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=13),
            showlegend=False,
            yaxis=dict(gridcolor="#f0f0f0"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, width="stretch")

        # Dynamic insight
        insights = {
            "R² (higher is better)": "The neural network achieves R² = 0.726 — substantially outperforming OLS (0.505) and Bayesian (0.504). This gap reflects meaningful nonlinear structure in NYC pricing that linear methods cannot capture.",
            "MAE — log scale": "The neural network's log-scale MAE of 0.301 beats the linear models (~0.397). On the log scale, a MAE of ~0.30 corresponds to errors of roughly 35% of the listing price.",
            "RMSE — log scale": "Neural network RMSE 0.405 vs linear models' ~0.495. RMSE penalizes large errors more heavily, confirming the NN better handles the dense mid-range of the distribution.",
            "MAE — USD scale": "All three models cluster within $0.50 of each other on dollar MAE (~$72). The NN's log-scale superiority is diluted because the median listing is in a moderate price range.",
            "RMSE — USD scale": "The linear models ($110) beat the neural network ($156) here. The NN makes larger errors on expensive listings; these amplify after exponentiation from log scale back to dollars.",
        }
        st.info(insights[selected_label])

    # The second tab: Scatter plot
    with tab2:
        st.markdown("#### Actual vs predicted — log price scale")
        st.markdown("Illustrative scatter showing how tightly each model's predictions track actual log-prices.")

        np.random.seed(42)
        n = 200
        true_vals = np.random.normal(5.0, 0.7, n)


        def make_preds(true_vals, noise_std, bias_fn=None):
            noise = np.random.normal(0, noise_std, len(true_vals))
            preds = true_vals + noise
            if bias_fn:
                preds += bias_fn(true_vals)
            return preds


        scatter_data = {
            "OLS (Manual)": make_preds(true_vals, 0.50),
            "Neural Network": make_preds(true_vals, 0.41),
            "Bayesian Regression": make_preds(true_vals, 0.50, lambda x: np.random.normal(0, 0.02, len(x))),
        }

        selected_models = st.multiselect(
            "Show models",
            list(scatter_data.keys()),
            default=list(scatter_data.keys())
        )

        fig2 = go.Figure()
        min_v, max_v = true_vals.min() - 0.3, true_vals.max() + 0.3
        fig2.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines", line=dict(color="#ccc", dash="dash", width=1.5),
            name="Perfect fit", showlegend=True
        ))
        for name in selected_models:
            preds = scatter_data[name]
            fig2.add_trace(go.Scatter(
                x=true_vals, y=preds, mode="markers",
                marker=dict(color=MODEL_RESULTS[name]["color"], size=5, opacity=0.55),
                name=name
            ))
        fig2.update_layout(
            height=420, margin=dict(t=20, b=40, l=40, r=20),
            xaxis_title="Actual log_price",
            yaxis_title="Predicted log_price",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#f0f0f0"),
            yaxis=dict(gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "Note: scatter points are illustrative — simulated to match each model's reported test-set RMSE. The neural network's tighter clustering reflects its higher R² of 0.726 vs ~0.504 for the linear models.")

    # The third tab graph: Bayesian coefficients
    with tab3:
        st.markdown("#### Bayesian posterior coefficients")
        st.markdown(
            "Posterior mean and 95% credible intervals for selected predictors. "
            "Coefficients are on the log-price scale and a coefficient of +0.20 means "
            "approximately +20% in listing price, all else equal."
        )

        df_coef = BAYESIAN_COEFS.copy()

        show_ci = st.toggle("Show 95% credible intervals", value=True)

        fig3 = go.Figure()

        if show_ci:
            fig3.add_trace(go.Scatter(
                x=list(df_coef["upper"]) + list(df_coef["lower"])[::-1],
                y=list(df_coef["feature"]) + list(df_coef["feature"])[::-1],
                fill="toself",
                fillcolor="rgba(29,158,117,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="95% credible interval",
                orientation="h"
            ))
            for _, row in df_coef.iterrows():
                fig3.add_trace(go.Scatter(
                    x=[row["lower"], row["upper"]],
                    y=[row["feature"], row["feature"]],
                    mode="lines",
                    line=dict(color="#9fe1cb" if row["mean"] >= 0 else "#f0997b", width=3),
                    showlegend=False, hoverinfo="skip"
                ))

        fig3.add_trace(go.Scatter(
            x=df_coef["mean"], y=df_coef["feature"],
            mode="markers",
            marker=dict(
                color=["#1d9e75" if v >= 0 else "#d85a30" for v in df_coef["mean"]],
                size=10, line=dict(width=1.5, color="white")
            ),
            name="Posterior mean",
            hovertemplate="<b>%{y}</b><br>Posterior mean: %{x:.3f}<extra></extra>"
        ))

        fig3.add_vline(x=0, line_width=1, line_dash="dash", line_color="#aaa")

        fig3.update_layout(
            height=420, margin=dict(t=20, b=40, l=160, r=40),
            xaxis_title="Coefficient value (log-price scale)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=13),
            showlegend=False,
            xaxis=dict(gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig3, width="stretch")

        st.markdown("**Coefficient table**")
        df_display = df_coef[["feature", "mean", "lower", "upper"]].rename(columns={
            "feature": "Predictor", "mean": "Posterior mean",
            "lower": "95% CI lower", "upper": "95% CI upper"
        }).sort_values("Posterior mean", ascending=False)
        df_display[["Posterior mean", "95% CI lower", "95% CI upper"]] = \
            df_display[["Posterior mean", "95% CI lower", "95% CI upper"]].round(4)
        st.dataframe(df_display.set_index("Predictor"), width="stretch")

        st.info(
            "Key takeaway: Accommodates has the strongest and most precise positive effect (+0.197, narrow CI). "
            "minimum_nights : is reliably negative (−0.068). Room-type dummies have wide CIs reflecting sparse data "
            "in some categories — intervals crossing zero indicate genuine uncertainty about that effect."
        )
