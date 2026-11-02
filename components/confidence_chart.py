"""TruthLens AI - Confidence Chart Component
Visualizes model probability scores, class distributions, and decision bounds using Plotly.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from ml.classifier import PredictionResult


def render_confidence_chart(result: PredictionResult) -> None:
    """Renders interactive Plotly visual probability bars and metrics."""
    ai_pct = result.ai_probability * 100.0
    real_pct = result.real_probability * 100.0

    # Horizontal Bar Comparison Chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["Model Probability"],
            x=[real_pct],
            name="Camera-Captured (Real)",
            orientation="h",
            marker=dict(
                color="#10b981",
                line=dict(color="#059669", width=1.5),
            ),
            text=[f"{real_pct:.1f}% Real"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=14, family="sans-serif"),
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Model Probability"],
            x=[ai_pct],
            name="AI-Generated (Synthetic)",
            orientation="h",
            marker=dict(
                color="#ef4444",
                line=dict(color="#dc2626", width=1.5),
            ),
            text=[f"{ai_pct:.1f}% AI"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=14, family="sans-serif"),
        )
    )

    fig.update_layout(
        barmode="stack",
        height=110,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="#334155",
            tickfont=dict(color="#94a3b8"),
            ticksuffix="%",
        ),
        yaxis=dict(showticklabels=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=12),
        ),
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Metric pills
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="AI Probability", value=f"{ai_pct:.1f}%")
    with col2:
        st.metric(label="Real Probability", value=f"{real_pct:.1f}%")
    with col3:
        st.metric(label="Decision Rule", value=f"AI ≥ {result.threshold_ai*100:.0f}%")
    with col4:
        st.metric(label="Inference Latency", value=f"{result.inference_time_ms:.1f} ms")
