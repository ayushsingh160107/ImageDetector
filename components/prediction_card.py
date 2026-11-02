"""TruthLens AI - Prediction Card Component
Renders decision badges, probability meters, plain-language forensic takeaway,
and uncertainty alerts.
"""

from __future__ import annotations

import streamlit as st
from ml.classifier import PredictionResult


def render_prediction_card(result: PredictionResult) -> None:
    """Renders the executive prediction verdict card."""
    is_ai = "AI" in result.predicted_label
    is_real = "Real" in result.predicted_label
    is_inconclusive = result.is_inconclusive

    if is_ai:
        bg_gradient = "linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%)"
        border_color = "#ef4444"
        badge_bg = "#dc2626"
        badge_text = "#ffffff"
        icon = "🤖"
        verdict_title = "Likely AI-Generated (Synthetic)"
    elif is_real:
        bg_gradient = "linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.25) 100%)"
        border_color = "#10b981"
        badge_bg = "#059669"
        badge_text = "#ffffff"
        icon = "📷"
        verdict_title = "Likely Real (Camera-Captured)"
    else:
        bg_gradient = "linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.25) 100%)"
        border_color = "#f59e0b"
        badge_bg = "#d97706"
        badge_text = "#ffffff"
        icon = "⚖️"
        verdict_title = "Inconclusive / Mixed Forensic Signals"

    st.markdown(
        f"""
        <div style="
            background: {bg_gradient};
            border: 2px solid {border_color};
            border-radius: 12px;
            padding: 22px 26px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 2rem;">{icon}</span>
                    <div>
                        <span style="
                            background: {badge_bg};
                            color: {badge_text};
                            font-size: 0.8rem;
                            font-weight: 700;
                            padding: 3px 10px;
                            border-radius: 12px;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;
                        ">Forensic Verdict</span>
                        <h2 style="margin: 4px 0 0 0; color: #ffffff; font-size: 1.6rem; font-weight: 700;">
                            {verdict_title}
                        </h2>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">Confidence Score</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">
                        {result.confidence_score * 100:.1f}%
                    </div>
                </div>
            </div>

            <div style="margin-top: 10px; background: rgba(15, 23, 42, 0.6); padding: 14px 18px; border-radius: 8px;">
                <p style="margin: 0; color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">
                    {result.plain_explanation}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.uncertainty_warning:
        st.warning(f"⚠️ **Forensic Notice:** {result.uncertainty_warning}")
