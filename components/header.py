"""TruthLens AI - UI Header Component
Renders the brand banner, subtitle, version badges, and status indicator.
"""

from __future__ import annotations

import streamlit as st


def render_header(
    title: str = "TruthLens AI",
    subtitle: str = "Deep Learning Forensics & AI-Generated Image Explainer",
    badge: str = "v1.2.0 • PyTorch Active",
) -> None:
    """Renders the top branding hero banner."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
            padding: 24px 28px;
            border-radius: 14px;
            border: 1px solid rgba(99, 102, 241, 0.25);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            margin-bottom: 24px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h1 style="
                        margin: 0;
                        padding: 0;
                        font-family: 'Inter', -apple-system, sans-serif;
                        font-weight: 800;
                        font-size: 2.2rem;
                        color: #ffffff;
                        letter-spacing: -0.5px;
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    ">
                        🔍 {title}
                    </h1>
                    <p style="
                        margin: 6px 0 0 0;
                        font-size: 1rem;
                        color: #cbd5e1;
                        font-weight: 400;
                    ">
                        {subtitle}
                    </p>
                </div>
                <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                    <span style="
                        background: rgba(99, 102, 241, 0.2);
                        color: #a5b4fc;
                        border: 1px solid rgba(165, 180, 252, 0.3);
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-size: 0.82rem;
                        font-weight: 600;
                        display: inline-block;
                    ">
                        {badge}
                    </span>
                    <span style="
                        background: rgba(16, 185, 129, 0.15);
                        color: #6ee7b7;
                        border: 1px solid rgba(16, 185, 129, 0.3);
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-size: 0.82rem;
                        font-weight: 600;
                        display: inline-block;
                    ">
                        ● Engine Online
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
