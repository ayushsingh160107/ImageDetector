"""TruthLens AI - Application Structure & Database Unit Tests"""

from __future__ import annotations

import os
import sqlite3
import pytest

from database.history import get_db_connection, init_db, save_analysis
from ml.gan_model import DCGANGenerator, generate_synthetic_images


def test_pages_and_app_exist():
    """Verifies that app.py and all 5 Streamlit multipage files exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.exists(os.path.join(base_dir, "app.py"))
    
    pages = [
        "1_Image_Detector.py",
        "2_Compare_Images.py",
        "3_How_It_Works.py",
        "4_Model_Performance.py",
        "5_GAN_Generator.py",
    ]
    for page in pages:
        page_path = os.path.join(base_dir, "pages", page)
        assert os.path.exists(page_path), f"Page missing: {page_path}"


def test_sqlite_history_logging(tmp_path):
    """Verifies SQLite history table initialization and record insertion."""
    test_db = str(tmp_path / "test_truthlens.db")
    init_db(test_db)
    
    rec_id = save_analysis(
        filename="test_sample.png",
        file_size_kb=124.5,
        dimensions="512x512",
        predicted_label="Likely AI-Generated",
        ai_probability=0.91,
        real_probability=0.09,
        confidence_score=0.91,
        inference_time_ms=12.4,
        model_name="TruthLens-Test",
        db_path=test_db,
    )
    assert rec_id > 0
    
    conn = get_db_connection(test_db)
    row = conn.execute("SELECT * FROM analysis_history WHERE id = ?", (rec_id,)).fetchone()
    assert row is not None
    assert row["filename"] == "test_sample.png"
    assert row["predicted_label"] == "Likely AI-Generated"
    conn.close()


def test_dcgan_generation():
    """Verifies DCGAN Generator forward pass produces PIL images of correct dimensions."""
    generator = DCGANGenerator(latent_dim=100)
    images = generate_synthetic_images(generator, num_samples=2, latent_dim=100, seed=123)
    assert len(images) == 2
    assert images[0].size == (64, 64)
    assert images[0].mode == "RGB"
