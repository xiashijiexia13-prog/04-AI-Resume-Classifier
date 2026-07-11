from pathlib import Path

import pytest

from predict import ResumeClassifier


def test_classifier_reports_missing_artifact(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="TF-IDF"):
        ResumeClassifier(model_dir=str(tmp_path))
