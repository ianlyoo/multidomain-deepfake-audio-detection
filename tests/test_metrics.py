import numpy as np
import pytest

from deepvoicehackathon.metrics import official_eer, official_score

def test_official_eer_perfect_ordering_is_zero():
    assert official_eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 0.0

def test_official_eer_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="NaN or infinity"):
        official_eer([0, 1], [0.1, np.nan])

def test_official_score_perfect_predictions_is_one():
    labels = {
        "FILE_FAKE": [0, 1, 0, 1, 0, 1],
        "VOICE_FAKE": [0, 1, 0, 1, 0, 1],
        "MUSIC_FAKE": [0, 1, 0, 1, 0, 1],
        "VOICE_PRESENT": [1, 1, 1, 1, 0, 0],
        "MUSIC_PRESENT": [0, 0, 1, 1, 1, 1],
    }
    predictions = {
        "FILE_FAKE_PROB": labels["FILE_FAKE"],
        "VOICE_FAKE_PROB": labels["VOICE_FAKE"],
        "MUSIC_FAKE_PROB": labels["MUSIC_FAKE"],
        "VOICE_PRESENT_PROB": labels["VOICE_PRESENT"],
        "MUSIC_PRESENT_PROB": labels["MUSIC_PRESENT"],
    }
    result = official_score(labels, predictions)
    assert result.score == pytest.approx(1.0)
    assert result.ads == pytest.approx(1.0)
    assert result.cps == pytest.approx(1.0)
