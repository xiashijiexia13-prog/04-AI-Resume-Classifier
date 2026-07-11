from train import build_error_analysis, evaluate_model


def test_error_analysis_identifies_weak_classes_and_confusions():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 1, 2]
    metrics = evaluate_model(y_true, y_pred, model=None, train_time=0)
    label_map = {"0": "A", "1": "B", "2": "C"}

    analysis = build_error_analysis(
        y_true, y_pred, label_map, metrics, top_n=2
    )

    assert len(analysis["weakest_classes"]) == 2
    assert analysis["most_common_confusions"] == [
        {"true_category": "A", "predicted_category": "B", "count": 1},
        {"true_category": "C", "predicted_category": "B", "count": 1},
    ]
