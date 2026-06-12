from lm_polygraph.generation_metrics.accuracy import AccuracyMetric


def test_accuracy_accepts_trailing_stop_punctuation():
    metric = AccuracyMetric()
    stats = {"greedy_texts": ["A.", "B\n\n", "C. \n\n", "D. $"]}

    scores = metric(stats, ["A", "B", "C", "D"])

    assert scores.tolist() == [1, 1, 1, 1]


def test_accuracy_does_not_accept_multiple_answers():
    metric = AccuracyMetric()
    stats = {"greedy_texts": ["A, B", "C and D"]}

    scores = metric(stats, ["A", "C"])

    assert scores.tolist() == [0, 0]
