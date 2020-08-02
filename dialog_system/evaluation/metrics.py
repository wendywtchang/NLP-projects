from sklearn.metrics import classification_report, f1_score
import numpy as np


def get_classification_report(gold, pred, classes):
    report = classification_report(y_true=gold, y_pred=pred, labels=classes)
    return report


def get_macro_f1(gold, pred):
    return f1_score(y_true=gold, y_pred=pred, average="macro")


def get_micro_f1(gold, pred):
    return f1_score(y_true=gold, y_pred=pred, average="micro")


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def hits_at_k(ranks, k=10):
    return float(len([r for r in ranks if r <= k])) / len(ranks)


def mrr(ranks):
    # nmean reciprocal rank
    return np.average([1.0 / r for r in ranks])


def mq(ranks, max_rank):
    # mean quantile
    return 1.0 - (np.average(ranks) / (max_rank - 1))
