"""Revision protocol: target-label-free selection and record-preserving splits."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss)
from sklearn.model_selection import GroupShuffleSplit


def binary_labels(values):
    text = pd.Series(values).astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(text, errors='coerce')
    if numeric.notna().all() and set(numeric.unique()) <= {0., 1.}:
        return numeric.astype(int).to_numpy()
    benign = text.isin(['benign', 'benigntraffic', 'normal', 'false'])
    if not benign.any():
        raise ValueError('No recognized benign label; inspect the dataset label dictionary.')
    known = benign | text.isin(['attack', 'malicious', 'anomaly', 'ddos', 'dos', 'true'])
    if not known.all():
        raise ValueError('Missing/unknown label must not silently become an attack.')
    return (~benign).astype(int).to_numpy()


def hashes(frame):
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)


def audit_projection(native, mapped, y):
    native_keys, mapped_keys = hashes(native), hashes(mapped)
    tab = pd.DataFrame({'native': native_keys, 'mapped': mapped_keys, 'y': y})
    conflicts = tab.groupby('mapped').y.nunique()
    return {
        'rows': len(tab), 'native_unique_predictors': int(tab.native.nunique()),
        'mapped_unique_predictors': int(tab.mapped.nunique()),
        'native_predictor_duplicate_rows': int(tab.native.duplicated(keep=False).sum()),
        'mapped_duplicate_rows': int(tab.mapped.duplicated(keep=False).sum()),
        'mapped_label_conflict_keys': int((conflicts > 1).sum()),
        'mapped_label_conflict_rows': int(tab.mapped.isin(conflicts[conflicts > 1].index).sum()),
        'projection_collision_keys': int((tab.groupby('mapped').native.nunique() > 1).sum()),
        'resolution': 'retain distinct native records; group identical mapped predictors before splitting',
    }


def group_split(x, y, fraction, seed, groups=None, require_both=True):
    groups = hashes(x) if groups is None else np.asarray(groups)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(x, y, groups))
    if set(groups[a]).intersection(groups[b]):
        raise AssertionError('Group overlap')
    if require_both and (len(np.unique(np.asarray(y)[a])) != 2 or len(np.unique(np.asarray(y)[b])) != 2):
        raise ValueError('A group split lacks one class; do not replace with a row split silently.')
    return a, b


def stable_uniform(row_ids, seed):
    return np.asarray([int.from_bytes(hashlib.blake2b(f'{seed}:{i}'.encode(), digest_size=8).digest(),
        'little') / 2**64 for i in row_ids])


@dataclass(frozen=True)
class AlarmRule:
    threshold: float
    tie_probability: float
    budget: float

    def predict(self, scores, row_ids, seed):
        scores = np.asarray(scores)
        return ((scores > self.threshold) | ((scores == self.threshold) &
            (stable_uniform(row_ids, seed) < self.tie_probability))).astype(int)


def alarm_rule(scores, budget):
    """Randomized boundary ties meet the adaptation budget in expectation, not an FPR bound."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0 or not np.isfinite(scores).all() or not 0 <= budget <= 1:
        raise ValueError('Finite nonempty scores and a budget in [0,1] are required.')
    if budget == 0:
        return AlarmRule(float('inf'), 0., budget)
    if budget == 1:
        return AlarmRule(float('-inf'), 0., budget)
    cutoff = float(np.quantile(scores, 1-budget, method='higher'))
    above, equal = np.sum(scores > cutoff), np.sum(scores == cutoff)
    probability = float(np.clip((budget * len(scores)-above) / max(equal, 1), 0, 1))
    return AlarmRule(cutoff, probability, budget)


def conservative_alarm_predict(scores, rule):
    return (np.asarray(scores) > rule.threshold).astype(int)


def select_on_source(candidates):
    """The accepted schema deliberately excludes all external evaluation fields."""
    allowed = {'id', 'source_f1', 'source_ba', 'complexity'}
    if any(set(c)-allowed for c in candidates):
        raise ValueError('Only source-calibration metrics and a declared tie-break are allowed.')
    return sorted(candidates, key=lambda c: (-c['source_f1'], -c['source_ba'],
        c.get('complexity', 0), c['id']))[0]['id']


def metrics(y, prediction, scores, probabilities=None):
    y, prediction = np.asarray(y), np.asarray(prediction)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    out = dict(accuracy=accuracy_score(y, prediction), macro_f1=f1_score(y, prediction,
        average='macro', zero_division=0), balanced_accuracy=balanced_accuracy_score(y, prediction),
        precision=precision_score(y, prediction, zero_division=0),
        recall=recall_score(y, prediction, zero_division=0),
        specificity=tn/max(tn+fp, 1), fpr=fp/max(tn+fp, 1),
        alarm_rate=float(prediction.mean()), false_alarms_per_10000=10000*fp/max(len(y), 1),
        false_alarms_per_10000_benign=10000*fp/max(tn+fp, 1),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), n=len(y),
        attack_prevalence=float(y.mean()))
    if len(np.unique(y)) == 2:
        out.update(roc_auc=roc_auc_score(y, scores), pr_auc=average_precision_score(y, scores))
    if probabilities is not None:
        out['brier'] = brier_score_loss(y, np.clip(probabilities, 0, 1))
    return out


def source_threshold(y, scores, max_fpr=None):
    choices = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
    choices = np.r_[choices, np.nextafter(max(scores), np.inf)]
    ranked = []
    for t in choices:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fpr = fp/max(tn+fp, 1)
        if max_fpr is None or fpr <= max_fpr:
            ranked.append((f1_score(y, pred, average='macro', zero_division=0), -fpr, float(t)))
    return max(ranked)[2]


def summaries(frame, keys, values):
    from scipy.stats import t
    rows = []
    for name, group in frame.groupby(keys, dropna=False):
        name = name if isinstance(name, tuple) else (name,)
        row = dict(zip(keys, name))
        for field in values:
            v = pd.to_numeric(group[field], errors='coerce').dropna().to_numpy()
            n = len(v)
            mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else float('nan')
            half = float(t.ppf(.975, n-1)*sd/np.sqrt(n)) if n > 1 else float('nan')
            row.update({field+'_mean': mean, field+'_sd': sd, field+'_ci_low': mean-half,
                field+'_ci_high': mean+half, field+'_n': n})
        rows.append(row)
    return pd.DataFrame(rows)
