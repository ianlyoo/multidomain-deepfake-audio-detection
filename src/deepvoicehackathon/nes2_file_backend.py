"""Version-seven FILE fusion: unchanged old17 plus two frozen Nes2 voice logits.

Invalid/missing auxiliary inputs raise ValueError; deployment callers must retain
the incumbent FILE fallback. This module does not alter component VOICE scores.
"""
import numpy as np
from scipy.special import expit

from .file_backend import FEATURES as BASE_FEATURES, EPSILON, features as base_features

VERSION = 7
MODEL_TYPE = 'gradient_boosted_trees_nes2voice19'
FEATURES = (*BASE_FEATURES, 'logit:nes2_fake', 'logit:voice_present*nes2_fake')


def features(rows):
    try:
        original = base_features(rows)
        probability = np.asarray([float(r['nes2_fake']) for r in rows], dtype=np.float64)
        presence = np.asarray([float(r['voice_present']) for r in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('missing/invalid Nes2 FILE feature input') from error
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Nes2 fake feature must be finite in [0,1]')
    if any(r.get('nes2_status', 'ok') != 'ok' or r.get('nes2_error', '') for r in rows):
        raise ValueError('failed Nes2 extraction')
    bounded = np.clip(np.column_stack((probability, presence * probability)), EPSILON, 1 - EPSILON)
    return np.column_stack((original, np.log(bounded) - np.log1p(-bounded)))


def predict(model, rows):
    if (model.get('version') != VERSION or model.get('model_type') != MODEL_TYPE
            or model.get('feature_names') != list(FEATURES) or model.get('epsilon') != EPSILON
            or model.get('input_dtype') != 'float32'):
        raise ValueError('incompatible Nes2voice19 export')
    x = features(rows).astype(np.float32)
    try:
        initial, rate = float(model['initial_log_odds']), float(model['learning_rate'])
        trees = model['trees']
        if not np.isfinite([initial, rate]).all() or rate <= 0 or not isinstance(trees, list) or not trees:
            raise ValueError('invalid tree parameters')
        raw = np.full(len(rows), initial, dtype=np.float64)
        for tree in trees:
            left, right, feature = (tree[k] for k in ('children_left', 'children_right', 'feature'))
            threshold, value = (np.asarray(tree[k], dtype=np.float64) for k in ('threshold', 'value'))
            n = len(left)
            if (not n or any(len(a) != n for a in (right, feature, threshold, value))
                    or not np.isfinite(threshold).all() or not np.isfinite(value).all()):
                raise ValueError('invalid tree arrays')
            for node in range(n):
                if any(type(a[node]) is not int for a in (left, right, feature)):
                    raise ValueError('noninteger tree coordinate')
                if left[node] == -1:
                    if right[node] != -1:
                        raise ValueError('invalid leaf')
                elif not (node < left[node] < n and node < right[node] < n and 0 <= feature[node] < len(FEATURES)):
                    raise ValueError('invalid/cyclic tree coordinate')
            for i, values in enumerate(x):
                node = 0
                while left[node] != -1:
                    node = left[node] if float(values[feature[node]]) <= threshold[node] else right[node]
                raw[i] += rate * value[node]
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('malformed Nes2voice19 tree export') from error
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite Nes2voice19 tree log odds')
    return expit(raw)
