"""Frozen FILE29 arithmetic and version-ten inference; failures raise ValueError.

No model loading, temporal forwards, fallback, or artifact mutation occurs here.
The runner owns retention of the complete unrounded incumbent FILE probability.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import evidence_file_backend as evidence

VERSION = 10
MODEL_TYPE = 'gradient_boosted_trees_evidence29'
TASK = 'FILE_FAKE only'
PROTOCOL_SHA = 'c73189a851a1016a71001944ace0c42b8d80e89cabcfc6e491d0ac21029bcf8a'
SOURCE44_SHA = '4c143dcf09155b24990bb0d638cb1432b90dfc061b6cef654b0e1ad408c9c841'
PREDECESSOR_SHA = '8a9dba6701417ad018457e70f538c9745b7c9095b1dc3130f91a1a98e110a9c5'
TEMPORAL_HELPER_SHA = 'fe04881a63b681b3b7e022bd05f445edacd2295ae9c7b4df41260760dd503b6d'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-evidence29'
EPSILON = evidence.EPSILON
SCORE_FIELDS = tuple(name.removeprefix('logit:') for name in evidence.FEATURES[:13])
FEATURES29 = (*evidence.FEATURES,
              'logit:eliya_temporal_mean', 'logit:voice_present*eliya_temporal_mean',
              'logit:eliya_temporal_max', 'logit:voice_present*eliya_temporal_max',
              'prob:eliya_temporal_range')
FEATURES = FEATURES29


def additional_features(voice_present, eliya_temporal_scores):
    """Five float64 statistics of one to three complete native probabilities.

    Crop uniqueness and center provenance are authenticated by the producer;
    repeated probability values are valid and must not be deduplicated.
    """
    try:
        if (not isinstance(eliya_temporal_scores, list)
                or not 1 <= len(eliya_temporal_scores) <= 3):
            raise ValueError('one to three Eliya probabilities in a list required')
        supplied = (voice_present, *eliya_temporal_scores)
        if any(not np.isscalar(v) or np.asarray(v).dtype.kind not in 'fiu' for v in supplied):
            raise ValueError('real numeric scalar probabilities required')
        values = np.asarray(supplied, dtype=np.float64)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('finite probabilities in [0,1] required')
        presence, scores = values[0], values[1:]
        mean = np.mean(np.sort(scores), dtype=np.float64)
        maximum = np.max(scores)
        bounded = np.clip([mean, presence * mean, maximum, presence * maximum],
                          EPSILON, 1 - EPSILON)
        return np.concatenate((np.log(bounded) - np.log1p(-bounded),
                               [maximum - np.min(scores)]))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid FILE29 temporal probabilities') from error


def features(rows):
    """Preserve the exact old24 prefix, including its original input contract."""
    try:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError('nonempty feature rows required')
        original = evidence.features(rows)
        added = np.asarray([additional_features(float(r['voice_present']),
                                                r['eliya_temporal_scores'])
                            for r in rows], dtype=np.float64)
        return np.column_stack((original, added))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('missing/invalid FILE29 feature record') from error


def validate_model(model):
    """Validate all metadata and all tree nodes before any traversal; return None."""
    try:
        fixed = dict(version=VERSION, model_type=MODEL_TYPE, task=TASK,
                     feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
                     learning_rate=.05, n_estimators=64, max_depth=2,
                     min_samples_leaf=24, seed=20260905, training_rows=1920, training_groups=40,
                     source44_sha256=SOURCE44_SHA, predecessor_sha256=PREDECESSOR_SHA,
                     protocol_sha256=PROTOCOL_SHA, temporal_helper_sha256=TEMPORAL_HELPER_SHA,
                     recipe=RECIPE)
        if not isinstance(model, dict) or any(model.get(k) != v for k, v in fixed.items()):
            raise ValueError('incompatible fixed FILE29 export')
        for key in ('version', 'n_estimators', 'max_depth', 'min_samples_leaf', 'seed',
                    'training_rows', 'training_groups'):
            if type(model[key]) is not int:
                raise ValueError('integer recipe metadata required')
        for key in ('epsilon', 'learning_rate', 'initial_log_odds'):
            if type(model[key]) not in (int, float) or not np.isfinite(model[key]):
                raise ValueError('finite numeric model metadata required')
        trees = model['trees']
        if not isinstance(trees, list) or len(trees) != 64:
            raise ValueError('exactly 64 trees required')
        for tree in trees:
            if not isinstance(tree, dict):
                raise ValueError('JSON tree object required')
            arrays = [tree[k] for k in ('children_left', 'children_right', 'feature', 'threshold', 'value')]
            if any(not isinstance(a, list) for a in arrays):
                raise ValueError('one-dimensional JSON tree arrays required')
            left, right, feature, threshold, value = arrays
            n = len(left)
            if not 1 <= n <= 7 or any(len(a) != n for a in arrays):
                raise ValueError('invalid depth-two tree size')
            if any(type(v) is not int for a in arrays[:3] for v in a):
                raise ValueError('integer tree coordinates required')
            if any(type(v) not in (int, float) or not np.isfinite(v) for a in arrays[3:] for v in a):
                raise ValueError('finite scalar tree data required')
            seen, pending = set(), [(0, 0)]
            while pending:
                node, depth = pending.pop()
                if node in seen or depth > 2:
                    raise ValueError('shared/cyclic/deep tree node')
                seen.add(node)
                if left[node] == -1:
                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:
                        raise ValueError('invalid sklearn leaf')
                else:
                    if (not 0 <= feature[node] < len(FEATURES)
                            or not node < left[node] < n or not node < right[node] < n):
                        raise ValueError('invalid tree split coordinate')
                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))
            if len(seen) != n:
                raise ValueError('unreachable tree nodes')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed FILE29 tree export') from error


def predict(model, rows):
    validate_model(model)
    x = features(rows).astype(np.float32)
    raw = np.full(len(rows), model['initial_log_odds'], dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        for tree in model['trees']:
            for i, values in enumerate(x):
                node = 0
                while tree['children_left'][node] != -1:
                    node = (tree['children_left'][node]
                            if float(values[tree['feature'][node]]) <= tree['threshold'][node]
                            else tree['children_right'][node])
                raw[i] += model['learning_rate'] * tree['value'][node]
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite FILE29 log odds')
    return expit(raw)
