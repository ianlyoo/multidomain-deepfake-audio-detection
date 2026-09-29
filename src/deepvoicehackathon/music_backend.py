"""Frozen MUSIC17 features and strict JSON inference; failures raise ValueError.

Only original component probabilities enter this head. The caller owns exact
incumbent MUSIC retention; this module never fills missing evidence.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .file_backend import SCORE_FIELDS, FEATURES, EPSILON, features as base_features

VERSION = 1
MODEL_TYPE = 'gradient_boosted_trees_music17'
PROTOCOL_SHA = '00741ffe13d7182892b985e758464f0f6d034e866add59319543a1fc314e92d3'
SOURCE39_SHA = 'd3137f8133ec46acfc1d9c8ada8bcf6f053531b72d76399cd65b0ea1cdc0d027'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-music17'


def features(rows):
    """Exact old17 transform, requiring real numeric original13 probabilities."""
    try:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError('nonempty original probability rows required')
        for row in rows:
            for name in SCORE_FIELDS:
                value = row[name]
                if (not np.isscalar(value) or np.asarray(value).dtype.kind not in 'fiu'
                        or not np.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError('finite numeric scalar probability required')
        return base_features(rows)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid MUSIC17 original probability record') from error


def validate_model(model):
    """Validate every node and recipe field without mutating the export."""
    try:
        fixed = dict(version=VERSION, model_type=MODEL_TYPE, task='MUSIC_FAKE only',
                     feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
                     learning_rate=.05, n_estimators=64, max_depth=2,
                     min_samples_leaf=24, seed=20260905, training_rows=1680,
                     training_groups=40, protocol_sha256=PROTOCOL_SHA,
                     source39_sha256=SOURCE39_SHA, recipe=RECIPE)
        if not isinstance(model, dict) or any(model.get(k) != v for k, v in fixed.items()):
            raise ValueError('incompatible fixed MUSIC17 export')
        for key in ('version', 'n_estimators', 'max_depth', 'min_samples_leaf',
                    'seed', 'training_rows', 'training_groups'):
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
            arrays = [tree[k] for k in ('children_left', 'children_right', 'feature',
                                      'threshold', 'value')]
            if any(not isinstance(a, list) for a in arrays):
                raise ValueError('one-dimensional JSON tree arrays required')
            left, right, feature, threshold, value = arrays
            n = len(left)
            if not 1 <= n <= 7 or any(len(a) != n for a in arrays):
                raise ValueError('invalid depth-two tree size')
            if any(type(v) is not int for a in arrays[:3] for v in a):
                raise ValueError('integer tree coordinates required')
            if any(type(v) not in (int, float) or not np.isfinite(v)
                   for a in arrays[3:] for v in a):
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
        raise ValueError('malformed MUSIC17 tree export') from error


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
        raise ValueError('nonfinite MUSIC17 log odds')
    return expit(raw)
