"""Frozen MUSIC24 inference; invalid inputs/models raise ValueError.

The caller owns exact sub44 retention. No neutral or partial evidence is filled.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import music_backend as music
from . import evidence_file_backend as evidence

SCORE_FIELDS = music.SCORE_FIELDS
EPSILON = music.EPSILON
FEATURES24 = tuple(evidence.FEATURES)
FEATURES = FEATURES24
VERSION = 2
MODEL_TYPE = 'gradient_boosted_trees_music24'
PROTOCOL_SHA = 'd1112dc1d9ac951a592bcea42831f3a6fa70fc768457dd5ecaa5346e1d82f693'
SOURCE44_SHA = '4c143dcf09155b24990bb0d638cb1432b90dfc061b6cef654b0e1ad408c9c841'
PREDECESSOR_SHA = '19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-music24'


def features(rows):
    """Strict original MUSIC17 plus the immutable seven FILE24 transforms."""
    try:
        original = music.features(rows)
        if any(r['evidence_status'] != 'ok' or r['evidence_error'] != '' for r in rows):
            raise ValueError('healthy complete evidence required')
        added = np.asarray([evidence.additional_features(
            r['voice_present'], r['eliya_fake'], r['spectra_scores']) for r in rows], dtype=np.float64)
        result = np.column_stack((original, added))
        if (result.shape != (len(rows), 24) or not np.isfinite(result).all()
                or not np.array_equal(result[:, :17], original)):
            raise ValueError('MUSIC24 feature contract drift')
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid MUSIC24 probability record') from error


def validate_model(model):
    """Validate the complete fixed recipe and every node without mutation."""
    try:
        fixed = dict(version=VERSION, model_type=MODEL_TYPE, task='MUSIC_FAKE only',
                     feature_names=list(FEATURES24), epsilon=EPSILON, input_dtype='float32',
                     learning_rate=.05, n_estimators=64, max_depth=2,
                     min_samples_leaf=24, seed=20260905, training_rows=1680,
                     training_groups=40, protocol_sha256=PROTOCOL_SHA,
                     source44_sha256=SOURCE44_SHA, predecessor_sha256=PREDECESSOR_SHA,
                     recipe=RECIPE)
        if not isinstance(model, dict) or any(model.get(k) != v for k, v in fixed.items()):
            raise ValueError('incompatible fixed MUSIC24 export')
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
                    if (not 0 <= feature[node] < len(FEATURES24)
                            or not node < left[node] < n or not node < right[node] < n):
                        raise ValueError('invalid tree split coordinate')
                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))
            if len(seen) != n:
                raise ValueError('unreachable tree nodes')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed MUSIC24 tree export') from error


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
        raise ValueError('nonfinite MUSIC24 log odds')
    return expit(raw)
