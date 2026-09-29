"""Frozen FILE24 features and strict version-eight JSON inference.

Errors are observable as ValueError. Callers retain the entire incumbent FILE
on failure; this module never substitutes neutral or partial evidence.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .file_backend import FEATURES as BASE_FEATURES, EPSILON, features as base_features

VERSION = 8
MODEL_TYPE = 'gradient_boosted_trees_evidence24'
PROTOCOL_SHA = 'fbdc4bc4532a894e122ce46a04b39f1d2be0d0ebc1f0d89b142e7af977387f63'
SOURCE38_SHA = 'e2f1c5d32fec53d4d1580d2aee5430de46545f519bda423b2422e0d2d1a95b3d'
PREDECESSOR_SHA = 'cf05f8ed590efb4d3baeb821b7a9445aef2b6445554d54d66b26444d7f9787a9'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-evidence24'
FEATURES = (*BASE_FEATURES,
            'logit:eliya_fake', 'logit:voice_present*eliya_fake',
            'logit:spectra_temporal_mean', 'logit:voice_present*spectra_temporal_mean',
            'logit:spectra_temporal_max', 'logit:voice_present*spectra_temporal_max',
            'prob:spectra_temporal_range')


def additional_features(voice_present, eliya_fake, spectra_scores):
    """Return seven float64 values, preserving supplied native probabilities."""
    try:
        if not isinstance(spectra_scores, list) or not 1 <= len(spectra_scores) <= 3:
            raise ValueError('one to three Spectra probabilities in a list required')
        if any(not np.isscalar(v) or np.asarray(v).dtype.kind not in 'fiu'
               for v in (voice_present, eliya_fake, *spectra_scores)):
            raise ValueError('real numeric scalar probabilities required')
        values = np.asarray([voice_present, eliya_fake, *spectra_scores], dtype=np.float64)
        if (values.shape != (2 + len(spectra_scores),) or not np.isfinite(values).all()
                or ((values < 0) | (values > 1)).any()):
            raise ValueError('finite scalar probabilities in [0,1] required')
        presence, eliya = values[:2]
        scores = values[2:]
        mean, maximum = np.mean(scores), np.max(scores)
        bounded = np.clip([eliya, presence * eliya, mean, presence * mean,
                           maximum, presence * maximum], EPSILON, 1 - EPSILON)
        return np.concatenate((np.log(bounded) - np.log1p(-bounded),
                               [maximum - np.min(scores)]))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid FILE24 auxiliary probabilities') from error


def features(rows):
    try:
        if any(r['evidence_status'] != 'ok' or r['evidence_error'] != '' for r in rows):
            raise ValueError('healthy complete evidence required')
        original = base_features(rows)
        added = np.asarray([additional_features(float(r['voice_present']), r['eliya_fake'], r['spectra_scores'])
                            for r in rows], dtype=np.float64)
        return np.column_stack((original, added))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('missing/invalid FILE24 feature record') from error


def validate_model(model):
    """Return None on success; validate without mutating or retaining the model."""
    try:
        fixed = dict(version=VERSION, model_type=MODEL_TYPE, feature_names=list(FEATURES),
                     epsilon=EPSILON, input_dtype='float32', task='FILE_FAKE only',
                     learning_rate=.05, n_estimators=64, max_depth=2,
                     min_samples_leaf=24, seed=20260905, training_rows=1920, training_groups=40,
                     source38_sha256=SOURCE38_SHA, predecessor_sha256=PREDECESSOR_SHA,
                     protocol_sha256=PROTOCOL_SHA, recipe=RECIPE)
        if not isinstance(model, dict) or any(model.get(k) != v for k, v in fixed.items()):
            raise ValueError('incompatible fixed FILE24 export')
        for key in ('version', 'n_estimators', 'max_depth', 'min_samples_leaf', 'seed',
                    'training_rows', 'training_groups'):
            if type(model[key]) is not int:
                raise ValueError('integer recipe metadata required')
        if type(model['initial_log_odds']) not in (int, float) or not np.isfinite(model['initial_log_odds']):
            raise ValueError('invalid initial log odds')
        trees = model['trees']
        if not isinstance(trees, list) or len(trees) != 64:
            raise ValueError('exactly 64 trees required')
        for tree in trees:
            names = ('children_left', 'children_right', 'feature', 'threshold', 'value')
            arrays = [tree[k] for k in names]
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
        raise ValueError('malformed FILE24 tree export') from error


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
        raise ValueError('nonfinite FILE24 log odds')
    return expit(raw)
