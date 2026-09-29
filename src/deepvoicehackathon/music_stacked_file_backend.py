"""Strict FILE24 plus learned MUSIC17 feature; caller owns incumbent fallback."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import evidence_file_backend as evidence

VERSION = 11
MODEL_TYPE = 'gradient_boosted_trees_file25_musicstack'
SCORE_FIELDS = ('df_voice_mean', 'df_music_mean', 'df_music_max', 'spectra', 'nii',
                'artifactnet', 'lofcz', 'current_voice', 'current_music', 'voice_present',
                'music_present', 'current_file', 'file_music')
EPSILON = evidence.EPSILON
FEATURES = (*evidence.FEATURES, 'logit:music17')
PROTOCOL_SHA = '4637b4583ea41413dfc0113b091fcea43ad20182f6b5e3c33e6f67e040134dc8'
SOURCE44_SHA = '4c143dcf09155b24990bb0d638cb1432b90dfc061b6cef654b0e1ad408c9c841'
PREDECESSOR_SHA = '8a9dba6701417ad018457e70f538c9745b7c9095b1dc3130f91a1a98e110a9c5'
MUSIC17_SHA = '19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-file25-nested-musicstack'
IMPLEMENTATIONS = ('src/deepvoicehackathon/music_stacked_file_backend.py', 'tools/train_sub50_music_stack.py')


def features(rows):
    try:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError('nonempty original FILE24 rows required')
        for row in rows:
            for key in SCORE_FIELDS:
                value = row[key]
                if (not np.isscalar(value) or np.asarray(value).dtype.kind not in 'fiu'
                        or not np.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError('finite real original13 probabilities required')
        original = evidence.features(rows)
        music = []
        for row in rows:
            value = row['music17']
            if (not np.isscalar(value) or np.asarray(value).dtype.kind not in 'fiu'
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('successful real MUSIC17 probability required')
            music.append(value)
        p = np.clip(np.asarray(music, dtype=np.float64), EPSILON, 1-EPSILON)
        return np.column_stack((original, np.log(p)-np.log1p(-p)))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid FILE25 MUSIC stacking evidence') from error


def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task='FILE_FAKE only',
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
        loss='log_loss', subsample=1., seed=20260905, training_rows=1920, training_groups=40,
        protocol_sha256=PROTOCOL_SHA, source44_sha256=SOURCE44_SHA,
        predecessor_sha256=PREDECESSOR_SHA, stage1_model_sha256=MUSIC17_SHA,
        stage1_oof_seed=20260905, inner_group_seed=20260915, recipe=RECIPE)


def validate_model(model):
    try:
        fixed = model_metadata()
        if type(model) is not dict or set(model) != set(fixed) | {'initial_log_odds', 'trees', 'implementation_sha256'}:
            raise ValueError('strict FILE25 model object required')
        for key, value in fixed.items():
            if type(model[key]) is not type(value) or model[key] != value:
                raise ValueError('fixed FILE25 recipe mismatch: ' + key)
        pins = model['implementation_sha256']
        if (type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any(
                type(p) is not str or len(p) != 64 or any(c not in '0123456789abcdef' for c in p)
                for p in pins.values())):
            raise ValueError('portable implementation SHA256 map required')
        if type(model['initial_log_odds']) not in (int, float) or not np.isfinite(model['initial_log_odds']):
            raise ValueError('finite initial log odds required')
        if type(model['trees']) is not list or len(model['trees']) != 64:
            raise ValueError('exactly64 trees required')
        names = ('children_left', 'children_right', 'feature', 'threshold', 'value')
        for tree in model['trees']:
            if type(tree) is not dict or set(tree) != set(names):
                raise ValueError('strict tree object required')
            arrays = [tree[k] for k in names]
            if any(type(a) is not list for a in arrays):
                raise ValueError('JSON tree arrays required')
            left, right, feature, threshold, value = arrays
            n = len(left)
            if not 1 <= n <= 7 or any(len(a) != n for a in arrays):
                raise ValueError('depth-two tree bounds required')
            if any(type(v) is not int for a in arrays[:3] for v in a):
                raise ValueError('integer tree coordinates required')
            if any(type(v) not in (int, float) or not np.isfinite(v) for a in arrays[3:] for v in a):
                raise ValueError('finite numeric tree data required')
            seen, pending = set(), [(0, 0)]
            while pending:
                node, depth = pending.pop()
                if node in seen or depth > 2:
                    raise ValueError('shared/cyclic/deep node')
                seen.add(node)
                if left[node] == -1:
                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:
                        raise ValueError('invalid sklearn leaf')
                else:
                    if not 0 <= feature[node] < 25 or not node < left[node] < n or not node < right[node] < n:
                        raise ValueError('invalid split coordinates')
                    pending.extend(((left[node], depth+1), (right[node], depth+1)))
            if len(seen) != n:
                raise ValueError('unreachable tree node')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed FILE25 music-stack model') from error


def predict(model, rows):
    validate_model(model)
    x = features(rows).astype(np.float32)
    raw = np.full(len(rows), model['initial_log_odds'], dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        for tree in model['trees']:
            for i, values in enumerate(x):
                node = 0
                while tree['children_left'][node] != -1:
                    node = (tree['children_left'][node] if float(values[tree['feature'][node]]) <= tree['threshold'][node]
                            else tree['children_right'][node])
                raw[i] += model['learning_rate'] * tree['value'][node]
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite FILE25 log odds')
    return expit(raw)
