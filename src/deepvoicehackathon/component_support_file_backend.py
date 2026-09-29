"""Strict FILE29 with final-component support features; callers own whole-raw54 fallback."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import native_fake_music_file_backend as baseline

VERSION = 14
MODEL_TYPE = 'gradient_boosted_trees_file29_component_support'
SCORE_FIELDS = baseline.SCORE_FIELDS
EPSILON = baseline.EPSILON
FEATURES = (*baseline.FEATURES, 'logit:voice38', 'logit:voice_present*voice38',
            'logit:music_present*music17', 'logit:max(voice_present*voice38,music_present*music17)')
PROTOCOL_SHA = 'f4d5175a0f0c7ba002fe0c0f30b4e61e0decaf736daac4b6e6c2c01825d337b0'
PROTOCOL_COMMIT = '5da86db'
SOURCE54_SHA = '707cc797728b18d6856a080dba413df297ccf3e665bc5dbce5aec754e064477f'
PREDECESSOR_SHA = 'ed34f05d680479eb487504cc69aa6ef05a979d7b0276100dfd7b60164840fb19'
MUSIC17_SHA = '19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-file29-component-support'
IMPLEMENTATIONS = ('src/deepvoicehackathon/component_support_file_backend.py',
                   'tools/train_sub55_component_support.py')
RECEIPT_HASHES = ('rows_sha256', 'groups_sha256', 'labels_sha256', 'weights_sha256',
                  'features_sha256', 'suppliers_sha256')


def _is_real_probability(value):
    if isinstance(value, (bool, np.bool_)):
        return False
    if not np.isscalar(value):
        return False
    if np.asarray(value).dtype.kind not in 'fiu':
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(number) and 0.0 <= number <= 1.0)


def _logit(value):
    clipped = min(max(float(value), EPSILON), 1.0 - EPSILON)
    return float(np.log(clipped) - np.log1p(-clipped))


def features(rows):
    try:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError('nonempty component-support FILE rows required')
        original = baseline.features(rows)
        if original.shape != (len(rows), 25):
            raise ValueError('original25 feature width drift')
        added = []
        for row in rows:
            for key in ('voice_present', 'music_present', 'music17', 'voice38'):
                if key not in row:
                    raise ValueError('missing component-support probability: ' + key)
                if not _is_real_probability(row[key]):
                    raise ValueError('finite real component-support probability required: ' + key)
            voice38 = float(row['voice38'])
            present_voice = float(row['voice_present'])
            present_music = float(row['music_present'])
            music17 = float(row['music17'])
            v = round(voice38, 10)
            vp10 = round(present_voice, 10)
            mp10 = round(present_music, 10)
            m10 = round(music17, 10)
            sv = round(vp10 * v, 10)
            sm = round(mp10 * m10, 10)
            peak = sv if sv >= sm else sm
            for candidate in (v, sv, sm, peak):
                if not np.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
                    raise ValueError('rounded component support out of range')
            added.append([_logit(v), _logit(sv), _logit(sm), _logit(peak)])
        return np.column_stack((original, np.asarray(added, dtype=np.float64)))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid FILE29 component-support evidence') from error


def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task='FILE_FAKE only',
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
        loss='log_loss', subsample=1., seed=20260905, training_rows=2316, training_groups=40,
        extra_training_rows=396, native_training_rows=264, fake_training_rows=132,
        protocol_sha256=PROTOCOL_SHA, protocol_commit=PROTOCOL_COMMIT,
        source54_sha256=SOURCE54_SHA, predecessor_sha256=PREDECESSOR_SHA,
        stage1_model_sha256=MUSIC17_SHA, stage1_oof_seed=20260905, inner_group_seed=20260915,
        recipe=RECIPE)


def _validate_receipt(receipt):
    return baseline._validate_receipt(receipt)


def validate_model(model):
    try:
        fixed = model_metadata()
        if type(model) is not dict or set(model) != set(fixed) | {'initial_log_odds', 'trees', 'implementation_sha256', 'fit_receipt'}:
            raise ValueError('strict component-support FILE29 model required')
        for key, value in fixed.items():
            if type(model[key]) is not type(value) or model[key] != value:
                raise ValueError('fixed FILE29 recipe mismatch: ' + key)
        pins = model['implementation_sha256']
        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any(not baseline._sha(v) for v in pins.values()):
            raise ValueError('portable implementation SHA256 map required')
        _validate_receipt(model['fit_receipt'])
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
                    if not 0 <= feature[node] < 29 or not node < left[node] < n or not node < right[node] < n:
                        raise ValueError('invalid split coordinates')
                    pending.extend(((left[node], depth+1), (right[node], depth+1)))
            if len(seen) != n:
                raise ValueError('unreachable tree node')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed FILE29 component-support model') from error


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
        raise ValueError('nonfinite FILE29 log odds')
    return expit(raw)
