"""Strict FILE31 with Nes2 auxiliary features; callers own whole-raw55 fallback."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import component_support_file_backend as baseline

VERSION = 15
MODEL_TYPE = 'gradient_boosted_trees_file31_nes2aux'
SCORE_FIELDS = baseline.SCORE_FIELDS
EPSILON = baseline.EPSILON
FEATURES = (*baseline.FEATURES, 'logit:nes2_fake', 'logit:voice_present*nes2_fake')
PROTOCOL_SHA = '231e45185a2956029eb96a7cba9331b218381eb18eb0da681949901b01d852a6'
PROTOCOL_COMMIT = '0c24eb7'
SOURCE55_SHA = '5ef3c3ce76cf06270e76e6498c395939d1ebcf828e46e4ce75f2a672a650f78d'
PREDECESSOR_SHA = '0f38b7c9006d8ac309c6157140f7c9b473b390939cecbb9b1bb91282d1e08c92'
MUSIC17_SHA = '19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-file31-nes2aux'
IMPLEMENTATIONS = ('src/deepvoicehackathon/nes2_aux_file_backend.py',
                   'tools/train_sub56_nes2_file.py')
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
            raise ValueError('nonempty nes2aux FILE rows required')
        original = baseline.features(rows)
        if original.shape != (len(rows), 29):
            raise ValueError('original29 feature width drift')
        added = []
        for row in rows:
            if 'nes2_fake' not in row:
                raise ValueError('missing nes2aux probability: nes2_fake')
            if not _is_real_probability(row['nes2_fake']):
                raise ValueError('finite real nes2aux probability required: nes2_fake')
            if not _is_real_probability(row['voice_present']):
                raise ValueError('finite real nes2aux probability required: voice_present')
            fake = float(row['nes2_fake'])
            present = float(row['voice_present'])
            pair = present * fake
            if not np.isfinite(pair) or not 0.0 <= pair <= 1.0:
                raise ValueError('unrounded nes2aux product out of range')
            added.append([_logit(fake), _logit(pair)])
        return np.column_stack((original, np.asarray(added, dtype=np.float64)))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid FILE31 nes2aux evidence') from error


def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task='FILE_FAKE only',
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
        loss='log_loss', subsample=1., seed=20260905, training_rows=2316, training_groups=40,
        extra_training_rows=396, native_training_rows=264, fake_training_rows=132,
        protocol_sha256=PROTOCOL_SHA, protocol_commit=PROTOCOL_COMMIT,
        source55_sha256=SOURCE55_SHA, predecessor_sha256=PREDECESSOR_SHA,
        stage1_model_sha256=MUSIC17_SHA, stage1_oof_seed=20260905, inner_group_seed=20260915,
        recipe=RECIPE)


def _validate_receipt(receipt):
    return baseline._validate_receipt(receipt)


def validate_model(model):
    try:
        fixed = model_metadata()
        if type(model) is not dict or set(model) != set(fixed) | {'initial_log_odds', 'trees', 'implementation_sha256', 'fit_receipt'}:
            raise ValueError('strict nes2aux FILE31 model required')
        for key, value in fixed.items():
            if type(model[key]) is not type(value) or model[key] != value:
                raise ValueError('fixed FILE31 recipe mismatch: ' + key)
        pins = model['implementation_sha256']
        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS):
            raise ValueError('portable implementation SHA256 map required')
        for digest in pins.values():
            if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
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
                    if not 0 <= feature[node] < 31 or not node < left[node] < n or not node < right[node] < n:
                        raise ValueError('invalid split coordinates')
                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))
            if len(seen) != n:
                raise ValueError('unreachable tree node')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed FILE31 nes2aux model') from error


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
        raise ValueError('nonfinite FILE31 log odds')
    return expit(raw)
