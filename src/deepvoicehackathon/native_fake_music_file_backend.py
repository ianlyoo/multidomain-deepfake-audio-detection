"""Strict native fake-music FILE25 trees; callers own whole-raw53 fallback."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import music_stacked_file_backend as stack

VERSION = 13
MODEL_TYPE = 'gradient_boosted_trees_file25_native_fake_music'
FEATURES = stack.FEATURES
SCORE_FIELDS = stack.SCORE_FIELDS
EPSILON = stack.EPSILON
PROTOCOL_SHA = '5e7f3c135bcb5aa31cdca6cf340b459b9f42115a82a27e1104ef505b61d03e20'
PROTOCOL_COMMIT = 'e7347b0'
SOURCE53_SHA = '5c6870c4d1b51e22207c2ae3aa23f90c87db37398f860fe1f09248f0c0673d0c'
PREDECESSOR_SHA = '2da7f008dab1a80be02f5f29560ddd967267b12a31108ec38e6c78f72135dd2b'
MUSIC17_SHA = '19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-file25-native-fake-music'
IMPLEMENTATIONS = ('src/deepvoicehackathon/native_fake_music_file_backend.py',
                   'tools/train_sub54_native_fake_music.py')
RECEIPT_HASHES = ('rows_sha256', 'groups_sha256', 'labels_sha256', 'weights_sha256',
                  'features_sha256', 'suppliers_sha256')


def features(rows):
    return stack.features(rows)


def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task='FILE_FAKE only',
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
        loss='log_loss', subsample=1., seed=20260905, training_rows=2316, training_groups=40,
        extra_training_rows=396, native_training_rows=264, fake_training_rows=132,
        protocol_sha256=PROTOCOL_SHA, protocol_commit=PROTOCOL_COMMIT,
        source53_sha256=SOURCE53_SHA, predecessor_sha256=PREDECESSOR_SHA,
        stage1_model_sha256=MUSIC17_SHA, stage1_oof_seed=20260905, inner_group_seed=20260915,
        recipe=RECIPE)


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _validate_receipt(receipt):
    keys = {'role', 'training_rows', 'training_groups', 'old_rows', 'native_rows',
            'fake_rows', 'extra_rows', 'class_counts', *RECEIPT_HASHES}
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError('strict actual fit receipt required')
    if receipt['role'] not in ('fold', 'full'):
        raise ValueError('explicit fold/full fit role required')
    if any(type(receipt[k]) is not int
           for k in ('training_rows', 'old_rows', 'native_rows', 'fake_rows', 'extra_rows')):
        raise ValueError('integer actual fit counts required')
    groups = receipt['training_groups']
    if (type(groups) is not list or not groups or any(type(g) is not str or not g for g in groups)
            or groups != sorted(set(groups))):
        raise ValueError('sorted unique actual training groups required')
    old, native, fake, extra = (receipt['old_rows'], receipt['native_rows'],
                                receipt['fake_rows'], receipt['extra_rows'])
    if old != 48 * len(groups):
        raise ValueError('actual old FILE view counts mismatch')
    if extra != native + fake or receipt['training_rows'] != old + extra:
        raise ValueError('actual native/fake FILE view counts mismatch')
    if not 0 < extra <= 396 or native % 12 or fake % 6 or native > 264 or fake > 132:
        raise ValueError('actual native/fake partition bounds mismatch')
    counts = receipt['class_counts']
    expected = {'0': 15 * len(groups) + native // 2,
                '1': 33 * len(groups) + native // 2 + fake}
    if (type(counts) is not dict or set(counts) != {'0', '1'}
            or any(type(v) is not int or v <= 0 for v in counts.values())
            or counts != expected):
        raise ValueError('actual FILE class counts mismatch')
    if receipt['role'] == 'full':
        if (old, native, fake, len(groups), counts) != (1920, 264, 132, 40, {'0': 732, '1': 1584}):
            raise ValueError('full receipt must prove2316/40')
    elif len(groups) not in (30, 35):
        raise ValueError('outer receipt cannot masquerade as full fit')
    if any(not _sha(receipt[k]) for k in RECEIPT_HASHES):
        raise ValueError('actual fit identity SHA256 required')


def validate_model(model):
    """Reject malformed metadata/trees without projection onto an old schema."""
    try:
        fixed = model_metadata()
        if type(model) is not dict or set(model) != set(fixed) | {'initial_log_odds', 'trees', 'implementation_sha256', 'fit_receipt'}:
            raise ValueError('strict native fake-music FILE25 model required')
        for key, value in fixed.items():
            if type(model[key]) is not type(value) or model[key] != value:
                raise ValueError('fixed FILE25 recipe mismatch: ' + key)
        pins = model['implementation_sha256']
        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any(not _sha(v) for v in pins.values()):
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
                    if not 0 <= feature[node] < 25 or not node < left[node] < n or not node < right[node] < n:
                        raise ValueError('invalid split coordinates')
                    pending.extend(((left[node], depth+1), (right[node], depth+1)))
            if len(seen) != n:
                raise ValueError('unreachable tree node')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed FILE25 native fake-music model') from error


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
