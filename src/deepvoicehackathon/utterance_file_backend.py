"""Strict FILE29 with whole-utterance coverage; callers own whole-raw55 fallback."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import component_support_file_backend as baseline

VERSION = 17
MODEL_TYPE = 'gradient_boosted_trees_file29_utterance'
SCORE_FIELDS = baseline.SCORE_FIELDS
EPSILON = baseline.EPSILON
FEATURES = baseline.FEATURES
PROTOCOL_SHA = '7cc3408be416a8f179e6dfb5cfdbdf523399afea22e647686cd6d51b468ef1ab'
PROTOCOL_COMMIT = '075dc9c'
SOURCE55_SHA = '5ef3c3ce76cf06270e76e6498c395939d1ebcf828e46e4ce75f2a672a650f78d'
PREDECESSOR_SHA = '0f38b7c9006d8ac309c6157140f7c9b473b390939cecbb9b1bb91282d1e08c92'
PLAN_SHA = '9b788332b9fcbe6dc0849c065fb29d84ca3d0d4d2fe65322922fb32718306f8c'
PLAN_FILE_SHA = 'a1c92751f073efdfc742e25004cf9000fe72384723093f8fa4af053aadb98c2e'
MUSIC17_SHA = '19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-file29-utterance'
IMPLEMENTATIONS = ('src/deepvoicehackathon/utterance_file_backend.py',
                   'tools/train_sub59_utterance.py')
RECEIPT_HASHES = ('rows_sha256', 'groups_sha256', 'labels_sha256', 'weights_sha256',
                  'features_sha256', 'suppliers_sha256')


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def features(rows):
    try:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError('nonempty utterance FILE rows required')
        original = baseline.features(rows)
        if original.shape != (len(rows), 29):
            raise ValueError('original29 feature width drift')
        return original
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid FILE29 utterance evidence') from error


def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task='FILE_FAKE only',
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
        loss='log_loss', subsample=1., seed=20260905, training_rows=3504, training_groups=40,
        old_rows=1920, native_rows=264, fake_rows=132, utterance_rows=1188, extra_rows=1584,
        protocol_sha256=PROTOCOL_SHA, protocol_commit=PROTOCOL_COMMIT,
        source55_sha256=SOURCE55_SHA, predecessor_sha256=PREDECESSOR_SHA,
        plan_sha256=PLAN_SHA, plan_file_sha256=PLAN_FILE_SHA,
        stage1_model_sha256=MUSIC17_SHA, stage1_oof_seed=20260905, inner_group_seed=20260915,
        recipe=RECIPE)


def _validate_receipt(receipt):
    keys = {'role', 'training_rows', 'training_groups', 'old_rows', 'native_rows',
            'fake_rows', 'utterance_rows', 'extra_rows', 'class_counts', *RECEIPT_HASHES}
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError('strict actual fit receipt required')
    if receipt['role'] not in ('fold', 'full'):
        raise ValueError('explicit fold/full fit role required')
    if any(type(receipt[k]) is not int
           for k in ('training_rows', 'old_rows', 'native_rows', 'fake_rows',
                     'utterance_rows', 'extra_rows')):
        raise ValueError('integer actual fit counts required')
    groups = receipt['training_groups']
    if (type(groups) is not list or not groups or any(type(g) is not str or not g for g in groups)
            or groups != sorted(set(groups))):
        raise ValueError('sorted unique actual training groups required')
    num = len(groups)
    old, native, fake, utterance, extra = (receipt['old_rows'], receipt['native_rows'],
        receipt['fake_rows'], receipt['utterance_rows'], receipt['extra_rows'])
    if old != 48 * num:
        raise ValueError('actual old FILE view counts mismatch')
    if extra != native + fake + utterance or receipt['training_rows'] != old + extra:
        raise ValueError('actual native/fake/utterance FILE view counts mismatch')
    if not 0 < extra <= 1584 or native % 12 or fake % 6 or utterance % 36:
        raise ValueError('actual native/fake/utterance partition bounds mismatch')
    if not 0 <= native <= 264 or not 0 <= fake <= 132 or not 0 < utterance <= 1188:
        raise ValueError('actual native/fake/utterance partition bounds mismatch')
    if utterance > 36 * num:
        raise ValueError('actual utterance rows exceed selected speaker capacity')
    counts = receipt['class_counts']
    expected = {'0': 15 * num + native // 2 + utterance // 3,
                '1': 33 * num + native // 2 + fake + 2 * utterance // 3}
    if (type(counts) is not dict or set(counts) != {'0', '1'}
            or any(type(v) is not int or v <= 0 for v in counts.values())
            or counts != expected):
        raise ValueError('actual FILE class counts mismatch')
    if receipt['role'] == 'full':
        if (old, native, fake, utterance, extra, num, counts) != (1920, 264, 132, 1188, 1584, 40, {'0': 1128, '1': 2376}):
            raise ValueError('full receipt must prove3504/40')
        if receipt['training_rows'] != 3504:
            raise ValueError('full receipt must prove3504/40')
    elif num not in (30, 35):
        raise ValueError('outer receipt cannot masquerade as full fit')
    if any(not _sha(receipt[k]) for k in RECEIPT_HASHES):
        raise ValueError('actual fit identity SHA256 required')


def validate_model(model):
    try:
        fixed = model_metadata()
        if type(model) is not dict or set(model) != set(fixed) | {'initial_log_odds', 'trees', 'implementation_sha256', 'fit_receipt'}:
            raise ValueError('strict utterance FILE29 model required')
        for key, value in fixed.items():
            if type(model[key]) is not type(value) or model[key] != value:
                raise ValueError('fixed FILE29 recipe mismatch: ' + key)
        pins = model['implementation_sha256']
        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS):
            raise ValueError('portable implementation SHA256 map required')
        for digest in pins.values():
            if not _sha(digest):
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
        raise ValueError('malformed FILE29 utterance model') from error


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
