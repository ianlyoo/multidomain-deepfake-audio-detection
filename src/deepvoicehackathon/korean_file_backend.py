"""Strict FILE29 retrained with an appended Korean family; callers own whole-raw55 fallback."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import component_support_file_backend as baseline

VERSION = 19
MODEL_TYPE = 'gradient_boosted_trees_file29_korean'
SCORE_FIELDS = baseline.SCORE_FIELDS
EPSILON = baseline.EPSILON
FEATURES = baseline.FEATURES
SOURCE55_SHA = '5ef3c3ce76cf06270e76e6498c395939d1ebcf828e46e4ce75f2a672a650f78d'
PREDECESSOR_SHA = '0f38b7c9006d8ac309c6157140f7c9b473b390939cecbb9b1bb91282d1e08c92'
MUSIC17_SHA = '19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-file29-korean'
KOREAN_GROUP_PREFIX = 'kor-'
KOREAN_WEIGHT = 1.0
# Rewritten only by `tools/build_sub67_korean_rows.py freeze` from a complete capture report.
KOREAN_PLAN = {'plan_sha256': 'e29e8ee5ac13bf53c4692994c0e5482ce8c5f9ce70136f7494b1f52d494d2ed4', 'training_rows': 940, 'training_groups': 17, 'class_counts': {'0': 408, '1': 532}}
IMPLEMENTATIONS = ('src/deepvoicehackathon/korean_file_backend.py',
                   'tools/train_sub67_korean_file.py')
RECEIPT_HASHES = ('rows_sha256', 'groups_sha256', 'labels_sha256', 'weights_sha256',
                  'features_sha256', 'suppliers_sha256')


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def frozen():
    plan = KOREAN_PLAN
    return (type(plan) is dict and _sha(plan.get('plan_sha256')) and type(plan.get('training_rows')) is int
            and plan['training_rows'] > 0 and type(plan.get('training_groups')) is int
            and plan['training_groups'] > 0 and type(plan.get('class_counts')) is dict
            and set(plan['class_counts']) == {'0', '1'}
            and all(type(v) is int and v > 0 for v in plan['class_counts'].values())
            and sum(plan['class_counts'].values()) == plan['training_rows'])


def features(rows):
    try:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError('nonempty Korean FILE rows required')
        original = baseline.features(rows)
        if original.shape != (len(rows), 29):
            raise ValueError('original29 feature width drift')
        return original
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid FILE29 Korean evidence') from error


def model_metadata():
    plan = KOREAN_PLAN
    return dict(version=VERSION, model_type=MODEL_TYPE, task='FILE_FAKE only',
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
        loss='log_loss', subsample=1., seed=20260905,
        training_rows=2316 + plan['training_rows'], training_groups=40 + plan['training_groups'],
        old_rows=1920, native_rows=264, fake_rows=132, korean_rows=plan['training_rows'],
        korean_groups=plan['training_groups'], korean_weight=KOREAN_WEIGHT,
        korean_plan_sha256=plan['plan_sha256'], source55_sha256=SOURCE55_SHA,
        predecessor_sha256=PREDECESSOR_SHA, stage1_model_sha256=MUSIC17_SHA,
        korean_music17='deployed_full_MUSIC17_out_of_sample', stage1_oof_seed=20260905,
        inner_group_seed=20260915, recipe=RECIPE)


def _validate_receipt(receipt):
    keys = {'role', 'training_rows', 'training_groups', 'old_rows', 'native_rows', 'fake_rows',
            'korean_rows', 'extra_rows', 'class_counts', 'korean_class_counts', *RECEIPT_HASHES}
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError('strict actual fit receipt required')
    if receipt['role'] not in ('fold', 'full'):
        raise ValueError('explicit fold/full fit role required')
    if any(type(receipt[k]) is not int
           for k in ('training_rows', 'old_rows', 'native_rows', 'fake_rows', 'korean_rows', 'extra_rows')):
        raise ValueError('integer actual fit counts required')
    groups = receipt['training_groups']
    if (type(groups) is not list or not groups or any(type(g) is not str or not g for g in groups)
            or groups != sorted(set(groups))):
        raise ValueError('sorted unique actual training groups required')
    korean_groups = [g for g in groups if g.startswith(KOREAN_GROUP_PREFIX)]
    num = len(groups) - len(korean_groups)
    old, native, fake, korean, extra = (receipt['old_rows'], receipt['native_rows'],
        receipt['fake_rows'], receipt['korean_rows'], receipt['extra_rows'])
    if old != 48 * num:
        raise ValueError('actual old FILE view counts mismatch')
    if extra != native + fake + korean or receipt['training_rows'] != old + extra:
        raise ValueError('actual native/fake/Korean FILE view counts mismatch')
    if native % 12 or fake % 6 or not 0 <= native <= 264 or not 0 <= fake <= 132:
        raise ValueError('actual native/fake partition bounds mismatch')
    plan = KOREAN_PLAN
    if not 0 < korean <= plan['training_rows'] or not 0 < len(korean_groups) <= plan['training_groups']:
        raise ValueError('actual Korean partition bounds mismatch')
    kc = receipt['korean_class_counts']
    if (type(kc) is not dict or set(kc) != {'0', '1'} or any(type(v) is not int or v < 0 for v in kc.values())
            or kc['0'] + kc['1'] != korean or kc['0'] > plan['class_counts']['0']
            or kc['1'] > plan['class_counts']['1']):
        raise ValueError('actual Korean class counts mismatch')
    counts = receipt['class_counts']
    expected = {'0': 15 * num + native // 2 + kc['0'],
                '1': 33 * num + native // 2 + fake + kc['1']}
    if (type(counts) is not dict or set(counts) != {'0', '1'}
            or any(type(v) is not int or v <= 0 for v in counts.values()) or counts != expected):
        raise ValueError('actual FILE class counts mismatch')
    if receipt['role'] == 'full':
        if ((old, native, fake, num, korean, len(korean_groups), kc)
                != (1920, 264, 132, 40, plan['training_rows'], plan['training_groups'], plan['class_counts'])):
            raise ValueError('full receipt must prove 2316 retained rows plus the frozen Korean plan')
    elif num not in (30, 35) or len(korean_groups) >= plan['training_groups']:
        raise ValueError('outer receipt cannot masquerade as full fit')
    if any(not _sha(receipt[k]) for k in RECEIPT_HASHES):
        raise ValueError('actual fit identity SHA256 required')


def validate_model(model):
    try:
        if not frozen():
            raise ValueError('frozen Korean plan required')
        fixed = model_metadata()
        if type(model) is not dict or set(model) != set(fixed) | {'initial_log_odds', 'trees', 'implementation_sha256', 'fit_receipt'}:
            raise ValueError('strict Korean FILE29 model required')
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
        raise ValueError('malformed FILE29 Korean model') from error


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
