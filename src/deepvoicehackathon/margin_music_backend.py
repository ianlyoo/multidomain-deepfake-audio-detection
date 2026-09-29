"""Strict MUSIC18 JSON inference. The caller owns incumbent retention on failure."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from . import music_backend as music

VERSION = 4
MODEL_TYPE = 'gradient_boosted_trees_music18_margin'
FEATURES = (*music.FEATURES, 'raw:lofcz_affine64')
SCORE_FIELDS = music.SCORE_FIELDS
EPSILON = music.EPSILON
PROTOCOL_SHA = '22744284cc609e5513c1c733e69933e487fac174997430a0172d6061500f9a70'
SOURCE44_SHA = '4c143dcf09155b24990bb0d638cb1432b90dfc061b6cef654b0e1ad408c9c841'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-music18-margin'
IMPLEMENTATIONS = ('src/deepvoicehackathon/margin_music_backend.py',
                   'tools/train_sub48_music_margin.py')


def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task='MUSIC_FAKE only',
                feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
                learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
                loss='log_loss', subsample=1., seed=20260905, training_rows=1680,
                training_groups=40, protocol_sha256=PROTOCOL_SHA,
                source44_sha256=SOURCE44_SHA, recipe=RECIPE)


def features(rows):
    """Keep the exact MUSIC17 prefix; append the unsaturated affine scalar."""
    try:
        prefix = music.features(rows)
        margins = []
        for row in rows:
            value = row['lofcz_z64']
            if (not np.isscalar(value) or np.asarray(value).dtype.kind not in 'fiu'
                    or not np.isfinite(value)):
                raise ValueError('finite real affine scalar required')
            margins.append(value)
        result = np.column_stack((prefix, np.asarray(margins, dtype=np.float64)))
        with np.errstate(over='ignore', invalid='ignore'):
            finite = np.isfinite(result.astype(np.float32)).all()
        if not finite:
            raise ValueError('finite float32 conversion required')
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid MUSIC18 probability/margin record') from error


def validate_model(model):
    """Validate all metadata and every node, including unvisited branches."""
    try:
        fixed = model_metadata()
        if (type(model) is not dict or set(model) != set(fixed) |
                {'initial_log_odds', 'trees', 'implementation_sha256'}):
            raise ValueError('strict model keys required')
        for key, value in fixed.items():
            if type(model[key]) is not type(value) or model[key] != value:
                raise ValueError('fixed recipe metadata mismatch: ' + key)
        pins = model['implementation_sha256']
        if (type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS)
                or any(type(v) is not str or len(v) != 64 or
                       any(c not in '0123456789abcdef' for c in v) for v in pins.values())):
            raise ValueError('portable implementation SHA256 map required')
        initial = model['initial_log_odds']
        if type(initial) not in (int, float) or not np.isfinite(initial):
            raise ValueError('finite initial log odds required')
        if type(model['trees']) is not list or len(model['trees']) != 64:
            raise ValueError('exactly 64 trees required')
        names = ('children_left', 'children_right', 'feature', 'threshold', 'value')
        for tree in model['trees']:
            if type(tree) is not dict or set(tree) != set(names):
                raise ValueError('strict tree object required')
            arrays = [tree[k] for k in names]
            if any(type(a) is not list for a in arrays):
                raise ValueError('JSON arrays required')
            left, right, feature, threshold, value = arrays
            n = len(left)
            if not 1 <= n <= 7 or any(len(a) != n for a in arrays):
                raise ValueError('depth-two tree size required')
            if any(type(v) is not int for a in arrays[:3] for v in a):
                raise ValueError('integer tree coordinates required')
            if any(type(v) not in (int, float) or not np.isfinite(v)
                   for a in arrays[3:] for v in a):
                raise ValueError('finite tree data required')
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
                    if (not 0 <= feature[node] < 18 or
                            not node < left[node] < n or not node < right[node] < n):
                        raise ValueError('invalid split bounds')
                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))
            if len(seen) != n:
                raise ValueError('unreachable nodes')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed MUSIC18 tree export') from error


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
        raise ValueError('nonfinite MUSIC18 log odds')
    return expit(raw)
