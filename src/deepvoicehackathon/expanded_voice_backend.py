"""Frozen expanded VOICE8 features and strict standalone JSON inference for sub60."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

VERSION = 2
MODEL_TYPE = 'gradient_boosted_trees_voice8_expanded'
EPSILON = 1e-6
FEATURES = ('logit:df_voice_mean', 'logit:nii', 'logit:eliya_fake',
            'logit:spectra_temporal_mean', 'logit:spectra_temporal_max',
            'logit:voice_present', 'logit:voice44', 'prob:spectra_temporal_range')
PROTOCOL_SHA = '543563b7ff9963c7fc87c78cbb1d61d9b5ab969a6385df22ee7a2850ba8f0feb'
PROTOCOL_COMMIT = 'ec90350'
SOURCE55_SHA = '5ef3c3ce76cf06270e76e6498c395939d1ebcf828e46e4ce75f2a672a650f78d'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-voice8-expanded'
IMPLEMENTATIONS = ('src/deepvoicehackathon/expanded_voice_backend.py', 'tools/train_sub60_expanded_voice.py')
RECEIPT_HASHES = ('rows_sha256', 'groups_sha256', 'labels_sha256', 'weights_sha256', 'features_sha256')


def probability(value):
    if (not np.isscalar(value) or np.asarray(value).dtype.kind not in 'fiu'
            or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('finite real scalar probability required')
    return float(value)


def spectra_values(scores):
    if not isinstance(scores, list) or not 1 <= len(scores) <= 3:
        raise ValueError('one to three native Spectra probabilities required')
    return np.asarray([probability(value) for value in scores], dtype=np.float64)


def incumbent_voice(df_mean, nii, eliya_fake, spectra_scores):
    """Exact44 fixed blend arithmetic, including original stable NII branches."""
    df, nii, eliya = [probability(v) for v in (df_mean, nii, eliya_fake)]
    spectra = spectra_values(spectra_scores)
    blended = float((1.0 - .625) * df + .625 * float(np.mean(spectra)))
    current = float(np.clip(blended, EPSILON, 1.0 - EPSILON))
    nii = float(np.clip(nii, EPSILON, 1.0 - EPSILON))
    current_logit = np.log(current / (1.0 - current))
    nii_logit = np.log(nii / (1.0 - nii))
    combined = (1.0 - .2) * current_logit
    combined += .2 * nii_logit
    if combined >= 0.0:
        exp_negative = np.exp(-combined)
        voice37 = float(1.0 / (1.0 + exp_negative))
    else:
        exp_positive = np.exp(combined)
        voice37 = float(exp_positive / (1.0 + exp_positive))
    return probability(.75 * float(voice37) + .25 * float(eliya))


def features(rows):
    try:
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError('nonempty voice rows required')
        result = []
        for row in rows:
            df, nii, eliya, presence, current = [probability(row[k]) for k in
                ('df_voice_mean', 'nii', 'eliya_fake', 'voice_present', 'voice44')]
            scores = spectra_values(row['spectra_scores'])
            if current != incumbent_voice(df, nii, eliya, row['spectra_scores']):
                raise ValueError('exact incumbent44 VOICE binding required')
            maximum, minimum = float(np.max(scores)), float(np.min(scores))
            bounded = np.clip([df, nii, eliya, float(np.mean(scores)), maximum, presence, current],
                              EPSILON, 1 - EPSILON)
            result.append(np.concatenate((np.log(bounded) - np.log1p(-bounded), [maximum-minimum])))
        return np.asarray(result, dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid VOICE8 evidence') from error


def model_metadata():
    return dict(version=VERSION, model_type=MODEL_TYPE, task='VOICE_FAKE only',
        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype='float32',
        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,
        loss='log_loss', subsample=1., seed=20260905, training_rows=2016, training_groups=40,
        old_rows=960, native_rows=264, utterance_rows=792, extra_rows=1056,
        protocol_sha256=PROTOCOL_SHA, protocol_commit=PROTOCOL_COMMIT,
        source55_sha256=SOURCE55_SHA, recipe=RECIPE)


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _validate_receipt(receipt):
    keys = {'role', 'training_rows', 'training_groups', 'old_rows', 'native_rows',
            'utterance_rows', 'class_counts', *RECEIPT_HASHES}
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError('strict actual fit receipt required')
    if receipt['role'] not in ('fold', 'full'):
        raise ValueError('explicit fold/full fit role required')
    if any(type(receipt[k]) is not int
           for k in ('training_rows', 'old_rows', 'native_rows', 'utterance_rows')):
        raise ValueError('integer actual fit counts required')
    groups = receipt['training_groups']
    if (type(groups) is not list or not groups
            or any(type(g) is not str or not g for g in groups)
            or groups != sorted(set(groups))):
        raise ValueError('sorted unique actual training groups required')
    num = len(groups)
    old, native, utterance = receipt['old_rows'], receipt['native_rows'], receipt['utterance_rows']
    if old != 24 * num:
        raise ValueError('actual old VOICE view counts mismatch')
    if native % 12 or utterance % 24:
        raise ValueError('actual native/utterance partition multiple mismatch')
    if not 0 <= native <= 264 or not 0 <= utterance <= 792:
        raise ValueError('actual native/utterance partition bounds mismatch')
    if receipt['training_rows'] != old + native + utterance:
        raise ValueError('actual old/native/utterance VOICE view counts mismatch')
    if receipt['training_rows'] % 2:
        raise ValueError('actual VOICE class counts require even training rows')
    counts = receipt['class_counts']
    expected = {'0': receipt['training_rows'] // 2, '1': receipt['training_rows'] // 2}
    if (type(counts) is not dict or set(counts) != {'0', '1'}
            or any(type(v) is not int or v <= 0 for v in counts.values())
            or counts != expected):
        raise ValueError('actual VOICE class counts mismatch')
    if receipt['role'] == 'full':
        if (old, native, utterance, num, receipt['training_rows'], counts) != (
                960, 264, 792, 40, 2016, {'0': 1008, '1': 1008}):
            raise ValueError('full receipt must prove2016/40')
    else:
        if num != 30:
            raise ValueError('outer receipt cannot masquerade as full fit')
        if receipt['training_rows'] > 2016:
            raise ValueError('fold receipt exceeds full2016 coverage')
    if any(not _sha(receipt[k]) for k in RECEIPT_HASHES):
        raise ValueError('actual fit identity SHA256 required')


def validate_model(model):
    try:
        fixed = model_metadata()
        if type(model) is not dict or set(model) != set(fixed) | {'initial_log_odds', 'trees', 'implementation_sha256', 'fit_receipt'}:
            raise ValueError('strict VOICE8 expanded model object required')
        for key, value in fixed.items():
            if type(model[key]) is not type(value) or model[key] != value:
                raise ValueError('fixed VOICE8 expanded recipe mismatch: ' + key)
        pins = model['implementation_sha256']
        if (type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any(
                type(p) is not str or len(p) != 64 or any(c not in '0123456789abcdef' for c in p)
                for p in pins.values())):
            raise ValueError('portable implementation SHA256 map required')
        _validate_receipt(model['fit_receipt'])
        if type(model['initial_log_odds']) not in (int, float) or not np.isfinite(model['initial_log_odds']):
            raise ValueError('finite initial log odds required')
        if type(model['trees']) is not list or len(model['trees']) != 64:
            raise ValueError('exact64 trees required')
        names = ('children_left', 'children_right', 'feature', 'threshold', 'value')
        for tree in model['trees']:
            if type(tree) is not dict or set(tree) != set(names):
                raise ValueError('strict tree keys required')
            arrays = [tree[k] for k in names]
            if any(type(a) is not list for a in arrays):
                raise ValueError('JSON arrays required')
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
                    if not 0 <= feature[node] < 8 or not node < left[node] < n or not node < right[node] < n:
                        raise ValueError('invalid split coordinates')
                    pending.extend(((left[node], depth+1), (right[node], depth+1)))
            if len(seen) != n:
                raise ValueError('unreachable tree node')
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        raise ValueError('malformed VOICE8 expanded model') from error


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
        raise ValueError('nonfinite VOICE8 expanded log odds')
    return expit(raw)
