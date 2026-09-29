"""Version-nine provenance gate around the immutable FILE24 arithmetic.

Failure is a ValueError; the runner owns fallback to the entire sub39 FILE.
The compatibility mapping changes a copy, never the supplied artifact.
"""
from __future__ import annotations

from . import evidence_file_backend as evidence

VERSION = 9
MODEL_TYPE = 'gradient_boosted_trees_evidence24_expanded'
PROTOCOL_SHA = '816329526cb0c0f75b30ea09f972d04eb21c197b851b50e4939d07b08ac57eb3'
SOURCE39_SHA = 'd3137f8133ec46acfc1d9c8ada8bcf6f053531b72d76399cd65b0ea1cdc0d027'
PREDECESSOR_SHA = '8a9dba6701417ad018457e70f538c9745b7c9095b1dc3130f91a1a98e110a9c5'
RECIPE = 'fixed64-depth2-lr0.05-leaf24-evidence24-expanded'
FEATURES = evidence.FEATURES
EPSILON = evidence.EPSILON
features = evidence.features


def compatible_model(model):
    """Strictly validate new metadata, then return an old-compatible copy.

Tree arrays are shared read-only with the caller; neither validator nor scorer
mutates them. Only explicitly changed provenance fields are translated.
"""
    fixed = dict(version=VERSION, model_type=MODEL_TYPE,
                 training_rows=2355, training_groups=69,
                 old_training_rows=1920, new_training_rows=435,
                 source39_sha256=SOURCE39_SHA, predecessor_sha256=PREDECESSOR_SHA,
                 protocol_sha256=PROTOCOL_SHA, recipe=RECIPE)
    if not isinstance(model, dict) or any(model.get(k) != v for k, v in fixed.items()):
        raise ValueError('incompatible expanded FILE24 provenance')
    if any(type(model[k]) is not int for k in (
            'version', 'training_rows', 'training_groups', 'old_training_rows', 'new_training_rows')):
        raise ValueError('integer expanded FILE24 population metadata required')
    compatible = dict(model)
    compatible.update(version=evidence.VERSION, model_type=evidence.MODEL_TYPE,
                      training_rows=1920, training_groups=40,
                      source38_sha256=evidence.SOURCE38_SHA,
                      predecessor_sha256=evidence.PREDECESSOR_SHA,
                      protocol_sha256=evidence.PROTOCOL_SHA, recipe=evidence.RECIPE)
    evidence.validate_model(compatible)
    return compatible


def validate_model(model):
    """Return None on success, without mutating or retaining the artifact."""
    compatible_model(model)


def predict(model, rows):
    return evidence.predict(compatible_model(model), rows)
