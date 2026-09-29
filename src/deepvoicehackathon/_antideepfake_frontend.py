"""Torch-native fairseq v0.12.2-faithful Wav2Vec2 front-end for AntiDeepfake.

Transcribed operation-for-operation from the fairseq 0.12.2 sdist module
fairseq.models.wav2vec.wav2vec2 (Wav2Vec2Model, ConvFeatureExtractionModel,
make_conv_pos, TransformerEncoder and its local TransformerSentenceEncoderLayer
with layer_norm_first=True, plus SamePad, TransposeLast, Fp32LayerNorm and the
exact-erf gelu from fairseq.modules). Only the inference path taken by the
model card is implemented: forward(source, mask=False, features_only=True)
with padding_mask=None, all dropouts at 0.0, feature_grad_mult == 1.0,
crop_seq_to_multiple == 1, quantize_input False and no target_glu.

State-dict keys are verbatim fairseq names (m_ssl.model.* plus proj_fc.*),
including the legacy manual weight-norm parameters
encoder.pos_conv.0.{weight_g, weight_v, bias}, so the checkpoint loads with
strict=True after dropping exactly the audited pretraining-only tensors.

This module never imports torch at module level: torch classes are built by
build_native_detector_class(torch), which keeps voice_eval importable without
torch. Key tables, shape tables and the header audit below are pure Python.
"""

from __future__ import annotations

# Version of the native forward graph and key table. Wired into the detector
# spec config so cached scores from an older frontend can never be reused.
# Bump on any change to the forward graph, key table or weight semantics.
NATIVE_FRONTEND_VERSION = 1

CONV_SPECS = (
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
    (512, 2, 2),
)
CONV_IN_CHANNELS = 1
FEATURE_DIM = 512
EMBED_DIM = 1024
FFN_DIM = 4096
NUM_HEADS = 16
HEAD_DIM = EMBED_DIM // NUM_HEADS
ATTN_SCALE = HEAD_DIM ** -0.5
NUM_LAYERS = 24
POS_CONV_KERNEL = 128
POS_CONV_GROUPS = 16
POS_CONV_PAD = POS_CONV_KERNEL // 2
REQUIRED_SEQ_LEN_MULTIPLE = 2
LAYER_NORM_EPS = 1e-5
NUM_CLASSES = 2

PRETRAINING_ONLY_KEYS = (
    "m_ssl.model.final_proj.weight",
    "m_ssl.model.final_proj.bias",
    "m_ssl.model.project_q.weight",
    "m_ssl.model.project_q.bias",
    "m_ssl.model.quantizer.vars",
    "m_ssl.model.quantizer.weight_proj.weight",
    "m_ssl.model.quantizer.weight_proj.bias",
    "m_ssl.model.mask_emb",
)

_SSL_PREFIX = "m_ssl.model"


def expected_key_shapes():
    shapes = {}
    in_dim = CONV_IN_CHANNELS
    for index, spec in enumerate(CONV_SPECS):
        dim, kernel, _stride = spec
        base = _SSL_PREFIX + ".feature_extractor.conv_layers." + str(index)
        shapes[base + ".0.weight"] = (dim, in_dim, kernel)
        shapes[base + ".0.bias"] = (dim,)
        shapes[base + ".2.1.weight"] = (dim,)
        shapes[base + ".2.1.bias"] = (dim,)
        in_dim = dim
    shapes[_SSL_PREFIX + ".layer_norm.weight"] = (FEATURE_DIM,)
    shapes[_SSL_PREFIX + ".layer_norm.bias"] = (FEATURE_DIM,)
    shapes[_SSL_PREFIX + ".post_extract_proj.weight"] = (EMBED_DIM, FEATURE_DIM)
    shapes[_SSL_PREFIX + ".post_extract_proj.bias"] = (EMBED_DIM,)
    shapes[_SSL_PREFIX + ".encoder.pos_conv.0.weight_g"] = (1, 1, POS_CONV_KERNEL)
    shapes[_SSL_PREFIX + ".encoder.pos_conv.0.weight_v"] = (
        EMBED_DIM,
        EMBED_DIM // POS_CONV_GROUPS,
        POS_CONV_KERNEL,
    )
    shapes[_SSL_PREFIX + ".encoder.pos_conv.0.bias"] = (EMBED_DIM,)
    for layer in range(NUM_LAYERS):
        base = _SSL_PREFIX + ".encoder.layers." + str(layer)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            shapes[base + ".self_attn." + proj + ".weight"] = (EMBED_DIM, EMBED_DIM)
            shapes[base + ".self_attn." + proj + ".bias"] = (EMBED_DIM,)
        shapes[base + ".self_attn_layer_norm.weight"] = (EMBED_DIM,)
        shapes[base + ".self_attn_layer_norm.bias"] = (EMBED_DIM,)
        shapes[base + ".fc1.weight"] = (FFN_DIM, EMBED_DIM)
        shapes[base + ".fc1.bias"] = (FFN_DIM,)
        shapes[base + ".fc2.weight"] = (EMBED_DIM, FFN_DIM)
        shapes[base + ".fc2.bias"] = (EMBED_DIM,)
        shapes[base + ".final_layer_norm.weight"] = (EMBED_DIM,)
        shapes[base + ".final_layer_norm.bias"] = (EMBED_DIM,)
    shapes[_SSL_PREFIX + ".encoder.layer_norm.weight"] = (EMBED_DIM,)
    shapes[_SSL_PREFIX + ".encoder.layer_norm.bias"] = (EMBED_DIM,)
    shapes["proj_fc.weight"] = (NUM_CLASSES, EMBED_DIM)
    shapes["proj_fc.bias"] = (NUM_CLASSES,)
    return shapes


def expected_inference_keys():
    return set(expected_key_shapes())


def read_safetensors_header(path):
    import json
    import struct

    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("not a safetensors file: header length missing")
        (length,) = struct.unpack("<Q", raw)
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("not a safetensors file: truncated JSON header")
        header = json.loads(payload.decode("utf-8"))
    shapes = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        shapes[str(key)] = tuple(int(item) for item in value["shape"])
    return shapes


def audit_checkpoint_keys(name_to_shape):
    expected = expected_key_shapes()
    have = {str(key): tuple(shape) for key, shape in dict(name_to_shape).items()}
    pretraining = set(PRETRAINING_ONLY_KEYS)
    expected_keys = set(expected)
    have_keys = set(have)
    missing = sorted(expected_keys - have_keys)
    unexpected = sorted(have_keys - expected_keys - pretraining)
    pretraining_found = sorted(pretraining & have_keys)
    pretraining_missing = sorted(pretraining - have_keys)
    shape_mismatches = {}
    for key in sorted(expected_keys & have_keys):
        if have[key] != expected[key]:
            shape_mismatches[key] = {"actual": have[key], "expected": expected[key]}
    complete = (
        not missing
        and not unexpected
        and not shape_mismatches
        and not pretraining_missing
    )
    return {
        "complete": complete,
        "n_header": len(have_keys),
        "n_inference_expected": len(expected_keys),
        "n_inference_found": len(expected_keys & have_keys),
        "n_pretraining_expected": len(pretraining),
        "n_pretraining_found": len(pretraining_found),
        "missing": missing,
        "unexpected": unexpected,
        "pretraining_found": pretraining_found,
        "pretraining_missing": pretraining_missing,
        "shape_mismatches": shape_mismatches,
    }


def format_audit_error(report, path):
    lines = ["checkpoint " + str(path) + " does not match the audited layout"]
    lines.append(
        "header tensors: "
        + str(report["n_header"])
        + ", inference expected/found: "
        + str(report["n_inference_expected"])
        + "/"
        + str(report["n_inference_found"])
        + ", pretraining expected/found: "
        + str(report["n_pretraining_expected"])
        + "/"
        + str(report["n_pretraining_found"])
    )
    for label in ("missing", "unexpected", "pretraining_missing"):
        items = report[label]
        if items:
            preview = ", ".join(items[:8])
            if len(items) > 8:
                preview += ", ... (+" + str(len(items) - 8) + " more)"
            lines.append(label + " (" + str(len(items)) + "): " + preview)
    mismatches = report["shape_mismatches"]
    if mismatches:
        first = sorted(mismatches)[:5]
        detail = ", ".join(
            key
            + " actual "
            + str(mismatches[key]["actual"])
            + " expected "
            + str(mismatches[key]["expected"])
            for key in first
        )
        lines.append("shape_mismatches (" + str(len(mismatches)) + "): " + detail)
    return "; ".join(lines)


def build_native_detector_class(torch):
    nn = torch.nn
    F = torch.nn.functional

    class TransposeLast(nn.Module):
        def forward(self, x):
            return x.transpose(-2, -1)

    class Fp32LayerNorm(nn.LayerNorm):
        def forward(self, input):
            weight = self.weight.float() if self.weight is not None else None
            bias = self.bias.float() if self.bias is not None else None
            output = F.layer_norm(
                input.float(), self.normalized_shape, weight, bias, self.eps
            )
            return output.type_as(input)

    class SamePad(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.remove = 1 if kernel_size % 2 == 0 else 0

        def forward(self, x):
            if self.remove > 0:
                x = x[:, :, : -self.remove]
            return x

    class WeightNormConv1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_g = nn.Parameter(torch.empty(1, 1, POS_CONV_KERNEL))
            self.weight_v = nn.Parameter(
                torch.empty(EMBED_DIM, EMBED_DIM // POS_CONV_GROUPS, POS_CONV_KERNEL)
            )
            self.bias = nn.Parameter(torch.empty(EMBED_DIM))

        def forward(self, x):
            norm = self.weight_v.norm(p=2, dim=(0, 1), keepdim=True)
            weight = self.weight_v * (self.weight_g / norm)
            return F.conv1d(
                x,
                weight,
                self.bias,
                stride=1,
                padding=POS_CONV_PAD,
                groups=POS_CONV_GROUPS,
            )

    class NativeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
            self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

        def forward(self, x, key_padding_mask=None):
            time_first = x.transpose(0, 1)
            batch, length, _ = time_first.shape
            query = (
                self.q_proj(time_first)
                .view(batch, length, NUM_HEADS, HEAD_DIM)
                .transpose(1, 2)
                * ATTN_SCALE
            )
            key = (
                self.k_proj(time_first)
                .view(batch, length, NUM_HEADS, HEAD_DIM)
                .transpose(1, 2)
            )
            value = (
                self.v_proj(time_first)
                .view(batch, length, NUM_HEADS, HEAD_DIM)
                .transpose(1, 2)
            )
            scores = torch.matmul(query, key.transpose(-2, -1))
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask[:, None, None, :], float("-inf")
                )
            probs = torch.softmax(scores.float(), dim=-1).type_as(scores)
            attended = torch.matmul(probs, value)
            attended = attended.transpose(1, 2).reshape(batch, length, EMBED_DIM)
            return self.out_proj(attended).transpose(0, 1), None

    class NativeEncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = NativeAttention()
            self.self_attn_layer_norm = nn.LayerNorm(EMBED_DIM, eps=LAYER_NORM_EPS)
            self.fc1 = nn.Linear(EMBED_DIM, FFN_DIM)
            self.fc2 = nn.Linear(FFN_DIM, EMBED_DIM)
            self.final_layer_norm = nn.LayerNorm(EMBED_DIM, eps=LAYER_NORM_EPS)
            self.dropout1 = nn.Dropout(0.0)
            self.dropout2 = nn.Dropout(0.0)
            self.dropout3 = nn.Dropout(0.0)

        def forward(self, x, self_attn_padding_mask=None):
            residual = x
            x = self.self_attn_layer_norm(x)
            x, _attn = self.self_attn(x, key_padding_mask=self_attn_padding_mask)
            x = self.dropout1(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            x = F.gelu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            return x

    class NativeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_conv = nn.Sequential(
                WeightNormConv1d(), SamePad(POS_CONV_KERNEL), nn.GELU()
            )
            self.layers = nn.ModuleList(
                [NativeEncoderLayer() for _ in range(NUM_LAYERS)]
            )
            self.layer_norm = nn.LayerNorm(EMBED_DIM, eps=LAYER_NORM_EPS)

        def forward(self, x, padding_mask=None):
            if padding_mask is not None:
                raise ValueError("padding masks are not supported by this frontend")
            conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + conv
            pad_length = (-x.size(1)) % REQUIRED_SEQ_LEN_MULTIPLE
            if pad_length:
                x = F.pad(x, (0, 0, 0, pad_length))
                padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
                padding_mask[:, -pad_length:] = True
            x = x.transpose(0, 1)
            for layer in self.layers:
                x = layer(x, self_attn_padding_mask=padding_mask)
            x = x.transpose(0, 1)
            if pad_length:
                x = x[:, :-pad_length]
            return self.layer_norm(x)

    def _conv_block(n_in, n_out, kernel, stride):
        return nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel, stride=stride, bias=True),
            nn.Dropout(p=0.0),
            nn.Sequential(
                TransposeLast(),
                Fp32LayerNorm(n_out, eps=LAYER_NORM_EPS),
                TransposeLast(),
            ),
            nn.GELU(),
        )

    class NativeFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            n_in = CONV_IN_CHANNELS
            for dim, kernel, stride in CONV_SPECS:
                blocks.append(_conv_block(n_in, dim, kernel, stride))
                n_in = dim
            self.conv_layers = nn.ModuleList(blocks)

        def forward(self, x):
            x = x.unsqueeze(1)
            for conv in self.conv_layers:
                x = conv(x)
            return x

    class NativeWav2Vec2(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = NativeFeatureExtractor()
            self.layer_norm = nn.LayerNorm(FEATURE_DIM, eps=LAYER_NORM_EPS)
            self.post_extract_proj = nn.Linear(FEATURE_DIM, EMBED_DIM)
            self.encoder = NativeEncoder()

        def forward(self, source, padding_mask=None, mask=False, features_only=False):
            if mask:
                raise ValueError("masking is a training path and is not supported")
            if not features_only:
                raise ValueError("only features_only inference is supported")
            if padding_mask is not None:
                raise ValueError("padding masks are not supported by this frontend")
            features = self.feature_extractor(source).transpose(1, 2)
            features = self.layer_norm(features)
            encoded = self.encoder(self.post_extract_proj(features))
            return {"x": encoded, "padding_mask": None}

    class NativeSSLModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = NativeWav2Vec2()

        def extract_feat(self, input_data):
            if input_data.ndim == 3:
                input_data = input_data[:, :, 0]
            return self.model(input_data, mask=False, features_only=True)["x"]

    class NativeDeepfakeDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.m_ssl = NativeSSLModel()
            self.adap_pool1d = nn.AdaptiveAvgPool1d(output_size=1)
            self.proj_fc = nn.Linear(EMBED_DIM, NUM_CLASSES)

        def forward(self, wav):
            emb = self.m_ssl.extract_feat(wav)
            emb = emb.transpose(1, 2)
            pooled = self.adap_pool1d(emb).squeeze(-1)
            return self.proj_fc(pooled)

    return NativeDeepfakeDetector
