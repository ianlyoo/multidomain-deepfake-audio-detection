#!/usr/bin/env python3
"""경진대회 테스트 데이터에 대한 5개 확률값을 생성한다."""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

# Load ORT before numerical/audio libraries. This avoids a native DLL collision
# on the local Windows smoke-test host; sessions still select providers below.
import onnxruntime as ort

# 추론에는 model 폴더에 포함된 로컬 파일만 사용한다.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
from scipy.ndimage import minimum_filter1d
from tqdm import tqdm


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"
ARTIFACTNET_DIR = MODEL_DIR / "artifactnet_v94"
SPECTRA_DIR = MODEL_DIR / "spectra_aasist3"
LOFCZ_DIR = MODEL_DIR / "lofcz_ai_music"
NII_DIR = MODEL_DIR / "nii_antideepfake"

DEFAULT_TEST_DIR = Path("data") / "test"
DEFAULT_SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = Path("output") / "submission.csv"

# 오디오 처리 설정
AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5
ARTIFACTNET_SAMPLE_RATE = 44_100
ARTIFACTNET_CHUNK_SAMPLES = 4 * ARTIFACTNET_SAMPLE_RATE
ARTIFACTNET_MUSIC_BLEND_WEIGHT = 0.25
ARTIFACTNET_FILE_MUSIC_BLEND_WEIGHT = 0.50
SPECTRA_SEGMENT_SAMPLES = 64_600
SPECTRA_PREEMPHASIS = 0.97
SPECTRA_VOICE_BLEND_WEIGHT = 0.625
NII_VOICE_BLEND_WEIGHT = 0.20
NII_MAX_AUDIO_SAMPLES = 30 * AUDIO_SAMPLE_RATE
NII_MIN_AUDIO_SAMPLES = 400
NII_LOGIT_EPS = 1e-6
LOFCZ_N_FFT = 8192
LOFCZ_MAX_SAMPLES = 300 * AUDIO_SAMPLE_RATE
LOFCZ_FREQ_MIN = 1000
LOFCZ_FREQ_MAX = 8000
LOFCZ_HULL_AREA = 10
LOFCZ_MIN_DB = -45
LOFCZ_MAX_DB = 5
LOFCZ_MUSIC_BLEND_WEIGHT = 0.75
# sub68: fine-tuned lofcz fakeprint MUSIC classifier (raw mixture, no new neural forward).
FT_MUSIC_PATH = MODEL_DIR / "ft_music_fakeprint_v2.json"
FT_MUSIC_SHA256 = "e9391e8097cefef3e2d0252eae1cacb467458a6d08b0b409b232d402a1f743fa"
FT_MUSIC_MAX_BYTES = 2000000
FT_MUSIC_BLEND_WEIGHT = 0.5
FT_MUSIC_FLOOR = "new"
# sub97b: segment-only FT MUSIC head (window-bag retrain, recentred bias).
FT_MUSIC_SEG_PATH = MODEL_DIR / "ft_music_seg_sub97b.json"
FT_MUSIC_SEG_SHA256 = "7fcc13317c6b776d74b5af31278254027654510338dd4c5a4999f205033a9e10"

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}


# -----------------------------------------------------------------------------
# 1. 입력 파일 및 제출 양식 확인
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the zero-shot audio deepfake baseline."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def select_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(device_name)


def find_audio_files(test_dir):
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    audio_files = []
    for path in test_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(path)
    audio_files.sort(key=lambda path: path.stem)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")

    audio_ids = [path.stem for path in audio_files]
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("Audio IDs must be unique")
    return audio_files


def read_sample_submission(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames
        rows = list(reader)

    if column_names is None or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")

    required_columns = ["ID"] + PREDICTION_COLUMNS
    missing_columns = [name for name in required_columns if name not in column_names]
    if missing_columns:
        raise ValueError(f"Sample submission is missing columns: {missing_columns}")

    seen_ids = set()
    for row in rows:
        audio_id = str(row["ID"]).strip()
        if not audio_id:
            raise ValueError("Sample submission contains an empty ID")
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate ID in sample submission: {audio_id}")
        seen_ids.add(audio_id)
        row["ID"] = audio_id

    return column_names, rows


def order_audio_files(audio_files, submission_rows):
    audio_by_id = {path.stem: path for path in audio_files}
    submission_ids = [row["ID"] for row in submission_rows]

    missing_ids = [audio_id for audio_id in submission_ids if audio_id not in audio_by_id]
    extra_ids = [audio_id for audio_id in audio_by_id if audio_id not in submission_ids]
    if missing_ids or extra_ids:
        raise ValueError(
            "Test audio and sample submission IDs do not match. "
            f"Missing: {missing_ids[:5]}, Extra: {extra_ids[:5]}"
        )

    return [audio_by_id[audio_id] for audio_id in submission_ids]


def load_audio(audio_path):
    audio, _ = librosa.load(
        audio_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {audio_path}")
    return audio


# -----------------------------------------------------------------------------
# 2. 오디오 구간 분할
# -----------------------------------------------------------------------------

def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]

    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:SEGMENT_SAMPLES].astype(np.float32)

    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# 3. PANNs를 이용한 음성·음악 존재 여부 추론
# -----------------------------------------------------------------------------

def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns_model(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging, labels

    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )

    config_path = PANNS_DIR / "component_labels.json"
    label_groups = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[label] for label in label_groups["voice"]]
    music_indices = [label_to_index[label] for label in label_groups["music"]]
    return model, voice_indices, music_indices


def make_panns_segments(audio):
    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment = librosa.resample(
            segment,
            orig_sr=AUDIO_SAMPLE_RATE,
            target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(segment.astype(np.float32))
    return np.stack(segments)


def predict_presence(model, voice_indices, music_indices, audio, segment_music=None):
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    voice_probability = float(predictions[:, voice_indices].max())
    music_probability = float(predictions[:, music_indices].max())
    if segment_music is not None:
        # Capture the existing PANNs forward; an optional diagnostic must not
        # alter the original file-level VOICE/MUSIC presence probabilities.
        try:
            starts = np.asarray(get_segment_starts(audio.size), dtype=np.int64)
            scores = np.asarray(predictions[:, music_indices].max(axis=1), dtype=np.float64)
            if scores.shape != starts.shape or not np.isfinite(scores).all():
                raise ValueError("Invalid PANNs segment music scores")
            segment_music["starts"] = starts
            segment_music["scores"] = scores
        except (Exception, SystemExit):
            segment_music.clear()
        # sub90: speech presence for segment VOICE; failure only drops this key.
        try:
            voice_scores = np.asarray(predictions[:, voice_indices].max(axis=1), dtype=np.float64)
            if "starts" not in segment_music or voice_scores.shape != segment_music["starts"].shape:
                raise ValueError("Invalid PANNs segment voice scores")
            segment_music["voice_scores"] = voice_scores
        except (Exception, SystemExit):
            segment_music.pop("voice_scores", None)
    return voice_probability, music_probability


def predict_presence_for_all_files(audio_files, device, music_segments=None):
    model, voice_indices, music_indices = load_panns_model(device)
    presence_scores = {}

    for audio_path in tqdm(audio_files, desc="Presence"):
        try:
            audio = load_audio(audio_path)
            segment_music = {} if music_segments is not None else None
            presence_scores[audio_path.stem] = predict_presence(
                model, voice_indices, music_indices, audio, segment_music
            )
            if segment_music:
                try:
                    music_segments[audio_path.stem] = segment_music
                except (Exception, SystemExit):
                    pass
        except Exception as error:
            print(
                f"WARNING: presence inference failed for {audio_path.name}: {error}",
                file=sys.stderr,
            )
            presence_scores[audio_path.stem] = (0.5, 0.5)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return presence_scores


# -----------------------------------------------------------------------------
# 4. HTDemucs를 이용한 음성·음악 분리
# -----------------------------------------------------------------------------

def load_htdemucs_model():
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        # PyTorch 2.6부터 바뀐 기본값에 맞춰 기존 체크포인트를 불러온다.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def separate_voice_and_music(audio_path, model, device):
    waveform = load_track(
        audio_path, model.audio_channels, model.samplerate
    ).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            normalized_waveform[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)

    music_sources = []
    for index, source_name in enumerate(model.sources):
        if source_name != "vocals":
            music_sources.append(sources[index])
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )


# -----------------------------------------------------------------------------
# 5. DF-Arena 1B를 이용한 성분별 Fake 추론
# -----------------------------------------------------------------------------

def load_df_arena_model(device):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous_directory = Path.cwd()
    os.chdir(DF_ARENA_DIR)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_DIR),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    finally:
        os.chdir(previous_directory)

    model = model.to(device).eval()
    fake_label_index = int(model.config.label2id["spoof"])
    return model, fake_label_index


def load_artifactnet_session():
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(
        str(ARTIFACTNET_DIR / "artifactnet_v94_full.onnx"),
        sess_options=options,
        providers=providers,
    )


def load_lofcz_session():
    """Load the pinned tiny fakeprint classifier on CPU."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(LOFCZ_DIR / "ai_music_detector.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def load_spectra_session(device):
    """Load Spectra on CUDA when available, with a measured-safe CPU fallback."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    requested_cuda = device.type == "cuda"
    if requested_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(
        str(SPECTRA_DIR / "spectra-aasist3.onnx"),
        sess_options=options,
        providers=providers,
    )
    if requested_cuda and session.get_providers()[0] != "CUDAExecutionProvider":
        print(
            "WARNING: Spectra is using its CPU fallback; expected runtime "
            "remains below the evaluation limit",
            file=sys.stderr,
        )
    return session


def load_nii_model(device):
    """Load the pinned NII AntiDeepfake model through its fairseq-free frontend."""
    if str(NII_DIR) not in sys.path:
        sys.path.insert(0, str(NII_DIR))
    from native_frontend import (
        PRETRAINING_ONLY_KEYS,
        audit_checkpoint_keys,
        build_native_detector_class,
        format_audit_error,
        read_safetensors_header,
    )
    from safetensors.torch import load_file

    weight_path = NII_DIR / "model.safetensors"
    coverage = audit_checkpoint_keys(read_safetensors_header(weight_path))
    if not coverage["complete"]:
        raise RuntimeError(format_audit_error(coverage, weight_path))
    model = build_native_detector_class(torch)()
    state_dict = load_file(str(weight_path), device="cpu")
    for key in PRETRAINING_ONLY_KEYS:
        state_dict.pop(key, None)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def make_artifactnet_chunks(audio):
    waveform = librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=AUDIO_SAMPLE_RATE,
        target_sr=ARTIFACTNET_SAMPLE_RATE,
        res_type="soxr_hq",
    ).astype(np.float32, copy=False)
    if waveform.size < ARTIFACTNET_CHUNK_SAMPLES:
        waveform = np.pad(
            waveform,
            (0, ARTIFACTNET_CHUNK_SAMPLES - waveform.size),
            mode="constant",
        )
    last_start = max(0, waveform.size - ARTIFACTNET_CHUNK_SAMPLES)
    starts = sorted({0, last_start // 2, last_start})
    return [
        waveform[start : start + ARTIFACTNET_CHUNK_SAMPLES][None]
        for start in starts
    ]


def predict_artifactnet_fake(session, audio):
    scores = []
    for chunk in make_artifactnet_chunks(audio):
        output = session.run(None, {"audio": chunk.astype(np.float32, copy=False)})
        score = float(np.asarray(output[0]).reshape(-1)[0])
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        raise ValueError("ArtifactNet returned no finite scores")
    return float(np.mean(scores))


def blend_music_fake(df_arena_fake, artifactnet_fake):
    """Anchor sub09 to the public winner while testing residual diversity."""
    weight = ARTIFACTNET_MUSIC_BLEND_WEIGHT
    return float((1.0 - weight) * df_arena_fake + weight * artifactnet_fake)


def blend_file_music_fake(df_arena_fake, artifactnet_fake):
    """Use the officially best sub11 ArtifactNet weight only inside FILE."""
    weight = ARTIFACTNET_FILE_MUSIC_BLEND_WEIGHT
    return float((1.0 - weight) * df_arena_fake + weight * artifactnet_fake)


def make_lofcz_fakeprint(audio):
    """Reproduce lofcz's pinned 16 kHz, 1--8 kHz fakeprint."""
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("lofcz received invalid audio")
    waveform = waveform[:LOFCZ_MAX_SAMPLES]
    tensor = torch.from_numpy(waveform).unsqueeze(0)
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=LOFCZ_N_FFT,
        power=2,
        normalized=False,
    )
    with torch.inference_mode():
        spectrum = spectrogram(tensor)
    spectrum_db = 10 * torch.log10(torch.clamp(spectrum, min=1e-10, max=1e6))
    mean_spectrum = spectrum_db.mean(dim=(0, 2)).cpu().numpy()
    frequencies = np.linspace(
        0.0,
        AUDIO_SAMPLE_RATE / 2.0,
        num=(LOFCZ_N_FFT // 2) + 1,
    )
    mask = (frequencies >= LOFCZ_FREQ_MIN) & (frequencies <= LOFCZ_FREQ_MAX)
    selected = mean_spectrum[mask]
    if selected.shape != (3585,):
        raise ValueError(f"Unexpected lofcz fakeprint shape: {selected.shape}")
    hull = minimum_filter1d(selected, size=LOFCZ_HULL_AREA, mode="nearest")
    hull = np.clip(hull, LOFCZ_MIN_DB, None)
    residue = np.clip(selected - hull, 0.0, LOFCZ_MAX_DB)
    return (residue / (float(np.max(residue)) + 1e-6)).astype(np.float32)


def load_lofcz_audio(audio_path):
    """Match the pinned lofcz torchaudio decode/resample path exactly."""
    waveform, source_rate = torchaudio.load(str(audio_path))
    waveform = waveform.float()
    if int(source_rate) != AUDIO_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(
            int(source_rate), AUDIO_SAMPLE_RATE
        )(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.reshape(-1)
    audio = waveform.cpu().numpy().astype(np.float32, copy=False)
    audio = audio[:LOFCZ_MAX_SAMPLES]
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid lofcz audio: {audio_path}")
    return audio


def predict_lofcz_fake(session, audio):
    fakeprint = make_lofcz_fakeprint(audio)
    output = session.run(None, {"fakeprint": fakeprint.reshape(1, -1)})
    score = float(np.asarray(output[0]).reshape(-1)[0])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Unexpected lofcz score: {score}")
    return score


def blend_lofcz_music_fake(current_music_fake, lofcz_fake):
    """sub18: raise the confirmed fakeprint share inside MUSIC only."""
    weight = LOFCZ_MUSIC_BLEND_WEIGHT
    return float((1.0 - weight) * current_music_fake + weight * lofcz_fake)


def make_spectra_input(audio):
    """Apply source-matched preemphasis then return the first fixed window."""
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Spectra received invalid audio")
    emphasized = np.empty_like(waveform)
    emphasized[0] = waveform[0]
    emphasized[1:] = waveform[1:] - SPECTRA_PREEMPHASIS * waveform[:-1]
    if emphasized.size < SPECTRA_SEGMENT_SAMPLES:
        repeats = SPECTRA_SEGMENT_SAMPLES // emphasized.size + 1
        emphasized = np.tile(emphasized, repeats)
    return emphasized[:SPECTRA_SEGMENT_SAMPLES][None].astype(
        np.float32, copy=False
    )


def predict_spectra_fake(session, audio):
    """Map Spectra's higher-is-bona-fide logit to a higher-is-fake score."""
    output = session.run(None, {"wav": make_spectra_input(audio)})
    logits = np.asarray(output[0], dtype=np.float64)
    if logits.shape != (1, 2) or not np.isfinite(logits).all():
        raise ValueError(f"Unexpected Spectra logits: shape={logits.shape}")
    bona_fide_logit = float(logits[0, 1])
    # This fixed transform exactly reverses the benchmark ranking while
    # producing a legal per-file value in [0, 1].
    if bona_fide_logit >= 0.0:
        exp_negative = np.exp(-bona_fide_logit)
        return float(exp_negative / (1.0 + exp_negative))
    exp_positive = np.exp(bona_fide_logit)
    return float(1.0 / (1.0 + exp_positive))


# BEGIN SUB37 VOICE TEMPORAL HELPERS
def spectra_temporal_starts(n):
    """Sorted unique starts for the frozen 64600-sample temporal rule."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("Spectra temporal length must be a positive integer")
    last = max(0, int(n) - SPECTRA_SEGMENT_SAMPLES)
    return sorted({0, last // 2, last})


def _sub37_valid_probability(value):
    """Accept only real numeric scalar probabilities, including exact endpoints."""
    return bool(np.isscalar(value) and np.asarray(value).dtype.kind in "fiu"
                and np.isfinite(value) and 0.0 <= value <= 1.0)


def _sub37_temporal_audio(audio):
    """Validate the actual 1D stem without flattening or dropping bad samples."""
    waveform = np.asarray(audio)
    if (waveform.ndim != 1 or waveform.size == 0 or waveform.dtype.kind not in "fiu"
            or not np.isfinite(waveform).all()):
        raise ValueError("Spectra temporal received invalid 1D audio")
    with np.errstate(over="raise", invalid="raise"):
        waveform = waveform.astype(np.float32, copy=False)
    if not np.isfinite(waveform).all():
        raise ValueError("Spectra temporal received non-finite float32 audio")
    return waveform


def predict_spectra_temporal(session, audio, first_score, evidence=None):
    """Reuse S0 and average all unique raw-crop probabilities; never a partial mean.

    Calls the unchanged predict_spectra_fake for each added window, allowing
    research instrumentation at that boundary. Errors propagate to the caller.
    For a valid short stem, return first_score itself without new arithmetic.
    """
    if not _sub37_valid_probability(first_score):
        raise ValueError("Spectra temporal received invalid first score")
    waveform = _sub37_temporal_audio(audio)
    if waveform.size <= SPECTRA_SEGMENT_SAMPLES:
        try:
            _sub39_capture(evidence, spectra_scores=[float(first_score)], spectra_starts=[0])
        except (Exception, SystemExit):
            if type(evidence) is dict:
                evidence.clear()
        return first_score
    scores = [float(first_score)]
    for start in spectra_temporal_starts(waveform.size)[1:]:
        score = predict_spectra_fake(
            session, waveform[start : start + SPECTRA_SEGMENT_SAMPLES]
        )
        if not _sub37_valid_probability(score):
            raise ValueError("Spectra temporal received invalid added score")
        scores.append(float(score))
    mean = float(np.mean(scores))
    if not _sub37_valid_probability(mean):
        raise ValueError("Spectra temporal produced invalid mean")
    try:
        _sub39_capture(evidence, spectra_scores=list(scores), spectra_starts=spectra_temporal_starts(waveform.size))
    except (Exception, SystemExit):
        if type(evidence) is dict:
            evidence.clear()
    return mean


def _sub37_final_voice(session, audio, current_voice, df_mean, s0, nii, evidence=None):
    """Commit only a complete healthy temporal VOICE; retain the entire old value."""
    try:
        if not all(_sub37_valid_probability(value) for value in (df_mean, s0, nii)):
            return current_voice
        waveform = _sub37_temporal_audio(audio)
        if waveform.size <= SPECTRA_SEGMENT_SAMPLES:
            try:
                _sub39_capture(evidence, spectra_scores=[float(s0)], spectra_starts=[0], temporal_ok=True)
            except (Exception, SystemExit):
                if type(evidence) is dict:
                    evidence.clear()
            return current_voice
        if evidence is None:
            temporal = predict_spectra_temporal(session, waveform, s0)
        else:
            temporal = predict_spectra_temporal(session, waveform, s0, evidence=evidence)
        if not _sub37_valid_probability(temporal):
            raise ValueError("Spectra temporal produced invalid mean")
        blended = blend_voice_fake(df_mean, temporal)
        if not _sub37_valid_probability(blended):
            raise ValueError("Spectra temporal produced invalid VOICE blend")
        candidate_voice = blend_nii_voice_fake(blended, nii)
        if not _sub37_valid_probability(candidate_voice):
            raise ValueError("Spectra temporal produced invalid NII blend")
        try:
            _sub39_capture(evidence, temporal_ok=True)
        except (Exception, SystemExit):
            if type(evidence) is dict:
                evidence.clear()
        return candidate_voice
    except (Exception, SystemExit) as temporal_error:
        print(f"WARNING: Spectra temporal VOICE failed: {temporal_error}; "
              "retaining sub33 final VOICE", file=sys.stderr)
        return current_voice
# END SUB37 VOICE TEMPORAL HELPERS


def blend_voice_fake(df_arena_fake, spectra_fake):
    """Probe sub15 near the fitted VOICE response vertex."""
    weight = SPECTRA_VOICE_BLEND_WEIGHT
    return float((1.0 - weight) * df_arena_fake + weight * spectra_fake)


def predict_nii_fake(model, audio, device):
    """Score at most the first 30 s of a 16 kHz Demucs vocal stem."""
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size < NII_MIN_AUDIO_SAMPLES or not np.isfinite(waveform).all():
        raise ValueError("NII received invalid or too-short voice audio")
    waveform = waveform[:NII_MAX_AUDIO_SAMPLES]
    values = waveform.astype(np.float64, copy=False)
    waveform = ((values - values.mean()) / np.sqrt(values.var() + 1e-5)).astype(
        np.float32
    )
    tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.softmax(logits.float(), dim=-1)[0, 0]
    score = float(probability)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Unexpected NII score: {score}")
    return score


def blend_nii_voice_fake(current_fake, nii_fake):
    """sub19: fixed per-file 80/20 logit blend; no cross-file statistics."""
    current = float(np.clip(current_fake, NII_LOGIT_EPS, 1.0 - NII_LOGIT_EPS))
    nii = float(np.clip(nii_fake, NII_LOGIT_EPS, 1.0 - NII_LOGIT_EPS))
    current_logit = np.log(current / (1.0 - current))
    nii_logit = np.log(nii / (1.0 - nii))
    combined = (1.0 - NII_VOICE_BLEND_WEIGHT) * current_logit
    combined += NII_VOICE_BLEND_WEIGHT * nii_logit
    if combined >= 0.0:
        exp_negative = np.exp(-combined)
        return float(1.0 / (1.0 + exp_negative))
    exp_positive = np.exp(combined)
    return float(exp_positive / (1.0 + exp_positive))


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def predict_fake(model, fake_label_index, audio, device, pooling="max"):
    if pooling not in ("mean", "max", "both"):
        raise ValueError(f"Unsupported pooling: {pooling}")
    if calculate_rms(audio) < SILENCE_RMS:
        return (0.0, 0.0) if pooling == "both" else 0.0

    segment_scores = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment_tensor = torch.from_numpy(segment).to(device)

        with torch.inference_mode():
            logits = model(input_values=segment_tensor)["logits"]
            probabilities = torch.softmax(logits.float(), dim=-1)
        segment_scores.append(float(probabilities[0, fake_label_index]))

    if pooling == "mean":
        return float(np.mean(segment_scores))
    if pooling == "max":
        return max(segment_scores)
    if pooling == "both":
        return max(segment_scores), float(np.mean(segment_scores))
    raise ValueError(f"Unsupported pooling: {pooling}")


# -----------------------------------------------------------------------------
# 6. 파일 단위 점수 계산 및 제출 파일 저장
# -----------------------------------------------------------------------------

def combine_file_fake_score(voice_fake, music_fake, voice_present, music_present):
    voice_score = voice_present * voice_fake
    music_score = music_present * music_fake
    return max(voice_score, music_score)

# ----------------------------------------------------------------------------
# sub21: frozen learned FILE fusion (FILE head only; other heads unchanged).
# The 17-feature standardized logistic backend was fitted on external
# development audio only (384 files / 24 source groups, seed 20260905, C=3)
# and is frozen. Inference uses per-file component scores plus these immutable
# training statistics only: no batch moments, no cross-file adaptation, and
# no evaluation-data learning. Any missing component or backend problem falls
# back to the current FILE formula for that file alone.
# ----------------------------------------------------------------------------
FILE_BACKEND_PATH = MODEL_DIR / "file_backend_joint_v1.json"
FILE_BACKEND_EPSILON = 1e-6
_FILE_BACKEND_CACHE = None


def _load_file_backend():
    """Load the frozen sub25 tree export once, validating before caching."""
    global _FILE_BACKEND_CACHE
    if _FILE_BACKEND_CACHE is not None:
        return _FILE_BACKEND_CACHE
    try:
        with FILE_BACKEND_PATH.open("r", encoding="utf-8") as handle:
            model = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("cannot load FILE tree backend") from error
    if not isinstance(model, dict):
        raise ValueError("FILE backend must be an object")
    expected_features = (
        "logit:df_voice_mean", "logit:df_music_mean", "logit:df_music_max",
        "logit:spectra", "logit:nii", "logit:artifactnet", "logit:lofcz",
        "logit:current_voice", "logit:current_music", "logit:voice_present",
        "logit:music_present", "logit:current_file", "logit:file_music",
        "logit:voice_present*df_voice_mean",
        "logit:voice_present*current_voice",
        "logit:music_present*file_music",
        "logit:music_present*current_music",
    )
    expected = {
        "version": 2, "model_type": "gradient_boosted_trees",
        "input_dtype": "float32", "n_estimators": 64, "max_depth": 2,
        "epsilon": FILE_BACKEND_EPSILON, "learning_rate": 0.05,
    }
    for key, value in expected.items():
        actual = model.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"FILE backend {key} mismatch")
    if model.get("feature_names") != list(expected_features):
        raise ValueError("FILE backend feature contract mismatch")

    def finite_number(value):
        return (type(value) in (int, float)
                and bool(np.isfinite(float(value))))

    try:
        initial = model.get("initial_log_odds")
        if not finite_number(initial):
            raise ValueError("FILE backend invalid initial log odds")
        trees = model.get("trees")
        if not isinstance(trees, list) or len(trees) != 64:
            raise ValueError("FILE backend requires 64 trees")
        validated = []
        for tree in trees:
            if not isinstance(tree, dict):
                raise ValueError("FILE backend tree must be an object")
            arrays = [tree.get(key) for key in (
                "children_left", "children_right", "feature", "threshold", "value"
            )]
            if not all(isinstance(array, list) for array in arrays):
                raise ValueError("FILE backend tree arrays required")
            left, right, feature, threshold, values = arrays
            size = len(left)
            if not 1 <= size <= 7 or any(len(array) != size for array in arrays):
                raise ValueError("FILE backend invalid tree array lengths")
            if not all(type(value) is int for array in arrays[:3] for value in array):
                raise ValueError("FILE backend indices must be integers")
            if not all(finite_number(value) for array in arrays[3:] for value in array):
                raise ValueError("FILE backend non-finite tree parameters")
            for node in range(size):
                if left[node] == -1 and right[node] == -1:
                    if feature[node] != -2:
                        raise ValueError("FILE backend invalid leaf feature sentinel")
                elif (not 0 <= left[node] < size or not 0 <= right[node] < size
                      or not 0 <= feature[node] < 17):
                    raise ValueError("FILE backend invalid child or feature index")
            seen = set()
            pending = [(0, 0)]
            while pending:
                node, depth = pending.pop()
                if node in seen or depth > 2:
                    raise ValueError("FILE backend repeated node, cycle or depth > 2")
                seen.add(node)
                if left[node] != -1:
                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))
            if len(seen) != size:
                raise ValueError("FILE backend disconnected nodes")
            validated.append(tuple(tuple(array) for array in arrays))
    except (TypeError, OverflowError) as error:
        raise ValueError("FILE backend invalid numeric parameters") from error
    _FILE_BACKEND_CACHE = (float(initial), 0.05, tuple(validated))
    return _FILE_BACKEND_CACHE


def _predict_file_tree_log_odds(features, backend):
    """Evaluate float32 inputs against the export's float64 thresholds."""
    initial, rate, trees = backend
    split_input = np.asarray(features, dtype=np.float32)
    if split_input.shape != (17,) or not bool(np.isfinite(split_input).all()):
        raise ValueError("FILE backend requires 17 finite features")
    total = initial
    for left, right, feature, threshold, values in trees:
        node = 0
        while left[node] != -1:
            # NumPy 2 scalar comparison can round the threshold to float32.
            # Promote the already-rounded input to preserve midpoint decisions.
            if float(split_input[feature[node]]) <= float(threshold[node]):
                node = left[node]
            else:
                node = right[node]
        total += rate * values[node]
    if not np.isfinite(total):
        raise ValueError("FILE backend produced non-finite log odds")
    return float(total)



def _sigmoid(value):
    """Stable logistic sigmoid returning a float in (0, 1)."""
    value = float(value)
    if value >= 0.0:
        exp_negative = np.exp(-value)
        return float(1.0 / (1.0 + exp_negative))
    exp_positive = np.exp(value)
    return float(exp_positive / (1.0 + exp_positive))


def predict_learned_file_fake(df_voice_mean, df_music_mean, df_music_max,
                              spectra, nii, artifactnet, lofcz,
                              current_voice, current_music,
                              voice_present, music_present,
                              current_file, file_music):
    """Score one file with the frozen learned backend.

    Raises on any missing, non-finite, or out-of-range component so the
    caller retains the current FILE score for that file.
    """
    values = (df_voice_mean, df_music_mean, df_music_max, spectra, nii,
              artifactnet, lofcz, current_voice, current_music,
              voice_present, music_present, current_file, file_music)
    if any(value is None for value in values):
        raise ValueError("learned FILE fusion is missing a component score")
    try:
        raw = np.asarray([float(value) for value in values], dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("learned FILE fusion received an invalid score") from error
    if not bool(np.isfinite(raw).all()) or bool(((raw < 0.0) | (raw > 1.0)).any()):
        raise ValueError("learned FILE fusion received an invalid score")
    interactions = np.asarray((
        raw[9] * raw[0], raw[9] * raw[7], raw[10] * raw[12], raw[10] * raw[8],
    ), dtype=np.float64)
    bounded = np.clip(
        np.concatenate((raw, interactions)),
        FILE_BACKEND_EPSILON, 1.0 - FILE_BACKEND_EPSILON,
    )
    features = np.log(bounded) - np.log1p(-bounded)
    score = _sigmoid(_predict_file_tree_log_odds(features, _load_file_backend()))
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("learned FILE fusion produced an invalid score")
    return float(score)


# BEGIN SUB38 ELIYA HELPERS
ELIYA_DIR = MODEL_DIR / "eliya_forensics"
ELIYA_VENDOR_DIR = MODEL_DIR / "_sub38_eliya"
ELIYA_VENDOR_SHA256 = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "eliya_forensics.py": "279ce521339495ca6cde144257ee1463bf0ad26c4ff59d56b7d32d47c5d31179",
    "eliya_runtime.py": "352cb7c8dd71d63c5a7708618f1ff1bfc346dee07b8fde3f5e6a557e957733e3",
}


def load_eliya_model(device):
    """Load authenticated vendored bytes with relative imports and no project install.

    Package registration is temporary and restored even after SystemExit. The
    returned model and predictor retain their module references. The runtime
    owns checkpoint authentication, CPU/meta audit, and explicit device transfer.
    Errors propagate to the independent optional-model startup handler.
    """
    import hashlib
    import importlib.util

    directory = ELIYA_VENDOR_DIR.resolve()
    sources = {}
    for filename, expected in ELIYA_VENDOR_SHA256.items():
        path = directory / filename
        if path.stat().st_size > 1_000_000:
            raise ValueError("Eliya vendor source exceeds size limit: " + filename)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Eliya vendor source SHA-256 mismatch: " + filename)
        sources[filename] = raw
    package_name = "_sub38_eliya"
    previous = {name: module for name, module in sys.modules.copy().items()
                if name == package_name or name.startswith(package_name + ".")}
    for name in previous:
        del sys.modules[name]
    try:
        package = None
        runtime = None
        for filename in ("__init__.py", "eliya_forensics.py", "eliya_runtime.py"):
            name = package_name if filename == "__init__.py" else package_name + "." + filename[:-3]
            path = directory / filename
            options = {"submodule_search_locations": [str(directory)]} if package is None else {}
            spec = importlib.util.spec_from_file_location(name, path, **options)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot initialize Eliya vendor module: " + name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            # Execute the very bytes authenticated above; do not consume a pyc.
            exec(compile(sources[filename], str(path), "exec"), module.__dict__)
            if package is None:
                package = module
            else:
                setattr(package, filename[:-3], module)
            if filename == "eliya_runtime.py":
                runtime = module
        if not callable(getattr(runtime, "load_eliya", None)) or not callable(
                getattr(runtime, "predict_eliya_fake", None)):
            raise ValueError("Eliya vendor runtime interface is invalid")
        model, receipt = runtime.load_eliya(ELIYA_DIR, device)
        if model is None or not isinstance(receipt, dict):
            raise ValueError("Eliya vendor loader returned invalid model/receipt")
        return {"model": model, "predict": runtime.predict_eliya_fake, "receipt": receipt}
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]
        sys.modules.update(previous)


def predict_eliya_fake(model, audio, device):
    """Delegate one raw float32 stem to the exact vendored runtime, without retry."""
    return model["predict"](model["model"], audio, device)


def _sub38_final_voice(model, audio, current_voice, df_mean, s0, nii, device, evidence=None):
    """Blend only a healthy independent score; retain entire sub37 on any failure."""
    try:
        if model is None:
            return current_voice
        if not all(_sub37_valid_probability(value) for value in (df_mean, s0, nii, current_voice)):
            return current_voice
        if (not isinstance(audio, np.ndarray) or audio.dtype != np.float32
                or audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all()):
            raise ValueError("Eliya requires finite nonempty 1D raw float32 audio")
        if not isinstance(model, dict) or model.get("model") is None or not callable(model.get("predict")):
            raise ValueError("Eliya model state is unavailable or invalid")
        score = predict_eliya_fake(model, audio, device)
        if not _sub37_valid_probability(score):
            raise ValueError("Eliya returned invalid scalar probability")
        candidate_voice = 0.75 * float(current_voice) + 0.25 * float(score)
        if not _sub37_valid_probability(candidate_voice):
            raise ValueError("Eliya produced invalid final VOICE blend")
        try:
            _sub39_capture(evidence, eliya_fake=score, eliya_ok=True)
        except (Exception, SystemExit):
            if type(evidence) is dict:
                evidence.clear()
        return candidate_voice
    except (Exception, SystemExit) as eliya_error:
        print(f"WARNING: Eliya VOICE failed: {eliya_error}; retaining sub37 final VOICE",
              file=sys.stderr)
        return current_voice
# END SUB38 ELIYA HELPERS


# BEGIN SUB39 EVIDENCE FILE HELPERS
EVIDENCE_FILE_BACKEND_PATH = MODEL_DIR / "file_backend_evidence24_v1.json"
EVIDENCE_FILE_BACKEND_SHA256 = "8a9dba6701417ad018457e70f538c9745b7c9095b1dc3130f91a1a98e110a9c5"
EVIDENCE_FILE_PROTOCOL_SHA256 = "fbdc4bc4532a894e122ce46a04b39f1d2be0d0ebc1f0d89b142e7af977387f63"
EVIDENCE_FILE_SOURCE_SHA256 = "704f201d6bfbccb4baa9d47608aacb44fa79d8e72152a1bc626337e166eb5699"
EVIDENCE_FILE_COMPILED_SOURCE = 'import numpy as np\n\nSCORE_FIELDS = (\n    "df_voice_mean", "df_music_mean", "df_music_max", "spectra", "nii",\n    "artifactnet", "lofcz", "current_voice", "current_music", "voice_present",\n    "music_present", "current_file", "file_music",\n)\n\nINTERACTIONS = (("voice_present", "df_voice_mean"), ("voice_present", "current_voice"),\n                ("music_present", "file_music"), ("music_present", "current_music"))\n\nFEATURES = tuple("logit:" + name for name in SCORE_FIELDS) + tuple(\n    "logit:" + left + "*" + right for left, right in INTERACTIONS)\n\nEPSILON = 1e-6\n\ndef features(rows):\n    if not rows:\n        raise ValueError("empty feature input")\n    values = np.asarray([[float(row[name]) for name in SCORE_FIELDS] for row in rows])\n    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():\n        raise ValueError("component probabilities must be finite and in [0, 1]")\n    indices = {name: index for index, name in enumerate(SCORE_FIELDS)}\n    products = np.column_stack([values[:, indices[a]] * values[:, indices[b]] for a, b in INTERACTIONS])\n    bounded = np.clip(np.column_stack((values, products)), EPSILON, 1 - EPSILON)\n    return np.log(bounded) - np.log1p(-bounded)\n\nBASE_FEATURES = FEATURES\nbase_features = features\n\n"""Frozen FILE24 features and strict version-eight JSON inference.\n\nErrors are observable as ValueError. Callers retain the entire incumbent FILE\non failure; this module never substitutes neutral or partial evidence.\n"""\n\nimport numpy as np\n\nfrom scipy.special import expit\n\nVERSION = 8\n\nMODEL_TYPE = \'gradient_boosted_trees_evidence24\'\n\nPROTOCOL_SHA = \'fbdc4bc4532a894e122ce46a04b39f1d2be0d0ebc1f0d89b142e7af977387f63\'\n\nSOURCE38_SHA = \'e2f1c5d32fec53d4d1580d2aee5430de46545f519bda423b2422e0d2d1a95b3d\'\n\nPREDECESSOR_SHA = \'cf05f8ed590efb4d3baeb821b7a9445aef2b6445554d54d66b26444d7f9787a9\'\n\nRECIPE = \'fixed64-depth2-lr0.05-leaf24-evidence24\'\n\nFEATURES = (*BASE_FEATURES,\n            \'logit:eliya_fake\', \'logit:voice_present*eliya_fake\',\n            \'logit:spectra_temporal_mean\', \'logit:voice_present*spectra_temporal_mean\',\n            \'logit:spectra_temporal_max\', \'logit:voice_present*spectra_temporal_max\',\n            \'prob:spectra_temporal_range\')\n\ndef additional_features(voice_present, eliya_fake, spectra_scores):\n    """Return seven float64 values, preserving supplied native probabilities."""\n    try:\n        if not isinstance(spectra_scores, list) or not 1 <= len(spectra_scores) <= 3:\n            raise ValueError(\'one to three Spectra probabilities in a list required\')\n        if any(not np.isscalar(v) or np.asarray(v).dtype.kind not in \'fiu\'\n               for v in (voice_present, eliya_fake, *spectra_scores)):\n            raise ValueError(\'real numeric scalar probabilities required\')\n        values = np.asarray([voice_present, eliya_fake, *spectra_scores], dtype=np.float64)\n        if (values.shape != (2 + len(spectra_scores),) or not np.isfinite(values).all()\n                or ((values < 0) | (values > 1)).any()):\n            raise ValueError(\'finite scalar probabilities in [0,1] required\')\n        presence, eliya = values[:2]\n        scores = values[2:]\n        mean, maximum = np.mean(scores), np.max(scores)\n        bounded = np.clip([eliya, presence * eliya, mean, presence * mean,\n                           maximum, presence * maximum], EPSILON, 1 - EPSILON)\n        return np.concatenate((np.log(bounded) - np.log1p(-bounded),\n                               [maximum - np.min(scores)]))\n    except (TypeError, ValueError, OverflowError) as error:\n        raise ValueError(\'invalid FILE24 auxiliary probabilities\') from error\n\ndef features(rows):\n    try:\n        if any(r[\'evidence_status\'] != \'ok\' or r[\'evidence_error\'] != \'\' for r in rows):\n            raise ValueError(\'healthy complete evidence required\')\n        original = base_features(rows)\n        added = np.asarray([additional_features(float(r[\'voice_present\']), r[\'eliya_fake\'], r[\'spectra_scores\'])\n                            for r in rows], dtype=np.float64)\n        return np.column_stack((original, added))\n    except (KeyError, TypeError, ValueError, OverflowError) as error:\n        raise ValueError(\'missing/invalid FILE24 feature record\') from error\n\ndef validate_model(model):\n    """Return None on success; validate without mutating or retaining the model."""\n    try:\n        fixed = dict(version=VERSION, model_type=MODEL_TYPE, feature_names=list(FEATURES),\n                     epsilon=EPSILON, input_dtype=\'float32\', task=\'FILE_FAKE only\',\n                     learning_rate=.05, n_estimators=64, max_depth=2,\n                     min_samples_leaf=24, seed=20260905, training_rows=1920, training_groups=40,\n                     source38_sha256=SOURCE38_SHA, predecessor_sha256=PREDECESSOR_SHA,\n                     protocol_sha256=PROTOCOL_SHA, recipe=RECIPE)\n        if not isinstance(model, dict) or any(model.get(k) != v for k, v in fixed.items()):\n            raise ValueError(\'incompatible fixed FILE24 export\')\n        for key in (\'version\', \'n_estimators\', \'max_depth\', \'min_samples_leaf\', \'seed\',\n                    \'training_rows\', \'training_groups\'):\n            if type(model[key]) is not int:\n                raise ValueError(\'integer recipe metadata required\')\n        if type(model[\'initial_log_odds\']) not in (int, float) or not np.isfinite(model[\'initial_log_odds\']):\n            raise ValueError(\'invalid initial log odds\')\n        trees = model[\'trees\']\n        if not isinstance(trees, list) or len(trees) != 64:\n            raise ValueError(\'exactly 64 trees required\')\n        for tree in trees:\n            names = (\'children_left\', \'children_right\', \'feature\', \'threshold\', \'value\')\n            arrays = [tree[k] for k in names]\n            if any(not isinstance(a, list) for a in arrays):\n                raise ValueError(\'one-dimensional JSON tree arrays required\')\n            left, right, feature, threshold, value = arrays\n            n = len(left)\n            if not 1 <= n <= 7 or any(len(a) != n for a in arrays):\n                raise ValueError(\'invalid depth-two tree size\')\n            if any(type(v) is not int for a in arrays[:3] for v in a):\n                raise ValueError(\'integer tree coordinates required\')\n            if any(type(v) not in (int, float) or not np.isfinite(v) for a in arrays[3:] for v in a):\n                raise ValueError(\'finite scalar tree data required\')\n            seen, pending = set(), [(0, 0)]\n            while pending:\n                node, depth = pending.pop()\n                if node in seen or depth > 2:\n                    raise ValueError(\'shared/cyclic/deep tree node\')\n                seen.add(node)\n                if left[node] == -1:\n                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:\n                        raise ValueError(\'invalid sklearn leaf\')\n                else:\n                    if (not 0 <= feature[node] < len(FEATURES)\n                            or not node < left[node] < n or not node < right[node] < n):\n                        raise ValueError(\'invalid tree split coordinate\')\n                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))\n            if len(seen) != n:\n                raise ValueError(\'unreachable tree nodes\')\n    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:\n        raise ValueError(\'malformed FILE24 tree export\') from error\n\ndef predict(model, rows):\n    validate_model(model)\n    x = features(rows).astype(np.float32)\n    raw = np.full(len(rows), model[\'initial_log_odds\'], dtype=np.float64)\n    with np.errstate(over=\'ignore\', invalid=\'ignore\'):\n        for tree in model[\'trees\']:\n            for i, values in enumerate(x):\n                node = 0\n                while tree[\'children_left\'][node] != -1:\n                    node = (tree[\'children_left\'][node]\n                            if float(values[tree[\'feature\'][node]]) <= tree[\'threshold\'][node]\n                            else tree[\'children_right\'][node])\n                raw[i] += model[\'learning_rate\'] * tree[\'value\'][node]\n    if not np.isfinite(raw).all():\n        raise ValueError(\'nonfinite FILE24 log odds\')\n    return expit(raw)\n'
_EVIDENCE_FILE_BACKEND_CACHE = None
_EVIDENCE_FILE_HELPER_CACHE = None


def _sub39_helpers():
    """Compile the embedded reviewed pure CPU implementation in an isolated namespace."""
    global _EVIDENCE_FILE_HELPER_CACHE
    if _EVIDENCE_FILE_HELPER_CACHE is None:
        namespace = {"__name__": "_sub39_evidence_file"}
        exec(compile(EVIDENCE_FILE_COMPILED_SOURCE, "<sub39-evidence-file>", "exec"), namespace)
        _EVIDENCE_FILE_HELPER_CACHE = namespace
    return _EVIDENCE_FILE_HELPER_CACHE


def _load_evidence_file_backend():
    """Bounded authenticated JSON; publish cache only after full tree validation."""
    import hashlib
    global _EVIDENCE_FILE_BACKEND_CACHE
    if _EVIDENCE_FILE_BACKEND_CACHE is not None:
        return _EVIDENCE_FILE_BACKEND_CACHE
    digest = EVIDENCE_FILE_BACKEND_SHA256
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("sub39 backend identity is not finalized")
    with EVIDENCE_FILE_BACKEND_PATH.open("rb") as handle:
        raw = handle.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("sub39 backend exceeds 1 MB limit")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("sub39 backend SHA-256 mismatch")
    model = json.loads(raw)
    _validate_evidence_file_backend(model)
    _EVIDENCE_FILE_BACKEND_CACHE = model
    return model


def _validate_evidence_file_backend(model):
    """Public builder/replay boundary: exact embedded full-tree validator."""
    return _sub39_helpers()["validate_model"](model)


def additional_evidence_features(voice_present, eliya_fake, spectra_scores):
    """Public replay API: the exact compiled seven-feature implementation."""
    return _sub39_helpers()["additional_features"](voice_present, eliya_fake, spectra_scores)


def predict_evidence_file_fake(original13, eliya_fake, spectra_scores, backend=None):
    """Public actual FILE scorer; raises on invalid input/model, with no inference."""
    if not isinstance(original13, (tuple, list)) or len(original13) != 13:
        raise ValueError("sub39 requires all original 13 FILE arguments")
    helper = _sub39_helpers()
    row = dict(zip(helper["SCORE_FIELDS"], original13))
    row.update(eliya_fake=eliya_fake, spectra_scores=spectra_scores,
               evidence_status="ok", evidence_error="")
    model = _load_evidence_file_backend() if backend is None else backend
    score = float(helper["predict"](model, [row])[0])
    if not _sub37_valid_probability(score):
        raise ValueError("sub39 produced invalid FILE probability")
    return score


def _sub39_capture(evidence, **fields):
    """Optional file-local collector; callers isolate even instrumentation faults."""
    if type(evidence) is dict:
        evidence.update(fields)


def _sub39_final_file(current_file, original13, learned_ok, evidence):
    """Commit only complete healthy FILE24; never change any protected head."""
    try:
        if (learned_ok is not True or type(evidence) is not dict
                or evidence.get("temporal_ok") is not True
                or evidence.get("eliya_ok") is not True):
            return current_file
        score = predict_evidence_file_fake(
            original13, evidence["eliya_fake"], evidence["spectra_scores"],
        )
        if not _sub37_valid_probability(score):
            raise ValueError("sub39 scorer returned invalid FILE probability")
        # sub50: observe original FILE24 success without changing its return.
        try:
            evidence["_sub50_file24_ok"] = True
        except (Exception, SystemExit):
            pass
        return score
    except (Exception, SystemExit) as evidence_error:
        print(f"WARNING: evidence FILE failed: {evidence_error}; retaining sub38 FILE",
              file=sys.stderr)
        return current_file
# END SUB39 EVIDENCE FILE HELPERS


# BEGIN SUB44 LEARNED MUSIC HELPERS
MUSIC_BACKEND_PATH = MODEL_DIR / "music_backend_joint_v1.json"
MUSIC_BACKEND_SHA256 = "19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7"
MUSIC_BACKEND_HELPER_SHA256 = "7f8327480fdc9ecb45637bea508a8cc996e4bb84e58fe5b046d664d28a71f933"
MUSIC_BACKEND_COMPILED_SOURCE = 'import numpy as np\n\nSCORE_FIELDS = (\n    "df_voice_mean", "df_music_mean", "df_music_max", "spectra", "nii",\n    "artifactnet", "lofcz", "current_voice", "current_music", "voice_present",\n    "music_present", "current_file", "file_music",\n)\n\nINTERACTIONS = (("voice_present", "df_voice_mean"), ("voice_present", "current_voice"),\n                ("music_present", "file_music"), ("music_present", "current_music"))\n\nFEATURES = tuple("logit:" + name for name in SCORE_FIELDS) + tuple(\n    "logit:" + left + "*" + right for left, right in INTERACTIONS)\n\nEPSILON = 1e-6\n\ndef features(rows):\n    if not rows:\n        raise ValueError("empty feature input")\n    values = np.asarray([[float(row[name]) for name in SCORE_FIELDS] for row in rows])\n    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():\n        raise ValueError("component probabilities must be finite and in [0, 1]")\n    indices = {name: index for index, name in enumerate(SCORE_FIELDS)}\n    products = np.column_stack([values[:, indices[a]] * values[:, indices[b]] for a, b in INTERACTIONS])\n    bounded = np.clip(np.column_stack((values, products)), EPSILON, 1 - EPSILON)\n    return np.log(bounded) - np.log1p(-bounded)\n\nBASE_FEATURES = FEATURES\nbase_features = features\n\n"""Frozen MUSIC17 features and strict JSON inference; failures raise ValueError.\n\nOnly original component probabilities enter this head. The caller owns exact\nincumbent MUSIC retention; this module never fills missing evidence.\n"""\n\nimport numpy as np\n\nfrom scipy.special import expit\n\nVERSION = 1\n\nMODEL_TYPE = \'gradient_boosted_trees_music17\'\n\nPROTOCOL_SHA = \'00741ffe13d7182892b985e758464f0f6d034e866add59319543a1fc314e92d3\'\n\nSOURCE39_SHA = \'d3137f8133ec46acfc1d9c8ada8bcf6f053531b72d76399cd65b0ea1cdc0d027\'\n\nRECIPE = \'fixed64-depth2-lr0.05-leaf24-music17\'\n\ndef features(rows):\n    """Exact old17 transform, requiring real numeric original13 probabilities."""\n    try:\n        if not isinstance(rows, (list, tuple)) or not rows:\n            raise ValueError(\'nonempty original probability rows required\')\n        for row in rows:\n            for name in SCORE_FIELDS:\n                value = row[name]\n                if (not np.isscalar(value) or np.asarray(value).dtype.kind not in \'fiu\'\n                        or not np.isfinite(value) or not 0 <= value <= 1):\n                    raise ValueError(\'finite numeric scalar probability required\')\n        return base_features(rows)\n    except (KeyError, TypeError, ValueError, OverflowError) as error:\n        raise ValueError(\'invalid MUSIC17 original probability record\') from error\n\ndef validate_model(model):\n    """Validate every node and recipe field without mutating the export."""\n    try:\n        fixed = dict(version=VERSION, model_type=MODEL_TYPE, task=\'MUSIC_FAKE only\',\n                     feature_names=list(FEATURES), epsilon=EPSILON, input_dtype=\'float32\',\n                     learning_rate=.05, n_estimators=64, max_depth=2,\n                     min_samples_leaf=24, seed=20260905, training_rows=1680,\n                     training_groups=40, protocol_sha256=PROTOCOL_SHA,\n                     source39_sha256=SOURCE39_SHA, recipe=RECIPE)\n        if not isinstance(model, dict) or any(model.get(k) != v for k, v in fixed.items()):\n            raise ValueError(\'incompatible fixed MUSIC17 export\')\n        for key in (\'version\', \'n_estimators\', \'max_depth\', \'min_samples_leaf\',\n                    \'seed\', \'training_rows\', \'training_groups\'):\n            if type(model[key]) is not int:\n                raise ValueError(\'integer recipe metadata required\')\n        for key in (\'epsilon\', \'learning_rate\', \'initial_log_odds\'):\n            if type(model[key]) not in (int, float) or not np.isfinite(model[key]):\n                raise ValueError(\'finite numeric model metadata required\')\n        trees = model[\'trees\']\n        if not isinstance(trees, list) or len(trees) != 64:\n            raise ValueError(\'exactly 64 trees required\')\n        for tree in trees:\n            if not isinstance(tree, dict):\n                raise ValueError(\'JSON tree object required\')\n            arrays = [tree[k] for k in (\'children_left\', \'children_right\', \'feature\',\n                                      \'threshold\', \'value\')]\n            if any(not isinstance(a, list) for a in arrays):\n                raise ValueError(\'one-dimensional JSON tree arrays required\')\n            left, right, feature, threshold, value = arrays\n            n = len(left)\n            if not 1 <= n <= 7 or any(len(a) != n for a in arrays):\n                raise ValueError(\'invalid depth-two tree size\')\n            if any(type(v) is not int for a in arrays[:3] for v in a):\n                raise ValueError(\'integer tree coordinates required\')\n            if any(type(v) not in (int, float) or not np.isfinite(v)\n                   for a in arrays[3:] for v in a):\n                raise ValueError(\'finite scalar tree data required\')\n            seen, pending = set(), [(0, 0)]\n            while pending:\n                node, depth = pending.pop()\n                if node in seen or depth > 2:\n                    raise ValueError(\'shared/cyclic/deep tree node\')\n                seen.add(node)\n                if left[node] == -1:\n                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:\n                        raise ValueError(\'invalid sklearn leaf\')\n                else:\n                    if (not 0 <= feature[node] < len(FEATURES)\n                            or not node < left[node] < n or not node < right[node] < n):\n                        raise ValueError(\'invalid tree split coordinate\')\n                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))\n            if len(seen) != n:\n                raise ValueError(\'unreachable tree nodes\')\n    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:\n        raise ValueError(\'malformed MUSIC17 tree export\') from error\n\ndef predict(model, rows):\n    validate_model(model)\n    x = features(rows).astype(np.float32)\n    raw = np.full(len(rows), model[\'initial_log_odds\'], dtype=np.float64)\n    with np.errstate(over=\'ignore\', invalid=\'ignore\'):\n        for tree in model[\'trees\']:\n            for i, values in enumerate(x):\n                node = 0\n                while tree[\'children_left\'][node] != -1:\n                    node = (tree[\'children_left\'][node]\n                            if float(values[tree[\'feature\'][node]]) <= tree[\'threshold\'][node]\n                            else tree[\'children_right\'][node])\n                raw[i] += model[\'learning_rate\'] * tree[\'value\'][node]\n    if not np.isfinite(raw).all():\n        raise ValueError(\'nonfinite MUSIC17 log odds\')\n    return expit(raw)\n'
_MUSIC_BACKEND_CACHE = None
_MUSIC_HELPER_CACHE = None


def _sub44_helpers():
    global _MUSIC_HELPER_CACHE
    if _MUSIC_HELPER_CACHE is None:
        namespace = {"__name__": "_sub44_learned_music"}
        exec(compile(MUSIC_BACKEND_COMPILED_SOURCE, "<sub44-learned-music>", "exec"), namespace)
        _MUSIC_HELPER_CACHE = namespace
    return _MUSIC_HELPER_CACHE


def _validate_music_backend(model):
    return _sub44_helpers()["validate_model"](model)


def _load_music_backend():
    import hashlib
    global _MUSIC_BACKEND_CACHE
    if _MUSIC_BACKEND_CACHE is not None:
        return _MUSIC_BACKEND_CACHE
    digest = MUSIC_BACKEND_SHA256
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("sub44 backend identity is not finalized")
    with MUSIC_BACKEND_PATH.open("rb") as handle:
        raw = handle.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("sub44 backend exceeds 1 MB limit")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("sub44 backend SHA256 mismatch")
    model = json.loads(raw)
    _validate_music_backend(model)
    _MUSIC_BACKEND_CACHE = model
    return model


def predict_learned_music_fake(original13, backend=None):
    if (not isinstance(original13, (tuple, list)) or len(original13) != 13
            or not all(_sub37_valid_probability(v) for v in original13)):
        raise ValueError("sub44 requires all13 healthy original probabilities")
    helper = _sub44_helpers()
    row = dict(zip(helper["SCORE_FIELDS"], original13))
    model = _load_music_backend() if backend is None else backend
    score = float(helper["predict"](model, [row])[0])
    if not _sub37_valid_probability(score):
        raise ValueError("sub44 invalid learned MUSIC probability")
    return score


def _sub44_final_music(current_music, original13, diagnostics=None):
    """Only original13 is required; keep all39 heads and old failure behavior."""
    try:
        if (not isinstance(original13, (tuple, list)) or len(original13) != 13
                or not _sub37_valid_probability(current_music)
                or not all(_sub37_valid_probability(v) for v in original13)):
            if type(diagnostics) is dict:
                diagnostics.update(status="unchanged_original_failure", candidate_attempts=0)
            return current_music
        expected = .03125 * original13[8] + .96875 * original13[6]
        if current_music != expected:
            raise ValueError("sub44 original MUSIC13/current binding mismatch")
        score = predict_learned_music_fake(original13)
        if not _sub37_valid_probability(score):
            raise ValueError("sub44 scorer returned invalid MUSIC probability")
        if type(diagnostics) is dict:
            diagnostics.update(status="ok", candidate_attempts=1)
        return score
    except (Exception, SystemExit) as error:
        if type(diagnostics) is dict:
            diagnostics.clear()
            diagnostics.update(status="new_failure", error=str(error))
        print(f"WARNING: learned MUSIC failed: {error}; retaining sub39 MUSIC", file=sys.stderr)
        return current_music
# END SUB44 LEARNED MUSIC HELPERS


# BEGIN SUB50 MUSIC STACKED FILE HELPERS
MUSIC_STACK_FILE_BACKEND_PATH = MODEL_DIR / "file_backend_musicstack25_v1.json"
MUSIC_STACK_FILE_BACKEND_SHA256 = "6fde2fad18dce4914d29b436cea7c8f6785c58ec84de58662765e107715aa25b"
MUSIC_STACK_FILE_HELPER_SHA256 = "5edcfb4ddba1ba21fecc6fb7e22b6bddb3cf86fa3a90e65a6d426274e2acaa55"
MUSIC_STACK_FILE_COMPILED_SOURCE = '"""Strict FILE24 plus learned MUSIC17 feature; caller owns incumbent fallback."""\n\nimport numpy as np\n\nfrom scipy.special import expit\n\nVERSION = 11\n\nMODEL_TYPE = \'gradient_boosted_trees_file25_musicstack\'\n\nSCORE_FIELDS = (\'df_voice_mean\', \'df_music_mean\', \'df_music_max\', \'spectra\', \'nii\',\n                \'artifactnet\', \'lofcz\', \'current_voice\', \'current_music\', \'voice_present\',\n                \'music_present\', \'current_file\', \'file_music\')\n\nEPSILON = evidence.EPSILON\n\nFEATURES = (*evidence.FEATURES, \'logit:music17\')\n\nPROTOCOL_SHA = \'4637b4583ea41413dfc0113b091fcea43ad20182f6b5e3c33e6f67e040134dc8\'\n\nSOURCE44_SHA = \'4c143dcf09155b24990bb0d638cb1432b90dfc061b6cef654b0e1ad408c9c841\'\n\nPREDECESSOR_SHA = \'8a9dba6701417ad018457e70f538c9745b7c9095b1dc3130f91a1a98e110a9c5\'\n\nMUSIC17_SHA = \'19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7\'\n\nRECIPE = \'fixed64-depth2-lr0.05-leaf24-file25-nested-musicstack\'\n\nIMPLEMENTATIONS = (\'src/deepvoicehackathon/music_stacked_file_backend.py\', \'tools/train_sub50_music_stack.py\')\n\ndef features(rows):\n    try:\n        if not isinstance(rows, (list, tuple)) or not rows:\n            raise ValueError(\'nonempty original FILE24 rows required\')\n        for row in rows:\n            for key in SCORE_FIELDS:\n                value = row[key]\n                if (not np.isscalar(value) or np.asarray(value).dtype.kind not in \'fiu\'\n                        or not np.isfinite(value) or not 0 <= value <= 1):\n                    raise ValueError(\'finite real original13 probabilities required\')\n        original = evidence.features(rows)\n        music = []\n        for row in rows:\n            value = row[\'music17\']\n            if (not np.isscalar(value) or np.asarray(value).dtype.kind not in \'fiu\'\n                    or not np.isfinite(value) or not 0 <= value <= 1):\n                raise ValueError(\'successful real MUSIC17 probability required\')\n            music.append(value)\n        p = np.clip(np.asarray(music, dtype=np.float64), EPSILON, 1-EPSILON)\n        return np.column_stack((original, np.log(p)-np.log1p(-p)))\n    except (KeyError, TypeError, ValueError, OverflowError) as error:\n        raise ValueError(\'invalid FILE25 MUSIC stacking evidence\') from error\n\ndef model_metadata():\n    return dict(version=VERSION, model_type=MODEL_TYPE, task=\'FILE_FAKE only\',\n        feature_names=list(FEATURES), epsilon=EPSILON, input_dtype=\'float32\',\n        learning_rate=.05, n_estimators=64, max_depth=2, min_samples_leaf=24,\n        loss=\'log_loss\', subsample=1., seed=20260905, training_rows=1920, training_groups=40,\n        protocol_sha256=PROTOCOL_SHA, source44_sha256=SOURCE44_SHA,\n        predecessor_sha256=PREDECESSOR_SHA, stage1_model_sha256=MUSIC17_SHA,\n        stage1_oof_seed=20260905, inner_group_seed=20260915, recipe=RECIPE)\n\ndef validate_model(model):\n    try:\n        fixed = model_metadata()\n        if type(model) is not dict or set(model) != set(fixed) | {\'initial_log_odds\', \'trees\', \'implementation_sha256\'}:\n            raise ValueError(\'strict FILE25 model object required\')\n        for key, value in fixed.items():\n            if type(model[key]) is not type(value) or model[key] != value:\n                raise ValueError(\'fixed FILE25 recipe mismatch: \' + key)\n        pins = model[\'implementation_sha256\']\n        if (type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any(\n                type(p) is not str or len(p) != 64 or any(c not in \'0123456789abcdef\' for c in p)\n                for p in pins.values())):\n            raise ValueError(\'portable implementation SHA256 map required\')\n        if type(model[\'initial_log_odds\']) not in (int, float) or not np.isfinite(model[\'initial_log_odds\']):\n            raise ValueError(\'finite initial log odds required\')\n        if type(model[\'trees\']) is not list or len(model[\'trees\']) != 64:\n            raise ValueError(\'exactly64 trees required\')\n        names = (\'children_left\', \'children_right\', \'feature\', \'threshold\', \'value\')\n        for tree in model[\'trees\']:\n            if type(tree) is not dict or set(tree) != set(names):\n                raise ValueError(\'strict tree object required\')\n            arrays = [tree[k] for k in names]\n            if any(type(a) is not list for a in arrays):\n                raise ValueError(\'JSON tree arrays required\')\n            left, right, feature, threshold, value = arrays\n            n = len(left)\n            if not 1 <= n <= 7 or any(len(a) != n for a in arrays):\n                raise ValueError(\'depth-two tree bounds required\')\n            if any(type(v) is not int for a in arrays[:3] for v in a):\n                raise ValueError(\'integer tree coordinates required\')\n            if any(type(v) not in (int, float) or not np.isfinite(v) for a in arrays[3:] for v in a):\n                raise ValueError(\'finite numeric tree data required\')\n            seen, pending = set(), [(0, 0)]\n            while pending:\n                node, depth = pending.pop()\n                if node in seen or depth > 2:\n                    raise ValueError(\'shared/cyclic/deep node\')\n                seen.add(node)\n                if left[node] == -1:\n                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:\n                        raise ValueError(\'invalid sklearn leaf\')\n                else:\n                    if not 0 <= feature[node] < 25 or not node < left[node] < n or not node < right[node] < n:\n                        raise ValueError(\'invalid split coordinates\')\n                    pending.extend(((left[node], depth+1), (right[node], depth+1)))\n            if len(seen) != n:\n                raise ValueError(\'unreachable tree node\')\n    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:\n        raise ValueError(\'malformed FILE25 music-stack model\') from error\n\ndef predict(model, rows):\n    validate_model(model)\n    x = features(rows).astype(np.float32)\n    raw = np.full(len(rows), model[\'initial_log_odds\'], dtype=np.float64)\n    with np.errstate(over=\'ignore\', invalid=\'ignore\'):\n        for tree in model[\'trees\']:\n            for i, values in enumerate(x):\n                node = 0\n                while tree[\'children_left\'][node] != -1:\n                    node = (tree[\'children_left\'][node] if float(values[tree[\'feature\'][node]]) <= tree[\'threshold\'][node]\n                            else tree[\'children_right\'][node])\n                raw[i] += model[\'learning_rate\'] * tree[\'value\'][node]\n    if not np.isfinite(raw).all():\n        raise ValueError(\'nonfinite FILE25 log odds\')\n    return expit(raw)\n'
_MUSIC_STACK_FILE_HELPER_CACHE = None
_MUSIC_STACK_FILE_BACKEND_CACHE = None


def _sub50_helpers():
    global _MUSIC_STACK_FILE_HELPER_CACHE
    if _MUSIC_STACK_FILE_HELPER_CACHE is None:
        from types import SimpleNamespace
        namespace = {"__name__": "_sub50_music_stacked_file", "evidence": SimpleNamespace(**_sub39_helpers())}
        exec(compile(MUSIC_STACK_FILE_COMPILED_SOURCE, "<sub50-music-stacked-file>", "exec"), namespace)
        _MUSIC_STACK_FILE_HELPER_CACHE = namespace
    return _MUSIC_STACK_FILE_HELPER_CACHE


def _validate_music_stack_file_backend(model):
    return _sub50_helpers()["validate_model"](model)


def _load_music_stack_file_backend():
    import hashlib
    global _MUSIC_STACK_FILE_BACKEND_CACHE
    if _MUSIC_STACK_FILE_BACKEND_CACHE is not None:
        return _MUSIC_STACK_FILE_BACKEND_CACHE
    digest = MUSIC_STACK_FILE_BACKEND_SHA256
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("sub50 backend identity is not finalized")
    with MUSIC_STACK_FILE_BACKEND_PATH.open("rb") as handle:
        raw = handle.read(1_000_001)
    if len(raw) > 1_000_000 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("sub50 model size/SHA256 mismatch")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate sub50 model key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite sub50 model constant: " + value)
    model = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _validate_music_stack_file_backend(model)
    _MUSIC_STACK_FILE_BACKEND_CACHE = model
    return model


def predict_music_stacked_file_fake(original13, eliya_fake, spectra_scores, music17, backend=None):
    if (not isinstance(original13, (tuple, list)) or len(original13) != 13
            or not all(_sub37_valid_probability(v) for v in original13)):
        raise ValueError("sub50 requires all13 healthy original probabilities")
    if (not isinstance(spectra_scores, list) or not 1 <= len(spectra_scores) <= 3
            or not all(_sub37_valid_probability(v) for v in spectra_scores)
            or spectra_scores[0] != original13[3]):
        raise ValueError("sub50 original Spectra S0 binding mismatch")
    helper = _sub50_helpers()
    row = dict(zip(helper["SCORE_FIELDS"], original13))
    row.update(eliya_fake=eliya_fake, spectra_scores=list(spectra_scores), music17=music17,
               evidence_status="ok", evidence_error="")
    model = _load_music_stack_file_backend() if backend is None else backend
    score = float(helper["predict"](model, [row])[0])
    if not _sub37_valid_probability(score):
        raise ValueError("sub50 invalid stacked FILE probability")
    return score


def _sub50_final_file(current_file, original13, learned_ok, evidence, music_score, music_state, diagnostics=None):
    """Use successful existing MUSIC once; new failures retain entire44 FILE."""
    try:
        if (not _sub37_valid_probability(current_file) or learned_ok is not True
                or type(evidence) is not dict or evidence.get("_sub50_file24_ok") is not True
                or evidence.get("temporal_ok") is not True or evidence.get("eliya_ok") is not True
                or type(music_state) is not dict or music_state.get("status") != "ok"
                or type(music_state.get("candidate_attempts")) is not int
                or music_state.get("candidate_attempts") != 1):
            if type(diagnostics) is dict:
                diagnostics.update(status="unchanged_incumbent_or_music_failure", candidate_attempts=0)
            return current_file
        score = predict_music_stacked_file_fake(original13, evidence["eliya_fake"],
            evidence["spectra_scores"], music_score)
        if not _sub37_valid_probability(score):
            raise ValueError("sub50 scorer returned invalid FILE probability")
        if type(diagnostics) is dict:
            diagnostics.update(status="ok", candidate_attempts=1)
        return score
    except (Exception, SystemExit) as error:
        if type(diagnostics) is dict:
            diagnostics.clear()
            diagnostics.update(status="new_failure", error=str(error))
        print(f"WARNING: music-stacked FILE failed: {error}; retaining sub44 FILE", file=sys.stderr)
        return current_file
# END SUB50 MUSIC STACKED FILE HELPERS


# sub52: new training coverage only; unchanged original25 feature computation.
NATIVE_DURATION_FILE_BACKEND_PATH = MODEL_DIR / "file_backend_native_duration25_v1.json"
NATIVE_DURATION_FILE_BACKEND_SHA256 = '2da7f008dab1a80be02f5f29560ddd967267b12a31108ec38e6c78f72135dd2b'
NATIVE_DURATION_FILE_HELPER_SHA256 = '5ef9d97c5af482d6e4326d8290a49d1d7767e59c74b8b2513576fb8fe0b31a26'
NATIVE_DURATION_FILE_COMPILED_SOURCE = '"""Strict native-duration FILE25 trees; callers own whole-sub50 fallback."""\nimport numpy as np\nfrom scipy.special import expit\nVERSION = 12\nMODEL_TYPE = \'gradient_boosted_trees_file25_native_duration\'\nFEATURES = stack.FEATURES\nSCORE_FIELDS = stack.SCORE_FIELDS\nEPSILON = stack.EPSILON\nPROTOCOL_SHA = \'69155b17ed1c836c92645f1f39660971bed13bcfc670534d4946910d2102c385\'\nSOURCE50_SHA = \'f3494938d11ae1948dd319de0c40aab197eb78c381089007914bf6daba21dc07\'\nPREDECESSOR_SHA = \'6fde2fad18dce4914d29b436cea7c8f6785c58ec84de58662765e107715aa25b\'\nMUSIC17_SHA = \'19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7\'\nRECIPE = \'fixed64-depth2-lr0.05-leaf24-file25-native-duration\'\nIMPLEMENTATIONS = (\'src/deepvoicehackathon/native_duration_file_backend.py\', \'tools/train_sub52_native_duration.py\')\nRECEIPT_HASHES = (\'rows_sha256\', \'groups_sha256\', \'labels_sha256\', \'weights_sha256\', \'features_sha256\', \'suppliers_sha256\')\n\ndef features(rows):\n    return stack.features(rows)\n\ndef model_metadata():\n    return dict(version=VERSION, model_type=MODEL_TYPE, task=\'FILE_FAKE only\', feature_names=list(FEATURES), epsilon=EPSILON, input_dtype=\'float32\', learning_rate=0.05, n_estimators=64, max_depth=2, min_samples_leaf=24, loss=\'log_loss\', subsample=1.0, seed=20260905, training_rows=2184, training_groups=40, extra_training_rows=264, protocol_sha256=PROTOCOL_SHA, source50_sha256=SOURCE50_SHA, predecessor_sha256=PREDECESSOR_SHA, stage1_model_sha256=MUSIC17_SHA, stage1_oof_seed=20260905, inner_group_seed=20260915, recipe=RECIPE)\n\ndef _sha(value):\n    return type(value) is str and len(value) == 64 and all((c in \'0123456789abcdef\' for c in value))\n\ndef _validate_receipt(receipt):\n    keys = {\'role\', \'training_rows\', \'training_groups\', \'old_rows\', \'extra_rows\', \'class_counts\', *RECEIPT_HASHES}\n    if type(receipt) is not dict or set(receipt) != keys:\n        raise ValueError(\'strict actual fit receipt required\')\n    if receipt[\'role\'] not in (\'fold\', \'full\'):\n        raise ValueError(\'explicit fold/full fit role required\')\n    if any((type(receipt[k]) is not int for k in (\'training_rows\', \'old_rows\', \'extra_rows\'))):\n        raise ValueError(\'integer actual fit counts required\')\n    groups = receipt[\'training_groups\']\n    if type(groups) is not list or not groups or any((type(g) is not str or not g for g in groups)) or (groups != sorted(set(groups))):\n        raise ValueError(\'sorted unique actual training groups required\')\n    old, extra = (receipt[\'old_rows\'], receipt[\'extra_rows\'])\n    if old != 48 * len(groups) or old + extra != receipt[\'training_rows\'] or (not 0 < extra <= 264) or extra % 12:\n        raise ValueError(\'actual old/new FILE view counts mismatch\')\n    counts = receipt[\'class_counts\']\n    if type(counts) is not dict or set(counts) != {\'0\', \'1\'} or any((type(v) is not int or v <= 0 for v in counts.values())) or (counts != {\'0\': 15 * len(groups) + extra // 2, \'1\': 33 * len(groups) + extra // 2}):\n        raise ValueError(\'actual FILE class counts mismatch\')\n    if receipt[\'role\'] == \'full\':\n        if (old, extra, len(groups), counts) != (1920, 264, 40, {\'0\': 732, \'1\': 1452}):\n            raise ValueError(\'full receipt must prove2184/40\')\n    elif len(groups) not in (30, 35):\n        raise ValueError(\'outer receipt cannot masquerade as full fit\')\n    if any((not _sha(receipt[k]) for k in RECEIPT_HASHES)):\n        raise ValueError(\'actual fit identity SHA256 required\')\n\ndef validate_model(model):\n    """Reject malformed metadata/trees without projection onto an old schema."""\n    try:\n        fixed = model_metadata()\n        if type(model) is not dict or set(model) != set(fixed) | {\'initial_log_odds\', \'trees\', \'implementation_sha256\', \'fit_receipt\'}:\n            raise ValueError(\'strict native-duration FILE25 model required\')\n        for key, value in fixed.items():\n            if type(model[key]) is not type(value) or model[key] != value:\n                raise ValueError(\'fixed FILE25 recipe mismatch: \' + key)\n        pins = model[\'implementation_sha256\']\n        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any((not _sha(v) for v in pins.values())):\n            raise ValueError(\'portable implementation SHA256 map required\')\n        _validate_receipt(model[\'fit_receipt\'])\n        if type(model[\'initial_log_odds\']) not in (int, float) or not np.isfinite(model[\'initial_log_odds\']):\n            raise ValueError(\'finite initial log odds required\')\n        if type(model[\'trees\']) is not list or len(model[\'trees\']) != 64:\n            raise ValueError(\'exactly64 trees required\')\n        names = (\'children_left\', \'children_right\', \'feature\', \'threshold\', \'value\')\n        for tree in model[\'trees\']:\n            if type(tree) is not dict or set(tree) != set(names):\n                raise ValueError(\'strict tree object required\')\n            arrays = [tree[k] for k in names]\n            if any((type(a) is not list for a in arrays)):\n                raise ValueError(\'JSON tree arrays required\')\n            left, right, feature, threshold, value = arrays\n            n = len(left)\n            if not 1 <= n <= 7 or any((len(a) != n for a in arrays)):\n                raise ValueError(\'depth-two tree bounds required\')\n            if any((type(v) is not int for a in arrays[:3] for v in a)):\n                raise ValueError(\'integer tree coordinates required\')\n            if any((type(v) not in (int, float) or not np.isfinite(v) for a in arrays[3:] for v in a)):\n                raise ValueError(\'finite numeric tree data required\')\n            seen, pending = (set(), [(0, 0)])\n            while pending:\n                node, depth = pending.pop()\n                if node in seen or depth > 2:\n                    raise ValueError(\'shared/cyclic/deep node\')\n                seen.add(node)\n                if left[node] == -1:\n                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:\n                        raise ValueError(\'invalid sklearn leaf\')\n                else:\n                    if not 0 <= feature[node] < 25 or not node < left[node] < n or (not node < right[node] < n):\n                        raise ValueError(\'invalid split coordinates\')\n                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))\n            if len(seen) != n:\n                raise ValueError(\'unreachable tree node\')\n    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:\n        raise ValueError(\'malformed FILE25 native-duration model\') from error\n\ndef predict(model, rows):\n    validate_model(model)\n    x = features(rows).astype(np.float32)\n    raw = np.full(len(rows), model[\'initial_log_odds\'], dtype=np.float64)\n    with np.errstate(over=\'ignore\', invalid=\'ignore\'):\n        for tree in model[\'trees\']:\n            for i, values in enumerate(x):\n                node = 0\n                while tree[\'children_left\'][node] != -1:\n                    node = tree[\'children_left\'][node] if float(values[tree[\'feature\'][node]]) <= tree[\'threshold\'][node] else tree[\'children_right\'][node]\n                raw[i] += model[\'learning_rate\'] * tree[\'value\'][node]\n    if not np.isfinite(raw).all():\n        raise ValueError(\'nonfinite FILE25 log odds\')\n    return expit(raw)\n'
_NATIVE_DURATION_FILE_HELPER_CACHE = None
_NATIVE_DURATION_FILE_BACKEND_CACHE = None


def _sub52_helpers():
    from types import SimpleNamespace
    global _NATIVE_DURATION_FILE_HELPER_CACHE
    if _NATIVE_DURATION_FILE_HELPER_CACHE is None:
        namespace = {"stack": SimpleNamespace(**_sub50_helpers())}
        exec(compile(NATIVE_DURATION_FILE_COMPILED_SOURCE, "<sub52-native-duration-file>", "exec"), namespace)
        _NATIVE_DURATION_FILE_HELPER_CACHE = namespace
    return _NATIVE_DURATION_FILE_HELPER_CACHE


def _validate_native_duration_backend(model):
    _sub52_helpers()["validate_model"](model)


def _load_native_duration_backend():
    import hashlib
    import json
    global _NATIVE_DURATION_FILE_BACKEND_CACHE
    if _NATIVE_DURATION_FILE_BACKEND_CACHE is not None:
        return _NATIVE_DURATION_FILE_BACKEND_CACHE
    expected = NATIVE_DURATION_FILE_BACKEND_SHA256
    if (type(expected) is not str or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("frozen native-duration FILE SHA required")
    with NATIVE_DURATION_FILE_BACKEND_PATH.open("rb") as handle:
        raw = handle.read(1_000_000)
    if len(raw) >= 1_000_000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("bounded native-duration FILE model/hash mismatch")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate native-duration JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite native-duration JSON constant")
    model = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _validate_native_duration_backend(model)
    if model["fit_receipt"]["role"] != "full":
        raise ValueError("deployed native-duration FILE requires full fit")
    _NATIVE_DURATION_FILE_BACKEND_CACHE = model
    return model


def predict_native_duration_file_fake(original13, eliya_fake, spectra_scores, music17, backend=None):
    if (not isinstance(original13, (tuple, list)) or len(original13) != 13
            or not all(_sub37_valid_probability(v) for v in original13)):
        raise ValueError("native-duration FILE requires healthy original13")
    if (type(spectra_scores) is not list or not 1 <= len(spectra_scores) <= 3
            or not all(_sub37_valid_probability(v) for v in spectra_scores)
            or spectra_scores[0] != original13[3]):
        raise ValueError("native-duration FILE original S0 binding mismatch")
    helper = _sub52_helpers()
    row = dict(zip(helper["SCORE_FIELDS"], original13))
    row.update(eliya_fake=eliya_fake, spectra_scores=list(spectra_scores), music17=music17,
               evidence_status="ok", evidence_error="")
    model = _load_native_duration_backend() if backend is None else backend
    score = float(helper["predict"](model, [row])[0])
    if not _sub37_valid_probability(score):
        raise ValueError("invalid native-duration FILE probability")
    return score


def _sub52_final_file(current_file, original13, evidence, music_score, file50_state, diagnostics=None):
    if type(diagnostics) is dict:
        diagnostics.update(status="unchanged_original_failure", candidate_attempts=0)
    if (not _sub37_valid_probability(current_file) or type(file50_state) is not dict
            or file50_state.get("status") != "ok"
            or type(file50_state.get("candidate_attempts")) is not int
            or file50_state["candidate_attempts"] != 1):
        return current_file
    try:
        if type(diagnostics) is dict:
            diagnostics.update(status="new_failure", candidate_attempts=1)
        if (type(evidence) is not dict or evidence.get("temporal_ok") is not True
                or evidence.get("eliya_ok") is not True):
            raise ValueError("complete original50 evidence required")
        score = predict_native_duration_file_fake(original13, evidence["eliya_fake"],
            evidence["spectra_scores"], music_score)
        if not _sub37_valid_probability(score):
            raise ValueError("invalid native-duration FILE result")
    except (Exception, SystemExit) as error:
        print(f"WARNING: native-duration FILE failed: {error}; retaining sub50 FILE", file=sys.stderr)
        return current_file
    if type(diagnostics) is dict:
        diagnostics.update(status="ok", candidate_attempts=1)
    return score


def _sub52_light_file(row):
    """Fixed presence-weighted component floor on finalized rounded outputs."""
    original = row.get("FILE_FAKE_PROB", 0.5)
    try:
        keys = ("FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
        raw = [row[key] for key in keys]
        if any(isinstance(value, (bool, np.bool_)) or
               not isinstance(value, (int, float, np.integer, np.floating))
               for value in raw):
            return original
        f, v, m, vp, mp = (float(value) for value in raw)
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0
                   for value in (f, v, m, vp, mp)):
            return original
        return round(max(f, vp * v, mp * m), 10)
    except (Exception, SystemExit):
        return original


# sub54: new fake-music training coverage only; unchanged original25 feature computation.
NATIVE_FAKE_MUSIC_FILE_BACKEND_PATH = MODEL_DIR / "file_backend_native_fake_music25_v1.json"
NATIVE_FAKE_MUSIC_FILE_BACKEND_SHA256 = 'ed34f05d680479eb487504cc69aa6ef05a979d7b0276100dfd7b60164840fb19'
NATIVE_FAKE_MUSIC_FILE_HELPER_SHA256 = '7af5483225c05d27bf4114b98993b66f1b174dd122c6f64620131ca6da6ce4c2'
NATIVE_FAKE_MUSIC_FILE_COMPILED_SOURCE = '"""Strict native fake-music FILE25 trees; callers own whole-raw53 fallback."""\nimport numpy as np\nfrom scipy.special import expit\nVERSION = 13\nMODEL_TYPE = \'gradient_boosted_trees_file25_native_fake_music\'\nFEATURES = stack.FEATURES\nSCORE_FIELDS = stack.SCORE_FIELDS\nEPSILON = stack.EPSILON\nPROTOCOL_SHA = \'5e7f3c135bcb5aa31cdca6cf340b459b9f42115a82a27e1104ef505b61d03e20\'\nPROTOCOL_COMMIT = \'e7347b0\'\nSOURCE53_SHA = \'5c6870c4d1b51e22207c2ae3aa23f90c87db37398f860fe1f09248f0c0673d0c\'\nPREDECESSOR_SHA = \'2da7f008dab1a80be02f5f29560ddd967267b12a31108ec38e6c78f72135dd2b\'\nMUSIC17_SHA = \'19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7\'\nRECIPE = \'fixed64-depth2-lr0.05-leaf24-file25-native-fake-music\'\nIMPLEMENTATIONS = (\'src/deepvoicehackathon/native_fake_music_file_backend.py\', \'tools/train_sub54_native_fake_music.py\')\nRECEIPT_HASHES = (\'rows_sha256\', \'groups_sha256\', \'labels_sha256\', \'weights_sha256\', \'features_sha256\', \'suppliers_sha256\')\n\ndef features(rows):\n    return stack.features(rows)\n\ndef model_metadata():\n    return dict(version=VERSION, model_type=MODEL_TYPE, task=\'FILE_FAKE only\', feature_names=list(FEATURES), epsilon=EPSILON, input_dtype=\'float32\', learning_rate=0.05, n_estimators=64, max_depth=2, min_samples_leaf=24, loss=\'log_loss\', subsample=1.0, seed=20260905, training_rows=2316, training_groups=40, extra_training_rows=396, native_training_rows=264, fake_training_rows=132, protocol_sha256=PROTOCOL_SHA, protocol_commit=PROTOCOL_COMMIT, source53_sha256=SOURCE53_SHA, predecessor_sha256=PREDECESSOR_SHA, stage1_model_sha256=MUSIC17_SHA, stage1_oof_seed=20260905, inner_group_seed=20260915, recipe=RECIPE)\n\ndef _sha(value):\n    return type(value) is str and len(value) == 64 and all((c in \'0123456789abcdef\' for c in value))\n\ndef _validate_receipt(receipt):\n    keys = {\'role\', \'training_rows\', \'training_groups\', \'old_rows\', \'native_rows\', \'fake_rows\', \'extra_rows\', \'class_counts\', *RECEIPT_HASHES}\n    if type(receipt) is not dict or set(receipt) != keys:\n        raise ValueError(\'strict actual fit receipt required\')\n    if receipt[\'role\'] not in (\'fold\', \'full\'):\n        raise ValueError(\'explicit fold/full fit role required\')\n    if any((type(receipt[k]) is not int for k in (\'training_rows\', \'old_rows\', \'native_rows\', \'fake_rows\', \'extra_rows\'))):\n        raise ValueError(\'integer actual fit counts required\')\n    groups = receipt[\'training_groups\']\n    if type(groups) is not list or not groups or any((type(g) is not str or not g for g in groups)) or (groups != sorted(set(groups))):\n        raise ValueError(\'sorted unique actual training groups required\')\n    old, native, fake, extra = (receipt[\'old_rows\'], receipt[\'native_rows\'], receipt[\'fake_rows\'], receipt[\'extra_rows\'])\n    if old != 48 * len(groups):\n        raise ValueError(\'actual old FILE view counts mismatch\')\n    if extra != native + fake or receipt[\'training_rows\'] != old + extra:\n        raise ValueError(\'actual native/fake FILE view counts mismatch\')\n    if not 0 < extra <= 396 or native % 12 or fake % 6 or (native > 264) or (fake > 132):\n        raise ValueError(\'actual native/fake partition bounds mismatch\')\n    counts = receipt[\'class_counts\']\n    expected = {\'0\': 15 * len(groups) + native // 2, \'1\': 33 * len(groups) + native // 2 + fake}\n    if type(counts) is not dict or set(counts) != {\'0\', \'1\'} or any((type(v) is not int or v <= 0 for v in counts.values())) or (counts != expected):\n        raise ValueError(\'actual FILE class counts mismatch\')\n    if receipt[\'role\'] == \'full\':\n        if (old, native, fake, len(groups), counts) != (1920, 264, 132, 40, {\'0\': 732, \'1\': 1584}):\n            raise ValueError(\'full receipt must prove2316/40\')\n    elif len(groups) not in (30, 35):\n        raise ValueError(\'outer receipt cannot masquerade as full fit\')\n    if any((not _sha(receipt[k]) for k in RECEIPT_HASHES)):\n        raise ValueError(\'actual fit identity SHA256 required\')\n\ndef validate_model(model):\n    """Reject malformed metadata/trees without projection onto an old schema."""\n    try:\n        fixed = model_metadata()\n        if type(model) is not dict or set(model) != set(fixed) | {\'initial_log_odds\', \'trees\', \'implementation_sha256\', \'fit_receipt\'}:\n            raise ValueError(\'strict native fake-music FILE25 model required\')\n        for key, value in fixed.items():\n            if type(model[key]) is not type(value) or model[key] != value:\n                raise ValueError(\'fixed FILE25 recipe mismatch: \' + key)\n        pins = model[\'implementation_sha256\']\n        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any((not _sha(v) for v in pins.values())):\n            raise ValueError(\'portable implementation SHA256 map required\')\n        _validate_receipt(model[\'fit_receipt\'])\n        if type(model[\'initial_log_odds\']) not in (int, float) or not np.isfinite(model[\'initial_log_odds\']):\n            raise ValueError(\'finite initial log odds required\')\n        if type(model[\'trees\']) is not list or len(model[\'trees\']) != 64:\n            raise ValueError(\'exactly64 trees required\')\n        names = (\'children_left\', \'children_right\', \'feature\', \'threshold\', \'value\')\n        for tree in model[\'trees\']:\n            if type(tree) is not dict or set(tree) != set(names):\n                raise ValueError(\'strict tree object required\')\n            arrays = [tree[k] for k in names]\n            if any((type(a) is not list for a in arrays)):\n                raise ValueError(\'JSON tree arrays required\')\n            left, right, feature, threshold, value = arrays\n            n = len(left)\n            if not 1 <= n <= 7 or any((len(a) != n for a in arrays)):\n                raise ValueError(\'depth-two tree bounds required\')\n            if any((type(v) is not int for a in arrays[:3] for v in a)):\n                raise ValueError(\'integer tree coordinates required\')\n            if any((type(v) not in (int, float) or not np.isfinite(v) for a in arrays[3:] for v in a)):\n                raise ValueError(\'finite numeric tree data required\')\n            seen, pending = (set(), [(0, 0)])\n            while pending:\n                node, depth = pending.pop()\n                if node in seen or depth > 2:\n                    raise ValueError(\'shared/cyclic/deep node\')\n                seen.add(node)\n                if left[node] == -1:\n                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:\n                        raise ValueError(\'invalid sklearn leaf\')\n                else:\n                    if not 0 <= feature[node] < 25 or not node < left[node] < n or (not node < right[node] < n):\n                        raise ValueError(\'invalid split coordinates\')\n                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))\n            if len(seen) != n:\n                raise ValueError(\'unreachable tree node\')\n    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:\n        raise ValueError(\'malformed FILE25 native fake-music model\') from error\n\ndef predict(model, rows):\n    validate_model(model)\n    x = features(rows).astype(np.float32)\n    raw = np.full(len(rows), model[\'initial_log_odds\'], dtype=np.float64)\n    with np.errstate(over=\'ignore\', invalid=\'ignore\'):\n        for tree in model[\'trees\']:\n            for i, values in enumerate(x):\n                node = 0\n                while tree[\'children_left\'][node] != -1:\n                    node = tree[\'children_left\'][node] if float(values[tree[\'feature\'][node]]) <= tree[\'threshold\'][node] else tree[\'children_right\'][node]\n                raw[i] += model[\'learning_rate\'] * tree[\'value\'][node]\n    if not np.isfinite(raw).all():\n        raise ValueError(\'nonfinite FILE25 log odds\')\n    return expit(raw)\n'
_NATIVE_FAKE_MUSIC_FILE_HELPER_CACHE = None
_NATIVE_FAKE_MUSIC_FILE_BACKEND_CACHE = None


def _sub54_helpers():
    from types import SimpleNamespace
    global _NATIVE_FAKE_MUSIC_FILE_HELPER_CACHE
    if _NATIVE_FAKE_MUSIC_FILE_HELPER_CACHE is None:
        namespace = {"stack": SimpleNamespace(**_sub50_helpers())}
        exec(compile(NATIVE_FAKE_MUSIC_FILE_COMPILED_SOURCE, "<sub54-native-fake-music-file>", "exec"), namespace)
        _NATIVE_FAKE_MUSIC_FILE_HELPER_CACHE = namespace
    return _NATIVE_FAKE_MUSIC_FILE_HELPER_CACHE


def _validate_native_fake_music_backend(model):
    _sub54_helpers()["validate_model"](model)


def _load_native_fake_music_backend():
    import hashlib
    import json
    global _NATIVE_FAKE_MUSIC_FILE_BACKEND_CACHE
    if _NATIVE_FAKE_MUSIC_FILE_BACKEND_CACHE is not None:
        return _NATIVE_FAKE_MUSIC_FILE_BACKEND_CACHE
    expected = NATIVE_FAKE_MUSIC_FILE_BACKEND_SHA256
    if (type(expected) is not str or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("frozen native fake-music FILE SHA required")
    try:
        with NATIVE_FAKE_MUSIC_FILE_BACKEND_PATH.open("rb") as handle:
            raw = handle.read(1_000_000)
    except OSError as error:
        raise ValueError(f"missing native fake-music FILE model: {error}") from error
    if len(raw) >= 1_000_000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("bounded native fake-music FILE model/hash mismatch")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate native fake-music JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite native fake-music JSON constant")
    model = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _validate_native_fake_music_backend(model)
    if model["fit_receipt"]["role"] != "full":
        raise ValueError("deployed native fake-music FILE requires full fit")
    _NATIVE_FAKE_MUSIC_FILE_BACKEND_CACHE = model
    return model


def predict_native_fake_music_file_fake(original13, eliya_fake, spectra_scores, music17, backend=None):
    if (not isinstance(original13, (tuple, list)) or len(original13) != 13
            or not all(_sub37_valid_probability(v) for v in original13)):
        raise ValueError("native fake-music FILE requires healthy original13")
    if (type(spectra_scores) is not list or not 1 <= len(spectra_scores) <= 3
            or not all(_sub37_valid_probability(v) for v in spectra_scores)
            or spectra_scores[0] != original13[3]):
        raise ValueError("native fake-music FILE original S0 binding mismatch")
    helper = _sub54_helpers()
    row = dict(zip(helper["SCORE_FIELDS"], original13))
    row.update(eliya_fake=eliya_fake, spectra_scores=list(spectra_scores), music17=music17,
               evidence_status="ok", evidence_error="")
    model = _load_native_fake_music_backend() if backend is None else backend
    score = float(helper["predict"](model, [row])[0])
    if not _sub37_valid_probability(score):
        raise ValueError("invalid native fake-music FILE probability")
    return score


def _sub54_final_file(current_file, original13, evidence, music_score, file53_state, diagnostics=None):
    if type(diagnostics) is dict:
        diagnostics.update(status="unchanged_original_failure", candidate_attempts=0)
    if (not _sub37_valid_probability(current_file) or type(file53_state) is not dict
            or file53_state.get("status") != "ok"
            or type(file53_state.get("candidate_attempts")) is not int
            or file53_state["candidate_attempts"] != 1):
        return current_file
    try:
        if type(diagnostics) is dict:
            diagnostics.update(status="new_failure", candidate_attempts=1)
        if (type(evidence) is not dict or evidence.get("temporal_ok") is not True
                or evidence.get("eliya_ok") is not True):
            raise ValueError("complete original50 evidence required")
        score = predict_native_fake_music_file_fake(original13, evidence["eliya_fake"],
            evidence["spectra_scores"], music_score)
        if not _sub37_valid_probability(score):
            raise ValueError("invalid native fake-music FILE result")
    except (Exception, SystemExit) as error:
        print(f"WARNING: native fake-music FILE failed: {error}; retaining raw53 FILE", file=sys.stderr)
        return current_file
    if type(diagnostics) is dict:
        diagnostics.update(status="ok", candidate_attempts=1)
    return score


# sub55: final-component support FILE29 only; unchanged original25 feature computation.
COMPONENT_SUPPORT_FILE_BACKEND_PATH = MODEL_DIR / "file_backend_component_support29_v1.json"
COMPONENT_SUPPORT_FILE_BACKEND_SHA256 = '0f38b7c9006d8ac309c6157140f7c9b473b390939cecbb9b1bb91282d1e08c92'
COMPONENT_SUPPORT_FILE_HELPER_SHA256 = 'b1f9669c410f4fd70c785111067198012b6805b8aa78ab90910e5c52edfbd15e'
COMPONENT_SUPPORT_FILE_COMPILED_SOURCE = '"""Strict FILE29 with final-component support features; callers own whole-raw54 fallback."""\nimport numpy as np\nfrom scipy.special import expit\nVERSION = 14\nMODEL_TYPE = \'gradient_boosted_trees_file29_component_support\'\nSCORE_FIELDS = baseline.SCORE_FIELDS\nEPSILON = baseline.EPSILON\nFEATURES = (*baseline.FEATURES, \'logit:voice38\', \'logit:voice_present*voice38\', \'logit:music_present*music17\', \'logit:max(voice_present*voice38,music_present*music17)\')\nPROTOCOL_SHA = \'f4d5175a0f0c7ba002fe0c0f30b4e61e0decaf736daac4b6e6c2c01825d337b0\'\nPROTOCOL_COMMIT = \'5da86db\'\nSOURCE54_SHA = \'707cc797728b18d6856a080dba413df297ccf3e665bc5dbce5aec754e064477f\'\nPREDECESSOR_SHA = \'ed34f05d680479eb487504cc69aa6ef05a979d7b0276100dfd7b60164840fb19\'\nMUSIC17_SHA = \'19816ba06811cb3b385802ea35f837a9a219891525f7dc2b02c71d098d6be9f7\'\nRECIPE = \'fixed64-depth2-lr0.05-leaf24-file29-component-support\'\nIMPLEMENTATIONS = (\'src/deepvoicehackathon/component_support_file_backend.py\', \'tools/train_sub55_component_support.py\')\nRECEIPT_HASHES = (\'rows_sha256\', \'groups_sha256\', \'labels_sha256\', \'weights_sha256\', \'features_sha256\', \'suppliers_sha256\')\n\ndef _is_real_probability(value):\n    if isinstance(value, (bool, np.bool_)):\n        return False\n    if not np.isscalar(value):\n        return False\n    if np.asarray(value).dtype.kind not in \'fiu\':\n        return False\n    try:\n        number = float(value)\n    except (TypeError, ValueError, OverflowError):\n        return False\n    return bool(np.isfinite(number) and 0.0 <= number <= 1.0)\n\ndef _logit(value):\n    clipped = min(max(float(value), EPSILON), 1.0 - EPSILON)\n    return float(np.log(clipped) - np.log1p(-clipped))\n\ndef features(rows):\n    try:\n        if not isinstance(rows, (list, tuple)) or not rows:\n            raise ValueError(\'nonempty component-support FILE rows required\')\n        original = baseline.features(rows)\n        if original.shape != (len(rows), 25):\n            raise ValueError(\'original25 feature width drift\')\n        added = []\n        for row in rows:\n            for key in (\'voice_present\', \'music_present\', \'music17\', \'voice38\'):\n                if key not in row:\n                    raise ValueError(\'missing component-support probability: \' + key)\n                if not _is_real_probability(row[key]):\n                    raise ValueError(\'finite real component-support probability required: \' + key)\n            voice38 = float(row[\'voice38\'])\n            present_voice = float(row[\'voice_present\'])\n            present_music = float(row[\'music_present\'])\n            music17 = float(row[\'music17\'])\n            v = round(voice38, 10)\n            vp10 = round(present_voice, 10)\n            mp10 = round(present_music, 10)\n            m10 = round(music17, 10)\n            sv = round(vp10 * v, 10)\n            sm = round(mp10 * m10, 10)\n            peak = sv if sv >= sm else sm\n            for candidate in (v, sv, sm, peak):\n                if not np.isfinite(candidate) or not 0.0 <= candidate <= 1.0:\n                    raise ValueError(\'rounded component support out of range\')\n            added.append([_logit(v), _logit(sv), _logit(sm), _logit(peak)])\n        return np.column_stack((original, np.asarray(added, dtype=np.float64)))\n    except (KeyError, TypeError, ValueError, OverflowError) as error:\n        raise ValueError(\'invalid FILE29 component-support evidence\') from error\n\ndef model_metadata():\n    return dict(version=VERSION, model_type=MODEL_TYPE, task=\'FILE_FAKE only\', feature_names=list(FEATURES), epsilon=EPSILON, input_dtype=\'float32\', learning_rate=0.05, n_estimators=64, max_depth=2, min_samples_leaf=24, loss=\'log_loss\', subsample=1.0, seed=20260905, training_rows=2316, training_groups=40, extra_training_rows=396, native_training_rows=264, fake_training_rows=132, protocol_sha256=PROTOCOL_SHA, protocol_commit=PROTOCOL_COMMIT, source54_sha256=SOURCE54_SHA, predecessor_sha256=PREDECESSOR_SHA, stage1_model_sha256=MUSIC17_SHA, stage1_oof_seed=20260905, inner_group_seed=20260915, recipe=RECIPE)\n\ndef _validate_receipt(receipt):\n    return baseline._validate_receipt(receipt)\n\ndef validate_model(model):\n    try:\n        fixed = model_metadata()\n        if type(model) is not dict or set(model) != set(fixed) | {\'initial_log_odds\', \'trees\', \'implementation_sha256\', \'fit_receipt\'}:\n            raise ValueError(\'strict component-support FILE29 model required\')\n        for key, value in fixed.items():\n            if type(model[key]) is not type(value) or model[key] != value:\n                raise ValueError(\'fixed FILE29 recipe mismatch: \' + key)\n        pins = model[\'implementation_sha256\']\n        if type(pins) is not dict or set(pins) != set(IMPLEMENTATIONS) or any((not baseline._sha(v) for v in pins.values())):\n            raise ValueError(\'portable implementation SHA256 map required\')\n        _validate_receipt(model[\'fit_receipt\'])\n        if type(model[\'initial_log_odds\']) not in (int, float) or not np.isfinite(model[\'initial_log_odds\']):\n            raise ValueError(\'finite initial log odds required\')\n        if type(model[\'trees\']) is not list or len(model[\'trees\']) != 64:\n            raise ValueError(\'exactly64 trees required\')\n        names = (\'children_left\', \'children_right\', \'feature\', \'threshold\', \'value\')\n        for tree in model[\'trees\']:\n            if type(tree) is not dict or set(tree) != set(names):\n                raise ValueError(\'strict tree object required\')\n            arrays = [tree[k] for k in names]\n            if any((type(a) is not list for a in arrays)):\n                raise ValueError(\'JSON tree arrays required\')\n            left, right, feature, threshold, value = arrays\n            n = len(left)\n            if not 1 <= n <= 7 or any((len(a) != n for a in arrays)):\n                raise ValueError(\'depth-two tree bounds required\')\n            if any((type(v) is not int for a in arrays[:3] for v in a)):\n                raise ValueError(\'integer tree coordinates required\')\n            if any((type(v) not in (int, float) or not np.isfinite(v) for a in arrays[3:] for v in a)):\n                raise ValueError(\'finite numeric tree data required\')\n            seen, pending = (set(), [(0, 0)])\n            while pending:\n                node, depth = pending.pop()\n                if node in seen or depth > 2:\n                    raise ValueError(\'shared/cyclic/deep node\')\n                seen.add(node)\n                if left[node] == -1:\n                    if right[node] != -1 or feature[node] != -2 or threshold[node] != -2:\n                        raise ValueError(\'invalid sklearn leaf\')\n                else:\n                    if not 0 <= feature[node] < 29 or not node < left[node] < n or (not node < right[node] < n):\n                        raise ValueError(\'invalid split coordinates\')\n                    pending.extend(((left[node], depth + 1), (right[node], depth + 1)))\n            if len(seen) != n:\n                raise ValueError(\'unreachable tree node\')\n    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:\n        raise ValueError(\'malformed FILE29 component-support model\') from error\n\ndef predict(model, rows):\n    validate_model(model)\n    x = features(rows).astype(np.float32)\n    raw = np.full(len(rows), model[\'initial_log_odds\'], dtype=np.float64)\n    with np.errstate(over=\'ignore\', invalid=\'ignore\'):\n        for tree in model[\'trees\']:\n            for i, values in enumerate(x):\n                node = 0\n                while tree[\'children_left\'][node] != -1:\n                    node = tree[\'children_left\'][node] if float(values[tree[\'feature\'][node]]) <= tree[\'threshold\'][node] else tree[\'children_right\'][node]\n                raw[i] += model[\'learning_rate\'] * tree[\'value\'][node]\n    if not np.isfinite(raw).all():\n        raise ValueError(\'nonfinite FILE29 log odds\')\n    return expit(raw)\n'
_COMPONENT_SUPPORT_FILE_HELPER_CACHE = None
_COMPONENT_SUPPORT_FILE_BACKEND_CACHE = None


def _sub55_helpers():
    from types import SimpleNamespace
    global _COMPONENT_SUPPORT_FILE_HELPER_CACHE
    if _COMPONENT_SUPPORT_FILE_HELPER_CACHE is None:
        namespace = {"baseline": SimpleNamespace(**_sub54_helpers())}
        exec(compile(COMPONENT_SUPPORT_FILE_COMPILED_SOURCE, "<sub55-component-support-file>", "exec"), namespace)
        _COMPONENT_SUPPORT_FILE_HELPER_CACHE = namespace
    return _COMPONENT_SUPPORT_FILE_HELPER_CACHE


def _validate_component_support_backend(model):
    _sub55_helpers()["validate_model"](model)


def _load_component_support_backend():
    import hashlib
    import json
    global _COMPONENT_SUPPORT_FILE_BACKEND_CACHE
    if _COMPONENT_SUPPORT_FILE_BACKEND_CACHE is not None:
        return _COMPONENT_SUPPORT_FILE_BACKEND_CACHE
    expected = COMPONENT_SUPPORT_FILE_BACKEND_SHA256
    if (type(expected) is not str or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("frozen component-support FILE SHA required")
    try:
        with COMPONENT_SUPPORT_FILE_BACKEND_PATH.open("rb") as handle:
            raw = handle.read(1_000_000)
    except OSError as error:
        raise ValueError(f"missing component-support FILE model: {error}") from error
    if len(raw) >= 1_000_000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("bounded component-support FILE model/hash mismatch")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate component-support JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite component-support JSON constant")
    model = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _validate_component_support_backend(model)
    if model["fit_receipt"]["role"] != "full":
        raise ValueError("deployed component-support FILE requires full fit")
    _COMPONENT_SUPPORT_FILE_BACKEND_CACHE = model
    return model


def predict_component_support_file_fake(original13, eliya_fake, spectra_scores, music17, voice38, backend=None):
    if (not isinstance(original13, (tuple, list)) or len(original13) != 13
            or not all(_sub37_valid_probability(v) for v in original13)):
        raise ValueError("component-support FILE requires healthy original13")
    if (type(spectra_scores) is not list or not 1 <= len(spectra_scores) <= 3
            or not all(_sub37_valid_probability(v) for v in spectra_scores)
            or spectra_scores[0] != original13[3]):
        raise ValueError("component-support FILE original S0 binding mismatch")
    helper = _sub55_helpers()
    row = dict(zip(helper["SCORE_FIELDS"], original13))
    row.update(eliya_fake=eliya_fake, spectra_scores=list(spectra_scores), music17=music17,
               voice38=voice38, evidence_status="ok", evidence_error="")
    model = _load_component_support_backend() if backend is None else backend
    score = float(helper["predict"](model, [row])[0])
    if not _sub37_valid_probability(score):
        raise ValueError("invalid component-support FILE probability")
    return score


def _sub55_final_file(current_file, original13, evidence, music_score, voice38, file54_state, diagnostics=None):
    if type(diagnostics) is dict:
        diagnostics.update(status="unchanged_original_failure", candidate_attempts=0)
    if (not _sub37_valid_probability(current_file) or type(file54_state) is not dict
            or file54_state.get("status") != "ok"
            or type(file54_state.get("candidate_attempts")) is not int
            or file54_state["candidate_attempts"] != 1):
        return current_file
    try:
        if type(diagnostics) is dict:
            diagnostics.update(status="new_failure", candidate_attempts=1)
        if (type(evidence) is not dict or evidence.get("temporal_ok") is not True
                or evidence.get("eliya_ok") is not True):
            raise ValueError("complete original50 evidence required")
        score = predict_component_support_file_fake(original13, evidence["eliya_fake"],
            evidence["spectra_scores"], music_score, voice38)
        if not _sub37_valid_probability(score):
            raise ValueError("invalid component-support FILE result")
    except (Exception, SystemExit) as error:
        print(f"WARNING: component-support FILE failed: {error}; retaining raw54 FILE", file=sys.stderr)
        return current_file
    if type(diagnostics) is dict:
        diagnostics.update(status="ok", candidate_attempts=1)
    return score


# BEGIN SUB68 FT MUSIC HELPERS
def ftm_validate_model(model):
    """Strict fine-tuned fakeprint MUSIC model check; raises ValueError."""
    if type(model) is not dict or model.get("format") != "ft-music-fakeprint-v1":
        raise ValueError("fine-tuned MUSIC model format mismatch")
    if model.get("model_type") not in ("lr", "lr+cnn"):
        raise ValueError("fine-tuned MUSIC model type mismatch")
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = float(model["bias"])
    if weights.shape != (3585,) or not np.isfinite(weights).all() or not np.isfinite(bias):
        raise ValueError("fine-tuned MUSIC linear weights invalid")
    cnn = model.get("cnn")
    if model["model_type"] == "lr+cnn":
        if type(cnn) is not dict:
            raise ValueError("fine-tuned MUSIC residual missing")
        c = int(cnn["channels"])
        k = int(cnn["kernel"])
        d = int(cnn["dilation"])
        shapes = {"conv1_w": (c, 1, k), "conv1_b": (c,), "conv2_w": (c, c, k),
                  "conv2_b": (c,), "head_w": (1, 2 * c), "head_b": (1,)}
        if not (1 <= c <= 64 and k % 2 == 1 and 1 <= k <= 63 and 1 <= d <= 16):
            raise ValueError("fine-tuned MUSIC residual geometry invalid")
        for name, shape in shapes.items():
            value = np.asarray(cnn[name], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("fine-tuned MUSIC residual tensor invalid: " + name)
    elif cnn is not None:
        raise ValueError("linear fine-tuned MUSIC model carries a residual")
    return True


def ftm_prepare(model):
    ftm_validate_model(model)
    prepared = {"weights": np.asarray(model["weights"], dtype=np.float64),
                "bias": float(model["bias"]), "cnn": None}
    if model["model_type"] == "lr+cnn":
        cnn = model["cnn"]
        prepared["cnn"] = {name: np.asarray(cnn[name], dtype=np.float64)
                           for name in ("conv1_w", "conv1_b", "conv2_w", "conv2_b", "head_w", "head_b")}
        prepared["cnn"]["dilation"] = int(cnn["dilation"])
    return prepared


def _ftm_conv1d_same(x, weight, bias, dilation):
    out_channels, in_channels, kernel = weight.shape
    pad = dilation * (kernel // 2)
    padded = np.pad(x, ((0, 0), (pad, pad)))
    length = x.shape[1]
    taps = np.stack([padded[:, j * dilation:j * dilation + length] for j in range(kernel)], axis=-1)
    return np.einsum("ilk,oik->ol", taps, weight) + bias[:, None]


def _ftm_gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def ftm_residual(prepared, fakeprint):
    cnn = prepared["cnn"]
    if cnn is None:
        return 0.0
    x = np.asarray(fakeprint, dtype=np.float64).reshape(1, -1)
    h = _ftm_gelu(_ftm_conv1d_same(x, cnn["conv1_w"], cnn["conv1_b"], 1))
    h = _ftm_gelu(_ftm_conv1d_same(h, cnn["conv2_w"], cnn["conv2_b"], cnn["dilation"]))
    pooled = np.concatenate([h.mean(axis=1), h.max(axis=1)])
    return float(cnn["head_w"][0] @ pooled + cnn["head_b"][0])


def ftm_logit(prepared, fakeprint):
    x = np.asarray(fakeprint, dtype=np.float64).reshape(-1)
    if x.shape != (3585,) or not np.isfinite(x).all():
        raise ValueError("fine-tuned MUSIC needs a finite 3585-bin fakeprint")
    z = float(prepared["weights"] @ x + prepared["bias"]) + ftm_residual(prepared, x)
    if not np.isfinite(z):
        raise ValueError("fine-tuned MUSIC logit is not finite")
    return z


def ftm_probability(prepared, fakeprint):
    z = ftm_logit(prepared, fakeprint)
    if z >= 0.0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = np.exp(z)
    return float(e / (1.0 + e))

def load_ft_music_model():
    """Digest-pinned, size-bounded, strictly validated fine-tuned MUSIC model."""
    import hashlib
    with FT_MUSIC_PATH.open("rb") as handle:
        raw = handle.read(FT_MUSIC_MAX_BYTES + 1)
    if len(raw) > FT_MUSIC_MAX_BYTES:
        raise ValueError("fine-tuned MUSIC model exceeds size bound")
    if hashlib.sha256(raw).hexdigest() != FT_MUSIC_SHA256:
        raise ValueError("fine-tuned MUSIC model digest mismatch")
    return ftm_prepare(json.loads(raw))


def load_ft_music_seg_model():
    """sub97b: digest-pinned, size-bounded segment-branch FT MUSIC head."""
    import hashlib
    with FT_MUSIC_SEG_PATH.open("rb") as handle:
        raw = handle.read(FT_MUSIC_MAX_BYTES + 1)
    if len(raw) > FT_MUSIC_MAX_BYTES:
        raise ValueError("segment FT MUSIC model exceeds size bound")
    if hashlib.sha256(raw).hexdigest() != FT_MUSIC_SEG_SHA256:
        raise ValueError("segment FT MUSIC model digest mismatch")
    return ftm_prepare(json.loads(raw))


def _sub68_logit_blend(current, score, weight):
    current = float(np.clip(current, NII_LOGIT_EPS, 1.0 - NII_LOGIT_EPS))
    score = float(np.clip(score, NII_LOGIT_EPS, 1.0 - NII_LOGIT_EPS))
    combined = (1.0 - weight) * np.log(current / (1.0 - current))
    combined += weight * np.log(score / (1.0 - score))
    if combined >= 0.0:
        return float(1.0 / (1.0 + np.exp(-combined)))
    exp_positive = np.exp(combined)
    return float(exp_positive / (1.0 + exp_positive))


def _sub68_final_music(prepared, audio_path, current_music):
    """New MUSIC or None; None (any failure) keeps the exact source MUSIC."""
    try:
        if prepared is None or not _sub37_valid_probability(current_music):
            return None
        fakeprint = make_lofcz_fakeprint(load_lofcz_audio(audio_path))
        score = ftm_probability(prepared, fakeprint)
        if not _sub37_valid_probability(score):
            raise ValueError("fine-tuned MUSIC returned invalid probability")
        candidate = _sub68_logit_blend(current_music, score, FT_MUSIC_BLEND_WEIGHT)
        if not _sub37_valid_probability(candidate):
            raise ValueError("fine-tuned MUSIC blend invalid")
        return candidate
    except (Exception, SystemExit) as ft_music_error:
        print(f"WARNING: fine-tuned MUSIC failed: {ft_music_error}; retaining source MUSIC",
              file=sys.stderr)
        return None
# END SUB68 FT MUSIC HELPERS


def _sub80_window_starts(n, width, hop):
    if n <= width:
        return [0]
    last = n - width
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _sub80_segment_ft_probability(prepared, audio_path, segment_music):
    """The eval-v2 PANNs overlap gate; None keeps sub70 whole-file MUSIC."""
    if prepared is None or not segment_music:
        return None
    pstarts = np.asarray(segment_music["starts"], dtype=np.int64)
    pscores = np.asarray(segment_music["scores"], dtype=np.float64)
    if (pstarts.ndim != 1 or pstarts.size == 0 or pscores.shape != pstarts.shape
            or not np.isfinite(pscores).all() or np.any(pscores < 0.0)
            or np.any(pscores > 1.0)):
        raise ValueError("Invalid PANNs segment presence")
    audio = load_lofcz_audio(audio_path)
    width, hop = 4 * AUDIO_SAMPLE_RATE, 2 * AUDIO_SAMPLE_RATE
    logits, presence = [], []
    for start in _sub80_window_starts(len(audio), width, hop):
        end = start + width
        overlap = np.maximum(0, np.minimum(end, pstarts + SEGMENT_SAMPLES)
                             - np.maximum(start, pstarts))
        if overlap.sum() <= 0:
            raise ValueError("PANNs segments do not cover fakeprint window")
        presence.append(float(np.sum(overlap * pscores) / np.sum(overlap)))
        logits.append(float(ftm_logit(prepared, make_lofcz_fakeprint(audio[start:end]))))
    weights = np.asarray(presence, dtype=np.float64)
    if not np.isfinite(weights).all() or weights.max() < 0.20:
        return None
    keep = weights >= 0.80 * weights.max()
    if not keep.any() or weights[keep].sum() <= 0:
        return None
    z = float(np.average(np.asarray(logits, dtype=np.float64)[keep], weights=weights[keep]))
    if not np.isfinite(z):
        raise ValueError("Invalid gated MUSIC logit")
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = np.exp(z)
    return float(e / (1.0 + e))


def _sub80_final_music(prepared, audio_path, music17, segment_music):
    """Exploratory MUSIC output; any failure retains sub70 whole-file MUSIC."""
    try:
        if not _sub37_valid_probability(music17):
            return None
        score = _sub80_segment_ft_probability(prepared, audio_path, segment_music)
        if score is None or not _sub37_valid_probability(score):
            return None
        candidate = _sub68_logit_blend(music17, score, FT_MUSIC_BLEND_WEIGHT)
        if not _sub37_valid_probability(candidate):
            raise ValueError("Invalid segment MUSIC blend")
        return candidate
    except (Exception, SystemExit) as error:
        print(f"WARNING: segment MUSIC failed: {error}; retaining sub70 MUSIC",
              file=sys.stderr)
        return None


SUB90_VOICE_BLEND_WEIGHT = 0.5


# sub99: trained window VOICE head (arm A_free of sub99_plan.json; heads sha256 1f3585b0f8bfe85ebfe19dc03417f178d79732ee359cd553df19148ec375e3e3).
SUB99_VOICE_HEAD_COEF = (0.24235476163554562, 0.20492590790397378, 0.1729559943355824, 0.5537457813724879)
SUB99_VOICE_HEAD_INTERCEPT = -0.2735022303019513


def _sub99_window_logit(df, spectra, nii, eliya):
    """Logistic head over the four window logits; replaces the fixed composition."""
    z = SUB99_VOICE_HEAD_INTERCEPT
    for c, value in zip(SUB99_VOICE_HEAD_COEF, (df, spectra, nii, eliya)):
        p = float(np.clip(value, NII_LOGIT_EPS, 1.0 - NII_LOGIT_EPS))
        z += c * float(np.log(p / (1.0 - p)))
    if not np.isfinite(z):
        raise ValueError("Invalid segment VOICE head logit")
    return z


def _sub90_segment_voice(models, voice_audio, segment_presence, device):
    """PANNs speech-gated vocal-stem windows; None keeps the exact sub70 VOICE."""
    _SUB94_CAPTURE.clear()
    df_model, fake_index, spectra_session, nii_model, eliya_model = models
    if nii_model is None or eliya_model is None or not segment_presence:
        return None
    if "voice_scores" not in segment_presence:
        return None
    pstarts = np.asarray(segment_presence["starts"], dtype=np.int64)
    pscores = np.asarray(segment_presence["voice_scores"], dtype=np.float64)
    if (pstarts.ndim != 1 or pstarts.size == 0 or pscores.shape != pstarts.shape
            or not np.isfinite(pscores).all() or np.any(pscores < 0.0)
            or np.any(pscores > 1.0)):
        raise ValueError("Invalid PANNs segment voice presence")
    waveform = np.asarray(voice_audio)
    if (waveform.ndim != 1 or waveform.size == 0 or waveform.dtype != np.float32
            or not np.isfinite(waveform).all()):
        raise ValueError("Segment VOICE requires finite 1D float32 stem")
    width = SPECTRA_SEGMENT_SAMPLES
    starts = _sub80_window_starts(waveform.size, width, width // 2)
    presence = []
    for start in starts:
        overlap = np.maximum(0, np.minimum(start + width, pstarts + SEGMENT_SAMPLES)
                             - np.maximum(start, pstarts))
        if overlap.sum() <= 0:
            raise ValueError("PANNs segments do not cover VOICE window")
        presence.append(float(np.sum(overlap * pscores) / np.sum(overlap)))
    weights = np.asarray(presence, dtype=np.float64)
    if not np.isfinite(weights).all() or weights.max() < 0.20:
        return None
    keep = np.flatnonzero(weights >= 0.80 * weights.max())
    if keep.size == 0 or weights[keep].sum() <= 0:
        return None
    logits = []
    for i in keep:
        crop = waveform[starts[i] : starts[i] + width]
        df = predict_fake(df_model, fake_index, crop, device, pooling="mean")
        spectra = predict_spectra_fake(spectra_session, crop)
        nii = predict_nii_fake(nii_model, crop, device)
        eliya = predict_eliya_fake(eliya_model, crop, device)
        if not all(_sub37_valid_probability(v) for v in (df, spectra, nii, eliya)):
            raise ValueError("Invalid segment VOICE component")
        _SUB94_CAPTURE.setdefault("windows", []).append(
            (df, spectra, nii, eliya, float(weights[i])))
        window = 0.75 * blend_nii_voice_fake(blend_voice_fake(df, spectra), nii) + 0.25 * float(eliya)
        window = float(np.clip(window, NII_LOGIT_EPS, 1.0 - NII_LOGIT_EPS))
        logits.append(np.log(window / (1.0 - window)))
        logits[-1] = _sub99_window_logit(df, spectra, nii, eliya)
    z = float(np.average(np.asarray(logits, dtype=np.float64), weights=weights[keep]))
    if not np.isfinite(z):
        raise ValueError("Invalid gated VOICE logit")
    _SUB94_CAPTURE["ok"] = True
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = np.exp(z)
    return float(e / (1.0 + e))


def _sub90_final_voice(models, voice_audio, voice38, segment_presence, device):
    """Exploratory VOICE output; any failure retains the entire sub70 VOICE."""
    try:
        if not _sub37_valid_probability(voice38):
            return None
        score = _sub90_segment_voice(models, voice_audio, segment_presence, device)
        if score is None or not _sub37_valid_probability(score):
            return None
        candidate = _sub68_logit_blend(voice38, score, SUB90_VOICE_BLEND_WEIGHT)
        if not _sub37_valid_probability(candidate):
            raise ValueError("Invalid segment VOICE blend")
        return candidate
    except (Exception, SystemExit) as error:
        print(f"WARNING: segment VOICE failed: {error}; retaining sub70 VOICE",
              file=sys.stderr)
        return None


# BEGIN SUB94 SEGMENT FILE HELPERS
# sub94: window components recorded by the sub90 VOICE loop for the current file.
SUB94_FILE_BLEND_WEIGHT = 0.5
_SUB94_CAPTURE = {}


def _sub94_logit(value):
    p = min(max(float(value), NII_LOGIT_EPS), 1.0 - NII_LOGIT_EPS)
    return float(np.log(p) - np.log1p(-p))


def _sub94_segment_file(original13, music17, capture):
    """Presence-weighted mean of per-window FILE29 logits; None keeps the pre-floor FILE."""
    windows = capture.get("windows") if capture.get("ok") is True else None
    if not windows:
        return None
    o = list(original13)
    voice_present, music_present, file_music = o[9], o[10], o[12]
    logits, weights = [], []
    for df, spectra, nii, eliya, weight in windows:
        prior = blend_nii_voice_fake(blend_voice_fake(df, spectra), nii)
        voice38 = 0.75 * prior + 0.25 * float(eliya)
        window = list(o)
        window[0], window[3], window[4], window[7] = df, spectra, nii, prior
        window[11] = combine_file_fake_score(df, file_music, voice_present, music_present)
        score = predict_component_support_file_fake(window, eliya, [spectra], music17, voice38)
        logits.append(_sub94_logit(score))
        weights.append(float(weight))
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Invalid segment FILE weights")
    z = float(np.average(np.asarray(logits, dtype=np.float64), weights=weights))
    if not np.isfinite(z):
        raise ValueError("Invalid segment FILE logit")
    if z >= 0:
        return float(1.0 / (1.0 + np.exp(-z)))
    e = np.exp(z)
    return float(e / (1.0 + e))


def _sub94_final_file(current_file, original13, music17, file55_state, capture):
    """Exploratory pre-floor FILE; any failure retains the exact sub90 pre-floor FILE."""
    try:
        if (not _sub37_valid_probability(current_file) or type(file55_state) is not dict
                or file55_state.get("status") != "ok" or not _sub37_valid_probability(music17)):
            return None
        score = _sub94_segment_file(original13, music17, capture)
        if score is None or not _sub37_valid_probability(score):
            return None
        candidate = _sub68_logit_blend(current_file, score, SUB94_FILE_BLEND_WEIGHT)
        if not _sub37_valid_probability(candidate):
            raise ValueError("Invalid segment FILE blend")
        return candidate
    except (Exception, SystemExit) as error:
        print(f"WARNING: segment FILE failed: {error}; retaining sub90 FILE", file=sys.stderr)
        return None
# END SUB94 SEGMENT FILE HELPERS


def predict_fake_scores_for_all_files(
    audio_files, submission_rows, presence_scores, device, music_segments=None
):
    df_arena_model, fake_label_index = load_df_arena_model(device)
    artifactnet_session = load_artifactnet_session()
    lofcz_session = load_lofcz_session()
    spectra_session = load_spectra_session(device)
    htdemucs_model = load_htdemucs_model()
    try:
        nii_model = load_nii_model(device)
    except (Exception, SystemExit) as nii_load_error:
        print(
            f"WARNING: NII model failed to load: {nii_load_error}; "
            "retaining sub18 voice scores",
            file=sys.stderr,
        )
        nii_model = None

    # sub38: the new optional model loads only after every old model attempt.
    try:
        eliya_model = load_eliya_model(device)
    except (Exception, SystemExit) as eliya_load_error:
        print(f"WARNING: Eliya model failed to load: {eliya_load_error}; "
              "retaining sub37 final VOICE", file=sys.stderr)
        eliya_model = None

    # sub68: the new optional model loads only after every source model attempt.
    try:
        ft_music_model = load_ft_music_model()
    except (Exception, SystemExit) as ft_music_load_error:
        print(f"WARNING: fine-tuned MUSIC model failed to load: {ft_music_load_error}; "
              "retaining source MUSIC", file=sys.stderr)
        ft_music_model = None

    # sub97b: the segment branch reads its own head; failure keeps release-v2 (exact sub90).
    try:
        ft_music_seg_model = load_ft_music_seg_model()
    except (Exception, SystemExit) as ft_music_seg_error:
        print(f"WARNING: segment MUSIC head failed to load: {ft_music_seg_error}; "
              "retaining release-v2 segment head", file=sys.stderr)
        ft_music_seg_model = ft_music_model

    for index, audio_path in enumerate(tqdm(audio_files, desc="Components")):
        voice_present, music_present = presence_scores[audio_path.stem]
        sub80_output_music = None
        sub90_output_voice = None
        try:
            voice_audio, music_audio = separate_voice_and_music(
                audio_path, htdemucs_model, device
            )
            voice_fake, voice_file_fake = predict_fake(
                df_arena_model, fake_label_index, voice_audio, device, pooling="both"
            )
            music_fake, music_file_fake = predict_fake(
                df_arena_model, fake_label_index, music_audio, device, pooling="both"
            )
            # sub13 starts from the sub11 rollback: MUSIC remains at its proven
            # 25% blend and FILE restores its officially best 50% blend.
            # The branch remains anchored to DF mean, proven superior for FILE.
            file_music_fake = music_file_fake
            df_voice_mean_raw = voice_file_fake
            df_music_mean_raw = music_file_fake
            df_music_max_raw = music_fake
            spectra_raw = None
            nii_raw = None
            artifactnet_raw = None
            lofcz_raw = None
            mixture_audio = None
            try:
                mixture_audio = load_audio(audio_path)
                artifactnet_music_fake = predict_artifactnet_fake(
                    artifactnet_session, mixture_audio
                )
                music_fake = blend_music_fake(music_fake, artifactnet_music_fake)
                file_music_fake = blend_file_music_fake(
                    music_file_fake, artifactnet_music_fake
                )
                artifactnet_raw = artifactnet_music_fake
            except (Exception, SystemExit) as music_error:
                print(
                    f"WARNING: ArtifactNet failed for {audio_path.name}: "
                    f"{music_error}; retaining sub06 music score",
                    file=sys.stderr,
                )
            try:
                lofcz_music_fake = predict_lofcz_fake(
                    lofcz_session, load_lofcz_audio(audio_path)
                )
                music_fake = blend_lofcz_music_fake(
                    music_fake, lofcz_music_fake
                )
                lofcz_raw = lofcz_music_fake
            except (Exception, SystemExit) as lofcz_error:
                print(
                    f"WARNING: lofcz failed for {audio_path.name}: "
                    f"{lofcz_error}; retaining sub15 music score",
                    file=sys.stderr,
                )
            # sub15 raises only submitted VOICE's Spectra share to 62.5%. FILE
            # deliberately keeps DF-Arena so the result remains identifiable.
            voice_fake = voice_file_fake
            try:
                spectra_voice_fake = predict_spectra_fake(
                    spectra_session, voice_audio
                )
                voice_fake = blend_voice_fake(
                    voice_file_fake, spectra_voice_fake
                )
                spectra_raw = spectra_voice_fake
            except (Exception, SystemExit) as voice_error:
                print(
                    f"WARNING: Spectra failed for {audio_path.name}: "
                    f"{voice_error}; retaining sub11 voice score",
                    file=sys.stderr,
                )
            if nii_model is not None:
                try:
                    nii_voice_fake = predict_nii_fake(
                        nii_model, voice_audio, device
                    )
                    voice_fake = blend_nii_voice_fake(
                        voice_fake, nii_voice_fake
                    )
                    nii_raw = nii_voice_fake
                except (Exception, SystemExit) as nii_error:
                    print(
                        f"WARNING: NII failed for {audio_path.name}: "
                        f"{nii_error}; retaining sub18 voice score",
                        file=sys.stderr,
                    )
            # Only this FILE music branch differs from sub09.
            current_file_fake = combine_file_fake_score(
                voice_file_fake, file_music_fake, voice_present, music_present
            )
            # sub21: FILE-only frozen learned fusion; every other head keeps
            # its exact sub20 value. Any missing component or backend problem
            # falls back to the current formula for this file only.
            file_fake = current_file_fake
            # sub39: preserve the original arguments before any final-head override.
            evidence = {}
            original_file13 = (
                df_voice_mean_raw, df_music_mean_raw, df_music_max_raw,
                spectra_raw, nii_raw, artifactnet_raw, lofcz_raw,
                voice_fake, music_fake, voice_present, music_present,
                current_file_fake, file_music_fake,
            )
            original_learned_file_ok = False
            try:
                learned_file_fake = predict_learned_file_fake(
                    df_voice_mean_raw, df_music_mean_raw, df_music_max_raw,
                    spectra_raw, nii_raw, artifactnet_raw, lofcz_raw,
                    voice_fake, music_fake, voice_present, music_present,
                    current_file_fake, file_music_fake,
                )
            except (Exception, SystemExit) as learned_error:
                print(
                    f"WARNING: learned FILE fusion failed for FILE_HEAD: "
                    f"{learned_error}; retaining current FILE score",
                    file=sys.stderr,
                )
            else:
                file_fake = learned_file_fake
                original_learned_file_ok = True
            # sub29: final MUSIC only; FILE consumed incumbent inputs above.
            if (lofcz_raw is not None and np.isfinite(lofcz_raw)
                    and 0.0 <= lofcz_raw <= 1.0):
                music_fake = 0.03125 * music_fake + 0.96875 * lofcz_raw
            # sub37: final VOICE only, after all unchanged FILE and MUSIC work.
            voice_fake = _sub37_final_voice(
                spectra_session, voice_audio, voice_fake,
                df_voice_mean_raw, spectra_raw, nii_raw, evidence=evidence,
            )
            # sub38: final VOICE only, after all original sub37 operations.
            voice_fake = _sub38_final_voice(
                eliya_model, voice_audio, voice_fake,
                df_voice_mean_raw, spectra_raw, nii_raw, device, evidence=evidence,
            )
            # sub39: new FILE only after every protected sub38 operation.
            try:
                file_fake = _sub39_final_file(
                    file_fake, original_file13, original_learned_file_ok, evidence,
                )
            except (Exception, SystemExit) as evidence_error:
                print(f"WARNING: evidence FILE failed: {evidence_error}; retaining sub38 FILE",
                      file=sys.stderr)
            # sub50: file-local success observer for the existing MUSIC17 call.
            sub50_music44_state = {}
            # sub44: MUSIC-only learned mapping after every original39 operation.
            try:
                music_fake = _sub44_final_music(
                    music_fake, original_file13, diagnostics=sub50_music44_state,
                )
            except (Exception, SystemExit) as music_backend_error:
                print(f"WARNING: learned MUSIC failed: {music_backend_error}; retaining sub39 MUSIC",
                      file=sys.stderr)
            # sub50: FILE25 only after all original44 work; no MUSIC rescoring.
            sub52_file50_state = {}
            try:
                file_fake = _sub50_final_file(file_fake, original_file13, original_learned_file_ok,
                    evidence, music_fake, sub50_music44_state, diagnostics=sub52_file50_state)
            except (Exception, SystemExit) as music_stack_file_error:
                print(f"WARNING: music-stacked FILE failed: {music_stack_file_error}; retaining sub44 FILE",
                      file=sys.stderr)
            # sub52: only final FILE changes, after every original50 operation.
            sub54_raw53_state = {}
            try:
                file_fake = _sub52_final_file(file_fake, original_file13, evidence,
                    music_fake, sub52_file50_state, diagnostics=sub54_raw53_state)
            except (Exception, SystemExit) as native_duration_error:
                print(f"WARNING: native-duration FILE boundary failed: {native_duration_error}; retaining sub50 FILE",
                      file=sys.stderr)
            # sub54: only final FILE changes, after every raw53 operation.
            sub54_file_state = {}
            try:
                file_fake = _sub54_final_file(file_fake, original_file13, evidence,
                    music_fake, sub54_raw53_state, diagnostics=sub54_file_state)
            except (Exception, SystemExit) as native_fake_music_error:
                print(f"WARNING: native fake-music FILE boundary failed: {native_fake_music_error}; retaining raw53 FILE",
                      file=sys.stderr)
            # sub55: only final FILE changes, after every raw54 operation.
            sub55_file_state = {}
            try:
                file_fake = _sub55_final_file(file_fake, original_file13, evidence,
                    music_fake, voice_fake, sub54_file_state, diagnostics=sub55_file_state)
            except (Exception, SystemExit) as component_support_error:
                print(f"WARNING: component-support FILE failed: {component_support_error}; retaining raw54 FILE",
                      file=sys.stderr)
            # sub68b: final MUSIC after every protected source FILE/MUSIC operation;
            # the unchanged output floor below then reads this MUSIC.
            music17 = music_fake
            sub68_candidate = _sub68_final_music(ft_music_model, audio_path, music_fake)
            if sub68_candidate is not None:
                music_fake = sub68_candidate
                try:
                    segment_music = (music_segments or {}).get(audio_path.stem)
                    sub80_output_music = _sub80_final_music(
                        ft_music_seg_model, audio_path, music17, segment_music)
                except (Exception, SystemExit) as segment_error:
                    print(f"WARNING: segment MUSIC failed: {segment_error}; retaining sub70 MUSIC",
                          file=sys.stderr)
            _SUB94_CAPTURE.clear()
            # sub90: pending VOICE after all original work; the floor below reads sub70 VOICE.
            sub90_output_voice = _sub90_final_voice(
                (df_arena_model, fake_label_index, spectra_session, nii_model, eliya_model),
                voice_audio, voice_fake, (music_segments or {}).get(audio_path.stem), device)
            # sub94: segment pre-floor FILE from the recorded sub90 windows; the floor reads it.
            sub94_file = _sub94_final_file(file_fake, original_file13, music17,
                                           sub55_file_state, _SUB94_CAPTURE)
            if sub94_file is not None:
                file_fake = sub94_file
        except (Exception, SystemExit) as error:
            print(
                f"WARNING: component inference failed for {audio_path.name}: {error}",
                file=sys.stderr,
            )
            file_fake = 0.5
            voice_fake = 0.5
            music_fake = 0.5
            voice_present = 0.5
            music_present = 0.5

        row = submission_rows[index]
        row["FILE_FAKE_PROB"] = round(file_fake, 10)
        row["VOICE_FAKE_PROB"] = round(voice_fake, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake, 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)
        row["FILE_FAKE_PROB"] = _sub52_light_file(row)
        # Exploratory sub80: publish gated MUSIC only after the exact sub70
        # floor has consumed its whole-file blended MUSIC.
        if sub80_output_music is not None:
            row["MUSIC_FAKE_PROB"] = round(sub80_output_music, 10)
        # Exploratory sub90: publish gated VOICE only after the same floor.
        if sub90_output_voice is not None:
            row["VOICE_FAKE_PROB"] = round(sub90_output_voice, 10)

    return submission_rows


def save_submission(output_path, column_names, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    device = select_device(args.device)

    # 1. 테스트 파일을 제출 양식의 ID 순서에 맞춘다.
    audio_files = find_audio_files(args.test_dir)
    column_names, submission_rows = read_sample_submission(args.sample_submission)
    audio_files = order_audio_files(audio_files, submission_rows)

    # 2. 파일별 음성·음악 존재 확률을 계산한다.
    music_segments = {}
    presence_scores = predict_presence_for_all_files(audio_files, device, music_segments)

    # 3. 음성과 음악을 분리한 뒤 성분별 Fake 확률을 계산한다.
    submission_rows = predict_fake_scores_for_all_files(
        audio_files, submission_rows, presence_scores, device, music_segments
    )

    # 4. 5개 예측값을 제출 파일로 저장한다.
    save_submission(args.output, column_names, submission_rows)
    print(f"Saved {len(submission_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
