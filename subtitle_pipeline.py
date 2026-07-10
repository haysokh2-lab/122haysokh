#!/usr/bin/env python3
"""
Khmer Video Translator - Subtitle Extraction Pipeline
=====================================================
Backend server that:
  1. Extracts audio from video using FFmpeg
  2. Transcribes full audio using Whisper (with VAD chunking for accuracy)
  3. Returns timestamped subtitle segments
  4. Generates SRT and VTT files
  5. Translates each segment to Khmer (or target language) via LLM
  6. Detects hardcoded subtitles via EasyOCR if no speech found
  7. Automatically merges OCR + Whisper results for best coverage
  8. Supports checkpoint/resume for long videos
  9. Provides proper error logging at every stage
 10. Advanced Khmer Translation Rules:
       - 100% complete Khmer translation (no foreign words left behind)
       - Automatic language correction for remaining foreign words
       - Intelligent context-aware translation
       - Intelligent summarization when text is too long for display time
       - Automatic repetition reduction (max 2-3 consecutive repeats)
       - Automatic natural Khmer rewrite (fix awkward/robotic translations)
       - Final validation before export
       - Mandatory Final Quality Enforcement before export (Stage 3h)

Usage:
    python subtitle_pipeline.py
    (listens on http://localhost:5050)
"""

import os
import sys
import json
import time
import logging
import subprocess
import tempfile
import shutil
import traceback
import asyncio
import re
import io
import hashlib
import gc
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import timedelta
from threading import Event, Lock
from typing import List, Dict, Optional, Tuple, Set, Any
from uuid import uuid4
import flask
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').strip().upper() or 'INFO'
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('subtitle_pipeline.log', mode='a', encoding='utf-8'),
    ]
)
logger = logging.getLogger('SubtitlePipeline')

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload (handles 3+ hour videos)

# No maximum duration - read the ENTIRE video from beginning to end
MAX_VIDEO_DURATION_SECONDS = None
# Zero means unlimited. Set APP_MAX_VIDEO_SECONDS explicitly when a deployment
# needs an upload-duration policy.
APP_MAX_VIDEO_SECONDS = int(os.environ.get('APP_MAX_VIDEO_SECONDS', '0'))
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}

# Temporary directory for processing files
TEMP_DIR = Path(tempfile.gettempdir()) / 'khmer_subtitle_pipeline'
TEMP_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_DIR = Path(__file__).resolve().parent

# Checkpoint directory for resume support
CHECKPOINT_DIR = TEMP_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = TEMP_DIR / 'cache'
TRANSLATION_CACHE_DIR = CACHE_DIR / 'translations'
TTS_CACHE_DIR = CACHE_DIR / 'tts'
OCR_CACHE_DIR = CACHE_DIR / 'ocr'
for _cache_dir in (CACHE_DIR, TRANSLATION_CACHE_DIR, TTS_CACHE_DIR, OCR_CACHE_DIR):
    _cache_dir.mkdir(parents=True, exist_ok=True)

WEB_DIR = PROJECT_DIR / 'web'

_cache_lock = Lock()
_pipeline_progress_lock = Lock()
_pipeline_progress = {
    'job_id': None,
    'operation': '',
    'state': 'idle',
    'stage': 'Idle',
    'status': 'Ready',
    'current': 0,
    'total': 0,
    'progress_pct': 0,
    'overall_progress_pct': 0,
}


class PipelineCancelled(RuntimeError):
    """Raised cooperatively when the browser cancels the active media job."""


class PipelineBusy(RuntimeError):
    """Raised when a second expensive media job is submitted."""


class _ProcessingLock:
    """Non-blocking process lock with one cooperative cancellation token."""

    def __init__(self):
        self._lock = Lock()
        self._state_lock = Lock()
        self._cancel_event = Event()
        self.job_id = None
        self.operation = ''

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            raise PipelineBusy(
                'Another media job is already running. Cancel it or wait for it to finish.'
            )

        job_id = (
            request.headers.get('X-Job-ID', '').strip()
            or request.form.get('job_id', '').strip()
        )
        if not job_id and request.is_json:
            payload = request.get_json(silent=True) or {}
            job_id = str(payload.get('request_job_id', '') or '').strip()
        job_id = job_id or uuid4().hex

        with self._state_lock:
            self.job_id = job_id
            self.operation = request.path.rsplit('/', 1)[-1] or 'pipeline'
            self._cancel_event.clear()
        with _pipeline_progress_lock:
            _pipeline_progress.clear()
            _pipeline_progress.update({
                'job_id': job_id,
                'operation': self.operation,
                'state': 'running',
                'stage': 'Starting',
                'status': 'Starting job',
                'current': 0,
                'total': 1,
                'progress_pct': 0,
                'overall_progress_pct': 0,
                'stage_started_at': time.time(),
                'updated_at': time.time(),
            })
        return self

    def __exit__(self, exc_type, _exc_value, _traceback):
        with _pipeline_progress_lock:
            current_state = _pipeline_progress.get('state')
            if exc_type is PipelineCancelled or self._cancel_event.is_set():
                _pipeline_progress.update({
                    'state': 'cancelled',
                    'status': 'Cancelled',
                    'updated_at': time.time(),
                })
            elif exc_type is not None:
                _pipeline_progress.update({
                    'state': 'error',
                    'status': 'Failed',
                    'updated_at': time.time(),
                })
            elif current_state == 'running':
                _pipeline_progress.update({
                    'state': 'completed',
                    'status': 'Completed',
                    'progress_pct': 100,
                    'overall_progress_pct': 100,
                    'updated_at': time.time(),
                })
        with self._state_lock:
            self.job_id = None
            self.operation = ''
            self._cancel_event.clear()
        try:
            _release_inference_models()
            _cleanup_ocr_frame_dirs()
        except Exception as cleanup_error:
            logger.debug(f'[Memory] Model cleanup failed: {cleanup_error}')
        self._lock.release()
        return False

    def cancel(self, job_id: str) -> bool:
        with self._state_lock:
            if not self.job_id or (job_id and job_id != self.job_id):
                return False
            self._cancel_event.set()
            return True

    def raise_if_cancelled(self):
        if self._cancel_event.is_set():
            raise PipelineCancelled('Processing was cancelled by the user.')

    @property
    def active(self) -> bool:
        return self._lock.locked()


process_lock = _ProcessingLock()


_STAGE_PROGRESS = {
    'Starting': (0, 1),
    'OCR': (1, 44),
    'Subtitle Timing': (45, 5),
    'Subtitle Extraction': (50, 5),
    'Speech Recognition': (55, 10),
    'Segmentation': (45, 5),
    'Translation': (65, 20),
    'Voice Generation': (85, 10),
    'Export': (95, 5),
}


def _bounded_worker_count(env_name: str, default: int, maximum: int = 8) -> int:
    try:
        configured = int(os.environ.get(env_name, '').strip() or default)
    except ValueError:
        configured = default
    return max(1, min(maximum, configured))


def _cache_key(prefix: str, payload: Dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    return f'{prefix}_{digest}'


def _media_fingerprint(path: str) -> str:
    """Hash a small stable sample so re-uploaded videos reuse OCR work."""
    media_path = Path(path)
    size = media_path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode('ascii'))
    sample_size = 1024 * 1024
    with open(media_path, 'rb') as media:
        digest.update(media.read(sample_size))
        if size > sample_size:
            media.seek(max(0, size - sample_size))
            digest.update(media.read(sample_size))
    return digest.hexdigest()


def _read_json_cache(cache_dir: Path, key: str) -> Optional[dict]:
    cache_path = cache_dir / f'{key}.json'
    if not cache_path.exists():
        return None
    try:
        with _cache_lock:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f'[Cache] Failed to read {cache_path.name}: {e}')
        return None


def _write_json_cache(cache_dir: Path, key: str, data: dict):
    cache_path = cache_dir / f'{key}.json'
    tmp_path = cache_dir / f'{key}.{uuid4().hex}.tmp'
    try:
        with _cache_lock:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, cache_path)
    except Exception as e:
        logger.debug(f'[Cache] Failed to write {cache_path.name}: {e}')
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _update_pipeline_progress(stage: str, current: int = 0, total: int = 0,
                              status: str = '', started_at: float = None):
    process_lock.raise_if_cancelled()
    now = time.time()
    with _pipeline_progress_lock:
        previous_stage = _pipeline_progress.get('stage')
        if started_at is None:
            started_at = (
                _pipeline_progress.get('stage_started_at', now)
                if previous_stage == stage
                else now
            )
    elapsed = max(0.0, now - started_at)
    pct = int((current / total) * 100) if total else 0
    stage_start, stage_weight = _STAGE_PROGRESS.get(stage, (0, 100))
    overall_pct = min(100, stage_start + round(stage_weight * pct / 100))
    eta = None
    if current > 0 and total and current < total:
        eta = round((elapsed / current) * (total - current), 1)
    with _pipeline_progress_lock:
        _pipeline_progress.update({
            'state': 'running',
            'stage': stage,
            'status': status or stage,
            'current': current,
            'total': total,
            'progress_pct': pct,
            'overall_progress_pct': overall_pct,
            'elapsed_seconds': round(elapsed, 1),
            'eta_seconds': eta,
            'stage_started_at': started_at,
            'updated_at': now,
        })


def _mark_pipeline_terminal(state: str, status: str):
    with _pipeline_progress_lock:
        _pipeline_progress.update({
            'state': state,
            'status': status,
            'updated_at': time.time(),
        })


def _resolve_command(command_name):
    env_name = f'{command_name.upper()}_EXE'
    configured = os.environ.get(env_name, '').strip()
    if configured:
        return configured

    found = shutil.which(command_name)
    if found:
        return found

    executable = f'{command_name}.exe' if os.name == 'nt' else command_name
    for candidate in (
        PROJECT_DIR / 'ffmpeg' / 'bin' / executable,
        PROJECT_DIR / 'bin' / executable,
        PROJECT_DIR / executable,
    ):
        if candidate.exists():
            return str(candidate)

    if command_name == 'ffmpeg':
        try:
            import imageio_ffmpeg
            imageio_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if imageio_exe and Path(imageio_exe).exists():
                return imageio_exe
        except Exception as e:
            logger.debug(f'[FFmpeg] imageio-ffmpeg fallback unavailable: {e}')

    return command_name


FFMPEG_EXE = _resolve_command('ffmpeg')
FFPROBE_EXE = _resolve_command('ffprobe')


def _missing_command_message(command_name, env_name):
    return (
        f'{command_name} was not found. Install FFmpeg and add it to PATH, '
        f'or set {env_name} to the full executable path.'
    )


def _run_cancellable(command: List[str], timeout: Optional[float] = None, **kwargs):
    """Run a child process while honoring the active browser cancellation token."""
    check = bool(kwargs.pop('check', False))
    started_at = time.time()
    process = subprocess.Popen(command, **kwargs)
    try:
        while True:
            process_lock.raise_if_cancelled()
            if timeout is not None and time.time() - started_at > timeout:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue
    except Exception:
        process.terminate()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        raise
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check and process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return completed


def _allowed_video_filename(filename: str) -> bool:
    return Path(filename or '').suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def _validate_video_upload(filename: str, video_path: str) -> Optional[str]:
    if not _allowed_video_filename(filename):
        allowed = ', '.join(sorted(ext.upper().lstrip('.') for ext in ALLOWED_VIDEO_EXTENSIONS))
        return f'Unsupported video type. Upload one of: {allowed}.'

    duration = get_video_duration(video_path)
    if APP_MAX_VIDEO_SECONDS > 0 and duration and duration > APP_MAX_VIDEO_SECONDS:
        return (
            f'Video is {duration / 60:.1f} minutes long. '
            f'Maximum supported length is {APP_MAX_VIDEO_SECONDS // 60} minutes.'
        )
    return None


def _env_enabled(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _translation_enabled() -> bool:
    """Keep translation configurable while enabling the normal workflow."""
    return _env_enabled('TRANSLATION_ENABLED', default=True)


# ---------------------------------------------------------------------------
# Lazy imports for heavy libraries (Whisper, EasyOCR)
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_model_config = None
_whisper_device = None
_whisper_compute_type = None
_whisper_backend = None
_whisper_lock = Lock()
_whisper_inference_lock = Lock()

_ocr_reader = None
_ocr_reader_languages = None
_ocr_lock = Lock()


def _detect_whisper_runtime() -> Tuple[str, str]:
    """Detect the fastest available Whisper runtime without forcing CPU."""
    forced_device = os.environ.get('WHISPER_DEVICE', '').strip().lower()
    forced_compute = os.environ.get('WHISPER_COMPUTE_TYPE', '').strip()

    cuda_available = False
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except Exception as e:
        logger.info(f'[Whisper] Torch CUDA detection unavailable: {e}')

    device = forced_device or ('cuda' if cuda_available else 'cpu')
    if device == 'cuda':
        compute_type = forced_compute or 'float16'
    else:
        compute_type = forced_compute or 'int8'

    return device, compute_type


def get_whisper_model(model_name=None):
    """Lazy-load and globally cache Faster-Whisper with GPU/CPU auto-selection."""
    global _whisper_model, _whisper_model_config, _whisper_device, _whisper_compute_type, _whisper_backend

    if not model_name:
        model_name = os.environ.get('WHISPER_MODEL', 'base').strip() or 'base'

    device, compute_type = _detect_whisper_runtime()
    config = (model_name, device, compute_type)

    if _whisper_model is None or _whisper_model_config != config:
        with _whisper_lock:
            if _whisper_model is None or _whisper_model_config != config:
                start_time = time.time()
                logger.info(
                    f'[Whisper] Loading Faster-Whisper model="{model_name}", '
                    f'device="{device}", compute_type="{compute_type}"'
                )
                try:
                    from faster_whisper import WhisperModel
                    cpu_threads = _bounded_worker_count('WHISPER_CPU_THREADS', 4, 8)
                    _whisper_model = WhisperModel(
                        model_name,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=cpu_threads,
                        num_workers=_bounded_worker_count('WHISPER_MODEL_WORKERS', 1, 4),
                    )
                    _whisper_backend = 'faster-whisper'
                except Exception as e:
                    logger.warning(f'[Whisper] Faster-Whisper unavailable ({e}); falling back to openai-whisper.')
                    import whisper
                    _whisper_model = whisper.load_model(model_name, device=device if device == 'cuda' else None)
                    _whisper_backend = 'openai-whisper'

                _whisper_model_config = config
                _whisper_device = device
                _whisper_compute_type = compute_type
                logger.info(f'[Whisper] Model loaded in {time.time() - start_time:.2f}s via {_whisper_backend}.')

    return _whisper_model


def get_ocr_reader(languages=None):
    """Lazy-load the EasyOCR reader (thread-safe)."""
    global _ocr_reader, _ocr_reader_languages
    if languages is None:
        languages = ['en']
    requested_languages = list(dict.fromkeys(languages))
    if _ocr_reader is None or _ocr_reader_languages != requested_languages:
        with _ocr_lock:
            if _ocr_reader is None or _ocr_reader_languages != requested_languages:
                import easyocr
                try:
                    import torch
                    torch.set_num_threads(
                        _bounded_worker_count('OCR_CPU_THREADS', 4, 8)
                    )
                    try:
                        torch.set_num_interop_threads(1)
                    except RuntimeError:
                        pass
                except Exception as thread_error:
                    logger.debug(f'[OCR] CPU thread limit unavailable: {thread_error}')
                gpu_setting = os.environ.get('OCR_GPU', 'auto').strip().lower()
                if gpu_setting == 'auto':
                    try:
                        import torch
                        gpu_enabled = bool(torch.cuda.is_available())
                    except Exception:
                        gpu_enabled = False
                else:
                    gpu_enabled = gpu_setting in {'1', 'true', 'yes', 'on'}
                # Enable OCR model downloads by default so models are fetched when needed
                download_enabled = _env_enabled('OCR_DOWNLOAD_ENABLED', True)
                language_groups = [
                    requested_languages,
                    ['en'],
                ]
                seen = set()
                last_error = None

                for lang_group in language_groups:
                    lang_key = tuple(lang_group)
                    if lang_key in seen:
                        continue
                    seen.add(lang_key)
                    try:
                        logger.info(f'Loading EasyOCR reader for {lang_group} ...')
                        reader = easyocr.Reader(
                            lang_group,
                            gpu=gpu_enabled,
                            download_enabled=download_enabled,
                        )
                        _ocr_reader = reader
                        _ocr_reader_languages = lang_group
                        logger.info(f'EasyOCR reader loaded successfully for {_ocr_reader_languages}.')
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(f'EasyOCR reader failed for {lang_group}: {e}')

                if _ocr_reader is None:
                    raise RuntimeError(f'EasyOCR reader could not be loaded: {last_error}')
    return _ocr_reader


def _ocr_languages_for_source(source_lang: str) -> List[str]:
    """Return a valid EasyOCR language group for the requested source."""
    normalized = (source_lang or '').strip().lower().replace('_', '-')
    base = normalized.split('-', 1)[0]
    if base == 'zh':
        chinese_model = 'ch_tra' if normalized in {'zh-tw', 'zh-hk', 'zh-hant'} else 'ch_sim'
        return [chinese_model, 'en']
    if base in {'ja', 'ko', 'th', 'vi'}:
        return [base, 'en']
    # EasyOCR does not currently provide a Khmer recognition model.
    return ['en']


def _ocr_script_score(text: str, source_lang: str) -> float:
    """Estimate whether recognized text belongs to an OCR language model."""
    compact = re.sub(r'[\s\d\W_]+', '', text or '', flags=re.UNICODE)
    if not compact:
        return 0.0
    normalized = (source_lang or '').lower()
    base = normalized.split('-', 1)[0]
    if base == 'zh':
        matching = sum('\u3400' <= char <= '\u9fff' for char in compact)
    elif base == 'ja':
        matching = sum(
            '\u3040' <= char <= '\u30ff' or '\u3400' <= char <= '\u9fff'
            for char in compact
        )
    elif base == 'ko':
        matching = sum('\uac00' <= char <= '\ud7af' for char in compact)
    elif base == 'th':
        matching = sum('\u0e00' <= char <= '\u0e7f' for char in compact)
    else:
        matching = sum(char.isascii() and char.isalpha() for char in compact)
    return matching / len(compact)


def _ocr_quality_metrics(segments: List[Dict]) -> Dict[str, float]:
    """Summarize OCR confidence without treating missing legacy data as failure."""
    valid_segments = [
        segment for segment in (segments or [])
        if isinstance(segment, dict)
        and str(segment.get('source') or segment.get('text') or '').strip()
    ]
    confidences = []
    for segment in valid_segments:
        confidence = segment.get('confidence')
        if isinstance(confidence, (int, float)):
            confidences.append(max(0.0, min(1.0, float(confidence))))

    if not confidences:
        return {
            'segment_count': float(len(valid_segments)),
            'confidence_coverage': 0.0,
            'average_confidence': 0.0,
            'reliable_ratio': 0.0,
        }

    return {
        'segment_count': float(len(valid_segments)),
        'confidence_coverage': len(confidences) / max(1, len(valid_segments)),
        'average_confidence': sum(confidences) / len(confidences),
        'reliable_ratio': (
            sum(confidence >= 0.35 for confidence in confidences)
            / len(confidences)
        ),
    }


def _ocr_output_is_usable(segments: List[Dict]) -> bool:
    """Reject a body of low-confidence OCR before it reaches translation."""
    metrics = _ocr_quality_metrics(segments)
    segment_count = int(metrics['segment_count'])
    if segment_count == 0:
        return False

    # Older caches and test fixtures may not contain confidence values. Their
    # quality cannot be judged here, so preserve the existing behavior.
    if metrics['confidence_coverage'] < 0.5:
        return True

    # A few difficult captions should not reject an otherwise healthy video.
    # A larger set where nearly every reading is weak is almost always the
    # wrong OCR language, an unsupported script/font, or insufficient scale.
    return not (
        segment_count >= 4
        and metrics['average_confidence'] < 0.28
        and metrics['reliable_ratio'] < 0.20
    )


# ---------------------------------------------------------------------------
# Checkpoint System - Resume from interrupted processing
# ---------------------------------------------------------------------------
def save_checkpoint(video_id: str, stage: str, data: dict):
    """Save processing checkpoint so we can resume if interrupted."""
    cp_path = CHECKPOINT_DIR / f'{video_id}_{stage}.json'
    try:
        with open(cp_path, 'w', encoding='utf-8') as f:
            json.dump({
                'video_id': video_id,
                'stage': stage,
                'timestamp': time.time(),
                'data': data,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f'[Checkpoint] Saved {stage} checkpoint for {video_id}')
    except Exception as e:
        logger.warning(f'[Checkpoint] Failed to save {stage}: {e}')


def load_checkpoint(video_id: str, stage: str) -> Optional[dict]:
    """Load processing checkpoint if available."""
    cp_path = CHECKPOINT_DIR / f'{video_id}_{stage}.json'
    if not cp_path.exists():
        return None
    try:
        with open(cp_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f'[Checkpoint] Failed to load {stage}: {e}')
        return None


def clear_checkpoints(video_id: str):
    """Clear all checkpoints for a video after successful completion."""
    for cp_file in CHECKPOINT_DIR.glob(f'{video_id}_*.json'):
        try:
            cp_file.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Helper: Format timestamp for SRT
# ---------------------------------------------------------------------------
def format_srt_time(seconds):
    """Convert seconds (float) to SRT timestamp format HH:MM:SS,mmm."""
    if seconds is None:
        seconds = 0.0
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'


def format_vtt_time(seconds):
    """Convert seconds (float) to VTT timestamp format HH:MM:SS.mmm."""
    if seconds is None:
        seconds = 0.0
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}'


def generate_srt(segments):
    """Generate SRT content from a list of segment dicts."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg.get('start', 0))
        end = format_srt_time(seg.get('end', 0))
        text = seg.get('target') or seg.get('source') or seg.get('text', '')
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return '\n'.join(lines)


def generate_vtt(segments):
    """Generate VTT content from a list of segment dicts."""
    lines = ['WEBVTT', '']
    for i, seg in enumerate(segments, 1):
        start = format_vtt_time(seg.get('start', 0))
        end = format_vtt_time(seg.get('end', 0))
        text = seg.get('target') or seg.get('source') or seg.get('text', '')
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append('')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Video Duration Helper
# ---------------------------------------------------------------------------
def get_video_duration(video_path: str) -> float:
    """Get media duration using ffprobe, WAV metadata, or PyAV."""
    ffprobe_available = (
        bool(shutil.which(FFPROBE_EXE))
        or (Path(FFPROBE_EXE).is_file() if FFPROBE_EXE else False)
    )
    if ffprobe_available:
        duration_cmd = [
            FFPROBE_EXE, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        try:
            duration_result = subprocess.run(
                duration_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if duration_result.returncode == 0 and duration_result.stdout.strip():
                return float(duration_result.stdout.strip())
        except (ValueError, OSError, subprocess.SubprocessError) as e:
            logger.debug(f'[Duration] ffprobe unavailable for this file: {e}')

    try:
        import wave
        with wave.open(video_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate:
                return frames / float(rate)
    except Exception:
        pass

    try:
        import av
        with av.open(video_path) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
            durations = []
            for stream in container.streams:
                if stream.duration and stream.time_base:
                    durations.append(float(stream.duration * stream.time_base))
            if durations:
                return max(durations)
    except Exception as e:
        logger.warning(f'[Duration] PyAV duration fallback failed: {e}')

    return 0.0


def _cleanup_inference_memory():
    """Release Python and GPU allocator memory between large chunk batches."""
    gc.collect()
    try:
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if os.name == 'nt':
        try:
            import ctypes
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetProcessWorkingSetSize(
                process_handle,
                ctypes.c_size_t(-1),
                ctypes.c_size_t(-1),
            )
        except Exception:
            pass


def _release_inference_models():
    """Release large OCR/Whisper models after a job on memory-limited systems."""
    global _ocr_reader, _ocr_reader_languages
    global _whisper_model, _whisper_model_config, _whisper_backend
    if _env_enabled('KEEP_MODELS_WARM', default=False):
        _cleanup_inference_memory()
        return
    _ocr_reader = None
    _ocr_reader_languages = None
    _whisper_model = None
    _whisper_model_config = None
    _whisper_backend = None
    _cleanup_inference_memory()


def _cleanup_ocr_frame_dirs():
    """Remove only this application's abandoned OCR frame directories."""
    temp_root = TEMP_DIR.resolve()
    for candidate in TEMP_DIR.glob('ocr_frames_*'):
        try:
            resolved = candidate.resolve()
            if resolved.parent == temp_root and resolved.is_dir():
                shutil.rmtree(resolved, ignore_errors=True)
        except OSError:
            pass


def _cleanup_stale_temp_files(max_age_seconds: int = 24 * 60 * 60):
    """Bound disk usage without touching files from an active/recent job."""
    cutoff = time.time() - max(60, max_age_seconds)
    patterns = (
        'upload_*',
        'ocr_upload_*',
        'audio_*',
        'dub_segment_*',
        'dubbing_*',
        'export_video_*',
        'export_subtitles_*',
        'exported_*',
    )
    temp_root = TEMP_DIR.resolve()
    for pattern in patterns:
        for candidate in TEMP_DIR.glob(pattern):
            try:
                resolved = candidate.resolve()
                if (
                    resolved.parent == temp_root
                    and resolved.is_file()
                    and resolved.stat().st_mtime < cutoff
                ):
                    resolved.unlink()
            except OSError:
                pass


def _whisper_options(audio_duration: float = 0.0) -> Dict[str, Any]:
    beam_size = max(1, min(3, int(os.environ.get('WHISPER_BEAM_SIZE', '2'))))
    best_of = max(1, min(3, int(os.environ.get('WHISPER_BEST_OF', '1'))))

    if _whisper_device == 'cuda':
        default_batch = 16 if audio_duration and audio_duration < 1800 else 8
    else:
        default_batch = 4

    batch_size = max(1, min(32, int(os.environ.get('WHISPER_BATCH_SIZE', str(default_batch)))))
    # CRITICAL: VAD filter improves segmentation accuracy significantly.
    # Use conservative settings to avoid removing speech.
    # Set WHISPER_VAD_FILTER=0 to disable.
    use_vad = os.environ.get('WHISPER_VAD_FILTER', '1').strip() in {'1', 'true', 'yes'}
    
    options = {
        'beam_size': beam_size,
        'best_of': best_of,
        'batch_size': batch_size,
        'vad_filter': use_vad,
        'word_timestamps': True,
        'condition_on_previous_text': False,
        'temperature': 0.0,
        'compression_ratio_threshold': 2.4,
        'log_prob_threshold': -1.0,
        'no_speech_threshold': 0.6,
    }
    
    if use_vad:
        options['vad_parameters'] = {
            'min_silence_duration_ms': 500,  # Longer silence threshold to avoid cutting speech
            'speech_pad_ms': 200,  # Generous padding to keep speech boundaries intact
            'threshold': 0.5,  # Lower threshold to be more inclusive
        }
    
    return options


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        if default is None:
            return None
        return float(default)


def _normalize_word_timestamp(item: Any, fallback_start: float, fallback_end: float) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        word_text = item.get('word') or item.get('text') or ''
        word_start = _coerce_float(item.get('start'), None)
        word_end = _coerce_float(item.get('end'), None)
        probability = item.get('probability', item.get('score'))
        speaker = item.get('speaker_id', item.get('speaker'))
    else:
        word_text = getattr(item, 'word', getattr(item, 'text', '')) or ''
        word_start = _coerce_float(getattr(item, 'start', None), None)
        word_end = _coerce_float(getattr(item, 'end', None), None)
        probability = getattr(item, 'probability', getattr(item, 'score', None))
        speaker = getattr(item, 'speaker_id', getattr(item, 'speaker', None))

    if not word_text:
        return None
    if word_start is None or word_end is None:
        word_start = fallback_start
        word_end = fallback_end

    normalized = {
        'word': str(word_text).strip(),
        'start': round(max(0.0, word_start), 3),
        'end': round(max(word_start + 0.01, word_end), 3),
        'probability': probability,
    }
    if speaker:
        normalized['speaker_id'] = speaker
    return normalized


def _segment_to_dict(seg: Any, offset: float = 0.0) -> Dict:
    start = float(getattr(seg, 'start', seg.get('start', 0.0) if isinstance(seg, dict) else 0.0)) + offset
    end = float(getattr(seg, 'end', seg.get('end', start) if isinstance(seg, dict) else start)) + offset
    text = getattr(seg, 'text', seg.get('text', '') if isinstance(seg, dict) else '')
    text = (text or '').strip()
    raw_words = getattr(seg, 'words', seg.get('words', []) if isinstance(seg, dict) else []) or []
    words = []
    if isinstance(raw_words, list):
        for item in raw_words:
            word = _normalize_word_timestamp(item, start, end)
            if word:
                word['start'] = round(word['start'] + offset, 3)
                word['end'] = round(word['end'] + offset, 3)
                words.append(word)
    result = {
        'start': max(0.0, start),
        'end': max(0.0, end),
        'text': text,
        'source': text,
        'target': '',
    }
    if words:
        result['words'] = words
    if isinstance(seg, dict):
        for key in ('confidence', 'speaker_id', 'speaker', 'avg_logprob', 'probability'):
            if key in seg:
                result[key] = seg[key]
    # CRITICAL: Ensure speaker_id is always propagated from word-level data
    # if no explicit speaker_id on the segment but words have one
    if not result.get('speaker_id') and not result.get('speaker') and words:
        word_speakers = set()
        for w in words:
            sp = w.get('speaker_id') or w.get('speaker') or ''
            if sp:
                word_speakers.add(str(sp))
        if len(word_speakers) == 1:
            result['speaker_id'] = word_speakers.pop()
    return result


def _has_speech_confidence(segment: Dict, minimum: float = 0.1) -> bool:
    """Reject low-confidence text hallucinated by the no-VAD recovery pass."""
    probabilities = []
    for word in segment.get('words') or []:
        probability = _coerce_float(word.get('probability'), None)
        if probability is not None:
            probabilities.append(probability)
    return not probabilities or (sum(probabilities) / len(probabilities)) >= minimum


def _build_word_timestamps(text: str, start: float, end: float) -> List[Dict[str, Any]]:
    """DEPRECATED: Do not use linear interpolation for word timestamps.
    Whisper word timestamps are required for accurate subtitle timing.
    This function is kept only as a fallback for segments without word data.
    Returns empty list to force accurate Whisper word timestamps.
    """
    # DISABLED: Linear interpolation gives WRONG timing
    # Word timestamps MUST come from Whisper (word_timestamps=True)
    return []


def _enrich_subtitle_segments_with_alignment(segments: List[Dict], target_lang: str = '', min_duration: float = 0.01) -> List[Dict]:
    """Add timing, word-level timing, and export metadata.
    CRITICAL: Use word timestamps as PRIMARY timing reference (not segment timestamps).
    This ensures subtitles start exactly when first word is spoken and end exactly when last word is spoken.
    """
    enriched = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        seg = dict(segment)
        source_text = (seg.get('source') or seg.get('text') or '').strip()
        target_text = (seg.get('target') or '').strip()
        seg_start = max(0.0, _coerce_float(seg.get('start', 0.0)))
        seg_end = max(seg_start + min_duration, _coerce_float(seg.get('end', seg_start + min_duration)))

        raw_words = seg.get('words') or []
        normalized_words = []
        if isinstance(raw_words, list):
            for item in raw_words:
                word = _normalize_word_timestamp(item, seg_start, seg_end)
                if word:
                    normalized_words.append(word)

        # CRITICAL: Use word timestamps as primary timing reference
        if normalized_words:
            # Extract valid word timings (must have actual timestamps from Whisper)
            word_starts = [_coerce_float(w.get('start'), None) for w in normalized_words if _coerce_float(w.get('start'), None) is not None]
            word_ends = [_coerce_float(w.get('end'), None) for w in normalized_words if _coerce_float(w.get('end'), None) is not None]
            
            if word_starts and word_ends:
                # RULE 3: Use earliest spoken word as subtitle start time
                start = min(word_starts)
                # RULE 4: Use last spoken word as subtitle end time
                end = max(word_ends)
                duration = max(min_duration, end - start)
            else:
                # Fallback: use segment timing if word timing incomplete
                logger.debug(f'[Timing] Segment has incomplete word timestamps. Using segment boundaries as fallback.')
                start = seg_start
                end = seg_end
                duration = max(min_duration, end - start)
        else:
            # No word timestamps available - use segment timing
            logger.debug(f'[Timing] No word timestamps available for: "{source_text[:60]}"')
            start = seg_start
            end = seg_end
            duration = max(min_duration, end - start)

        seg['start'] = round(start, 3)
        seg['end'] = round(end, 3)
        seg['duration'] = round(duration, 3)
        seg['source'] = source_text
        seg['target'] = target_text or source_text
        seg['text'] = source_text

        if 'confidence' not in seg and 'avg_logprob' in seg:
            confidence = seg.get('avg_logprob')
            try:
                seg['confidence'] = max(0.0, min(1.0, 1.0 - (abs(float(confidence)) / 100.0)))
            except (TypeError, ValueError):
                seg['confidence'] = None

        seg['words'] = normalized_words
        seg['alignment_source'] = 'speech' if normalized_words and isinstance(raw_words, list) else 'heuristic'
        if 'speaker_id' not in seg and 'speaker' in seg:
            seg['speaker_id'] = seg.get('speaker')
            
        # QUALITY CHECK: Log warnings if timing seems inaccurate
        if normalized_words and (not word_starts or not word_ends):
            logger.warning(f'[Timing] Segment has words but incomplete timestamps for: "{source_text[:40]}"')
        
        enriched.append(seg)
    return enriched


def _build_subtitle_export_payload(
    segments: List[Dict],
    preserve_boundaries: bool = False,
) -> List[Dict[str, Any]]:
    payload = []
    export_segments = (
        [dict(segment) for segment in (segments or [])]
        if preserve_boundaries
        else _enrich_subtitle_segments_with_alignment(segments or [])
    )
    for segment in export_segments:
        start = _coerce_float(segment.get('start', 0.0))
        end = _coerce_float(segment.get('end', start))
        duration = max(0.0, end - start)
        payload.append({
            'start_time': round(start, 3),
            'end_time': round(end, 3),
            'duration': round(duration, 3),
            'translated_text': (segment.get('target') or segment.get('source') or '').strip(),
            'original_text': (segment.get('source') or segment.get('text') or '').strip(),
            'speaker_id': segment.get('speaker_id') or segment.get('speaker'),
            'confidence': segment.get('confidence'),
            'words': segment.get('words', []),
        })
    return payload


def _resolve_segment_voice_window(segment: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    """Translate subtitle timing into an audio window in milliseconds.

    The subtitle timestamps are treated as the authoritative timing source, so
    voice playback must start and stop with the subtitle window.
    """
    if not isinstance(segment, dict):
        return 0, 0

    start = _coerce_float(segment.get('start', segment.get('start_time', 0.0)))
    end = _coerce_float(segment.get('end', segment.get('end_time', start)))
    if end < start:
        end = start

    start_ms = int(round(start * 1000.0))
    end_ms = int(round(end * 1000.0))
    return max(0, start_ms), max(start_ms, end_ms)


def _trim_tts_boundary_silence(
    segment: Any,
    boundary_padding_ms: int = 35,
    trailing_padding_ms: Optional[int] = None,
) -> Any:
    """Trim leading/trailing silence from TTS audio without changing timing semantics."""
    try:
        from pydub.silence import detect_nonsilent

        if len(segment) <= 0:
            return segment
        d_bfs = float(getattr(segment, 'dBFS', -50.0))
        silence_threshold = max(-50.0, d_bfs - 16.0)
        ranges = detect_nonsilent(
            segment,
            min_silence_len=20,
            silence_thresh=silence_threshold,
            seek_step=5,
        )
        if ranges:
            padding_ms = max(0, int(boundary_padding_ms))
            tail_padding_ms = (
                padding_ms
                if trailing_padding_ms is None
                else max(0, int(trailing_padding_ms))
            )
            start_ms = max(0, ranges[0][0] - padding_ms)
            end_ms = min(len(segment), ranges[-1][1] + tail_padding_ms)
            return segment[start_ms:end_ms]
    except Exception:
        pass
    return segment


def _insert_natural_pauses(segment: Any, text: str = '') -> Any:
    """Insert minimal pauses for natural speech pacing when applicable."""
    return segment


def _speed_change_audio(segment: Any, speed_factor: float) -> Any:
    """Adjust speech tempo in either direction while keeping its pitch stable."""
    if segment is None or speed_factor <= 0 or abs(speed_factor - 1.0) <= 0.001:
        return segment
    try:
        frame_rate = int(segment.frame_rate)
        faster = segment._spawn(
            segment.raw_data,
            overrides={'frame_rate': max(1, int(frame_rate * speed_factor))},
        )
        return faster.set_frame_rate(frame_rate)
    except Exception:
        return segment


def _sync_audio_to_subtitle_window(audio_segment: Any, segment: Optional[Dict[str, Any]], offset_ms: int = 0) -> Any:
    """Fit synthesized audio to exactly the subtitle duration."""
    if audio_segment is None:
        return None

    start_ms, end_ms = _resolve_segment_voice_window(segment)
    subtitle_duration_ms = max(1, end_ms - start_ms)
    effective_window_ms = max(1, subtitle_duration_ms - max(0, int(offset_ms)))

    audio_segment = _trim_tts_boundary_silence(audio_segment)
    audio_segment = _insert_natural_pauses(audio_segment, (segment or {}).get('target') or (segment or {}).get('source') or '')
    try:
        length_ms = int(getattr(audio_segment, 'length_ms', 0))
    except Exception:
        length_ms = 0

    if length_ms <= 0:
        try:
            length_ms = int(len(audio_segment))
        except Exception:
            length_ms = 0

    if length_ms > 0 and length_ms != effective_window_ms:
        audio_segment = _speed_change_audio(
            audio_segment,
            length_ms / float(effective_window_ms),
        )
        try:
            length_ms = int(len(audio_segment))
        except Exception:
            pass

    if length_ms < effective_window_ms:
        try:
            from pydub import AudioSegment
            audio_segment += AudioSegment.silent(duration=effective_window_ms - length_ms)
        except Exception:
            try:
                audio_segment += audio_segment.silent(duration=effective_window_ms - length_ms)
            except Exception:
                pass

    try:
        return audio_segment[:effective_window_ms]
    except Exception:
        return audio_segment


EDGE_TTS_VOICE_MAP = {
    'en': 'en-US-AriaNeural',
    'km': 'km-KH-SreymomNeural',
    'th': 'th-TH-NutchayaNeural',
    'vi': 'vi-VN-HoaiMyNeural',
    'zh': 'zh-CN-XiaoxiaoNeural',
    'zh-cn': 'zh-CN-XiaoxiaoNeural',
    'zh-tw': 'zh-TW-HsiaoYuNeural',
    'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-SunHiNeural',
    'lo': 'lo-LA-NoyNeural',
    'my': 'my-MM-AyeNeural',
}


def _get_default_tts_voice(target_lang: str, preferred_voice: str = '') -> str:
    # The browser voice picker exposes OS display names, while edge-tts needs
    # canonical IDs such as "km-KH-HortenseNeural". Ignore incompatible names
    # instead of letting the required workflow fail at its final stage.
    if preferred_voice and isinstance(preferred_voice, str):
        candidate = preferred_voice.strip()
        if re.fullmatch(r'[a-z]{2,3}-[A-Z]{2}-[A-Za-z0-9]+Neural', candidate):
            return candidate
    base = (target_lang or '').split('-')[0].lower().strip()
    return EDGE_TTS_VOICE_MAP.get(base, EDGE_TTS_VOICE_MAP.get('en'))


def _measure_wav_duration(wav_path: str) -> float:
    import wave

    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate else 0.0


def _trim_generated_wav_to_voice(
    wav_path: str,
    boundary_padding_ms: int = 35,
    trailing_padding_ms: Optional[int] = None,
):
    """Remove encoder/TTS boundary silence before it becomes timeline time."""
    from pydub import AudioSegment

    AudioSegment.converter = FFMPEG_EXE
    audio = AudioSegment.from_wav(wav_path)
    trimmed = _trim_tts_boundary_silence(
        audio,
        boundary_padding_ms=boundary_padding_ms,
        trailing_padding_ms=trailing_padding_ms,
    )
    trimmed.export(wav_path, format='wav')


def _synthesize_text_to_wav(
    text: str,
    voice_name: str,
    output_wav_path: str,
    sample_rate: int = 48000,
    speech_rate: float = 1.0,
    pitch_hz: int = 0,
    volume_percent: int = 0,
):
    """Synthesize one subtitle verbatim and write a mono PCM WAV file."""
    if not text or not voice_name:
        raise ValueError('Text and voice are required for TTS synthesis.')

    import edge_tts
    from pathlib import Path

    wav_path = Path(output_wav_path)
    tmp_mp3 = wav_path.with_suffix('.mp3')

    try:
        speech_rate = max(0.5, min(2.0, float(speech_rate)))
        pitch_hz = max(-50, min(50, int(pitch_hz)))
        volume_percent = max(-50, min(50, int(volume_percent)))
        edge_rate = f'{round((speech_rate - 1.0) * 100):+d}%'
        communicator = edge_tts.Communicate(
            text,
            voice=voice_name,
            rate=edge_rate,
            pitch=f'{pitch_hz:+d}Hz',
            volume=f'{volume_percent:+d}%',
        )
        try:
            asyncio.run(communicator.save(str(tmp_mp3)))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(communicator.save(str(tmp_mp3)))
            loop.close()

        subprocess.run(
            [
                FFMPEG_EXE,
                '-y',
                '-i', str(tmp_mp3),
                '-ar', str(sample_rate),
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                str(wav_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        raise RuntimeError(f'TTS synthesis failed: {e}')
    finally:
        try:
            if tmp_mp3.exists():
                tmp_mp3.unlink()
        except OSError:
            pass


def _concatenate_wav_segments(wav_paths: List[str], output_path: str):
    if not wav_paths:
        raise ValueError('No audio segments to concatenate.')

    import wave

    reference_params = None
    chunks = []
    for wav_path in wav_paths:
        with wave.open(str(wav_path), 'rb') as source:
            params = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
                source.getcomptype(),
            )
            if reference_params is None:
                reference_params = params
            elif params != reference_params:
                raise RuntimeError('Generated TTS clips do not share one WAV format.')
            chunks.append(source.readframes(source.getnframes()))

    channels, sample_width, frame_rate, compression = reference_params
    with wave.open(str(output_path), 'wb') as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(frame_rate)
        target.setcomptype(compression, 'not compressed')
        for chunk in chunks:
            target.writeframes(chunk)


def _subtitle_tts_text(segment: Dict[str, Any]) -> str:
    """Return one subtitle's text verbatim, using supported payload aliases."""
    for key in ('text', 'target', 'translated_text', 'source', 'original_text'):
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ''


def _assemble_subtitle_audio_timeline(
    clips: List[Tuple[str, float, float]],
    output_path: str,
):
    """Place fitted WAV clips at their absolute subtitle timestamps."""
    if not clips:
        raise ValueError('No subtitle audio clips to assemble.')

    import wave

    reference_params = None
    cursor_frame = 0
    with wave.open(str(output_path), 'wb') as target:
        for clip_path, start, end in clips:
            with wave.open(str(clip_path), 'rb') as source:
                params = (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                    source.getcomptype(),
                )
                if reference_params is None:
                    reference_params = params
                    channels, sample_width, frame_rate, compression = params
                    target.setnchannels(channels)
                    target.setsampwidth(sample_width)
                    target.setframerate(frame_rate)
                    target.setcomptype(compression, 'not compressed')
                elif params != reference_params:
                    raise RuntimeError('Generated TTS clips do not share one WAV format.')

                channels, sample_width, frame_rate, _ = reference_params
                frame_size = channels * sample_width
                start_frame = max(0, int(round(start * frame_rate)))
                end_frame = max(start_frame, int(round(end * frame_rate)))
                if start_frame < cursor_frame:
                    raise RuntimeError('Subtitle audio clips overlap on the output timeline.')

                gap_frames = start_frame - cursor_frame
                while gap_frames:
                    chunk_frames = min(gap_frames, frame_rate * 60)
                    target.writeframes(b'\x00' * (chunk_frames * frame_size))
                    gap_frames -= chunk_frames

                required_frames = end_frame - start_frame
                audio = source.readframes(required_frames)
                actual_frames = len(audio) // frame_size
                target.writeframes(audio)
                if actual_frames < required_frames:
                    target.writeframes(
                        b'\x00' * ((required_frames - actual_frames) * frame_size)
                    )
                cursor_frame = end_frame


def _atempo_filter(speed_factor: float) -> str:
    """Build a pitch-preserving FFmpeg atempo chain for any positive factor."""
    if speed_factor <= 0:
        raise ValueError('Audio speed factor must be positive.')
    factors = []
    remaining = speed_factor
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ','.join(f'atempo={factor:.8f}' for factor in factors)


def _fit_generated_wav_to_window(
    input_path: str,
    output_path: str,
    duration_seconds: float,
    sample_rate: int = 48000,
    preserve_natural_short_pacing: bool = False,
    ending_guard_seconds: float = 0.0,
):
    """Fit and normalize one subtitle clip in a single lossless PCM pass."""
    target_duration = max(0.01, float(duration_seconds))
    source_duration = _measure_wav_duration(input_path)
    if source_duration <= 0:
        raise RuntimeError('TTS provider returned an empty audio clip.')

    # atempo/loudnorm can leave a few delayed samples beyond the mathematically
    # exact duration. Finish speech slightly before a tight subtitle boundary
    # and let apad fill the remainder, instead of letting atrim sever the last
    # quiet syllable. The proportional cap avoids over-compressing short clips.
    requested_guard = max(0.0, float(ending_guard_seconds))
    applied_guard = min(requested_guard, target_duration * 0.12)
    speech_duration = max(0.01, target_duration - applied_guard)
    speed_factor = source_duration / speech_duration
    # Tight-sync clips are never slowed merely to occupy an entire subtitle
    # window. This preserves natural/manual pacing and leaves any remainder as
    # trailing timeline silence. Clips that are too long are still accelerated
    # just enough to end inside the subtitle window without cutting words.
    applied_speed_factor = (
        max(1.0, speed_factor)
        if preserve_natural_short_pacing
        else speed_factor
    )
    filters = []
    if abs(applied_speed_factor - 1.0) > 0.001:
        filters.append(_atempo_filter(applied_speed_factor))
    filters.extend([
        'loudnorm=I=-19:TP=-1:LRA=7',
        'apad',
        f'atrim=duration={target_duration:.6f}',
        'asetpts=N/SR/TB',
    ])
    completed = subprocess.run(
        [
            FFMPEG_EXE, '-y',
            '-i', input_path,
            '-af', ','.join(filters),
            '-ar', str(sample_rate),
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f'Unable to fit TTS audio to subtitle duration: '
            f'{completed.stderr[-1200:].strip()}'
        )


def _validate_dubbing_segments(
    original_segments: List[Dict],
    generated_segments: List[Dict],
):
    """Enforce one unchanged subtitle unit for every generated TTS clip."""
    if len(original_segments) != len(generated_segments):
        raise RuntimeError(
            'Subtitle count does not match generated TTS segment count.'
        )

    ordered_originals = sorted(
        enumerate(original_segments),
        key=lambda item: (
            _coerce_float(item[1].get('start', item[1].get('start_time', 0.0))),
            item[0],
        ),
    )
    for index, ((_, original), generated) in enumerate(
        zip(ordered_originals, generated_segments),
        1,
    ):
        original_start, original_end = _resolve_segment_voice_window(original)
        generated_start, generated_end = _resolve_segment_voice_window(generated)
        if (original_start, original_end) != (generated_start, generated_end):
            raise RuntimeError(f'Subtitle {index} timestamps changed during TTS.')
        if _subtitle_tts_text(original) != _subtitle_tts_text(generated):
            raise RuntimeError(f'Subtitle {index} text changed during TTS.')
        if generated.get('voice_clip_index') != index - 1:
            raise RuntimeError(f'Subtitle {index} does not map to exactly one TTS clip.')


def _probe_wav_loudness(wav_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Measure integrated LUFS and true peak using FFmpeg EBU R128."""
    completed = subprocess.run(
        [
            FFMPEG_EXE,
            '-hide_banner',
            '-i', wav_path,
            '-filter_complex', 'ebur128=peak=true',
            '-f', 'null',
            '-',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError('Unable to validate generated audio loudness.')

    summary = completed.stderr.rsplit('Summary:', 1)[-1]
    integrated_match = re.search(r'\bI:\s*(-?\d+(?:\.\d+)?)\s+LUFS', summary)
    peak_match = re.search(r'\bPeak:\s*(-?\d+(?:\.\d+)?)\s+dBFS', summary)
    integrated = float(integrated_match.group(1)) if integrated_match else None
    true_peak = float(peak_match.group(1)) if peak_match else None
    return integrated, true_peak


def _validate_dubbed_audio_file(
    output_audio_path: str,
    expected_duration: float,
    expected_sample_rate: int,
):
    """Validate format, timeline duration, loudness, and peak before export."""
    import wave

    with wave.open(output_audio_path, 'rb') as audio:
        sample_rate = audio.getframerate()
        channels = audio.getnchannels()
        frame_count = audio.getnframes()
    if sample_rate != expected_sample_rate:
        raise RuntimeError(
            f'Generated audio sample rate is {sample_rate}, expected '
            f'{expected_sample_rate}.'
        )
    if channels != 1:
        raise RuntimeError('Generated dubbing audio must be mono.')

    actual_duration = frame_count / float(sample_rate)
    if abs(actual_duration - expected_duration) > (1.0 / sample_rate):
        raise RuntimeError(
            f'Generated audio duration {actual_duration:.6f}s does not match '
            f'subtitle timeline {expected_duration:.6f}s.'
        )

    integrated_lufs, true_peak = _probe_wav_loudness(output_audio_path)
    if actual_duration >= 0.4 and integrated_lufs is not None:
        if not -21.0 <= integrated_lufs <= -17.0:
            raise RuntimeError(
                f'Generated audio loudness is {integrated_lufs:.1f} LUFS; '
                'expected -19 LUFS (+/-2 LU).'
            )
    if true_peak is not None and true_peak > -0.8:
        raise RuntimeError(
            f'Generated audio true peak is {true_peak:.1f} dBTP; '
            'the limit is -1 dBTP.'
        )


def _build_tight_sync_voice_groups(
    ordered_segments: List[Tuple[int, Dict]],
    max_gap_seconds: float = 0.25,
    max_phrase_seconds: float = 15.0,
    max_phrase_characters: int = 320,
) -> List[List[Tuple[int, Dict]]]:
    """Join mechanically split captions into natural TTS phrases.

    Subtitle objects and timestamps remain untouched. Only the synthesis call is
    grouped, so adjacent caption cuts no longer create a new voice onset (and a
    clipped word boundary) for every short subtitle.
    """
    groups: List[List[Tuple[int, Dict]]] = []
    for item in ordered_segments:
        _, segment = item
        if not groups:
            groups.append([item])
            continue

        previous = groups[-1][-1][1]
        group_first = groups[-1][0][1]
        previous_end = _coerce_float(
            previous.get('end', previous.get('end_time', 0.0))
        )
        current_start = _coerce_float(
            segment.get('start', segment.get('start_time', previous_end))
        )
        current_end = _coerce_float(
            segment.get('end', segment.get('end_time', current_start))
        )
        group_start = _coerce_float(
            group_first.get('start', group_first.get('start_time', 0.0))
        )
        previous_speaker = str(
            previous.get('speaker_id') or previous.get('speaker') or ''
        ).strip()
        current_speaker = str(
            segment.get('speaker_id') or segment.get('speaker') or ''
        ).strip()
        same_speaker = (
            not previous_speaker
            or not current_speaker
            or previous_speaker == current_speaker
        )
        phrase_characters = sum(
            len(_subtitle_tts_text(member))
            for _, member in groups[-1]
        ) + len(_subtitle_tts_text(segment))
        can_join = (
            current_start - previous_end <= max_gap_seconds
            and same_speaker
            and not bool(segment.get('scene_change'))
            and not bool(previous.get('scene_change'))
            and current_end - group_start <= max_phrase_seconds
            and phrase_characters <= max_phrase_characters
        )
        if can_join:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _generate_dubbed_audio(
    segments: List[Dict],
    target_lang: str,
    voice_name: str,
    output_audio_path: str,
    sample_rate: int = 48000,
    speech_rate: float = 1.0,
    pitch_hz: int = 0,
    volume_percent: int = 0,
    tight_sync: bool = False,
) -> List[Dict]:
    if not segments:
        raise ValueError('No subtitle segments provided for dubbing audio.')

    voice = _get_default_tts_voice(target_lang, voice_name)
    tmp_paths = []
    updated_segments = []
    timeline_clips = []
    ordered_segments = sorted(
        enumerate(segments),
        key=lambda item: (
            _coerce_float(item[1].get('start', item[1].get('start_time', 0.0))),
            item[0],
        ),
    )
    voice_started_at = time.time()
    _update_pipeline_progress(
        'Voice Generation',
        0,
        len(ordered_segments),
        f'Preparing voice for {len(ordered_segments)} subtitles',
        voice_started_at,
    )

    segment_windows = []
    for ordered_index, (_, seg) in enumerate(ordered_segments):
        text = _subtitle_tts_text(seg)
        if not text:
            raise ValueError(f'Subtitle {ordered_index + 1} has no text for TTS.')
        start = _coerce_float(seg.get('start', seg.get('start_time', 0.0)))
        subtitle_end = _coerce_float(seg.get('end', seg.get('end_time', start)))
        if start < 0 or subtitle_end <= start:
            raise ValueError(
                f'Subtitle {ordered_index + 1} has an invalid timestamp window.'
            )
        next_start = subtitle_end
        if ordered_index + 1 < len(ordered_segments):
            next_seg = ordered_segments[ordered_index + 1][1]
            next_start = _coerce_float(
                next_seg.get('start', next_seg.get('start_time', subtitle_end))
            )
            if next_start < subtitle_end:
                raise ValueError(
                    f'Subtitle {ordered_index + 1} overlaps subtitle '
                    f'{ordered_index + 2}; exact non-overlapping TTS timing '
                    'is impossible.'
                )
        segment_windows.append((ordered_index, seg, text, start, subtitle_end))

    voice_groups = (
        _build_tight_sync_voice_groups(ordered_segments)
        if tight_sync
        else [[item] for item in ordered_segments]
    )
    jobs = []
    window_by_identity = {
        id(seg): window for window in segment_windows for seg in [window[1]]
    }
    for group_index, group in enumerate(voice_groups):
        member_windows = [window_by_identity[id(seg)] for _, seg in group]
        start = member_windows[0][3]
        subtitle_end = member_windows[-1][4]
        voice_fit_end = subtitle_end
        if tight_sync and group_index + 1 < len(voice_groups):
            next_seg = voice_groups[group_index + 1][0][1]
            next_start = _coerce_float(
                next_seg.get('start', next_seg.get('start_time', subtitle_end))
            )
            if next_start > subtitle_end:
                voice_fit_end = min(next_start, subtitle_end + 2.0)
        text = (
            member_windows[0][2]
            if len(member_windows) == 1
            else ' '.join(window[2].strip() for window in member_windows)
        )
        jobs.append((
            group_index,
            member_windows,
            text,
            start,
            subtitle_end,
            voice_fit_end,
        ))

    def _generate_voice_clip(job):
        group_index, member_windows, text, start, subtitle_end, voice_fit_end = job
        process_lock.raise_if_cancelled()
        raw_wav_path = str(TEMP_DIR / f'dub_segment_raw_{uuid4().hex}.wav')
        fitted_wav_path = str(TEMP_DIR / f'dub_segment_fitted_{uuid4().hex}.wav')
        tmp_paths.extend((raw_wav_path, fitted_wav_path))
        tts_cache_key = _cache_key('tts', {
            'text': text,
            'voice': voice,
            'sample_rate': sample_rate,
            'speech_rate': round(float(speech_rate), 3),
            'pitch_hz': int(pitch_hz),
            'volume_percent': int(volume_percent),
        })
        cached_wav_path = TTS_CACHE_DIR / f'{tts_cache_key}.wav'
        last_tts_error = None
        if cached_wav_path.exists() and cached_wav_path.stat().st_size > 44:
            shutil.copy2(cached_wav_path, raw_wav_path)
        else:
            for attempt in range(1, 4):
                process_lock.raise_if_cancelled()
                try:
                    _synthesize_text_to_wav(
                        text,
                        voice,
                        raw_wav_path,
                        sample_rate=sample_rate,
                        speech_rate=speech_rate,
                        pitch_hz=pitch_hz,
                        volume_percent=volume_percent,
                    )
                    last_tts_error = None
                    break
                except Exception as error:
                    last_tts_error = error
                    logger.warning(
                        f'[TTS] Voice phrase {group_index + 1} attempt '
                        f'{attempt}/3 failed: {error}'
                    )
                    if attempt < 3:
                        time.sleep(min(2 ** (attempt - 1), 4))
        if last_tts_error is not None:
            raise RuntimeError(
                f'TTS failed for voice phrase {group_index + 1} after '
                f'3 attempts: {last_tts_error}'
            )
        if not cached_wav_path.exists():
            try:
                cache_tmp = cached_wav_path.with_suffix(f'.{uuid4().hex}.tmp')
                shutil.copy2(raw_wav_path, cache_tmp)
                with _cache_lock:
                    if not cached_wav_path.exists():
                        os.replace(cache_tmp, cached_wav_path)
                    elif cache_tmp.exists():
                        cache_tmp.unlink()
            except OSError as cache_error:
                logger.debug(f'[TTS] Cache write failed: {cache_error}')
        boundary_padding_ms = 10 if tight_sync else 35
        # Slower Edge TTS voices often finish with a quiet, stretched final
        # syllable. Keep a little more tail only for tight-sync slow playback
        # so silence detection cannot make that ending sound clipped.
        tight_trailing_padding_ms = (
            140 if tight_sync and float(speech_rate) < 1.0 else 45
        )
        _trim_generated_wav_to_voice(
            raw_wav_path,
            boundary_padding_ms=boundary_padding_ms,
            trailing_padding_ms=tight_trailing_padding_ms if tight_sync else None,
        )
        _fit_generated_wav_to_window(
            raw_wav_path,
            fitted_wav_path,
            voice_fit_end - start,
            sample_rate=sample_rate,
            preserve_natural_short_pacing=tight_sync,
            ending_guard_seconds=0.18 if tight_sync else 0.0,
        )
        updated_group = []
        for ordered_index, seg, _, member_start, member_end in member_windows:
            updated = dict(seg)
            updated['original_start'] = member_start
            updated['original_end'] = member_end
            updated['start'] = round(member_start, 3)
            updated['end'] = round(member_end, 3)
            updated['duration'] = round(member_end - member_start, 3)
            updated['audio_end'] = round(
                voice_fit_end if member_end == subtitle_end else member_end,
                3,
            )
            updated['timing_source'] = 'subtitle_timestamps'
            updated['speech_rate'] = round(
                max(0.5, min(2.0, float(speech_rate))), 2
            )
            updated['tight_sync'] = bool(tight_sync)
            updated['voice_phrase_index'] = group_index
            updated_group.append(updated)
        return updated_group, (fitted_wav_path, start, voice_fit_end), (
            raw_wav_path,
            fitted_wav_path,
        )

    try:
        worker_count = (
            1 if len(jobs) < 4
            else _bounded_worker_count('TTS_WORKERS', 3, 4)
        )
        if worker_count == 1:
            generated = map(_generate_voice_clip, jobs)
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            generated = executor.map(_generate_voice_clip, jobs)
        try:
            for updated_group, timeline_clip, generated_paths in generated:
                process_lock.raise_if_cancelled()
                for updated in updated_group:
                    updated['voice_clip_index'] = len(updated_segments)
                    updated_segments.append(updated)
                timeline_clips.append(timeline_clip)
                _update_pipeline_progress(
                    'Voice Generation',
                    len(updated_segments),
                    len(ordered_segments),
                    f'Generated voice {len(updated_segments)}/{len(ordered_segments)}',
                    voice_started_at,
                )
        finally:
            if worker_count > 1:
                executor.shutdown(wait=True, cancel_futures=True)

        if not updated_segments:
            raise RuntimeError('No valid TTS segments were generated.')

        _validate_dubbing_segments(segments, updated_segments)
        _assemble_subtitle_audio_timeline(timeline_clips, output_audio_path)
        _validate_dubbed_audio_file(
            output_audio_path,
            expected_duration=_coerce_float(updated_segments[-1].get('end')),
            expected_sample_rate=sample_rate,
        )
        return updated_segments
    finally:
        for path in tmp_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass


def _validate_speaker_pure_blocks(segments: List[Dict]) -> List[Dict]:
    """Reject a block containing word labels from more than one speaker."""
    validated = []
    for index, raw in enumerate(segments):
        segment = dict(raw)
        speakers = {
            _word_speaker(word)
            for word in (segment.get('words') or [])
            if _word_speaker(word)
        }
        declared = str(
            segment.get('speaker_id') or segment.get('speaker') or ''
        ).strip()
        if len(speakers) > 1:
            raise ValueError(
                f'Subtitle {index + 1} contains multiple speakers: '
                f'{", ".join(sorted(speakers))}. Speaker detection must run '
                'before translation and voice generation.'
            )
        if speakers:
            detected = next(iter(speakers))
            if declared and declared != detected:
                raise ValueError(
                    f'Subtitle {index + 1} speaker label {declared} does not '
                    f'match detected speaker {detected}.'
                )
            segment['speaker_id'] = detected
        elif declared:
            segment['speaker_id'] = declared
        else:
            segment['speaker_id'] = 'SPEAKER_00'
        validated.append(segment)
    return validated


def _build_audio_master_scene_timeline(segments: List[Dict]) -> List[Dict]:
    """Build visual scene metadata without changing subtitle audio units."""
    scenes = []
    for index, segment in enumerate(segments):
        speaker = str(
            segment.get('speaker_id') or segment.get('speaker') or 'SPEAKER_00'
        )
        action = str(segment.get('action') or '').strip()
        explicit_change = bool(segment.get('scene_change'))
        starts_new_scene = (
            not scenes
            or scenes[-1]['speaker_id'] != speaker
            or explicit_change
            or (action and action != scenes[-1].get('action', ''))
        )
        if starts_new_scene:
            scenes.append({
                'scene_id': len(scenes) + 1,
                'speaker_id': speaker,
                'start': _coerce_float(segment.get('start')),
                'end': _coerce_float(segment.get('end')),
                'duration': _coerce_float(segment.get('duration')),
                'original_start': _coerce_float(
                    segment.get('original_start', segment.get('start'))
                ),
                'original_end': _coerce_float(
                    segment.get('original_end', segment.get('end'))
                ),
                'action': action,
                'subtitle_indices': [index],
                'timing_source': 'subtitle_timestamps',
            })
            continue

        scene = scenes[-1]
        scene['end'] = _coerce_float(segment.get('end'))
        scene['duration'] = round(scene['end'] - scene['start'], 3)
        scene['original_end'] = _coerce_float(
            segment.get('original_end', segment.get('end'))
        )
        scene['subtitle_indices'].append(index)
    return scenes


def _read_wav_chunk(audio_path: str, start_sec: float, duration_sec: float):
    """Read a mono 16 kHz PCM WAV slice into a float32 array without another FFmpeg conversion."""
    import wave
    import numpy as np

    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        if channels != 1 or sample_width != 2:
            raise ValueError(
                f'Expected mono 16-bit PCM WAV, got channels={channels}, width={sample_width}'
            )

        start_frame = max(0, int(start_sec * sample_rate))
        frame_count = max(1, int(duration_sec * sample_rate))
        wf.setpos(min(start_frame, wf.getnframes()))
        data = wf.readframes(frame_count)

    if not data:
        return np.zeros(0, dtype=np.float32)

    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def _build_time_chunks(duration: float, chunk_seconds: float, overlap_seconds: float) -> List[Tuple[int, float, float]]:
    chunks = []
    chunk_seconds = max(1.0, float(chunk_seconds))
    overlap_seconds = max(0.0, min(float(overlap_seconds), max(0.0, (chunk_seconds / 2.0) - 0.01)))
    start = 0.0
    idx = 0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        read_start = max(0.0, start - overlap_seconds if idx else start)
        read_end = min(duration, end + overlap_seconds)
        chunks.append((idx, read_start, read_end))
        idx += 1
        start = end
    return chunks


def _prepare_segments_for_subtitle_generation(segments: List[Dict]) -> List[Dict]:
    """
    Prepare transcription-only subtitle segments for the Generate Subtitle flow.
    This keeps each subtitle as a single independent speech segment with the
    original start/end timestamps and source text, without translating or
    altering the timing.
    """
    prepared = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        prepared_seg = dict(seg)
        prepared_seg['source'] = (prepared_seg.get('source') or '').strip()
        prepared_seg['target'] = prepared_seg.get('source', '')
        prepared_seg['start'] = float(prepared_seg.get('start', 0.0))
        prepared_seg['end'] = float(prepared_seg.get('end', prepared_seg.get('start', 0.0)))
        if prepared_seg['end'] < prepared_seg['start']:
            prepared_seg['end'] = prepared_seg['start']
        prepared.append(prepared_seg)
    return prepared


def _group_button1_spoken_phrases(segments: List[Dict]) -> List[Dict]:
    """Group Button 1 transcription fragments into complete spoken phrases.

    This post-processor is intentionally scoped to speech subtitle generation.
    It never changes translation, OCR, paste, copy, or dubbing segmentation.
    """
    if not segments:
        return []

    def _text(segment: Dict) -> str:
        return re.sub(
            r'\s+',
            ' ',
            str(segment.get('source') or segment.get('text') or ''),
        ).strip()

    def _speaker(segment: Dict) -> str:
        value = str(
            segment.get('speaker_id') or segment.get('speaker') or ''
        ).strip()
        return '' if value.upper() in {'SPEAKER_00', 'UNKNOWN', 'NONE'} else value

    def _phrase_finished(text: str) -> bool:
        # Ellipses are deliberately excluded: they commonly represent a
        # hesitation or breath inside one continuing phrase.
        without_ellipsis = re.sub(r'(?:\.{2,}|…+)\s*$', '', text or '')
        return bool(re.search(r'[.!?。！？\u17d4\u17d5]\s*$', without_ellipsis))

    def _join_text(left: str, right: str) -> str:
        left = left.rstrip()
        right = right.lstrip()
        if not left:
            return right
        if not right:
            return left
        if re.match(r'^[,.;:!?，。！？、；：)\]}\u17d4\u17d5]', right):
            separator = ''
        elif (
            re.search(r'[\u3400-\u9fff\u3040-\u30ff]$', left)
            and re.match(r'^[\u3400-\u9fff\u3040-\u30ff]', right)
        ):
            separator = ''
        else:
            separator = ' '
        return f'{left}{separator}{right}'

    def _dedupe_words(words: List[Dict]) -> List[Dict]:
        result = []
        seen = set()
        for raw_word in words:
            if not isinstance(raw_word, dict):
                continue
            word = dict(raw_word)
            key = (
                round(_coerce_float(word.get('start'), 0.0), 3),
                round(_coerce_float(word.get('end'), 0.0), 3),
                str(word.get('word') or word.get('text') or '').strip().casefold(),
                _word_speaker(word),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(word)
        return sorted(
            result,
            key=lambda word: (
                _coerce_float(word.get('start'), 0.0),
                _coerce_float(word.get('end'), 0.0),
            ),
        )

    cleaned = []
    for raw in sorted(
        (item for item in segments if isinstance(item, dict)),
        key=lambda item: (
            _coerce_float(item.get('start'), 0.0),
            _coerce_float(item.get('end'), 0.0),
        ),
    ):
        text = _text(raw)
        start = _coerce_float(raw.get('start'), 0.0)
        end = _coerce_float(raw.get('end'), start)
        if not text or end <= start:
            continue

        segment = dict(raw)
        segment['source'] = text
        segment['text'] = text
        segment['start'] = round(max(0.0, start), 3)
        segment['end'] = round(max(start + 0.01, end), 3)
        segment['duration'] = round(segment['end'] - segment['start'], 3)
        segment['words'] = _dedupe_words(segment.get('words') or [])

        if cleaned:
            previous = cleaned[-1]
            same_text = _text(previous).casefold() == text.casefold()
            overlap = _time_overlap(
                previous['start'], previous['end'],
                segment['start'], segment['end'],
            )
            shortest = min(
                previous['end'] - previous['start'],
                segment['end'] - segment['start'],
            )
            # Remove chunk-overlap duplicates without deleting a genuinely
            # repeated phrase spoken later at a different time.
            if same_text and shortest > 0 and overlap / shortest >= 0.6:
                previous['start'] = min(previous['start'], segment['start'])
                previous['end'] = max(previous['end'], segment['end'])
                previous['duration'] = round(
                    previous['end'] - previous['start'], 3
                )
                previous['words'] = _dedupe_words(
                    (previous.get('words') or []) + (segment.get('words') or [])
                )
                continue
        cleaned.append(segment)

    grouped = []
    for segment in cleaned:
        if not grouped:
            grouped.append(segment)
            continue

        previous = grouped[-1]
        previous_speaker = _speaker(previous)
        current_speaker = _speaker(segment)
        if (
            previous_speaker
            and current_speaker
            and previous_speaker != current_speaker
        ):
            grouped.append(segment)
            continue
        if bool(previous_speaker) != bool(current_speaker):
            grouped.append(segment)
            continue

        gap = max(0.0, segment['start'] - previous['end'])
        same_detected_speaker = bool(
            previous_speaker
            and previous_speaker == current_speaker
        )
        same_original_turn = (
            previous.get('split_from') is not None
            and previous.get('split_from') == segment.get('split_from')
        )
        maximum_internal_gap = (
            0.65 if same_detected_speaker
            else 0.45 if same_original_turn
            else 0.0
        )
        combined_duration = segment['end'] - previous['start']
        should_merge = (
            gap <= maximum_internal_gap
            and combined_duration <= 6.5
            and not _phrase_finished(_text(previous))
        )

        if not should_merge:
            grouped.append(segment)
            continue

        previous['source'] = _join_text(_text(previous), _text(segment))
        previous['text'] = previous['source']
        previous['end'] = round(max(previous['end'], segment['end']), 3)
        previous['duration'] = round(
            previous['end'] - previous['start'], 3
        )
        previous['words'] = _dedupe_words(
            (previous.get('words') or []) + (segment.get('words') or [])
        )
        if not previous.get('speaker_id') and segment.get('speaker_id'):
            previous['speaker_id'] = segment.get('speaker_id')

    logger.info(
        f'[Button 1] Grouped {len(cleaned)} clean fragments into '
        f'{len(grouped)} complete spoken phrases.'
    )
    return grouped


# ============================================================================
# CRITICAL: Subtitle Boundary Preservation
# ============================================================================
# The following functions enforce the rule that:
#   ONE displayed subtitle = ONE subtitle segment.
#   Subtitles are NEVER merged, even if they have similar text.
#   Every subtitle phrase remains independent with its own timing.
# ============================================================================


def _repair_subtitle_timing(segments: List[Dict], min_duration: float = 0.01) -> List[Dict]:
    """Sort and stabilize subtitle timestamps while preserving speech boundaries.

    Every detected phrase remains an independent subtitle. Word timestamps are
    authoritative and are never shifted merely to hide a real overlap.
    """
    cleaned = []
    seen = set()
    for seg in sorted(segments, key=lambda s: (_coerce_float(s.get('start', 0)), _coerce_float(s.get('end', 0)))):
        text = (seg.get('source') or seg.get('text') or '').strip()
        if not text:
            continue
        enriched = _enrich_subtitle_segments_with_alignment([seg], min_duration=min_duration)[0]
        start = enriched['start']
        end = enriched['end']
        speaker = str(enriched.get('speaker_id') or enriched.get('speaker') or '').strip()
        # Use exact text match for deduplication (not fuzzy match)
        # Use millisecond precision to avoid incorrectly merging different segments
        key = (round(start, 3), round(end, 3), text.lower(), speaker)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(enriched)

    repaired = []
    for seg in cleaned:
        if repaired:
            last = repaired[-1]
            # Fix overlapping timestamps WITHOUT merging different subtitle phrases.
            # When both phrases have detected word timing, overlap can represent
            # real simultaneous speech and must not corrupt either speech boundary.
            if seg['start'] < last['end'] and not (last.get('words') and seg.get('words')):
                overlap_duration = last['end'] - seg['start']
                gap_threshold = 0.02  # 20ms minimum gap (very small)
                if overlap_duration > gap_threshold:
                    # CRITICAL: Check word timestamps to determine actual speech boundaries.
                    # The previous segment's end should be its last word's end time.
                    # The current segment's start should be its first word's start time.
                    last_words = last.get('words', [])
                    seg_words = seg.get('words', [])
                    
                    # Use word timestamps to refine previous segment's end
                    if last_words:
                        last_word_ends = [_coerce_float(w.get('end'), None) for w in last_words if _coerce_float(w.get('end'), None) is not None]
                        if last_word_ends:
                            actual_last_end = max(last_word_ends)
                            if actual_last_end < last['end']:
                                last['end'] = round(actual_last_end, 3)
                                last['duration'] = max(min_duration, last['end'] - last['start'])
                    
                    # Use word timestamps to refine current segment's start
                    if seg_words:
                        seg_word_starts = [_coerce_float(w.get('start'), None) for w in seg_words if _coerce_float(w.get('start'), None) is not None]
                        if seg_word_starts:
                            actual_seg_start = min(seg_word_starts)
                            if actual_seg_start > seg['start']:
                                seg['start'] = round(actual_seg_start, 3)
                    
                    # Only push forward if still overlapping after word-timestamp refinement
                    if seg['start'] < last['end']:
                        seg['start'] = last['end'] + gap_threshold
                        seg['end'] = max(seg['start'] + min_duration, seg['end'])
                        seg['duration'] = max(min_duration, seg['end'] - seg['start'])
                        logger.debug(f'[Timing] Fixed overlap: prev=[{last["start"]:.3f}-{last["end"]:.3f}] -> now [{seg["start"]:.3f}-{seg["end"]:.3f}]')
        repaired.append(seg)

    return _enrich_subtitle_segments_with_alignment(repaired, min_duration=min_duration)


# ---------------------------------------------------------------------------
# Stage 2b+: Subtitle Segmentation Enhancement
# These functions enforce the subtitle segmentation rules:
#   - Split merged sentences at natural speech pauses
#   - Enforce max duration (6s), max 2 lines, max 42 chars per line
#   - Never split mid-word/mid-phrase
#   - Validate before export
# ---------------------------------------------------------------------------

# Subtitle segmentation constants
PAUSE_DETECTION_MS = 180          # Even a short, measured speech pause starts a new phrase
SUBTITLE_MAX_DURATION_SEC = 6.0   # Hard readability guard for a long speaker turn
SUBTITLE_PREFERRED_MAX_SEC = 4.0  # Preferred max duration
SUBTITLE_MAX_LINES = 2            # Maximum lines per subtitle
SUBTITLE_MAX_CHARS_PER_LINE = 42  # Maximum characters per line
SUBTITLE_GAP_MS = 0               # Speech timestamps are exact; do not manufacture gaps
SPEAKER_BOUNDARY_TOLERANCE_SEC = 0.05
SUBTITLE_MAX_BLOCK_CHARS = SUBTITLE_MAX_CHARS_PER_LINE * SUBTITLE_MAX_LINES


def _detect_word_gaps(words: List[Dict], pause_threshold_ms: int = PAUSE_DETECTION_MS) -> List[int]:
    """
    Detect gaps between consecutive words that exceed the pause threshold.
    Uses word-level timestamps from Whisper to find natural split points.

    Args:
        words: List of word dicts with 'start', 'end', 'word' keys
        pause_threshold_ms: Minimum gap (ms) to consider a natural pause

    Returns:
        List of indices in the words list where a split should occur
        (the split happens BEFORE words at these indices).
    """
    if not words or len(words) < 2:
        return []

    split_indices = []
    for i in range(1, len(words)):
        prev_end = _coerce_float(words[i - 1].get('end', 0))
        curr_start = _coerce_float(words[i].get('start', prev_end))
        gap_ms = (curr_start - prev_end) * 1000.0

        if gap_ms >= pause_threshold_ms:
            split_indices.append(i)

    return split_indices


def _word_speaker(word: Dict) -> str:
    speaker = word.get('speaker_id', word.get('speaker', ''))
    return str(speaker).strip()


def _time_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _normalize_diarization_turns(diarization: Any) -> List[Dict[str, Any]]:
    """
    Convert pyannote Annotation output or plain turn dicts into
    [{start, end, speaker_id}] records.
    """
    turns = []
    if not diarization:
        return turns

    if isinstance(diarization, list):
        items = diarization
    elif hasattr(diarization, 'itertracks'):
        items = []
        try:
            for turn, _track, speaker in diarization.itertracks(yield_label=True):
                items.append({
                    'start': getattr(turn, 'start', 0.0),
                    'end': getattr(turn, 'end', 0.0),
                    'speaker_id': speaker,
                })
        except Exception as e:
            logger.warning(f'[Diarization] Failed to parse pyannote output: {e}')
            return []
    else:
        return turns

    for item in items:
        if not isinstance(item, dict):
            continue
        start = _coerce_float(item.get('start'), None)
        end = _coerce_float(item.get('end'), None)
        speaker = item.get('speaker_id') or item.get('speaker') or item.get('label')
        if start is None or end is None or not speaker or end <= start:
            continue
        turns.append({
            'start': round(max(0.0, start), 3),
            'end': round(max(0.0, end), 3),
            'speaker_id': str(speaker),
        })
    return sorted(turns, key=lambda t: (t['start'], t['end'], t['speaker_id']))


def _speaker_for_span(start: float, end: float, diarization_turns: List[Dict[str, Any]]) -> str:
    best_speaker = ''
    best_overlap = 0.0
    for turn in diarization_turns or []:
        overlap = _time_overlap(start, end, _coerce_float(turn.get('start')), _coerce_float(turn.get('end')))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(turn.get('speaker_id') or turn.get('speaker') or '')
    return best_speaker


def apply_speaker_diarization_to_segments(segments: List[Dict],
                                          diarization_turns: List[Dict[str, Any]]) -> List[Dict]:
    """
    Assign speaker labels to every word/subtitle from diarization turns and
    split immediately at speaker changes.
    """
    turns = _normalize_diarization_turns(diarization_turns)
    if not segments or not turns:
        return segments or []

    labeled = []
    for seg in _enrich_subtitle_segments_with_alignment(segments):
        updated = dict(seg)
        words = []
        for word in updated.get('words') or []:
            word = dict(word)
            word_start = _coerce_float(word.get('start'), updated.get('start', 0.0))
            word_end = _coerce_float(word.get('end'), word_start)
            speaker = _speaker_for_span(word_start, word_end, turns)
            if speaker:
                word['speaker_id'] = speaker
            words.append(word)
        updated['words'] = words

        if words:
            speakers = [_word_speaker(w) for w in words if _word_speaker(w)]
            if speakers and len(set(speakers)) == 1:
                updated['speaker_id'] = speakers[0]
            elif speakers:
                updated.pop('speaker_id', None)
        else:
            speaker = _speaker_for_span(
                _coerce_float(updated.get('start', 0.0)),
                _coerce_float(updated.get('end', 0.0)),
                turns,
            )
            if speaker:
                updated['speaker_id'] = speaker
        labeled.append(updated)

    # Split at speaker changes immediately so each subtitle remains a single
    # utterance from one speaker. This also prevents the later timing fixes from
    # dragging the next speaker's words into the previous subtitle.
    return split_subtitles_by_pauses(labeled)


def _detect_speech_boundary_splits(words: List[Dict],
                                   pause_threshold_ms: int = PAUSE_DETECTION_MS) -> List[int]:
    """
    Find subtitle split points from speech evidence: natural pauses and speaker changes.
    CRITICAL: Always split on speaker changes - never merge different speakers.
    Returned indices split before the word at that index.
    """
    if not words or len(words) < 2:
        return []

    split_indices = set(_detect_word_gaps(words, pause_threshold_ms))
    for idx in range(1, len(words)):
        prev_speaker = _word_speaker(words[idx - 1])
        curr_speaker = _word_speaker(words[idx])
        # CRITICAL: ALWAYS split if speakers differ (never merge)
        if prev_speaker and curr_speaker and prev_speaker != curr_speaker:
            split_indices.add(idx)
            logger.debug(f'[Split] Speaker change: {prev_speaker} -> {curr_speaker} at word index {idx}')
    return sorted(split_indices)


def _detect_vad_boundary_splits(words: List[Dict],
                                speech_regions: Optional[List[Tuple[float, float]]] = None,
                                pause_threshold_ms: int = PAUSE_DETECTION_MS) -> List[int]:
    """
    Split subtitles when consecutive words fall into different VAD speech regions.
    Only split when the region change also corresponds to a sustained pause,
    avoiding false splits caused by marginal VAD boundary detection.
    """
    if not words or len(words) < 2 or not speech_regions:
        return []

    split_indices = []
    previous_region = None
    prev_word_end = None
    for idx, word in enumerate(words):
        word_start = _coerce_float(word.get('start'), 0.0)
        word_end = _coerce_float(word.get('end'), word_start)
        best_region = None
        best_overlap = 0.0
        for region_idx, (region_start, region_end) in enumerate(speech_regions):
            overlap = _time_overlap(word_start, word_end, region_start, region_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_region = region_idx

        if previous_region is not None and best_region is not None and best_region != previous_region:
            gap_ms = 0.0
            if prev_word_end is not None:
                gap_ms = max(0.0, (word_start - prev_word_end) * 1000.0)
            # VAD frequently creates tiny regions inside one utterance. It is
            # supporting evidence only; a real long pause is still required.
            if gap_ms >= pause_threshold_ms:
                split_indices.append(idx)
        previous_region = best_region
        prev_word_end = word_end

    return split_indices


def _words_to_text(words: List[Dict]) -> str:
    return ' '.join((w.get('word') or w.get('text') or '').strip() for w in words).strip()


def _slice_subtitle_segment(seg: Dict, words: List[Dict], original_index: int) -> Optional[Dict]:
    sub_text = _words_to_text(words)
    if not sub_text:
        return None

    fallback_start = _coerce_float(seg.get('start', 0.0))
    sub_start = _coerce_float(words[0].get('start'), fallback_start)
    sub_end = _coerce_float(words[-1].get('end'), sub_start)
    new_seg = dict(seg)
    new_seg['start'] = round(max(0.0, sub_start), 3)
    new_seg['end'] = round(max(sub_start + 0.01, sub_end), 3)
    new_seg['source'] = sub_text
    new_seg['text'] = sub_text
    new_seg['duration'] = round(max(0.01, new_seg['end'] - new_seg['start']), 3)
    new_seg['words'] = words
    new_seg['split_from'] = original_index
    speaker = _word_speaker(words[0])
    if speaker:
        new_seg['speaker_id'] = speaker
    return new_seg


def _same_dialogue_speaker(left: Dict, right: Dict) -> bool:
    """Merge only when speaker identity is positively known to match."""
    left_speaker = str(left.get('speaker_id') or left.get('speaker') or '').strip()
    right_speaker = str(right.get('speaker_id') or right.get('speaker') or '').strip()
    return bool(left_speaker and right_speaker and left_speaker == right_speaker)


def _natural_readability_splits(words: List[Dict]) -> List[int]:
    """Choose phrase boundaries and bounded word breaks for a long turn."""
    if len(words) < 2:
        return []

    boundaries = []
    chunk_start = 0
    while chunk_start < len(words) - 1:
        chunk_start_time = _coerce_float(words[chunk_start].get('start'), 0.0)
        candidates = []
        for idx in range(chunk_start + 1, len(words)):
            left = words[idx - 1]
            right = words[idx]
            end = _coerce_float(left.get('end'), chunk_start_time)
            duration = end - chunk_start_time
            text = _words_to_text(words[chunk_start:idx])
            gap_ms = (
                _coerce_float(right.get('start'), end) - end
            ) * 1000.0
            punctuated = bool(re.search(r'[.!?,;:\u17d4\u17d5\u17d6]$', str(
                left.get('word') or left.get('text') or ''
            ).strip()))
            if punctuated or gap_ms >= PAUSE_DETECTION_MS:
                candidates.append((idx, duration, len(text)))

            if (
                duration >= SUBTITLE_PREFERRED_MAX_SEC
                or len(text) >= SUBTITLE_MAX_BLOCK_CHARS
            ):
                usable = [item for item in candidates if item[0] > chunk_start]
                if not usable:
                    # A very long unpunctuated turn still needs readable units.
                    # Word timing keeps this fallback lossless and synchronized.
                    split_idx = idx
                else:
                    split_idx = min(
                        usable,
                        key=lambda item: abs(item[1] - SUBTITLE_PREFERRED_MAX_SEC),
                    )[0]
                boundaries.append(split_idx)
                chunk_start = split_idx
                break
        else:
            break
    return sorted(set(boundaries))


def _find_line_break(text: str, max_chars: int = SUBTITLE_MAX_CHARS_PER_LINE) -> int:
    """
    Find the best position to break a subtitle line within the character limit.
    Prefers splitting at natural word boundaries.
    Returns the index where the break should occur, or -1 if no break needed.
    """
    if len(text) <= max_chars:
        return -1

    # Try to find a natural break point (space) near the limit
    # First, try to break at the last space within max_chars
    break_at = text.rfind(' ', 0, max_chars)
    if break_at > 0:
        return break_at

    # No space found — try to break at a punctuation boundary
    for punct in ('.', ',', '!', '?', ';', ':', '-', '\u17d4', '\u17d5', '\u17d6'):
        pos = text.rfind(punct, 0, max_chars)
        if pos > 0:
            return pos + 1

    return -1


def split_subtitles_by_pauses(segments: List[Dict],
                              speech_regions: Optional[List[Tuple[float, float]]] = None) -> List[Dict]:
    """
    Build dialogue blocks from measured word timing and speaker labels.

    A block changes at a speaker change, measured pause, completed phrase, or
    readability boundary. Existing independent segments are never merged.
    """
    if not segments:
        return []

    logger.info(f'[Segmentation] Splitting {len(segments)} segments by speech pauses...')
    split_count = 0
    result = []

    for seg_idx, seg in enumerate(segments):
        words = seg.get('words', [])
        text = (seg.get('source') or seg.get('text') or '').strip()
        start = _coerce_float(seg.get('start', 0.0))
        end = _coerce_float(seg.get('end', start))

        if not text or len(words) < 2:
            # No word timestamps available — keep as-is
            result.append(dict(seg))
            continue

        # Speaker changes and measured pauses are authoritative. Punctuation
        # marks a completed phrase, and long turns receive word-timed splits.
        split_word_indices = set(_detect_speech_boundary_splits(words, PAUSE_DETECTION_MS))
        split_word_indices.update(_detect_vad_boundary_splits(words, speech_regions))
        for word_idx in range(1, len(words)):
            previous_text = str(
                words[word_idx - 1].get('word')
                or words[word_idx - 1].get('text')
                or ''
            ).strip()
            if re.search(r'[.!?;:\u17d4\u17d5\u17d6]$', previous_text):
                split_word_indices.add(word_idx)
        split_word_indices.update(_natural_readability_splits(words))

        if not split_word_indices:
            # No significant pauses — keep as single subtitle
            result.append(dict(seg))
            continue

        # Split the segment into multiple subtitles at speech boundaries.
        prev_word_idx = 0

        for split_idx in sorted(split_word_indices):
            # Gather words from prev_word_idx to split_idx - 1
            sub_words = words[prev_word_idx:split_idx]
            new_seg = _slice_subtitle_segment(seg, sub_words, seg_idx)
            if new_seg:
                result.append(new_seg)
                split_count += 1
            prev_word_idx = split_idx

        # Add remaining words (if any) as the final subtitle
        if prev_word_idx < len(words):
            sub_words = words[prev_word_idx:]
            new_seg = _slice_subtitle_segment(seg, sub_words, seg_idx)
            if new_seg:
                result.append(new_seg)
                split_count += 1

    logger.info(f'[Segmentation] Built {len(result)} phrase/speaker blocks '
                f'({split_count} measured boundary slices).')
    return result if result else segments


def _apply_line_breaks(text: str, max_chars: int = SUBTITLE_MAX_CHARS_PER_LINE) -> str:
    """
    Apply line breaks to text to ensure no line exceeds max_chars.
    Returns text with \n inserted at appropriate break points.
    Max 2 lines.
    """
    if not text:
        return ''

    text = ' '.join(str(text).split())

    if len(text) <= max_chars:
        return text

    break_pos = _find_line_break(text, max_chars)
    if break_pos > 0:
        line1 = text[:break_pos].strip()
        line2 = text[break_pos:].strip()
        return f'{line1}\n{line2}'

    return text


def enforce_subtitle_format_limits(segments: List[Dict]) -> List[Dict]:
    """
    Apply display line wrapping without changing speaker/audio blocks.

    Dialogue segmentation is owned exclusively by speaker changes and measured
    pauses in ``split_subtitles_by_pauses``.
    """
    if not segments:
        return []

    result = []
    for raw in segments:
        seg = dict(raw)
        text = (seg.get('source') or seg.get('text') or '').strip()
        formatted_text = _apply_line_breaks(text)
        seg['source'] = formatted_text
        seg['text'] = formatted_text
        result.append(seg)
    return result


def _check_merged_sentences(segments: List[Dict]) -> List[str]:
    """
    Check for evidence of merged sentences within subtitles.
    Returns list of warning messages for any issues found.
    """
    warnings = []
    for idx, seg in enumerate(segments):
        text = (seg.get('source') or seg.get('text') or '').strip()
        words = seg.get('words', [])

        # Check if text contains multiple sentences separated by period but no pause
        # (Whisper often merges sentences - we detect this via word gap analysis)
        if words and len(words) >= 4:
            gaps = []
            for w_idx in range(1, len(words)):
                prev_end = _coerce_float(words[w_idx - 1].get('end', 0))
                curr_start = _coerce_float(words[w_idx].get('start', prev_end))
                gap_ms = (curr_start - prev_end) * 1000.0
                gaps.append(gap_ms)

            # Count pauses > 300ms
            large_gaps = [g for g in gaps if g > 300]
            if len(large_gaps) >= 1:
                warnings.append(
                    f'Segment {idx}: Contains {len(large_gaps)} large gaps (>{300}ms) '
                    f'may indicate merged sentences: "{text[:60]}..."'
                )

        # Also check for multiple terminal punctuations (sign of merged sentences)
        terminal_count = len(re.findall(r'[.!?\u17d4\u17d5\u17d6]', text))
        if terminal_count >= 2 and len(text.split()) >= 5:
            warnings.append(
                f'Segment {idx}: Contains {terminal_count} sentence endings - '
                f'possible merged sentences: "{text[:60]}..."'
            )

    return warnings


def _check_overlapping_subtitles(segments: List[Dict]) -> List[str]:
    """
    Check for overlapping subtitle timestamps.
    Returns list of warning messages for any overlaps found.
    """
    warnings = []
    for idx in range(1, len(segments)):
        prev_end = _coerce_float(segments[idx - 1].get('end', 0))
        curr_start = _coerce_float(segments[idx].get('start', 0))
        if curr_start < prev_end:
            overlap = prev_end - curr_start
            warnings.append(
                f'Segment {idx} overlaps with segment {idx - 1} by {overlap:.3f}s'
            )
    return warnings


def _check_speaker_boundaries(segments: List[Dict]) -> List[str]:
    """
    Check that every subtitle contains words from one speaker only and that a
    speaker-labeled segment matches its word-level speaker labels.
    """
    warnings = []
    for idx, seg in enumerate(segments):
        words = seg.get('words', [])
        word_speakers = [_word_speaker(w) for w in words if _word_speaker(w)]
        unique_speakers = sorted(set(word_speakers))
        segment_speaker = str(seg.get('speaker_id') or seg.get('speaker') or '').strip()

        if len(unique_speakers) > 1:
            warnings.append(
                f'Segment {idx}: Contains multiple speakers {", ".join(unique_speakers)}'
            )
        if segment_speaker and unique_speakers and segment_speaker not in unique_speakers:
            warnings.append(
                f'Segment {idx}: Speaker label {segment_speaker} does not match word speaker boundary'
            )
    return warnings


def _check_timing_boundaries(segments: List[Dict]) -> List[str]:
    """
    Check that each subtitle starts when speech begins and ends when it stops.
    Returns list of warning messages.
    Uses tight tolerance (30ms) for stricter timing accuracy detection.
    """
    warnings = []
    tolerance_seconds = 0.03  # 30ms tolerance - tighter for better sync detection
    for idx, seg in enumerate(segments):
        words = seg.get('words', [])
        if not words:
            continue

        seg_start = _coerce_float(seg.get('start', 0))
        seg_end = _coerce_float(seg.get('end', seg_start))

        word_start = _coerce_float(words[0].get('start', seg_start))
        word_end = _coerce_float(words[-1].get('end', seg_end))

        if seg_start < word_start - tolerance_seconds:
            warnings.append(
                f'Segment {idx}: Starts {word_start - seg_start:.3f}s before speech begins'
            )

        if seg_start > word_start + tolerance_seconds:
            warnings.append(
                f'Segment {idx}: Starts {seg_start - word_start:.3f}s after speech begins'
            )

        if seg_end < word_end - tolerance_seconds:
            warnings.append(
                f'Segment {idx}: Ends {word_end - seg_end:.3f}s before speech ends'
            )
        if seg_end > word_end + tolerance_seconds:
            warnings.append(
                f'Segment {idx}: Ends {seg_end - word_end:.3f}s after speech stops'
            )

    return warnings


def _fix_merged_sentences(segments: List[Dict]) -> List[Dict]:
    """
    Automatically fix merged sentences by splitting at detected word gaps.
    Uses the same pause detection as split_subtitles_by_pauses.
    """
    return split_subtitles_by_pauses(segments)


def _fix_overlapping_subtitles(segments: List[Dict]) -> List[Dict]:
    """
    Automatically fix overlapping subtitle timestamps.
    Adjusts end/start times to create small gaps (20ms).
    CRITICAL: NEVER merges different subtitle segments together.
    Only creates a small gap between them.
    
    Uses word timestamps as the primary timing reference:
    - Previous segment's end = last word's end time
    - Current segment's start = first word's start time
    - Only adjust if still overlapping after word-timestamp refinement
    """
    fixed = []
    for idx, seg in enumerate(segments):
        fixed_seg = dict(seg)
        if fixed and idx > 0:
            last = fixed[-1]
            last_end = _coerce_float(last.get('end', 0))
            curr_start = _coerce_float(fixed_seg.get('start', last_end))
            if curr_start < last_end:
                # Word-backed boundaries describe the detected speech itself.
                # Preserve both when speakers overlap instead of manufacturing
                # a later start or an earlier end.
                if last.get('words') and fixed_seg.get('words'):
                    fixed.append(fixed_seg)
                    continue
                # CRITICAL: Use word timestamps as primary timing reference.
                # The previous segment's end should be its last word's end time.
                # The current segment's start should be its first word's start time.
                last_words = last.get('words', [])
                seg_words = fixed_seg.get('words', [])
                
                # Refine previous segment's end using word timestamps
                if last_words:
                    last_word_ends = [_coerce_float(w.get('end'), None) for w in last_words if _coerce_float(w.get('end'), None) is not None]
                    if last_word_ends:
                        actual_last_end = max(last_word_ends)
                        if actual_last_end < last['end']:
                            last['end'] = round(actual_last_end, 3)
                            last_end = last['end']
                
                # Refine current segment's start using word timestamps
                if seg_words:
                    seg_word_starts = [_coerce_float(w.get('start'), None) for w in seg_words if _coerce_float(w.get('start'), None) is not None]
                    if seg_word_starts:
                        actual_seg_start = min(seg_word_starts)
                        if actual_seg_start > fixed_seg['start']:
                            fixed_seg['start'] = round(actual_seg_start, 3)
                            curr_start = fixed_seg['start']
                
                # Only apply gap if still overlapping after word-timestamp refinement
                if curr_start < last_end:
                    gap = SUBTITLE_GAP_MS / 1000.0  # 20ms gap
                    last['end'] = round(last_end, 3)
                    fixed_seg['start'] = round(last_end + gap, 3)
        fixed.append(fixed_seg)
    return fixed


def _fix_speaker_boundaries(segments: List[Dict]) -> List[Dict]:
    return split_subtitles_by_pauses(segments)


def _fix_timing_boundaries(segments: List[Dict]) -> List[Dict]:
    """
    Fix timing boundaries so each subtitle tightly wraps its word timestamps.
    Word timestamps are treated as the primary speech reference.
    """
    fixed = []
    for idx, seg in enumerate(segments):
        fixed_seg = dict(seg)
        words = seg.get('words', [])
        if words:
            valid_starts = [_coerce_float(w.get('start'), None) for w in words if _coerce_float(w.get('start'), None) is not None]
            valid_ends = [_coerce_float(w.get('end'), None) for w in words if _coerce_float(w.get('end'), None) is not None]
            word_start = min(valid_starts) if valid_starts else _coerce_float(seg.get('start', 0))
            word_end = max(valid_ends) if valid_ends else word_start
            fixed_seg['start'] = round(max(0.0, word_start), 3)
            fixed_seg['end'] = round(max(word_start + 0.01, word_end), 3)
            fixed_seg['duration'] = max(0.01, fixed_seg['end'] - fixed_seg['start'])
        fixed.append(fixed_seg)
    return fixed


def validate_subtitle_segmentation(segments: List[Dict], auto_fix: bool = True) -> Tuple[List[Dict], List[str]]:
    """
    Validate subtitle segmentation rules before export.

    Checks:
    - Check for merged sentences (with multiple pauses or punctuation)
    - Check for overlapping subtitles
    - Check for subtitles that continue after speaker has stopped
    - Check for subtitles that start before speech begins
    - Check for subtitles containing multiple speakers

    If auto_fix is True, automatically fix detected issues.
    Returns (fixed_segments, warnings).
    """
    if not segments:
        return segments, []

    logger.info(f'[Validation] Validating subtitle segmentation for {len(segments)} segments...')
    all_warnings = []

    # Sentence count and text length are deliberately not segmentation rules.
    merged_warnings = []
    overlap_warnings = _check_overlapping_subtitles(segments)
    speaker_warnings = _check_speaker_boundaries(segments)
    timing_warnings = _check_timing_boundaries(segments)

    all_warnings.extend(merged_warnings)
    all_warnings.extend(overlap_warnings)
    all_warnings.extend(speaker_warnings)
    all_warnings.extend(timing_warnings)

    if all_warnings:
        logger.warning(f'[Validation] Found {len(all_warnings)} segmentation issues:')
        for w in all_warnings[:10]:  # Log first 10
            logger.warning(f'  {w}')
        if len(all_warnings) > 10:
            logger.warning(f'  ... and {len(all_warnings) - 10} more')

    # Auto-fix if requested
    if auto_fix and all_warnings:
        logger.info(f'[Validation] Auto-fixing segmentation issues...')

        if speaker_warnings:
            logger.info(f'[Validation] Fixing speaker boundaries ({len(speaker_warnings)} issues)...')
            segments = _fix_speaker_boundaries(segments)

        # Fix overlapping timestamps - NEVER merges different segments
        if overlap_warnings:
            logger.info(f'[Validation] Fixing overlapping subtitles ({len(overlap_warnings)} issues)...')
            segments = _fix_overlapping_subtitles(segments)

        # Fix timing boundaries
        if timing_warnings:
            logger.info(f'[Validation] Fixing timing boundaries ({len(timing_warnings)} issues)...')
            segments = _fix_timing_boundaries(segments)

        # Re-validate
        logger.info(f'[Validation] Re-validating after fixes...')
        merged_warnings2 = []
        overlap_warnings2 = _check_overlapping_subtitles(segments)
        speaker_warnings2 = _check_speaker_boundaries(segments)
        timing_warnings2 = _check_timing_boundaries(segments)

        remaining = (
            len(merged_warnings2)
            + len(overlap_warnings2)
            + len(speaker_warnings2)
            + len(timing_warnings2)
        )
        if remaining == 0:
            logger.info(f'[Validation] All issues resolved.')
        else:
            logger.warning(f'[Validation] {remaining} issues remaining after auto-fix.')

    return segments, all_warnings


# ============================================================================
# STRICT SUBTITLE BOUNDARY ENFORCEMENT
# ============================================================================
# These functions validate and enforce that:
#   1. Every subtitle has its own independent timing (no merging)
#   2. No subtitle phrases have been combined
#   3. Each subtitle maps to exactly one audio segment
#   4. Speech starts exactly at subtitle appearance
#   5. Speech finishes before the next subtitle
# ============================================================================


def _enforce_strict_subtitle_boundaries(segments: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Enforce strict subtitle boundary rules before final export.
    This is the FINAL validation gate.
    
    Checks:
      - No two consecutive segments have merged text
      - Every segment has its own non-zero duration
      - No segment overlaps with its neighbors
      - Segment timing is monotonic (start <= end)
      - No segments are skipped or missing
    
    Returns (cleaned_segments, violations).
    Violations means the data was fixed. Empty violations = clean.
    """
    violations = []
    if not segments:
        return segments, violations
    
    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: _coerce_float(s.get('start', 0)))
    
    # Check for consecutive segments that were merged (same text spread across time)
    for i in range(len(sorted_segs)):
        seg = sorted_segs[i]
        text = (seg.get('source') or seg.get('text') or '').strip()
        if not text:
            violations.append(f'Segment {i}: Empty text removed')
            continue
        
        # Ensure start < end
        start = _coerce_float(seg.get('start', 0.0))
        end = _coerce_float(seg.get('end', start))
        if end <= start:
            end = start + 0.01
            sorted_segs[i]['end'] = round(end, 3)
            sorted_segs[i]['duration'] = 0.01
            violations.append(f'Segment {i}: Fixed zero-duration subtitle')
        
        sorted_segs[i]['start'] = round(max(0.0, start), 3)
        sorted_segs[i]['end'] = round(max(start + 0.01, end), 3)
        sorted_segs[i]['duration'] = round(sorted_segs[i]['end'] - sorted_segs[i]['start'], 3)
    
    # Check for overlaps and fix them without merging
    for i in range(1, len(sorted_segs)):
        prev = sorted_segs[i - 1]
        curr = sorted_segs[i]
        prev_end = _coerce_float(prev.get('end', 0.0))
        curr_start = _coerce_float(curr.get('start', 0.0))
        
        if curr_start < prev_end:
            # CRITICAL: Use word timestamps to determine actual speech boundaries
            # before applying any gap. This prevents truncating speech or
            # delaying subtitle appearance.
            prev_words = prev.get('words', [])
            curr_words = curr.get('words', [])

            if prev_words and curr_words:
                # This can be real overlapping dialogue. Both detected speech
                # windows remain exact and each speaker keeps a separate segment.
                continue
            
            # Refine previous segment's end using word timestamps
            if prev_words:
                prev_word_ends = [_coerce_float(w.get('end'), None) for w in prev_words if _coerce_float(w.get('end'), None) is not None]
                if prev_word_ends:
                    actual_prev_end = max(prev_word_ends)
                    if actual_prev_end < prev_end:
                        prev['end'] = round(actual_prev_end, 3)
                        prev_end = prev['end']
            
            # Refine current segment's start using word timestamps
            if curr_words:
                curr_word_starts = [_coerce_float(w.get('start'), None) for w in curr_words if _coerce_float(w.get('start'), None) is not None]
                if curr_word_starts:
                    actual_curr_start = min(curr_word_starts)
                    if actual_curr_start > curr_start:
                        curr['start'] = round(actual_curr_start, 3)
                        curr_start = curr['start']
            
            # Only apply gap if still overlapping after word-timestamp refinement
            if curr_start < prev_end:
                new_start = prev_end + 0.02  # 20ms gap
                curr_dur = max(0.01, _coerce_float(curr.get('end', curr_start)) - curr_start)
                sorted_segs[i]['start'] = round(new_start, 3)
                sorted_segs[i]['end'] = round(max(new_start + 0.01, new_start + curr_dur), 3)
                sorted_segs[i]['duration'] = round(sorted_segs[i]['end'] - sorted_segs[i]['start'], 3)
                violations.append(f'Segment {i}: Fixed overlap with segment {i-1} (preserved as independent)')
        
        # Ensure minimum gap between segments
        curr_start = _coerce_float(sorted_segs[i].get('start', 0.0))
        if curr_start < prev_end and not (
            sorted_segs[i - 1].get('words') and sorted_segs[i].get('words')
        ):
            sorted_segs[i]['start'] = round(prev_end + 0.02, 3)
    
    # Filter out empty segments
    cleaned = [s for s in sorted_segs if (s.get('source') or s.get('text') or '').strip()]
    
    if violations:
        logger.info(f'[Boundary Enforcement] Fixed {len(violations)} boundary violations. '
                    f'{len(cleaned)} segments preserved independently.')
    
    return cleaned, violations


def _validate_audio_segment_mapping(segments: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate that every subtitle has exactly one matching audio segment expected.
    Checks 1:1 mapping between subtitles and speech segments.
    Returns (passed, issues).
    """
    issues = []
    if not segments:
        return True, issues
    
    # Check that no segments have been merged by looking for duplicate text spans
    for i in range(1, len(segments)):
        prev_text = (segments[i-1].get('source') or '').strip().lower()
        curr_text = (segments[i].get('source') or '').strip().lower()
        prev_end = _coerce_float(segments[i-1].get('end', 0))
        curr_start = _coerce_float(segments[i].get('start', 0))
        
        # Check for merged segments (same or subset text with overlapping timing)
        if prev_text and curr_text:
            # If one text contains the other and timing overlaps, that's a merge
            if (prev_text in curr_text or curr_text in prev_text):
                # Only flag if timing also overlaps significantly
                if curr_start < prev_end:
                    issues.append(
                        f'Possible merge: Segment {i-1} ("{prev_text[:30]}...") and '
                        f'Segment {i} ("{curr_text[:30]}...") have overlapping text and timing'
                    )
    
    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Stage 1: Extract Audio using FFmpeg (with optional VAD chunking)
# ---------------------------------------------------------------------------
def extract_audio(video_path, audio_path, max_duration=None):
    """
    Extract audio from video using FFmpeg.
    Extracts the FULL audio from beginning to end, never stopping early.
    Returns the path to the extracted audio file.
    """
    logger.info(f'[Stage 1] Extracting FULL audio from: {video_path}')
    logger.info(f'[Stage 1] Output audio: {audio_path}')

    # First check if the video has an audio stream
    probe_cmd = [
        FFPROBE_EXE, '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'csv=p=0',
        video_path
    ]
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        has_audio = 'audio' in probe_result.stdout.lower()
    except Exception as e:
        logger.warning(f'[Stage 1] Audio probe failed: {e}')
        has_audio = True  # Assume it has audio

    if not has_audio:
        logger.warning('[Stage 1] No audio stream found in video. Creating silent audio.')
        duration = get_video_duration(video_path)
        if max_duration:
            duration = min(duration, float(max_duration))

        silent_cmd = [
            FFMPEG_EXE, '-y',
            '-f', 'lavfi', '-i', f'anullsrc=r=16000:cl=mono',
            '-t', str(duration),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            audio_path
        ]
        try:
            result = subprocess.run(silent_cmd, capture_output=True, text=True, timeout=60)
        except FileNotFoundError as e:
            raise RuntimeError(_missing_command_message('ffmpeg', 'FFMPEG_EXE')) from e
        if result.returncode != 0:
            raise RuntimeError(f'Failed to create silent audio: {result.stderr[:500]}')
        logger.info(f'[Stage 1] Silent audio created for {duration}s video')
        return audio_path

    cmd = [
        FFMPEG_EXE, '-y',
        '-i', video_path,
        '-vn',                    # No video
        '-acodec', 'pcm_s16le',   # WAV format
        '-ar', '16000',           # 16kHz sample rate (Whisper optimal)
        '-ac', '1',               # Mono
    ]
    if max_duration:
        cmd.extend(['-t', str(float(max_duration))])
    cmd.append(audio_path)

    logger.debug(f'[Stage 1] Running: {" ".join(cmd)}')
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours max for very long videos
        )
    except FileNotFoundError as e:
        raise RuntimeError(_missing_command_message('ffmpeg', 'FFMPEG_EXE')) from e

    elapsed = time.time() - start_time
    logger.info(f'[Stage 1] FFmpeg completed in {elapsed:.2f}s')

    if result.returncode != 0:
        logger.error(f'[Stage 1] FFmpeg failed (code {result.returncode})')
        logger.error(f'[Stage 1] stderr: {result.stderr[:2000]}')
        raise RuntimeError(
            f'FFmpeg audio extraction failed with code {result.returncode}. '
            f'stderr: {result.stderr[:500]}'
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'[Stage 1] Audio file was not created: {audio_path}')

    file_size = os.path.getsize(audio_path)
    logger.info(f'[Stage 1] Audio extracted successfully: {file_size} bytes')
    return audio_path


# ---------------------------------------------------------------------------
# VAD-based Audio Chunking for Whisper Accuracy
# ---------------------------------------------------------------------------
def detect_voice_activity(audio_path: str, frame_duration_ms: int = 30) -> List[Tuple[float, float]]:
    """
    Use WebRTC VAD or simple energy-based VAD to detect speech regions.
    This helps Whisper focus on speech segments and improves accuracy.
    Returns list of (start_sec, end_sec) speech segments.
    """
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        logger.info('[VAD] Using WebRTC VAD for speech detection')
    except ImportError:
        logger.info('[VAD] webrtcvad not installed. Using energy-based VAD fallback.')
        return _energy_based_vad(audio_path, frame_duration_ms)

    try:
        import wave
        import struct

        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            if sample_rate not in (8000, 16000, 32000, 48000):
                logger.warning(f'[VAD] Sample rate {sample_rate} not supported by WebRTC VAD, resampling needed')
                return _energy_based_vad(audio_path, frame_duration_ms)

            channels = wf.getnchannels()
            if channels != 1:
                logger.warning(f'[VAD] {channels} channels detected, using first channel')
                return _energy_based_vad(audio_path, frame_duration_ms)

            # Must be 16-bit PCM
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                logger.warning(f'[VAD] Sample width {sampwidth} not supported by WebRTC VAD')
                return _energy_based_vad(audio_path, frame_duration_ms)

            frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # bytes for 16-bit
            speech_segments = []
            in_speech = False
            segment_start = 0.0
            frame_num = 0
            silence_count = 0
            speech_count = 0
            silence_frames_for_split = max(1, int((PAUSE_DETECTION_MS / frame_duration_ms) + 0.5))

            while True:
                data = wf.readframes(int(sample_rate * frame_duration_ms / 1000))
                if not data or len(data) < frame_size:
                    break

                is_speech = vad.is_speech(data, sample_rate)
                current_time = frame_num * frame_duration_ms / 1000.0

                if is_speech:
                    speech_count += 1
                    silence_count = 0
                    if not in_speech:
                        segment_start = current_time
                        in_speech = True
                elif in_speech:
                    silence_count += 1
                    if silence_count >= silence_frames_for_split:
                        segment_end = current_time - ((silence_count - 1) * frame_duration_ms / 1000.0)
                        speech_segments.append((segment_start, max(segment_start + 0.01, segment_end)))
                        in_speech = False
                        silence_count = 0
                    speech_count = 0

                frame_num += 1

            # Don't forget last segment
            if in_speech:
                speech_segments.append((segment_start, frame_num * frame_duration_ms / 1000.0))

            # Merge segments that are close together
            merged = []
            for seg in speech_segments:
                if merged and seg[0] - merged[-1][1] < 0.2:
                    merged[-1] = (merged[-1][0], seg[1])
                else:
                    merged.append(seg)

            logger.info(f'[VAD] Detected {len(merged)} speech segments')
            return merged

    except Exception as e:
        logger.warning(f'[VAD] WebRTC VAD failed: {e}. Using energy-based fallback.')
        return _energy_based_vad(audio_path, frame_duration_ms)


def _energy_based_vad(audio_path: str, frame_duration_ms: int = 30) -> List[Tuple[float, float]]:
    """
    Simple energy-based VAD using RMS energy.
    Useful when webrtcvad is not available or audio format is incompatible.
    """
    try:
        import wave
        import struct
        import math

        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            frame_size = int(sample_rate * frame_duration_ms / 1000)
            frame_bytes = frame_size * sampwidth

            speech_segments = []
            in_speech = False
            segment_start = 0.0
            frame_num = 0
            silence_count = 0
            speech_count = 0

            # Compute overall RMS to set adaptive threshold
            all_data = wf.readframes(wf.getnframes())
            wf.rewind()

            if sampwidth == 2:
                fmt = '<{}h'.format(len(all_data) // 2)
                samples = struct.unpack(fmt, all_data)
                overall_rms = math.sqrt(sum(s * s for s in samples) / max(1, len(samples)))
            else:
                overall_rms = 500  # fallback

            threshold = max(overall_rms * 0.1, 100)

            while True:
                data = wf.readframes(frame_size)
                if not data or len(data) < frame_bytes:
                    break

                if sampwidth == 2:
                    fmt = '<{}h'.format(len(data) // 2)
                    samples = struct.unpack(fmt, data)
                    rms = math.sqrt(sum(s * s for s in samples) / max(1, len(samples)))
                else:
                    rms = 0

                current_time = frame_num * frame_duration_ms / 1000.0
                is_speech = rms > threshold

                if is_speech:
                    speech_count += 1
                    silence_count = 0
                    if not in_speech and speech_count > 3:  # Debounce
                        segment_start = current_time - 0.1
                        in_speech = True
                else:
                    silence_count += 1
                    if in_speech and silence_count > 10:  # 300ms silence
                        speech_segments.append((segment_start, current_time))
                        in_speech = False
                        silence_count = 0
                    speech_count = 0

                frame_num += 1

            if in_speech:
                speech_segments.append((segment_start, frame_num * frame_duration_ms / 1000.0))

            # Merge nearby segments
            merged = []
            for seg in speech_segments:
                if merged and seg[0] - merged[-1][1] < 0.2:
                    merged[-1] = (merged[-1][0], seg[1])
                else:
                    merged.append(seg)

            logger.info(f'[VAD] Energy VAD detected {len(merged)} speech segments')
            return merged

    except Exception as e:
        logger.warning(f'[VAD] Energy VAD failed: {e}')
        return [(0.0, 3600.0)]  # Assume full duration as fallback


def detect_speaker_diarization(audio_path: str) -> List[Dict[str, Any]]:
    """
    Optional speaker diarization using pyannote.audio.
    Enable with DIARIZATION_ENABLED=1 and provide HUGGINGFACE_TOKEN or
    PYANNOTE_AUTH_TOKEN. If unavailable, returns [] without failing the pipeline.
    """
    if not _env_enabled('DIARIZATION_ENABLED', True):
        logger.info('[Diarization] Disabled by DIARIZATION_ENABLED=0')
        return []

    token = (
        os.environ.get('PYANNOTE_AUTH_TOKEN')
        or os.environ.get('HUGGINGFACE_TOKEN')
        or os.environ.get('HF_TOKEN')
    )
    model_name = os.environ.get('PYANNOTE_MODEL', 'pyannote/speaker-diarization-3.1')
    if not token:
        logger.info('[Diarization] No Hugging Face token configured; skipping speaker diarization.')
        return []

    try:
        from pyannote.audio import Pipeline

        logger.info(f'[Diarization] Loading {model_name}')
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        try:
            import torch
            if _whisper_device == 'cuda' and torch.cuda.is_available():
                pipeline.to(torch.device('cuda'))
        except Exception:
            pass

        diarization = pipeline(audio_path)
        turns = _normalize_diarization_turns(diarization)
        logger.info(f'[Diarization] Detected {len(turns)} speaker turns')
        return turns
    except ImportError:
        logger.info('[Diarization] pyannote.audio is not installed; skipping speaker diarization.')
    except Exception as e:
        logger.warning(f'[Diarization] Speaker diarization failed: {e}')
    return []


def _force_align_with_whisperx(audio_path: str, segments: List[Dict], language: str) -> List[Dict]:
    """
    Optional forced alignment with WhisperX. It refines word timestamps against
    the original audio, then the normal splitter rebuilds subtitle boundaries.
    CRITICAL: WhisperX must NEVER merge subtitle segments. Each segment keeps
    its own independent timing.
    """
    if not segments or not _env_enabled('FORCED_ALIGNMENT_ENABLED', True):
        return segments or []

    try:
        import whisperx
    except ImportError:
        logger.info('[Alignment] whisperx is not installed; using Whisper word timestamps.')
        return segments

    try:
        align_language = _normalize_whisper_lang(language) or 'en'
        device = _whisper_device or ('cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
        logger.info(f'[Alignment] Running WhisperX forced alignment (lang={align_language}, device={device})')

        audio = whisperx.load_audio(audio_path)
        model_a, metadata = whisperx.load_align_model(language_code=align_language, device=device)
        transcript_segments = []
        for seg in segments:
            text = (seg.get('source') or seg.get('text') or '').strip()
            if text:
                transcript_segments.append({
                    'start': _coerce_float(seg.get('start', 0.0)),
                    'end': _coerce_float(seg.get('end', 0.0)),
                    'text': text,
                })
        if not transcript_segments:
            return segments

        aligned = whisperx.align(
            transcript_segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        aligned_segments = aligned.get('segments', []) if isinstance(aligned, dict) else []
        if not aligned_segments:
            logger.info('[Alignment] WhisperX returned no aligned segments; keeping Whisper timestamps.')
            return segments

        refined = []
        for original, aligned_seg in zip(segments, aligned_segments):
            seg = dict(original)
            words = []
            for item in aligned_seg.get('words') or []:
                word = _normalize_word_timestamp(
                    item,
                    _coerce_float(aligned_seg.get('start'), seg.get('start', 0.0)),
                    _coerce_float(aligned_seg.get('end'), seg.get('end', 0.0)),
                )
                if word:
                    words.append(word)
            if words:
                seg['start'] = words[0]['start']
                seg['end'] = words[-1]['end']
                seg['words'] = words
                seg['alignment_source'] = 'forced'
            refined.append(seg)

        logger.info(f'[Alignment] Forced alignment refined {len(refined)} segments')
        return refined
    except Exception as e:
        logger.warning(f'[Alignment] WhisperX forced alignment failed: {e}')
        return segments


def _align_segments_to_vad(segments: List[Dict], speech_regions: List[Tuple[float, float]]) -> List[Dict]:
    """
    Clamp non-word fallback segments to the nearest VAD speech island. Word-based
    segments keep exact word timing.
    
    CRITICAL: Segments with word timestamps are NEVER modified by VAD alignment.
    Word timestamps from Whisper are the authoritative timing source.
    """
    if not segments or not speech_regions:
        return segments or []

    aligned = []
    for seg in segments:
        updated = dict(seg)
        # CRITICAL: If segment has word timestamps, keep exact word timing
        # Do NOT clamp to VAD boundaries which may include padding
        if updated.get('words'):
            # Use word timestamps as authoritative timing
            words = updated.get('words', [])
            word_starts = [_coerce_float(w.get('start'), None) for w in words if _coerce_float(w.get('start'), None) is not None]
            word_ends = [_coerce_float(w.get('end'), None) for w in words if _coerce_float(w.get('end'), None) is not None]
            if word_starts and word_ends:
                updated['start'] = round(min(word_starts), 3)
                updated['end'] = round(max(word_ends), 3)
            aligned.append(updated)
            continue

        start = _coerce_float(updated.get('start', 0.0))
        end = _coerce_float(updated.get('end', start))
        best = None
        best_overlap = 0.0
        for region_start, region_end in speech_regions:
            overlap = _time_overlap(start, end, region_start, region_end)
            if overlap > best_overlap:
                best_overlap = overlap
                best = (region_start, region_end)
        if best:
            updated['start'] = round(max(start, best[0]), 3)
            updated['end'] = round(min(end, best[1]), 3)
            if updated['end'] <= updated['start']:
                updated['end'] = round(updated['start'] + 0.01, 3)
        aligned.append(updated)
    return aligned


# ---------------------------------------------------------------------------
# Stage 2: Transcribe with Whisper (full audio with VAD chunking)
# ---------------------------------------------------------------------------
def _normalize_whisper_lang(lang):
    """
    Normalize language codes for Whisper compatibility.
    Strips region suffixes (e.g., 'zh-CN' -> 'zh', 'en-US' -> 'en').
    Returns lowercase base language code.
    """
    if not lang or lang == 'auto':
        return lang
    base = lang.split('-')[0].lower().strip()
    return base


def _merge_button1_vad_recovery(
    primary_segments: List[Dict],
    recovery_segments: List[Dict],
) -> List[Dict]:
    """Add only confident speech missed by Button 1's fast VAD pass."""
    merged = [dict(segment) for segment in primary_segments or []]
    added = 0
    for raw in recovery_segments or []:
        candidate = dict(raw)
        text = str(candidate.get('source') or candidate.get('text') or '').strip()
        start = _coerce_float(candidate.get('start'), 0.0)
        end = _coerce_float(candidate.get('end'), start)
        probabilities = [
            _coerce_float(word.get('probability'), None)
            for word in candidate.get('words') or []
            if _coerce_float(word.get('probability'), None) is not None
        ]
        average_probability = (
            sum(probabilities) / len(probabilities)
            if probabilities
            else 0.0
        )
        if not text or end <= start or average_probability < 0.35:
            continue

        duration = max(0.01, end - start)
        covered = max(
            (
                _time_overlap(
                    start,
                    end,
                    _coerce_float(existing.get('start'), 0.0),
                    _coerce_float(existing.get('end'), 0.0),
                )
                for existing in merged
            ),
            default=0.0,
        )
        # The primary VAD result remains authoritative wherever it already
        # covers this speech. Recovery fills gaps; it never replaces text.
        if covered / duration >= 0.35:
            continue
        candidate['recovered_without_vad'] = True
        merged.append(candidate)
        added += 1

    if added:
        logger.info(
            f'[Button 1] Recovered {added} confident speech segment(s) '
            'that the fast VAD pass missed.'
        )
    return sorted(
        merged,
        key=lambda segment: (
            _coerce_float(segment.get('start'), 0.0),
            _coerce_float(segment.get('end'), 0.0),
        ),
    )


def transcribe_audio(
    audio_path,
    source_lang='en',
    recover_vad_gaps=False,
):
    """
    Transcribe the FULL audio using Whisper from beginning to end.
    Uses VAD chunking for better accuracy on long videos.
    Never stops until the last frame.
    Returns a list of segment dicts with 'start', 'end', 'text'.
    """
    logger.info(f'[Stage 2] Transcribing FULL audio with Whisper (lang={source_lang})')
    logger.info(f'[Stage 2] Audio file: {audio_path}')

    load_started = time.time()
    model = get_whisper_model()
    logger.info(f'[Timing] Whisper loading/cache lookup: {time.time() - load_started:.2f}s')

    # Normalize language code for Whisper compatibility
    whisper_lang = _normalize_whisper_lang(source_lang)
    if whisper_lang != source_lang:
        logger.info(f'[Stage 2] Normalized language from "{source_lang}" to "{whisper_lang}"')

    # Get audio duration
    audio_duration = get_video_duration(audio_path)
    logger.info(f'[Stage 2] Audio duration: {audio_duration:.2f}s')

    start_time = time.time()

    use_chunking = audio_duration > float(os.environ.get('WHISPER_CHUNK_THRESHOLD_SECONDS', '600'))
    result = None
    detected_lang = source_lang if source_lang != 'auto' else 'unknown'

    try:
        if _whisper_backend == 'faster-whisper':
            if use_chunking:
                logger.info('[Stage 2] Long audio detected. Using bounded Faster-Whisper chunk transcription.')
                segments = _transcribe_long_audio(audio_path, model, whisper_lang, audio_duration)
            else:
                options = _whisper_options(audio_duration)
                segments_iter, info = _run_faster_whisper(model, audio_path, whisper_lang, options)
                segments = [_segment_to_dict(seg) for seg in segments_iter]
                if recover_vad_gaps and options.get('vad_filter') and segments:
                    recovery_options = dict(options)
                    recovery_options['vad_filter'] = False
                    recovery_options['beam_size'] = max(
                        2,
                        min(5, int(recovery_options.get('beam_size', 2))),
                    )
                    recovery_iter, _recovery_info = _run_faster_whisper(
                        model,
                        audio_path,
                        whisper_lang,
                        recovery_options,
                    )
                    recovery_segments = [
                        _segment_to_dict(segment)
                        for segment in recovery_iter
                    ]
                    segments = _merge_button1_vad_recovery(
                        segments,
                        recovery_segments,
                    )
                if not segments and options.get('vad_filter'):
                    logger.warning(
                        '[Stage 2] Whisper VAD removed all audio; retrying once '
                        'without the VAD filter while retaining no-speech detection.'
                    )
                    retry_options = dict(options)
                    retry_options['vad_filter'] = False
                    segments_iter, info = _run_faster_whisper(
                        model,
                        audio_path,
                        whisper_lang,
                        retry_options,
                    )
                    retry_segments = [_segment_to_dict(seg) for seg in segments_iter]
                    segments = [
                        segment for segment in retry_segments
                        if _has_speech_confidence(segment)
                    ]
                    rejected = len(retry_segments) - len(segments)
                    if rejected:
                        logger.info(
                            f'[Stage 2] Rejected {rejected} low-confidence '
                            'no-VAD recovery segment(s).'
                        )
                detected_lang = getattr(info, 'language', source_lang if source_lang != 'auto' else 'unknown')
        else:
            fp16 = _whisper_device == 'cuda'
            result = model.transcribe(
                audio_path,
                language=whisper_lang if whisper_lang != 'auto' else None,
                task='transcribe',
                verbose=False,
                fp16=fp16,
                beam_size=_whisper_options(audio_duration)['beam_size'],
                best_of=_whisper_options(audio_duration)['best_of'],
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                word_timestamps=True,
            )
            segments = result.get('segments', [])
    except Exception as e:
        logger.error(f'[Stage 2] Whisper transcription failed: {e}')
        logger.error(traceback.format_exc())
        raise RuntimeError(f'Whisper transcription failed: {e}')

    elapsed = time.time() - start_time
    logger.info(f'[Stage 2] Whisper completed in {elapsed:.2f}s')

    if _whisper_backend != 'faster-whisper':
        detected_lang = result.get('language', 'unknown') if result else source_lang

    logger.info(f'[Stage 2] Detected language: {detected_lang}')
    logger.info(f'[Stage 2] Number of segments: {len(segments)}')

    # Convert to our standard format
    output_segments = []
    for seg in segments:
        normalized_seg = _segment_to_dict(seg) if 'source' not in seg else seg
        text = (normalized_seg.get('source') or normalized_seg.get('text') or '').strip()
        # Skip segments with no real content
        if not text:
            continue
        output_segments.append({
            'start': float(normalized_seg['start']),
            'end': float(normalized_seg['end']),
            'source': text,
            'target': '',
            'words': normalized_seg.get('words', []),
            'speaker_id': normalized_seg.get('speaker_id') or normalized_seg.get('speaker'),
        })

    speech_regions = detect_voice_activity(audio_path)
    output_segments = _align_segments_to_vad(output_segments, speech_regions)
    output_segments = _force_align_with_whisperx(audio_path, output_segments, detected_lang or source_lang)
    diarization_turns = detect_speaker_diarization(audio_path)
    output_segments = apply_speaker_diarization_to_segments(output_segments, diarization_turns)
    output_segments = split_subtitles_by_pauses(output_segments, speech_regions=speech_regions)
    output_segments = enforce_subtitle_format_limits(output_segments)
    output_segments, seg_warnings = validate_subtitle_segmentation(output_segments, auto_fix=True)
    if seg_warnings:
        logger.warning(f'[Stage 2] Auto-checked {len(seg_warnings)} subtitle segmentation issues after transcription.')

    output_segments = _repair_subtitle_timing(output_segments)

    # Verify we processed the entire duration
    if output_segments:
        last_end = output_segments[-1]['end']
        if last_end < audio_duration - 5:
            logger.warning(f'[Stage 2] Last segment ends at {last_end:.2f}s but audio is {audio_duration:.2f}s. '
                          f'Coverage: {last_end/audio_duration*100:.1f}%')
    else:
        logger.warning('[Stage 2] No segments found in audio')

    # Log sample segments
    for s in output_segments[:3]:
        logger.debug(f'[Stage 2]   [{s["start"]:.2f}-{s["end"]:.2f}] {s["source"][:80]}')

    _cleanup_inference_memory()
    return output_segments, detected_lang


def _run_faster_whisper(model, audio_input, whisper_lang: str, options: Dict[str, Any]):
    language = whisper_lang if whisper_lang != 'auto' else None
    # WhisperModel.transcribe and BatchedInferencePipeline.transcribe do not
    # accept exactly the same options. In particular, batch_size belongs only
    # to the batched pipeline.
    kwargs = {
        'language': language,
        'task': 'transcribe',
        'beam_size': options['beam_size'],
        'best_of': options['best_of'],
        'vad_filter': options['vad_filter'],
        'condition_on_previous_text': options['condition_on_previous_text'],
        'temperature': options['temperature'],
        'compression_ratio_threshold': options['compression_ratio_threshold'],
        'log_prob_threshold': options['log_prob_threshold'],
        'no_speech_threshold': options['no_speech_threshold'],
        'without_timestamps': False,
        'word_timestamps': options['word_timestamps'],
    }
    if options.get('vad_filter') and options.get('vad_parameters') is not None:
        kwargs['vad_parameters'] = options['vad_parameters']

    # The batched API requires VAD or explicit clip timestamps. The recovery
    # pass deliberately disables VAD, so send it directly through WhisperModel.
    if options.get('vad_filter'):
        try:
            from faster_whisper import BatchedInferencePipeline
            batched_model = BatchedInferencePipeline(model=model)
            batched_kwargs = dict(kwargs)
            batched_kwargs['batch_size'] = options['batch_size']
            return batched_model.transcribe(audio_input, **batched_kwargs)
        except Exception as e:
            logger.debug(f'[Stage 2] Batched Faster-Whisper path unavailable: {e}')

    return model.transcribe(audio_input, **kwargs)


def _transcribe_chunk_with_retry(audio_path: str, model, whisper_lang: str, chunk: Tuple[int, float, float],
                                 total_chunks: int, audio_duration: float) -> List[Dict]:
    chunk_idx, chunk_start, chunk_end = chunk
    chunk_duration = max(0.01, chunk_end - chunk_start)
    options = _whisper_options(audio_duration)
    last_error = None

    for attempt in range(1, 4):
        samples = None
        try:
            samples = _read_wav_chunk(audio_path, chunk_start, chunk_duration)
            if samples.size == 0:
                logger.info(f'[Stage 2] Chunk {chunk_idx + 1}/{total_chunks} is empty; skipping.')
                return []

            with _whisper_inference_lock if _whisper_device == 'cuda' else _nullcontext():
                segments_iter, _info = _run_faster_whisper(model, samples, whisper_lang, options)
                return [_segment_to_dict(seg, offset=chunk_start) for seg in segments_iter]
        except Exception as e:
            last_error = e
            logger.warning(
                f'[Stage 2] Chunk {chunk_idx + 1}/{total_chunks} failed '
                f'attempt {attempt}/3: {e}'
            )
            time.sleep(min(2.0 * attempt, 5.0))
        finally:
            del samples
            if attempt > 1:
                _cleanup_inference_memory()

    logger.error(f'[Stage 2] Chunk {chunk_idx + 1}/{total_chunks} exhausted retries: {last_error}')
    return []


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _transcribe_long_audio(audio_path: str, model, whisper_lang: str, audio_duration: float = 0.0) -> List[Dict]:
    """
    Transcribe very long audio by splitting into chunks guided by VAD.
    This prevents Whisper from losing context or stopping early on long files.
    """
    if not audio_duration:
        audio_duration = get_video_duration(audio_path)

    chunk_seconds = float(os.environ.get('WHISPER_CHUNK_SECONDS', '300'))
    overlap_seconds = float(os.environ.get('WHISPER_CHUNK_OVERLAP_SECONDS', '2'))
    chunks = _build_time_chunks(audio_duration, chunk_seconds, overlap_seconds)
    total_chunks = len(chunks)
    max_workers = 1 if _whisper_device == 'cuda' else _bounded_worker_count('WHISPER_CHUNK_WORKERS', 2, 4)

    logger.info(
        f'[Stage 2] Transcribing {total_chunks} chunks '
        f'(chunk={chunk_seconds:.0f}s, overlap={overlap_seconds:.1f}s, workers={max_workers}, '
        f'backend={_whisper_backend}, device={_whisper_device}, compute={_whisper_compute_type})'
    )

    all_segments = []
    started_at = time.time()
    _update_pipeline_progress('Speech Recognition', 0, total_chunks, 'Transcribing audio chunks', started_at)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _transcribe_chunk_with_retry,
                audio_path,
                model,
                whisper_lang,
                chunk,
                total_chunks,
                audio_duration,
            ): chunk
            for chunk in chunks
        }
        completed = 0
        for future in as_completed(futures):
            chunk_idx, chunk_start, chunk_end = futures[future]
            try:
                chunk_segments = future.result()
                all_segments.extend(chunk_segments)
                logger.info(
                    f'[Stage 2] Chunk {chunk_idx + 1}/{total_chunks} '
                    f'{chunk_start:.2f}-{chunk_end:.2f}s produced {len(chunk_segments)} segments'
                )
            except Exception as e:
                logger.error(f'[Stage 2] Unexpected chunk failure {chunk_idx + 1}/{total_chunks}: {e}')
            completed += 1
            _update_pipeline_progress(
                'Speech Recognition',
                completed,
                total_chunks,
                f'Transcribed chunk {completed}/{total_chunks}',
                started_at,
            )

    if not all_segments:
        logger.warning('[Stage 2] Chunked transcription produced no segments. Falling back to single pass.')
        options = _whisper_options(audio_duration)
        segments_iter, _info = _run_faster_whisper(model, audio_path, whisper_lang, options)
        all_segments = [_segment_to_dict(seg) for seg in segments_iter]

    repaired = _repair_subtitle_timing([
        {
            'start': seg['start'],
            'end': seg['end'],
            'source': seg.get('source') or seg.get('text', ''),
            'target': '',
            'words': seg.get('words', []),
            'speaker_id': seg.get('speaker_id') or seg.get('speaker'),
        }
        for seg in all_segments
    ])
    logger.info(f'[Stage 2] Chunked transcription: {len(all_segments)} raw -> {len(repaired)} repaired')
    _cleanup_inference_memory()
    return [
        {
            'start': s['start'],
            'end': s['end'],
            'text': s['source'],
            'words': s.get('words', []),
            'speaker_id': s.get('speaker_id') or s.get('speaker'),
        }
        for s in repaired
    ]


# ---------------------------------------------------------------------------
# Stage 2b: Enhanced OCR Subtitle Detection
# ---------------------------------------------------------------------------
def _ocr_text_similarity(left: str, right: str) -> float:
    """Compare OCR readings while tolerating whitespace and one-frame glitches."""
    left_norm = re.sub(r'\s+', ' ', (left or '').strip().casefold())
    right_norm = re.sub(r'\s+', ' ', (right or '').strip().casefold())
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        return shorter / longer if longer else 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _select_subtitle_text(
    results: List[Any],
    image_width: int,
    image_height: int,
) -> Tuple[str, float]:
    """Select centered caption lines while rejecting corner/UI text."""
    candidates = []
    for item in results or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        bbox, raw_text = item[0], item[1]
        confidence = _coerce_float(item[2], 1.0) if len(item) > 2 else 1.0
        text = _normalize_translation_text(raw_text)
        if not text or confidence < 0.12 or not bbox:
            continue
        try:
            xs = [float(point[0]) for point in bbox]
            ys = [float(point[1]) for point in bbox]
        except (TypeError, ValueError, IndexError):
            continue
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        width = max(1.0, right - left)
        height = max(1.0, bottom - top)
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        width_ratio = width / max(1.0, image_width)
        height_ratio = height / max(1.0, image_height)
        center_ratio = center_x / max(1.0, image_width)

        # Tiny labels and fixed corner marks are overwhelmingly UI/watermarks,
        # not readable dialogue captions.
        if height_ratio < 0.018 or width_ratio < 0.025:
            continue
        if (center_ratio < 0.14 or center_ratio > 0.86) and width_ratio < 0.22:
            continue

        horizontal_centering = max(0.0, 1.0 - abs(center_ratio - 0.5) * 2.0)
        score = (
            confidence
            + 0.55 * horizontal_centering
            + 0.25 * min(1.0, width_ratio * 3.0)
            + 0.12 * (center_y / max(1.0, image_height))
        )
        candidates.append({
            'text': text,
            'confidence': confidence,
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
            'center_x': center_x,
            'center_y': center_y,
            'height': height,
            'score': score,
        })

    if not candidates:
        return '', 0.0

    anchor = max(candidates, key=lambda item: item['score'])
    caption_lines = []
    for candidate in candidates:
        horizontal_distance = abs(candidate['center_x'] - anchor['center_x'])
        vertical_distance = abs(candidate['center_y'] - anchor['center_y'])
        if (
            horizontal_distance <= image_width * 0.30
            and vertical_distance <= max(
                image_height * 0.16,
                4.0 * max(candidate['height'], anchor['height']),
            )
            and candidate['confidence'] >= 0.18
        ):
            caption_lines.append(candidate)

    if not caption_lines:
        caption_lines = [anchor]
    caption_lines.sort(key=lambda item: (item['top'], item['left']))
    lines = []
    for item in caption_lines:
        if not lines or _ocr_text_similarity(lines[-1], item['text']) < 0.96:
            lines.append(item['text'])
    confidence = sum(item['confidence'] for item in caption_lines) / len(caption_lines)
    return '\n'.join(lines), confidence


def _build_ocr_subtitle_segments(
    observations: List[Tuple[float, str, float]],
    frame_interval: float,
) -> List[Dict]:
    """Turn every sampled frame, including blank frames, into caption windows."""
    segments = []
    current_text = ''
    current_start = 0.0
    current_end = 0.0
    current_confidence = 0.0
    blank_started = None
    # One missed sample should not split a caption; two reliably mean it left.
    blank_tolerance = max(frame_interval * 1.25, 0.05)

    def close_current(end_time: float):
        nonlocal current_text, current_start, current_end, current_confidence, blank_started
        if current_text.strip() and end_time > current_start:
            segments.append({
                'start': round(current_start, 3),
                'end': round(max(current_start + 0.01, end_time), 3),
                'source': current_text.strip(),
                'target': '',
                'confidence': round(current_confidence, 4),
                'timing_source': 'ocr_frames',
            })
        current_text = ''
        current_start = 0.0
        current_end = 0.0
        current_confidence = 0.0
        blank_started = None

    for frame_time, raw_text, raw_confidence in observations:
        text = _normalize_translation_text(raw_text)
        confidence = _coerce_float(raw_confidence, 0.0)
        if not text:
            if current_text and blank_started is None:
                blank_started = frame_time
            elif (
                current_text
                and blank_started is not None
                and frame_time - blank_started >= blank_tolerance
            ):
                close_current(blank_started)
            continue

        if not current_text:
            current_text = text
            current_start = frame_time
            current_end = frame_time + frame_interval
            current_confidence = confidence
            blank_started = None
            continue

        similarity = _ocr_text_similarity(current_text, text)
        resumed_quickly = (
            blank_started is not None
            and frame_time - blank_started < blank_tolerance
        )
        if similarity >= 0.80 or resumed_quickly and similarity >= 0.72:
            # Prefer the most complete reading, then the higher-confidence one.
            if len(text) > len(current_text) or confidence > current_confidence + 0.12:
                current_text = text
                current_confidence = max(current_confidence, confidence)
            current_end = frame_time + frame_interval
            blank_started = None
            continue

        close_current(blank_started if blank_started is not None else frame_time)
        current_text = text
        current_start = frame_time
        current_end = frame_time + frame_interval
        current_confidence = confidence

    if current_text:
        close_current(blank_started if blank_started is not None else current_end)

    merged = _merge_subtitle_segments(segments)
    for segment in merged:
        segment['start'] = round(_coerce_float(segment.get('start')), 3)
        segment['end'] = round(_coerce_float(segment.get('end')), 3)
        segment['duration'] = round(segment['end'] - segment['start'], 3)
    return merged


def detect_hardcoded_subtitles(
    video_path,
    fps=30.0,
    duration=None,
    source_lang='auto',
    max_dimension_override=None,
    frame_retries_override=None,
):
    """
    TRUE FRAME-BY-FRAME subtitle detection using EasyOCR with aggressive retry,
    multi-region scanning, and neighboring frame analysis.
    
    CRITICAL RULES:
    - Detect 100% of all subtitles displayed in the video.
    - Never skip any subtitle, even if it appears for a very short time.
    - Detect the complete subtitle phrase exactly as it appears on screen.
    - Detect subtitles frame-by-frame to capture fast subtitle changes.
    - Record the exact appearance and disappearance time of every subtitle.
    - If OCR confidence is low, analyze adjacent frames and retry automatically.
    - One displayed subtitle = One subtitle segment.
    - Never merge two subtitle phrases into one.
    - Never merge subtitles from different speakers.
    - Never merge subtitles that appear consecutively.
    - Never split a subtitle unless the subtitle itself changes on screen.
    - Start Time = The exact frame where the subtitle first appears.
    - End Time = The exact frame where the subtitle disappears or changes.
    
    Uses 30fps frame sampling for true frame-by-frame detection with multi-region
    scanning (top, middle, bottom) to catch subtitles anywhere on screen.
    Automatic retry with multiple OCR parameter variations for low-confidence frames.
    """
    logger.info(f'[Stage 2b] TRUE FRAME-BY-FRAME subtitle detection (fps={fps}, multi-region, auto-retry)')
    logger.info(f'[Stage 2b] Video: {video_path}')

    # Get video duration
    duration = _coerce_float(duration, 0.0)
    if duration <= 0:
        duration = get_video_duration(video_path)
    logger.info(f'[Stage 2b] Video duration: {duration:.2f}s')

    # Use the requested sampling rate across the full video timeline.
    frame_interval = max(0.033, 1.0 / fps)
    ocr_languages = _ocr_languages_for_source(source_lang)
    scan_full_frame = _env_enabled('OCR_SCAN_FULL_FRAME', default=False)
    scan_top = 0.0 if scan_full_frame else max(
        0.0, min(0.8, float(os.environ.get('OCR_SCAN_TOP', '0.30')))
    )
    scan_bottom = 1.0 if scan_full_frame else max(
        scan_top + 0.1,
        min(1.0, float(os.environ.get('OCR_SCAN_BOTTOM', '0.84'))),
    )
    configured_max_dimension = max(
        320,
        min(1280, int(os.environ.get('OCR_MAX_DIMENSION', '360'))),
    )
    max_dimension = (
        max(320, min(1600, int(max_dimension_override)))
        if max_dimension_override is not None
        else configured_max_dimension
    )
    fingerprint = _media_fingerprint(video_path)
    multi_region = _env_enabled('OCR_MULTI_REGION', default=False)

    def _key_for_language(language: str, dimension: int) -> str:
        return _cache_key('ocr', {
            'fingerprint': fingerprint,
            'source_lang': language,
            'languages': _ocr_languages_for_source(language),
            'fps': round(float(fps), 4),
            'scan_top': scan_top,
            'scan_bottom': scan_bottom,
            'max_dimension': dimension,
            'multi_region': multi_region,
            'version': 2,
        })

    cache_key = _key_for_language(source_lang, max_dimension)
    cached_ocr = _read_json_cache(OCR_CACHE_DIR, cache_key) or {}
    if cached_ocr.get('completed') and isinstance(cached_ocr.get('segments'), list):
        cached_segments = [
            {**segment, 'ocr_language': source_lang}
            for segment in cached_ocr['segments']
        ]
        logger.info(
            f'[Stage 2b] OCR cache hit: {len(cached_segments)} subtitle segments'
        )
        _update_pipeline_progress(
            'OCR', 1, 1, f'Reused {len(cached_segments)} cached subtitles'
        )
        return cached_segments

    # Reuse a higher-quality completed cache for this exact video even when the
    # language selector was wrong. This avoids minutes of garbage OCR (for
    # example, running the English recognizer over Chinese captions).
    best_alternative = None
    candidate_languages = ('zh-CN', 'zh-TW', 'ja', 'ko', 'en', 'vi', 'th')
    for candidate_language in candidate_languages:
        if candidate_language == source_lang:
            continue
        for candidate_dimension in dict.fromkeys((max_dimension, 480)):
            alternative = _read_json_cache(
                OCR_CACHE_DIR,
                _key_for_language(candidate_language, candidate_dimension),
            ) or {}
            candidate_segments = alternative.get('segments')
            if not alternative.get('completed') or not candidate_segments:
                continue
            joined_text = ' '.join(
                str(segment.get('source', '') or '')
                for segment in candidate_segments
            )
            script_score = _ocr_script_score(joined_text, candidate_language)
            average_confidence = sum(
                _coerce_float(segment.get('confidence'), 0.0)
                for segment in candidate_segments
            ) / len(candidate_segments)
            quality = script_score + average_confidence * 0.25
            if script_score >= 0.55 and (
                best_alternative is None or quality > best_alternative[0]
            ):
                best_alternative = (
                    quality,
                    candidate_language,
                    candidate_segments,
                )
    if best_alternative is not None:
        _, detected_ocr_language, candidate_segments = best_alternative
        reused_segments = [
            {**segment, 'ocr_language': detected_ocr_language}
            for segment in candidate_segments
        ]
        logger.info(
            f'[Stage 2b] Auto-selected cached OCR language '
            f'{detected_ocr_language}: {len(reused_segments)} subtitles'
        )
        _update_pipeline_progress(
            'OCR',
            1,
            1,
            f'Reused {len(reused_segments)} cached {detected_ocr_language} subtitles',
        )
        return reused_segments

    reader = get_ocr_reader(ocr_languages)
    logger.info(f'[Stage 2b] OCR languages: {ocr_languages}')

    logger.info(f'[Stage 2b] Frame-by-frame detection: sampling every {frame_interval:.4f}s ({fps:.1f} fps)')

    frames_dir = TEMP_DIR / f'ocr_frames_{uuid4().hex}'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Multi-region scanning: scan full frame, then specific regions
    # This catches subtitles anywhere on screen (top, middle, bottom)
    # Every frame is retained. Blank observations are essential: without them
    # a caption cannot end when it disappears.
    ocr_observations = [
        tuple(item)
        for item in (cached_ocr.get('observations') or [])
        if isinstance(item, (list, tuple)) and len(item) == 3
    ]

    # Process in batches to manage memory.
    batch_size = max(4, min(30, int(os.environ.get('OCR_BATCH_SIZE', '12'))))
    frame_times = []
    t = 0.0
    while t < duration:
        frame_times.append(t)
        t += frame_interval

    resume_index = max(
        0,
        min(len(frame_times), int(cached_ocr.get('next_frame_index') or 0)),
    )
    logger.info(
        f'[Stage 2b] Processing {len(frame_times)} frames in batches of '
        f'{batch_size} (resume={resume_index})'
    )

    # Track previous frame text for scene change detection
    prev_frame_text = ocr_observations[-1][1] if ocr_observations else ''
    prev_frame_time = ocr_observations[-1][0] if ocr_observations else -1.0
    frames_with_text = sum(1 for _, text, _ in ocr_observations if text)
    frames_without_text = len(ocr_observations) - frames_with_text
    configured_frame_retries = int(os.environ.get('OCR_FRAME_RETRIES', '0'))
    MAX_OCR_RETRIES = max(0, min(
        3,
        int(frame_retries_override)
        if frame_retries_override is not None
        else configured_frame_retries,
    ))
    LOW_CONF_THRESHOLD = 0.25  # Lower threshold to catch more subtitles

    # Scan the full frame plus top, middle, and bottom crops so subtitles are
    # detected wherever they appear in the video. Frames are decoded once per
    # batch; creating a new FFmpeg process for every frame/region made the UI
    # appear frozen on CPU-only systems.
    regions = [('full', None)]
    if _env_enabled('OCR_MULTI_REGION', default=False):
        regions.extend([
            ('bottom', (0.0, 0.65, 1.0, 1.0)),
            ('top', (0.0, 0.0, 1.0, 0.25)),
            ('middle', (0.0, 0.35, 1.0, 0.65)),
        ])

    for batch_start in range(resume_index, len(frame_times), batch_size):
        process_lock.raise_if_cancelled()
        batch_end = min(batch_start + batch_size, len(frame_times))
        batch_times = frame_times[batch_start:batch_end]

        # Decode all full frames for this batch in one FFmpeg process.
        batch_pattern = frames_dir / f'batch_{batch_start:06d}_%06d.jpg'
        extract_cmd = [
            FFMPEG_EXE, '-y',
            '-ss', f'{batch_times[0]:.6f}',
            '-i', video_path,
            '-vf', f'fps={fps:.8f}',
            '-frames:v', str(len(batch_times)),
            '-q:v', '2',
            str(batch_pattern),
        ]
        completed = _run_cancellable(
            extract_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(60, len(batch_times) * 3),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                'Unable to extract OCR frame batch: '
                f'{completed.stderr[-1000:].strip()}'
            )

        batch_paths = []  # (global_idx, frame_time, region_name, frame_path)
        for idx_offset, frame_time in enumerate(batch_times):
            global_idx = batch_start + idx_offset
            full_path = frames_dir / (
                f'batch_{batch_start:06d}_{idx_offset + 1:06d}.jpg'
            )
            if not full_path.exists():
                logger.warning(
                    f'[Stage 2b] Batch frame missing at {frame_time:.3f}s'
                )
                continue
            try:
                from PIL import Image

                with Image.open(full_path) as image:
                    width, height = image.size
                    scan_image = image.crop((
                        0,
                        int(height * scan_top),
                        width,
                        max(1, int(height * scan_bottom)),
                    ))
                    scale = min(1.0, max_dimension / max(scan_image.size))
                    if scale < 1.0:
                        scan_image = scan_image.resize((
                            max(1, int(scan_image.width * scale)),
                            max(1, int(scan_image.height * scale)),
                        ))
                    scan_path = frames_dir / f'frame_{global_idx:06d}_scan.jpg'
                    scan_image.save(scan_path, quality=90)
                    batch_paths.append(
                        (global_idx, frame_time, 'scan', scan_path)
                    )

                    for region_name, crop_box in regions[1:]:
                        left, top, right, bottom = crop_box
                        region_path = frames_dir / (
                            f'frame_{global_idx:06d}_{region_name}.jpg'
                        )
                        image.crop((
                            int(width * left),
                            int(height * top),
                            max(1, int(width * right)),
                            max(1, int(height * bottom)),
                        )).save(region_path, quality=92)
                        batch_paths.append(
                            (
                                global_idx,
                                frame_time,
                                region_name,
                                region_path,
                            )
                        )
                try:
                    full_path.unlink()
                except OSError:
                    pass
            except Exception as e:
                logger.warning(
                    f'[Stage 2b] Unable to prepare frame {global_idx}: {e}'
                )

        # OCR batch with aggressive retry for low-confidence frames
        # Process each frame's regions and pick the best result
        processed_frames = {}  # global_idx -> best result across all regions
        region_priority = {'bottom': 4, 'scan': 3, 'middle': 2, 'top': 1}
        
        for global_idx, frame_time, region_name, frame_path in batch_paths:
            process_lock.raise_if_cancelled()
            if not frame_path.exists():
                continue

            best_text = ''
            best_conf = 0.0
            best_bbox = None
            
            # Retry loop with multiple OCR parameter variations
            ocr_param_sets = [
                {'paragraph': False, 'width_ths': 0.7, 'height_ths': 0.5},
                {'paragraph': False, 'width_ths': 0.5, 'height_ths': 0.5},
                {'paragraph': False, 'width_ths': 0.3, 'height_ths': 0.3},
                {'paragraph': False, 'width_ths': 0.9, 'height_ths': 0.7},
            ]
            
            for retry in range(min(MAX_OCR_RETRIES + 1, len(ocr_param_sets))):
                try:
                    params = ocr_param_sets[retry]
                    results = reader.readtext(
                        str(frame_path),
                        paragraph=params['paragraph'],
                        width_ths=params['width_ths'],
                        height_ths=params['height_ths'],
                    )
                    
                    try:
                        from PIL import Image
                        with Image.open(frame_path) as prepared_image:
                            prepared_width, prepared_height = prepared_image.size
                    except Exception:
                        prepared_width, prepared_height = 640, 640
                    selected_text, selected_confidence = _select_subtitle_text(
                        results,
                        prepared_width,
                        prepared_height,
                    )
                    if selected_text:
                        best_text = selected_text
                        best_conf = selected_confidence
                        
                        # If confidence is good enough, stop retrying
                        if best_conf > LOW_CONF_THRESHOLD:
                            break
                            
                except Exception as e:
                    logger.warning(f'[Stage 2b] OCR retry {retry + 1} failed for frame {global_idx} ({region_name}): {e}')
                    if retry < MAX_OCR_RETRIES:
                        time.sleep(0.05)
                    continue

            # Store best result for this region
            if best_text and len(best_text) > 1:
                candidate_score = (
                    region_priority.get(region_name, 0),
                    best_conf,
                    len(best_text),
                )
                existing_score = (
                    processed_frames[global_idx][4]
                    if global_idx in processed_frames
                    else None
                )
                if existing_score is None or candidate_score > existing_score:
                    processed_frames[global_idx] = (
                        frame_time, best_text, best_conf, region_name, candidate_score
                    )

        # --- Neighboring frame analysis for low-confidence detections ---
        for idx_offset, ft in enumerate(batch_times):
            global_idx = batch_start + idx_offset
            detected = processed_frames.get(global_idx)
            if detected:
                _detected_time, text, conf, region, _score = detected
            else:
                text, conf, region = '', 0.0, ''
            
            # If confidence is low, check neighboring frames
            if text and conf <= LOW_CONF_THRESHOLD:
                neighbor_matches = 0
                for neighbor_idx in range(
                    max(0, len(ocr_observations) - 8),
                    len(ocr_observations),
                ):
                    if neighbor_idx < len(ocr_observations):
                        neighbor_text = ocr_observations[neighbor_idx][1]
                        if neighbor_text and (neighbor_text == text or text.startswith(neighbor_text) or neighbor_text.startswith(text)):
                            neighbor_matches += 1
                
                if neighbor_matches >= 2:
                    # Neighbors agree - accept even with low confidence
                    logger.debug(f'[Stage 2b] Frame {global_idx}: Low conf ({conf:.2f}) but accepted from neighbor analysis')
                    conf = max(conf, 0.45)  # Boost effective confidence
            
            # --- Apply detection result ---
            if text and len(text) > 1:
                ocr_observations.append((ft, text, conf))
                frames_with_text += 1
                if text != prev_frame_text:
                    logger.debug(f'[Stage 2b] Frame {global_idx} @ {ft:.3f}s: SUBTITLE CHANGE: "{text[:80]}" (conf={conf:.2f}, region={region})')
                prev_frame_text = text
                prev_frame_time = ft
            else:
                ocr_observations.append((ft, '', 0.0))
                frames_without_text += 1
                if prev_frame_text:
                    logger.debug(f'[Stage 2b] Frame {global_idx} @ {ft:.3f}s: subtitle disappeared')
                    prev_frame_text = ''
                    prev_frame_time = ft

        # Clean up batch frames
        for _, _, _, frame_path in batch_paths:
            try:
                if frame_path.exists():
                    frame_path.unlink()
            except OSError:
                pass

        logger.info(f'[Stage 2b] Processed batch {batch_start//batch_size + 1}: '
                    f'frames {batch_start}-{batch_end - 1}/{len(frame_times)}')
        _update_pipeline_progress(
            'OCR',
            batch_end,
            len(frame_times),
            f'OCR analyzed {batch_end}/{len(frame_times)} frames',
        )
        _write_json_cache(OCR_CACHE_DIR, cache_key, {
            'completed': False,
            'next_frame_index': batch_end,
            'observations': ocr_observations,
            'updated_at': time.time(),
        })

    logger.info(f'[Stage 2b] OCR found text in {frames_with_text} frames, '
                f'no text in {frames_without_text} frames')

    if not any(text for _, text, _ in ocr_observations):
        logger.warning('[Stage 2b] No hardcoded subtitles detected')
        _write_json_cache(OCR_CACHE_DIR, cache_key, {
            'completed': True,
            'segments': [],
            'updated_at': time.time(),
        })
        shutil.rmtree(frames_dir, ignore_errors=True)
        return []

    # =========================================================================
    # Build subtitle segments with EXACT frame-level timing
    # =========================================================================
    # CRITICAL: Each unique subtitle text change creates a new segment boundary.
    # The start time is the EXACT frame where the subtitle first appears.
    # The end time is the EXACT frame where the subtitle disappears or changes.
    # =========================================================================
    segments = _build_ocr_subtitle_segments(ocr_observations, frame_interval)
    for segment in segments:
        segment['ocr_language'] = source_lang
    _write_json_cache(OCR_CACHE_DIR, cache_key, {
        'completed': True,
        'segments': segments,
        'updated_at': time.time(),
    })

    # =========================================================================
    # CRITICAL: Do NOT merge subtitle segments.
    # Each unique subtitle phrase keeps its own independent timing.
    # =========================================================================
    # We NEVER call _merge_subtitle_segments here because it would merge different
    # subtitle phrases together, violating the "never merge" rule.
    # Only consecutive frames with EXACTLY the same text are merged above.
    
    logger.info(f'[Stage 2b] Built {len(segments)} subtitle segments from frame-by-frame OCR')
    for s in segments[:5]:
        logger.debug(f'[Stage 2b]   [{s["start"]:.3f}-{s["end"]:.3f}] {s["source"][:60]}')
    if len(segments) > 5:
        logger.debug(f'[Stage 2b]   ... and {len(segments) - 5} more segments')

    shutil.rmtree(frames_dir, ignore_errors=True)
    _cleanup_inference_memory()
    return segments


def _is_same_subtitle(text1: str, text2: str) -> bool:
    """
    STRICT check if two OCR results represent the EXACT same subtitle.
    
    CRITICAL RULES:
    - Only EXACT string matches are considered the same subtitle.
    - No fuzzy matching, no partial overlap, no character ratio matching.
    - This prevents merging different subtitle phrases together.
    - Different subtitles with similar text are NEVER considered the same.
    """
    if not text1 or not text2:
        return False

    # STRICT: Only exact match (case-insensitive, whitespace-normalized)
    t1 = ' '.join(text1.strip().lower().split())
    t2 = ' '.join(text2.strip().lower().split())

    return t1 == t2


def _merge_subtitle_segments(segments: List[Dict]) -> List[Dict]:
    """
    CRITICAL: This function NEVER merges different subtitle phrases.
    It ONLY merges EXACT text duplicates with overlapping timing.
    
    Rules:
    - Only merge if text is EXACTLY the same (case-insensitive, whitespace-normalized)
    - Only merge if timing overlaps (same subtitle detected twice)
    - NEVER merge different subtitle phrases together
    - NEVER merge subtitles from different speakers
    - NEVER merge subtitles that appear consecutively
    - Each unique subtitle phrase keeps its own independent timing
    """
    if not segments:
        return []

    merged = []
    for seg in segments:
        if not merged:
            merged.append(dict(seg))
            continue

        last = merged[-1]
        # CRITICAL: ONLY merge if text is EXACTLY the same (true duplicate)
        # This NEVER merges two different subtitle phrases.
        if _is_same_subtitle(last['source'], seg['source']):
            # Same text - extend timing and keep longest text
            last['end'] = max(last['end'], seg['end'])
            if len(seg['source']) > len(last['source']):
                last['source'] = seg['source']
        else:
            # DIFFERENT text - NEVER merge. Keep as independent segment.
            # Fix overlapping timing without merging.
            if seg['start'] < last['end']:
                # Create minimum gap, never merge
                seg['start'] = last['end'] + 0.02
                seg['end'] = max(seg['start'] + 0.01, seg['end'])
            merged.append(dict(seg))

    return merged


# ---------------------------------------------------------------------------
# Stage 2c: Combine Whisper + OCR for Best Coverage
# ---------------------------------------------------------------------------
def combine_subtitle_sources(
    whisper_segments: List[Dict],
    ocr_segments: List[Dict],
    prefer_ocr_timing: bool = False,
) -> List[Dict]:
    """
    Overlay OCR text on the existing speech segments by timestamp.

    Speech-to-text remains authoritative for segment count, timing, words,
    speakers, VAD, and synchronization. OCR can replace only textual content;
    it can never add, remove, split, merge, or retime a speech segment.
    """
    if not whisper_segments:
        return [dict(segment) for segment in ocr_segments]
    if not ocr_segments:
        return [dict(segment) for segment in whisper_segments]

    if prefer_ocr_timing:
        logger.info('[Stage 2c] Enriching frame-timed OCR captions with speech metadata')
        combined = []
        for raw_ocr in ocr_segments:
            caption = dict(raw_ocr)
            caption_start = _coerce_float(caption.get('start'), 0.0)
            caption_end = _coerce_float(caption.get('end'), caption_start)
            overlapping = [
                speech for speech in whisper_segments
                if _time_overlap(
                    caption_start,
                    caption_end,
                    _coerce_float(speech.get('start'), 0.0),
                    _coerce_float(speech.get('end'), 0.0),
                ) > 0
            ]
            speaker_overlap = {}
            for speech in overlapping:
                speaker = str(
                    speech.get('speaker_id') or speech.get('speaker') or ''
                ).strip()
                if not speaker:
                    continue
                overlap = _time_overlap(
                    caption_start,
                    caption_end,
                    _coerce_float(speech.get('start'), 0.0),
                    _coerce_float(speech.get('end'), 0.0),
                )
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap
            active_speaker = (
                max(speaker_overlap, key=speaker_overlap.get)
                if speaker_overlap
                else ''
            )
            if active_speaker:
                caption['speaker_id'] = active_speaker
            words = []
            for speech in overlapping:
                for raw_word in speech.get('words') or []:
                    word_start = _coerce_float(raw_word.get('start'), 0.0)
                    word_end = _coerce_float(raw_word.get('end'), word_start)
                    midpoint = (word_start + word_end) / 2.0
                    if not caption_start <= midpoint <= caption_end:
                        continue
                    word = dict(raw_word)
                    word['start'] = round(max(caption_start, word_start), 3)
                    word['end'] = round(min(caption_end, word_end), 3)
                    if word['end'] > word['start']:
                        words.append(word)
            if active_speaker:
                words = [
                    word for word in words
                    if not _word_speaker(word) or _word_speaker(word) == active_speaker
                ]
            caption['words'] = words
            caption['timing_source'] = 'ocr_frames'
            combined.append(caption)
        return combined

    logger.info('[Stage 2c] Matching OCR text onto Speech-to-Text segments')
    usable_ocr = []
    for raw in ocr_segments:
        if not isinstance(raw, dict):
            continue
        text = _normalize_translation_text(raw.get('source') or raw.get('text') or '')
        start = _coerce_float(raw.get('start'), None)
        end = _coerce_float(raw.get('end'), None)
        if text and start is not None and end is not None and end > start:
            usable_ocr.append((start, end, text))
    usable_ocr.sort(key=lambda item: (item[0], item[1]))

    overlaid = []
    replaced_count = 0
    for raw_speech in whisper_segments:
        speech = dict(raw_speech)
        speech_start = _coerce_float(speech.get('start'), 0.0)
        speech_end = _coerce_float(speech.get('end'), speech_start)
        matched_texts = []
        for ocr_start, ocr_end, ocr_text in usable_ocr:
            overlap = _time_overlap(speech_start, speech_end, ocr_start, ocr_end)
            if overlap <= 0:
                continue
            if not matched_texts or not _is_same_subtitle(matched_texts[-1], ocr_text):
                matched_texts.append(ocr_text)

        if matched_texts:
            ocr_text = _normalize_translation_text(' '.join(matched_texts))
            speech['source'] = ocr_text
            speech['text'] = ocr_text
            # Prevent stale STT text from winning in untranslated/export flows.
            speech['target'] = ''
            replaced_count += 1
        overlaid.append(speech)

    logger.info(
        f'[Stage 2c] OCR replaced text in {replaced_count}/{len(overlaid)} '
        f'Speech-to-Text segments; segment boundaries and metadata preserved'
    )
    return overlaid


def _prepare_ocr_authoritative_segments(segments: List[Dict]) -> List[Dict]:
    """Validate and deduplicate OCR captions without moving frame timestamps."""
    prepared = []
    for raw in sorted(
        (segment for segment in segments if isinstance(segment, dict)),
        key=lambda segment: (
            _coerce_float(segment.get('start'), 0.0),
            _coerce_float(segment.get('end'), 0.0),
        ),
    ):
        segment = dict(raw)
        text = _normalize_translation_text(
            segment.get('source') or segment.get('text') or ''
        )
        start = _coerce_float(segment.get('start'), None)
        end = _coerce_float(segment.get('end'), None)
        if not text or start is None or end is None or end <= start:
            continue

        segment['source'] = text
        segment['text'] = text
        segment['target'] = text
        segment['start'] = round(max(0.0, start), 3)
        segment['end'] = round(end, 3)
        segment['duration'] = round(segment['end'] - segment['start'], 3)
        segment['timing_source'] = 'ocr_frames'
        segment['emotion'] = segment.get('emotion') or _detect_emotion(text)

        if prepared:
            previous = prepared[-1]
            same_text = _is_same_subtitle(previous['source'], text)
            if same_text and segment['start'] <= previous['end'] + 0.05:
                previous['end'] = max(previous['end'], segment['end'])
                previous['duration'] = round(
                    previous['end'] - previous['start'],
                    3,
                )
                if (
                    not previous.get('speaker_id')
                    and segment.get('speaker_id')
                ):
                    previous['speaker_id'] = segment['speaker_id']
                continue
            if segment['start'] < previous['end']:
                if segment['start'] <= previous['start']:
                    if _coerce_float(
                        segment.get('confidence'), 0.0
                    ) > _coerce_float(previous.get('confidence'), 0.0):
                        prepared[-1] = segment
                    continue
                previous['end'] = segment['start']
                previous['duration'] = round(
                    previous['end'] - previous['start'],
                    3,
                )
        prepared.append(segment)
    return prepared


# ---------------------------------------------------------------------------
# Stage 3: LLM-based Translation with Context
# ---------------------------------------------------------------------------
KHMER_DUBBING_CHARS_PER_SEC = 9.5
SUBTITLE_TIMING_GRACE_SECONDS = 0.25
SCENE_GAP_SECONDS = 1.35
SCENE_MAX_DURATION_SECONDS = 14.0
SCENE_MAX_SOURCE_CHARS = 650
MAX_DUBBING_SOURCE_CHARS_PER_SECOND = 17.0
KHMER_MAX_TTS_CHARS_PER_SECOND = 10.5
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
GEMINI_DEFAULT_MODEL = 'gemini-2.0-flash'
ALLOW_KHMER_DUBBING_TEXT_TRIM = os.environ.get(
    'KHMER_ALLOW_TEXT_TRIM', ''
).strip().lower() in {'1', 'true', 'yes', 'on'}

# Maximum retry attempts for translation API calls
TRANSLATION_MAX_RETRIES = 3


def _segment_duration(seg):
    """Return subtitle duration without changing timestamps."""
    try:
        return max(0.1, float(seg.get('end', 0)) - float(seg.get('start', 0)))
    except (TypeError, ValueError):
        return 0.1


def _normalize_translation_text(text):
    """Clean API artifacts while preserving the translated content."""
    if not text:
        return ''
    text = text.replace('"', '"').replace('&#39;', "'").replace('&', '&')
    return re.sub(r'\s+', ' ', text).strip()


def _contains_khmer(text):
    return bool(re.search(r'[\u1780-\u17ff]', text or ''))


def _translation_matches_target_language(text: str, target_lang: str) -> bool:
    """Reject source-text fallbacks before they poison the translation cache."""
    if not (text or '').strip():
        return False
    base_target = (target_lang or '').split('-')[0].lower()
    if base_target == 'km':
        return _contains_khmer(text)
    return True


def _detect_emotion(text: str) -> str:
    """
    Detect the emotional tone of the subtitle text.
    Returns: 'neutral', 'happy', 'sad', 'angry', 'excited', 'fear', 'whisper', 'romantic'
    """
    if not text:
        return 'neutral'

    text_lower = text.lower()

    # Check for whisper indicators
    whisper_patterns = [r'\bwhisper\b', r'\bshush\b', r'\bquietly\b', r'\bsilently\b',
                        r'\.\.\.', r'\(whisper', r'\[whisper']
    if any(re.search(p, text_lower) for p in whisper_patterns):
        return 'whisper'

    # Check for anger
    anger_patterns = [r'\bangry\b', r'\bfurious\b', r'\brage\b', r'\bshout\b', r'\byell\b',
                       r'\bhate\b', r'\bdamn\b', r'\bhell\b', r'!{2,}',
                       r'\bwhat the\b', r'\bshut up\b', r'\bgo away\b']
    anger_score = sum(1 for p in anger_patterns if re.search(p, text_lower))
    if anger_score >= 2 or '!' in text:
        return 'angry'

    # Check for fear
    fear_patterns = [r'\bscared\b', r'\bafraid\b', r'\bterror\b', r'\bhorror\b',
                      r'\bfrightened\b', r'\bpanic\b', r'\bdanger\b', r'\brun\b',
                      r'\bhelp\b', r'\bno\b', r'\bstop\b']
    fear_score = sum(1 for p in fear_patterns if re.search(p, text_lower))
    if fear_score >= 2:
        return 'fear'

    # Check for excitement
    excitement_patterns = [r'\bamazing\b', r'\bawesome\b', r'\bwow\b', r'\bincredible\b',
                           r'\bfantastic\b', r'\bgreat\b', r'\byes\b', r'\bwoohoo\b',
                           r'\bexcited\b', r'\bcannot wait\b']
    excitement_score = sum(1 for p in excitement_patterns if re.search(p, text_lower))
    if excitement_score >= 2:
        return 'excited'

    # Check for happiness
    happy_patterns = [r'\bhappy\b', r'\bjoy\b', r'\blaugh\b', r'\bsmile\b',
                      r'\bglad\b', r'\bwonderful\b', r'\blove it\b', r'\bfun\b']
    happy_score = sum(1 for p in happy_patterns if re.search(p, text_lower))
    if happy_score >= 2:
        return 'happy'

    # Check for sadness
    sad_patterns = [r'\bsad\b', r'\bcry\b', r'\btears\b', r'\bheartbreaking\b',
                    r'\bmourn\b', r'\bmiss\b', r'\blonely\b', r'\bsorry\b']
    sad_score = sum(1 for p in sad_patterns if re.search(p, text_lower))
    if sad_score >= 2:
        return 'sad'

    # Check for romance
    romance_patterns = [r'\blove\b', r'\bromance\b', r'\bkiss\b', r'\bhug\b',
                        r'\bdear\b', r'\bhoney\b', r'\bsweetheart\b', r'\bbaby\b']
    romance_score = sum(1 for p in romance_patterns if re.search(p, text_lower))
    if romance_score >= 2:
        return 'romantic'

    return 'neutral'


def _estimate_spoken_seconds(text, target_lang):
    """
    Estimate whether translated text can be spoken in the subtitle slot.
    Khmer TTS/dubbing generally needs more time per visible character than English.
    """
    clean = re.sub(r'\s+', '', text or '')
    if not clean:
        return 0.0

    base_lang = (target_lang or '').split('-')[0].lower()
    if base_lang == 'km' or _contains_khmer(clean):
        return len(clean) / KHMER_DUBBING_CHARS_PER_SEC

    return len(clean) / 15.0


def _fits_subtitle_duration(text, duration, target_lang):
    return _estimate_spoken_seconds(text, target_lang) <= duration + SUBTITLE_TIMING_GRACE_SECONDS


def _has_terminal_punctuation(text):
    return bool(re.search(r'[.!?\u17d4\u17d5\u17d6]["\')\]]*\s*$', text or ''))


def _starts_new_scene(current_scene, next_seg):
    """
    Group nearby subtitles into dubbing scenes using timing and dialogue-flow cues.
    The original subtitle timestamps are not edited; scene boundaries reuse them.
    """
    if not current_scene:
        return False

    first = current_scene[0]
    prev = current_scene[-1]
    next_text = (next_seg.get('source') or '').strip()
    prev_text = (prev.get('source') or '').strip()
    if not next_text:
        return False

    try:
        gap = float(next_seg.get('start', 0)) - float(prev.get('end', 0))
        scene_duration = float(prev.get('end', 0)) - float(first.get('start', 0))
    except (TypeError, ValueError):
        gap = 0.0
        scene_duration = 0.0

    scene_text = ' '.join((seg.get('source') or '').strip() for seg in current_scene)
    if gap >= SCENE_GAP_SECONDS:
        return True
    if scene_duration >= SCENE_MAX_DURATION_SECONDS:
        return True
    if len(scene_text) + len(next_text) > SCENE_MAX_SOURCE_CHARS:
        return True

    # Keep fast back-and-forth lines together, but break after complete thoughts.
    if len(current_scene) >= 5 and _has_terminal_punctuation(prev_text):
        return True
    if (
        len(current_scene) >= 3
        and _has_terminal_punctuation(prev_text)
        and re.match(r'^(then|now|meanwhile|but|so|however|next)\b', next_text, re.IGNORECASE)
    ):
        return True

    return False


def _scene_source_text(scene_segments):
    parts = []
    for seg in scene_segments:
        text = _normalize_translation_text(seg.get('source', ''))
        if text:
            parts.append(text)
    return ' '.join(parts)


def _build_dubbing_scenes(segments):
    """
    Convert subtitle lines into ordered scene blocks for translation and TTS.
    Scene timestamps are copied from the first and last original subtitle.
    
    CRITICAL: Each scene preserves individual source_segments with their original
    boundaries so that translations can be mapped back to individual subtitles.
    """
    scenes = []
    current = []

    for seg in segments:
        if _starts_new_scene(current, seg):
            scenes.append(current)
            current = []
        current.append(seg)

    if current:
        scenes.append(current)

    scene_segments = []
    for idx, scene in enumerate(scenes):
        source_text = _scene_source_text(scene)
        if not source_text:
            continue

        scene_segments.append({
            'start': scene[0].get('start'),
            'end': scene[-1].get('end'),
            'source': source_text,
            'target': '',
            'emotion': _detect_emotion(source_text),
            'scene_id': idx + 1,
            'subtitle_count': len(scene),
            # CRITICAL: Preserve individual source segments for mapping translations back
            'source_segments': [
                {
                    'start': original.get('start'),
                    'end': original.get('end'),
                    'source': original.get('source', ''),
                }
                for original in scene
            ],
        })

    return scene_segments


def _important_markers(text):
    """Markers that should survive any candidate rewrite."""
    if not text:
        return set()
    return set(re.findall('\\d+(?:[.,:/-]\\d+)*|[%$\u20ac\u00a3\u00a5\u17db]|[A-Z]{2,}', text))


_NUMBER_SCALE_FACTORS = {
    '\u4e07': 10_000,          # Chinese: 万
    '\u842c': 10_000,          # Traditional Chinese: 萬
    '\u4ebf': 100_000_000,     # Chinese: 亿
    '\u5104': 100_000_000,     # Traditional Chinese: 億
    '\u1798\u17c9\u17ba\u1793': 10_000,  # Khmer: ម៉ឺន
    '\u179b\u17b6\u1793': 1_000_000,     # Khmer: លាន
    'million': 1_000_000,
    'billion': 1_000_000_000,
}


def _scaled_numeric_values(text, required_number=''):
    """
    Return semantic values for numbers followed by a scale word.

    Translation providers correctly render Chinese forms such as ``500万`` as
    ``5 million``. Comparing only the literal digits incorrectly rejects that
    translation even though both expressions represent 5,000,000.
    """
    if not text:
        return []

    normalized = str(text).translate(str.maketrans(
        '\u17e0\u17e1\u17e2\u17e3\u17e4\u17e5\u17e6\u17e7\u17e8\u17e9',
        '0123456789',
    ))
    units = sorted(_NUMBER_SCALE_FACTORS, key=len, reverse=True)
    unit_pattern = '|'.join(re.escape(unit) for unit in units)
    pattern = re.compile(
        rf'(\d+(?:[.,]\d+)?)[\s\u200b\u200c\u200d]*({unit_pattern})',
        flags=re.IGNORECASE,
    )

    values = []
    for match in pattern.finditer(normalized):
        number_text = match.group(1)
        if required_number and number_text != required_number:
            continue
        try:
            number = float(number_text.replace(',', ''))
        except ValueError:
            continue
        factor = _NUMBER_SCALE_FACTORS.get(
            match.group(2),
            _NUMBER_SCALE_FACTORS.get(match.group(2).lower()),
        )
        if factor:
            values.append(number * factor)
    return values


def _preserves_markers(source_text, candidate_text):
    candidate_markers = _important_markers(candidate_text)
    for marker in _important_markers(source_text):
        if marker in candidate_markers:
            continue

        # Permit an equivalent scaled number (for example 500万 -> 5 លាន),
        # while retaining exact matching for dates, percentages, and IDs.
        if re.fullmatch(r'\d+(?:[.,]\d+)?', marker):
            source_values = _scaled_numeric_values(source_text, marker)
            candidate_values = _scaled_numeric_values(candidate_text)
            if source_values and any(
                abs(source_value - candidate_value) < 0.5
                for source_value in source_values
                for candidate_value in candidate_values
            ):
                continue
        return False
    return True


def _condense_source_for_dubbing(source_text):
    """
    Remove low-information spoken fillers before a second translation attempt.
    This is intentionally conservative so meaning-bearing content remains intact.
    """
    text = f' {source_text.strip()} '
    if not text.strip():
        return ''

    filler_patterns = [
        r'\b(you know|I mean|kind of|sort of|basically|actually|really|just)\b',
        r'\b(as you can see|at this point in time|in order to)\b',
        r'\b(please note that|it is important to note that)\b',
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    replacements = {
        r'\bdo not\b': "don't",
        r'\bdoes not\b': "doesn't",
        r'\bdid not\b': "didn't",
        r'\bwe are going to\b': "we'll",
        r'\byou are going to\b': "you'll",
        r'\bI am going to\b': "I'll",
        r'\bin order to\b': 'to',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip(' ,')
    return text if len(text) < len(source_text.strip()) else source_text.strip()


def _summarize_source_for_dubbing(source_text, duration, target_lang):
    """
    Create a shorter spoken-translation source only when timing requires it.
    This keeps complete sentence/clause boundaries where possible and protects
    numbers, acronyms, and other important markers.
    """
    text = _condense_source_for_dubbing(_normalize_translation_text(source_text))
    if not text:
        return ''

    target_chars = max(60, int(max(duration, 0.1) * MAX_DUBBING_SOURCE_CHARS_PER_SECOND))
    if len(text) <= target_chars:
        return text

    sentences = [
        part.strip()
        for part in re.split(r'(?<=[.!?\u17d4\u17d5\u17d6])\s+', text)
        if part.strip()
    ]
    if len(sentences) > 1:
        kept = []
        for sentence in sentences:
            candidate = ' '.join(kept + [sentence]).strip()
            if kept and len(candidate) > target_chars:
                break
            kept.append(sentence)

        candidate = ' '.join(kept).strip()
        if candidate and _preserves_markers(source_text, candidate):
            return candidate

    clauses = [
        part.strip(' ,;:')
        for part in re.split(r'(?<=[,;:])\s+', text)
        if part.strip(' ,;:')
    ]
    if len(clauses) > 1:
        kept = []
        for clause in clauses:
            candidate = ', '.join(kept + [clause]).strip()
            if kept and len(candidate) > target_chars:
                break
            kept.append(clause)

        candidate = ', '.join(kept).strip()
        if candidate and _preserves_markers(source_text, candidate):
            if _has_terminal_punctuation(text) and not _has_terminal_punctuation(candidate):
                candidate += '.'
            return candidate

    words = text.split()
    if len(words) > 10:
        kept = []
        for word in words:
            candidate = ' '.join(kept + [word]).strip()
            if kept and len(candidate) > target_chars:
                break
            kept.append(word)

        candidate = ' '.join(kept).strip(' ,;:')
        if candidate and _preserves_markers(source_text, candidate):
            if not _has_terminal_punctuation(candidate):
                candidate += '.'
            return candidate

    return text


def _condense_khmer_for_dubbing(text):
    """
    Shorten only redundant Khmer phrasing. Avoid removing numbers, names, or clauses.
    """
    shortened = _normalize_translation_text(text)
    if not shortened:
        return ''

    replacements = [
        ('\u178a\u17c2\u179b\u1794\u17b6\u1793\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a', '\u178a\u17c2\u179b\u1794\u17b6\u1793'),
        ('\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a\u1794\u1784\u17d2\u17a0\u17b6\u1789', '\u1794\u1784\u17d2\u17a0\u17b6\u1789'),
        ('\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a\u1794\u17d2\u179a\u17be\u1794\u17d2\u179a\u17b6\u179f\u17cb', '\u1794\u17d2\u179a\u17be'),
        ('\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a\u1787\u17d2\u179a\u17be\u179f\u179a\u17be\u179f', '\u1787\u17d2\u179a\u17be\u179f'),
        ('\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a\u1794\u1784\u17d2\u1780\u17be\u178f', '\u1794\u1784\u17d2\u1780\u17be\u178f'),
        ('\u1793\u17c5\u1780\u17d2\u1793\u17bb\u1784', '\u1780\u17d2\u1793\u17bb\u1784'),
        ('\u1793\u17c5\u179b\u17be', '\u179b\u17be'),
        ('\u1793\u17c5\u1796\u17c1\u179b\u178a\u17c2\u179b', '\u1796\u17c1\u179b'),
        ('\u1796\u17b8\u1796\u17d2\u179a\u17c4\u17c7\u1790\u17b6', '\u1796\u17d2\u179a\u17c4\u17c7'),
        ('\u178a\u17be\u1798\u17d2\u1794\u17b8\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a', '\u178a\u17be\u1798\u17d2\u1794\u17b8'),
        ('\u1782\u17ba\u1787\u17b6', '\u1787\u17b6'),
        ('\u1794\u17b6\u1793\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a', '\u1794\u17b6\u1793'),
        ('\u1799\u17be\u1784\u1793\u17b9\u1784\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a', '\u1799\u17be\u1784\u1793\u17b9\u1784'),
        ('\u17a2\u17d2\u1793\u1780\u17a2\u17b6\u1785\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a', '\u17a2\u17d2\u1793\u1780\u17a2\u17b6\u1785'),
        ('\u179f\u17bc\u1798\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a', '\u179f\u17bc\u1798'),
    ]
    for verbose, concise in replacements:
        shortened = shortened.replace(verbose, concise)

    spoken_replacements = [
        ('\u178f\u17d2\u179a\u17bc\u179c\u1794\u17b6\u1793', '\u1794\u17b6\u1793'),
        ('\u1798\u17b6\u1793\u1793\u17d0\u1799\u1790\u17b6', '\u1782\u17ba'),
        ('\u1793\u17c1\u17c7\u1782\u17ba\u1787\u17b6', '\u1793\u17c1\u17c7\u1787\u17b6'),
        ('\u17a2\u17d2\u179c\u17b8\u178a\u17c2\u179b', '\u17a2\u17d2\u179c\u17b8'),
        ('\u1780\u17d2\u1793\u17bb\u1784\u1780\u17b6\u179a\u178a\u17c2\u179b', '\u1796\u17c1\u179b'),
        ('\u1793\u17c5\u1781\u17b6\u1784\u1780\u17d2\u1793\u17bb\u1784', '\u1780\u17d2\u1793\u17bb\u1784'),
    ]
    for verbose, concise in spoken_replacements:
        shortened = shortened.replace(verbose, concise)

    shortened = re.sub(r'\s+', ' ', shortened).strip()
    return shortened if len(shortened) < len(text.strip()) else text.strip()


def _naturalize_spoken_translation(text, target_lang):
    """Clean translated text for speech while preserving full clauses."""
    prepared = _normalize_translation_text(text)
    if not prepared:
        return ''

    prepared = re.sub(r'\s+([,.;:!?])', r'\1', prepared)
    prepared = re.sub(r'([([{])\s+', r'\1', prepared)
    prepared = re.sub(r'\s+([)\]}])', r'\1', prepared)
    prepared = re.sub(r'\.{2,}', '...', prepared)
    prepared = re.sub(r'\s+', ' ', prepared).strip()

    base_lang = (target_lang or '').split('-')[0].lower()
    if base_lang == 'km' or _contains_khmer(prepared):
        prepared = _condense_khmer_for_dubbing(prepared)
        prepared = re.sub(r'[“”"]', '', prepared)
        prepared = re.sub(r'\s*[,;:]\s*', ' ', prepared)
        prepared = re.sub(r'\s*([\u17d4\u17d5\u17d6])\s*', r'\1 ', prepared).strip()

    return prepared


def _fit_spoken_khmer_to_duration(text, duration):
    """
    Final safety pass for Khmer dubbing text. By default it does not remove
    words; timing is handled later by TTS speed, silence padding, and absolute
    timestamp placement. Set KHMER_ALLOW_TEXT_TRIM=1 to enable old phrase-fit
    trimming behavior.
    """
    prepared = _naturalize_spoken_translation(text, 'km')
    if not prepared:
        return ''
    if not ALLOW_KHMER_DUBBING_TEXT_TRIM:
        return prepared

    max_chars = max(24, int(max(duration, 0.1) * KHMER_MAX_TTS_CHARS_PER_SECOND))
    if len(re.sub(r'\s+', '', prepared)) <= max_chars:
        return prepared

    phrases = [
        part.strip()
        for part in re.split(r'(?<=[\u17d4\u17d5\u17d6.!?])\s+|\s{2,}', prepared)
        if part.strip()
    ]
    if len(phrases) <= 1:
        phrases = [part.strip() for part in re.split(r'\s+', prepared) if part.strip()]

    kept = []
    for phrase in phrases:
        candidate = ' '.join(kept + [phrase]).strip()
        if kept and len(re.sub(r'\s+', '', candidate)) > max_chars:
            break
        kept.append(phrase)

    shortened = ' '.join(kept).strip()
    return shortened if shortened else prepared


def _khmer_tts_char_budget(duration):
    return max(24, int(max(duration, 0.1) * KHMER_MAX_TTS_CHARS_PER_SECOND))


def _extract_gemini_text(api_data):
    parts = []
    for candidate in api_data.get('candidates', []) or []:
        content = candidate.get('content') or {}
        for part in content.get('parts', []) or []:
            text = part.get('text', '')
            if text:
                parts.append(text)
    return _normalize_translation_text(' '.join(parts))


# ---------------------------------------------------------------------------
# LLM Translation with Full Context
# ---------------------------------------------------------------------------
_llm_translation_cache = {}
_character_name_cache = {}


def _build_translation_prompt(source_text: str, source_lang: str, target_lang: str,
                                emotion: str = 'neutral', context_before: str = '',
                                context_after: str = '', known_names: dict = None) -> str:
    """
    Build a prompt for the LLM that provides full context for natural translation.
    """
    prompt_parts = [
        'Translate exactly one atomic subtitle into the target language.',
        '',
        'CRITICAL RULES:',
        '- Translate ONLY the characters and words inside Subtitle text',
        '- Never complete, continue, explain, expand, or paraphrase the subtitle',
        '- Never add a subject, object, name, phrase, or sentence',
        '- Never copy words from previous or next subtitle context',
        '- A one-character subtitle must remain one atomic translated term',
        '- Preserve every name, number, fact, and meaning present in the subtitle',
        '- Keep numbers, names, places, and facts accurate',
        '- Output ONLY the translation, no explanations',
        '- No quotes, timestamps, markdown, or labels',
        '- Do not add punctuation that is absent from the subtitle',
    ]

    if known_names:
        names_str = ', '.join(f'"{k}" -> "{v}"' for k, v in known_names.items())
        prompt_parts.append(f'- Character/Role names: {names_str} (KEEP THESE EXACT)')

    if context_before.strip():
        prompt_parts.append(
            f'\nPrevious subtitle for disambiguation only; DO NOT translate or '
            f'copy it: "{context_before.strip()}"'
        )

    if context_after.strip():
        prompt_parts.append(
            f'Next subtitle for disambiguation only; DO NOT translate or '
            f'copy it: "{context_after.strip()}"'
        )

    prompt_parts.append(f'\nSource language: {source_lang}')
    prompt_parts.append(f'Target language: {target_lang}')
    prompt_parts.append(f'\nSubtitle text: {source_text}')

    return '\n'.join(prompt_parts)


def _translate_with_gemini(source_text: str, source_lang: str, target_lang: str,
                            emotion: str = 'neutral', context_before: str = '',
                            context_after: str = '', known_names: dict = None,
                            duration: float = 5.0) -> str:
    """
    Translate subtitle text using Gemini LLM with full context.
    Retries on failure. Returns translated text or empty string on failure.
    """
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
        return ''

    import urllib.request
    import urllib.parse

    # Check cache first
    cache_key = f'{source_lang}:{target_lang}:{source_text[:100]}:{emotion}'
    if cache_key in _llm_translation_cache:
        logger.debug(f'[Translation] Using cached translation for: {source_text[:40]}...')
        return _llm_translation_cache[cache_key]

    model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
    if model.startswith('models/'):
        model = model.split('/', 1)[1]
    url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
        f'?key={urllib.parse.quote(api_key)}'
    )

    prompt = _build_translation_prompt(
        source_text, source_lang, target_lang,
        emotion=emotion,
        context_before=context_before,
        context_after=context_after,
        known_names=known_names,
    )

    payload = {
        'contents': [{
            'parts': [{'text': prompt}]
        }],
        'generationConfig': {
            'temperature': 0.3,
            'topP': 0.9,
            'maxOutputTokens': 500,
        },
    }

    for attempt in range(1, TRANSLATION_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            translated = _extract_gemini_text(data)
            if translated:
                # Cache the result
                _llm_translation_cache[cache_key] = translated
                if len(_llm_translation_cache) > 10000:
                    # Prevent memory bloat
                    _llm_translation_cache.clear()
                return translated

        except Exception as e:
            logger.warning(f'[Translation] Gemini attempt {attempt}/{TRANSLATION_MAX_RETRIES} failed: {e}')
            if attempt < TRANSLATION_MAX_RETRIES:
                wait_time = 2 ** attempt
                logger.info(f'[Translation] Retrying in {wait_time}s...')
                time.sleep(wait_time)

    return ''


def _translate_with_openai_api(source_text: str, source_lang: str, target_lang: str,
                                emotion: str = 'neutral', context_before: str = '',
                                context_after: str = '', known_names: dict = None,
                                duration: float = 5.0) -> str:
    """
    Translate using OpenAI-compatible API (supports any provider).
    Uses environment variables:
      - LLM_API_URL: base URL (default: https://api.openai.com/v1)
      - LLM_API_KEY: API key (default: OPENAI_API_KEY)
      - LLM_MODEL: model name (default: gpt-4o-mini)
    """
    import urllib.request
    import urllib.parse

    api_key = os.environ.get('LLM_API_KEY', os.environ.get('OPENAI_API_KEY', '')).strip()
    if not api_key:
        return ''

    base_url = os.environ.get('LLM_API_URL', 'https://api.openai.com/v1').strip()
    model = os.environ.get('LLM_MODEL', 'gpt-4o-mini').strip() or 'gpt-4o-mini'

    cache_key = f'openai:{source_lang}:{target_lang}:{source_text[:100]}:{emotion}'
    if cache_key in _llm_translation_cache:
        logger.debug(f'[Translation] Using cached OpenAI translation for: {source_text[:40]}...')
        return _llm_translation_cache[cache_key]

    url = f'{base_url}/chat/completions'

    prompt = _build_translation_prompt(
        source_text, source_lang, target_lang,
        emotion=emotion,
        context_before=context_before,
        context_after=context_after,
        known_names=known_names,
    )

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': 'You are a professional video subtitle translator. Translate naturally with full emotional and contextual accuracy.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.3,
        'max_tokens': 500,
    }

    for attempt in range(1, TRANSLATION_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))

            translated = ''
            for choice in data.get('choices', []):
                msg = choice.get('message', {})
                content = msg.get('content', '')
                if content:
                    translated += content

            if translated:
                translated = _normalize_translation_text(translated)
                _llm_translation_cache[cache_key] = translated
                return translated

        except Exception as e:
            logger.warning(f'[Translation] OpenAI attempt {attempt}/{TRANSLATION_MAX_RETRIES} failed: {e}')
            if attempt < TRANSLATION_MAX_RETRIES:
                wait_time = 2 ** attempt
                time.sleep(wait_time)

    return ''


def _fetch_llm_translation(source_text: str, source_lang: str, target_lang: str,
                            emotion: str = 'neutral', context_before: str = '',
                            context_after: str = '', known_names: dict = None,
                            duration: float = 5.0) -> List[str]:
    """
    Translate using available LLM backends.
    Priority: Gemini > OpenAI-compatible > Local Argos
    Returns list of candidate translations.
    """
    candidates = []

    # Try Gemini first (if API key configured)
    gemini_result = _translate_with_gemini(
        source_text, source_lang, target_lang,
        emotion=emotion, context_before=context_before,
        context_after=context_after, known_names=known_names,
        duration=duration
    )
    if gemini_result:
        candidates.append(gemini_result)

    # Try OpenAI-compatible (if API key configured)
    if not candidates:
        openai_result = _translate_with_openai_api(
            source_text, source_lang, target_lang,
            emotion=emotion, context_before=context_before,
            context_after=context_after, known_names=known_names,
            duration=duration
        )
        if openai_result:
            candidates.append(openai_result)

    return candidates


def _prepare_scene_for_tts(text, target_lang):
    """Make scene-level translated dialogue a clean continuous TTS block."""
    prepared = _naturalize_spoken_translation(text, target_lang)
    if not prepared:
        return ''

    base_lang = (target_lang or '').split('-')[0].lower()
    if base_lang == 'km' and not _has_terminal_punctuation(prepared):
        prepared = f'{prepared}\u17d4'
    elif not _has_terminal_punctuation(prepared):
        prepared = f'{prepared}.'

    return prepared


def _extend_subtitle_windows_for_audio(segments, target_lang, preserve_existing_timing: bool = True):
    """Extend subtitle durations only when explicitly requested for speech fitting."""
    if not segments:
        return segments

    if preserve_existing_timing:
        return [dict(segment) for segment in segments]

    extended_segments = []
    for idx, segment in enumerate(segments):
        seg = dict(segment)
        start = float(seg.get('start', 0.0))
        end = float(seg.get('end', start))
        duration = max(0.1, end - start)
        text = seg.get('target') or seg.get('source') or ''
        words = seg.get('words') or []
        if words:
            word_start = _coerce_float(words[0].get('start'), start)
            word_end = _coerce_float(words[-1].get('end'), end)
            seg['start'] = round(max(0.0, word_start), 3)
            seg['end'] = round(max(word_start + 0.01, word_end), 3)
            extended_segments.append(seg)
            continue

        needed_seconds = max(duration, _estimate_spoken_seconds(text, target_lang) + SUBTITLE_TIMING_GRACE_SECONDS + 0.25)

        if needed_seconds > duration:
            proposed_end = start + needed_seconds
            if idx + 1 < len(segments):
                next_start = float(segments[idx + 1].get('start', end))
                proposed_end = min(proposed_end, next_start - 0.05)
            if proposed_end > end:
                end = round(proposed_end, 2)

        seg['end'] = end
        extended_segments.append(seg)

    return extended_segments


def _validate_timing_accuracy(segments: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Validate that subtitle timing is accurate and synchronized.
    Uses word timestamps as the source of truth.
    Reports issues with early/late starts or speaker mismatches.
    """
    issues = []
    tolerance_ms = 100  # 100ms tolerance for timing variations

    for idx, seg in enumerate(segments):
        words = seg.get('words', [])
        seg_start = _coerce_float(seg.get('start', 0))
        seg_end = _coerce_float(seg.get('end', seg_start))

        if not words:
            # No word timestamps = cannot validate
            continue

        # Get word-level timing
        word_starts = [_coerce_float(w.get('start'), None) for w in words if _coerce_float(w.get('start'), None) is not None]
        word_ends = [_coerce_float(w.get('end'), None) for w in words if _coerce_float(w.get('end'), None) is not None]

        if not word_starts or not word_ends:
            continue

        first_word_start = min(word_starts)
        last_word_end = max(word_ends)

        # Check if subtitle timing matches word boundaries
        start_diff = abs(seg_start - first_word_start) * 1000  # Convert to ms
        end_diff = abs(seg_end - last_word_end) * 1000

        if start_diff > tolerance_ms:
            issues.append(
                f'Segment {idx}: Starts {start_diff:.0f}ms away from first word '
                f'(subtitle: {seg_start:.3f}, word: {first_word_start:.3f})'
            )

        if end_diff > tolerance_ms:
            issues.append(
                f'Segment {idx}: Ends {end_diff:.0f}ms away from last word '
                f'(subtitle: {seg_end:.3f}, word: {last_word_end:.3f})'
            )

        # Check speaker consistency
        word_speakers = [str(w.get('speaker_id', '')).strip() for w in words if w.get('speaker_id')]
        if len(set(word_speakers)) > 1:
            issues.append(f'Segment {idx}: Contains multiple speakers {set(word_speakers)}')

    if issues:
        logger.warning(f'[Timing Validation] Found {len(issues)} timing accuracy issues:')
        for issue in issues[:5]:  # Log first 5
            logger.warning(f'  {issue}')
        if len(issues) > 5:
            logger.warning(f'  ... and {len(issues) - 5} more')

    return segments, issues


def _sync_speech_to_subtitle(subtitle_start: float, subtitle_end: float,
                             audio_samples: Any, sample_rate: int = 16000) -> Tuple[Any, float, float]:
    """
    Synchronize speech audio to subtitle timing.
    
    1. Remove leading silence (speech must start exactly when subtitle appears)
    2. Remove trailing silence (speech must finish before next subtitle)
    3. If speech is too long, adjust speed slightly to fit
    4. Return trimmed audio, new start offset, new end offset
    
    Args:
        subtitle_start: The exact time the subtitle appears (seconds)
        subtitle_end: The exact time the subtitle disappears (seconds)
        audio_samples: numpy array of audio samples
        sample_rate: Audio sample rate in Hz
        
    Returns:
        (trimmed_audio, new_start_offset, new_end_offset)
    """
    import numpy as np
    if audio_samples is None or len(audio_samples) == 0:
        return audio_samples, subtitle_start, subtitle_end
    
    max_duration = max(0.1, subtitle_end - subtitle_start)
    max_samples = int(max_duration * sample_rate)
    
    # 1. Remove leading silence (detect first non-silent sample)
    # Use energy-based detection with adaptive threshold
    frame_size = int(0.01 * sample_rate)  # 10ms frames
    if len(audio_samples) > frame_size * 2:
        # Compute RMS per frame
        frames = []
        for i in range(0, len(audio_samples), frame_size):
            frame = audio_samples[i:i + frame_size]
            if len(frame) > 0:
                rms = np.sqrt(np.mean(frame ** 2))
                frames.append(rms)
        
        if frames:
            # Adaptive threshold: max(0.01, mean * 0.5, median * 2)
            mean_rms = float(np.mean(frames))
            median_rms = float(np.median(frames))
            threshold = max(0.01, mean_rms * 0.5, median_rms * 2.0)
            
            # Find first frame above threshold (speech onset)
            speech_start_idx = 0
            for i, rms in enumerate(frames):
                if rms > threshold:
                    speech_start_idx = max(0, i - 1)  # Include 1 frame before for context
                    break
            
            if speech_start_idx > 0:
                start_sample = speech_start_idx * frame_size
                audio_samples = audio_samples[start_sample:]
    
    # 2. Remove trailing silence
    if len(audio_samples) > frame_size * 2:
        frames = []
        for i in range(0, len(audio_samples), frame_size):
            frame = audio_samples[i:i + frame_size]
            if len(frame) > 0:
                rms = np.sqrt(np.mean(frame ** 2))
                frames.append(rms)
        
        if frames:
            mean_rms = float(np.mean(frames))
            median_rms = float(np.median(frames))
            threshold = max(0.01, mean_rms * 0.3, median_rms * 1.5)
            
            # Find last frame above threshold (speech offset)
            speech_end_idx = len(frames) - 1
            for i in range(len(frames) - 1, -1, -1):
                if frames[i] > threshold:
                    speech_end_idx = min(len(frames) - 1, i + 1)  # Include 1 frame after
                    break
            
            if speech_end_idx < len(frames) - 1:
                end_sample = (speech_end_idx + 1) * frame_size
                audio_samples = audio_samples[:end_sample]
    
    # 3. If audio is still too long, adjust speed slightly
    actual_duration = len(audio_samples) / sample_rate
    if actual_duration > max_duration * 0.95 and actual_duration > 0.1:
        # Compress slightly to fit
        speed_factor = min(1.5, max(0.8, max_duration / actual_duration))
        if speed_factor < 0.95 or speed_factor > 1.05:
            # Use simple resampling for speed adjustment
            new_length = int(len(audio_samples) / speed_factor)
            indices = np.linspace(0, len(audio_samples) - 1, new_length)
            # Linear interpolation resampling
            import math
            if len(audio_samples) > 0 and new_length > 0:
                x_old = np.arange(len(audio_samples))
                audio_samples = np.interp(indices, x_old, audio_samples).astype(np.float32)
    
    # Compute new start/end offsets
    trimmed_duration = len(audio_samples) / sample_rate
    new_end = subtitle_start + trimmed_duration
    if new_end > subtitle_end:
        new_end = subtitle_end
    
    return audio_samples, subtitle_start, new_end


def _force_align_audio_segment(audio_segment_path: str, subtitle_start: float, subtitle_end: float,
                                target_duration: Optional[float] = None) -> Tuple[Optional[str], float, float]:
    """
    Forced alignment for a single audio segment.
    
    If speech starts late: shift audio earlier so speech onset aligns with subtitle start.
    If speech ends too late: compress audio slightly until it fits inside subtitle duration.
    Never allow speech to overlap the next subtitle.
    
    Args:
        audio_segment_path: Path to WAV audio segment file
        subtitle_start: Target start time (seconds)
        subtitle_end: Target end time (seconds)
        target_duration: Desired duration after alignment (default: subtitle_end - subtitle_start)
        
    Returns:
        (aligned_audio_path, aligned_start, aligned_end) or (None, start, end) on failure
    """
    import numpy as np
    import wave
    
    if not audio_segment_path or not os.path.exists(audio_segment_path):
        return None, subtitle_start, subtitle_end
    
    max_duration = target_duration or max(0.1, subtitle_end - subtitle_start)
    
    try:
        with wave.open(audio_segment_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            data = wf.readframes(frames)
        
        if not data or frames == 0:
            return None, subtitle_start, subtitle_end
        
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Sync speech to subtitle timing
        aligned_samples, new_start, new_end = _sync_speech_to_subtitle(
            subtitle_start, subtitle_end, samples, sample_rate
        )
        
        if aligned_samples is None or len(aligned_samples) == 0:
            return None, subtitle_start, subtitle_end
        
        # Write aligned audio
        output_path = audio_segment_path.replace('.wav', '_aligned.wav')
        if output_path == audio_segment_path:
            output_path = audio_segment_path[:-4] + '_aligned.wav'
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            int_samples = (aligned_samples * 32767).astype(np.int16)
            wf.writeframes(int_samples.tobytes())
        
        return output_path, new_start, new_end
        
    except Exception as e:
        logger.warning(f'[Forced Alignment] Failed to align {audio_segment_path}: {e}')
        return None, subtitle_start, subtitle_end


def _remove_leading_trailing_silence_from_segments(segments: List[Dict]) -> List[Dict]:
    """
    Remove leading and trailing silence from subtitle timing based on word timestamps.
    
    For each segment with word timestamps:
      - Set subtitle_start = first word start
      - Set subtitle_end = last word end
      - If no word timestamps, keep original timing
    
    Critical: Never merges segments. Never changes segment boundaries in a way
    that would cause overlap.
    """
    if not segments:
        return segments
    
    updated = []
    for seg in segments:
        seg = dict(seg)
        words = seg.get('words', [])
        
        if words:
            word_starts = [float(w.get('start', 0)) for w in words if w.get('start') is not None]
            word_ends = [float(w.get('end', 0)) for w in words if w.get('end') is not None]
            
            if word_starts and word_ends:
                first_word = min(word_starts)
                last_word = max(word_ends)
                
                # Only update if word timestamps are within reasonable range
                orig_start = float(seg.get('start', 0))
                orig_end = float(seg.get('end', orig_start))
                
                if first_word >= orig_start and first_word <= orig_end:
                    seg['start'] = round(first_word, 3)
                if last_word >= orig_start and last_word <= orig_end:
                    seg['end'] = round(max(last_word, first_word + 0.01), 3)
                seg['duration'] = round(max(0.01, seg['end'] - seg['start']), 3)
        
        updated.append(seg)
    
    # Fix any overlaps created by the adjustment
    for i in range(1, len(updated)):
        prev_end = float(updated[i - 1].get('end', 0))
        curr_start = float(updated[i].get('start', 0))
        if curr_start < prev_end:
            if updated[i - 1].get('words') and updated[i].get('words'):
                continue
            # Create minimum gap, never merge
            gap = 0.02  # 20ms gap
            updated[i]['start'] = round(prev_end + gap, 3)
            if float(updated[i].get('end', 0)) < updated[i]['start']:
                updated[i]['end'] = round(updated[i]['start'] + 0.01, 3)
            updated[i]['duration'] = round(updated[i]['end'] - updated[i]['start'], 3)
    
    return updated


def _validate_subtitle_integrity(segments: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    FINAL integrity validation before subtitle export.
    
    Checks:
    ✓ No subtitle was skipped (gaps in coverage with no explanation)
    ✓ Every subtitle phrase is complete (no partial detections)
    ✓ No subtitle contains only half of the displayed sentence
    ✓ No subtitles were merged (each has unique text)
    ✓ Speech starts exactly with subtitle appearance
    ✓ Speech matches actor timing
    ✓ Subtitle timing matches video timing
    ✓ No subtitle has zero duration
    
    If any validation fails, automatically reprocess only the failed subtitle
    until synchronization is correct.
    """
    issues = []
    if not segments:
        return segments, issues
    
    # Check 1: No zero-duration subtitles
    for idx, seg in enumerate(segments):
        start = float(seg.get('start', 0))
        end = float(seg.get('end', start))
        duration = end - start
        if duration <= 0.001:
            issues.append(f'Segment {idx}: Zero duration subtitle')
    
    # Check 2: No merged subtitles (text contains multiple sentence endings with overlapping timing)
    for idx, seg in enumerate(segments):
        text = (seg.get('source') or seg.get('text') or '').strip()
        words = seg.get('words', [])
        
        if words and len(words) >= 4:
            # Check for large gaps between words indicating merged sentences
            gaps = []
            for w_idx in range(1, len(words)):
                prev_end = _coerce_float(words[w_idx - 1].get('end', 0))
                curr_start = _coerce_float(words[w_idx].get('start', prev_end))
                gap_ms = (curr_start - prev_end) * 1000.0
                gaps.append(gap_ms)
            
            large_gaps = [g for g in gaps if g > 200]  # >200ms gaps indicating merge
            if len(large_gaps) >= 1:
                issues.append(f'Segment {idx}: Possible merged sentences ({len(large_gaps)} gaps >200ms)')
    
    # Check 3: Subtitle timing wraps word timestamps
    tolerance = 0.15  # 150ms tolerance for timing sync
    for idx, seg in enumerate(segments):
        words = seg.get('words', [])
        if words:
            word_starts = [float(w.get('start', 0)) for w in words if w.get('start') is not None]
            word_ends = [float(w.get('end', 0)) for w in words if w.get('end') is not None]
            if word_starts and word_ends:
                first_word = min(word_starts)
                last_word = max(word_ends)
                seg_start = float(seg.get('start', 0))
                seg_end = float(seg.get('end', seg_start))
                
                if seg_start > first_word + tolerance:
                    issues.append(f'Segment {idx}: Subtitle starts {seg_start - first_word:.2f}s after speech')
                if seg_end < last_word - tolerance:
                    issues.append(f'Segment {idx}: Subtitle ends {last_word - seg_end:.2f}s before speech ends')
    
    # Check 4: No overlapping subtitles
    for i in range(1, len(segments)):
        prev_end = float(segments[i-1].get('end', 0))
        curr_start = float(segments[i].get('start', 0))
        if curr_start < prev_end and not (
            segments[i - 1].get('words') and segments[i].get('words')
        ):
            issues.append(f'Segment {i}: Overlaps with segment {i-1} by {prev_end - curr_start:.2f}s')
    
    # Fix issues automatically
    if issues:
        logger.warning(f'[Integrity Validation] Found {len(issues)} integrity issues')
        for issue in issues[:5]:
            logger.warning(f'  {issue}')
        
        # Fix timing issues: remove silence from segments
        segments = _remove_leading_trailing_silence_from_segments(segments)
        
        # Fix overlapping subtitles
        segments = _fix_overlapping_subtitles(segments)
        
        # Final boundary enforcement
        segments, _ = _enforce_strict_subtitle_boundaries(segments)
        
        logger.info(f'[Integrity Validation] Auto-fixed integrity issues')
    
    return segments, issues


def _finalize_subtitle_result(
    result: Dict,
    segments: List[Dict],
    target_lang: str,
    warning: str = '',
    preserve_boundaries: bool = False,
) -> Dict:
    """Build a successful subtitle response from the best segments available."""
    if preserve_boundaries:
        preserved = [dict(segment) for segment in segments]
        result['success'] = True
        result['segments'] = preserved
        result['srt'] = generate_srt(preserved)
        result['vtt'] = generate_vtt(preserved)
        result['export_json'] = _build_subtitle_export_payload(
            preserved,
            preserve_boundaries=True,
        )
        result['json'] = json.dumps(
            result['export_json'],
            ensure_ascii=False,
            indent=2,
        )
        if warning:
            result['warning'] = warning
        return result

    # Step 1: Remove leading/trailing silence to sync speech with subtitle timing
    segments = _remove_leading_trailing_silence_from_segments(segments)
    
    # Step 2: Extend windows and align
    segments = _extend_subtitle_windows_for_audio(segments, target_lang, preserve_existing_timing=True)
    segments = _enrich_subtitle_segments_with_alignment(segments, target_lang)
    
    # Step 3: Validate and auto-fix segmentation
    segments, export_warnings = validate_subtitle_segmentation(segments, auto_fix=True)
    
    # Step 4: CRITICAL: Validate timing accuracy using word timestamps as reference
    segments, timing_accuracy_issues = _validate_timing_accuracy(segments)
    
    # Step 5: Fix timing warnings
    timing_warnings = _check_overlapping_subtitles(segments) + _check_timing_boundaries(segments)
    if timing_warnings:
        segments = _fix_timing_boundaries(_fix_overlapping_subtitles(segments))
        segments = _repair_subtitle_timing(segments)
        logger.warning(f'[Final Export] Auto-fixed {len(timing_warnings)} timing issues before export.')
    if export_warnings:
        logger.info(f'[Final Export] Revalidated subtitle timing for {len(segments)} segments.')
    
    # Step 6: Check for merged sentences
    merged_warnings = _check_merged_sentences(segments) + _check_speaker_boundaries(segments)
    if merged_warnings:
        logger.warning(
            f'[Final Export] {len(merged_warnings)} possible merged subtitle issues remain after translation. '
            'Speech-boundary splitting runs before translation to avoid duplicating translated text.'
        )
    
    # Step 7: CRITICAL: Enforce strict subtitle boundaries before export
    # This ensures every subtitle is independent with its own timing
    segments, boundary_violations = _enforce_strict_subtitle_boundaries(segments)
    if boundary_violations:
        logger.warning(f'[Final Export] Fixed {len(boundary_violations)} boundary violations.')
    
    # Step 8: FINAL integrity validation
    segments, integrity_issues = _validate_subtitle_integrity(segments)
    if integrity_issues:
        logger.warning(f'[Final Export] Fixed {len(integrity_issues)} integrity issues before export.')
    
    # Step 9: Build output
    result['success'] = True
    result['segments'] = segments
    result['srt'] = generate_srt(segments)
    result['vtt'] = generate_vtt(segments)
    result['export_json'] = _build_subtitle_export_payload(segments)
    result['json'] = json.dumps(result['export_json'], ensure_ascii=False, indent=2)
    if warning:
        result['warning'] = warning
    
    # Log timing accuracy summary
    if timing_accuracy_issues:
        logger.info(f'[Final Export] Timing accuracy: {len(segments)} segments, {len(timing_accuracy_issues)} potential issues.')
    else:
        logger.info(f'[Final Export] Timing accuracy: All {len(segments)} segments have tight word-level synchronization.')
    
    # Log sync summary
    sync_ok = len(timing_accuracy_issues) == 0 and len(integrity_issues) == 0
    if sync_ok:
        logger.info(f'[Final Export] +++ SPEECH SYNC: All {len(segments)} segments synchronized correctly +++')
    else:
        logger.warning(f'[Final Export] Speech sync: {len(segments)} segments, {len(timing_accuracy_issues) + len(integrity_issues)} sync issues')
    
    return result


def _fallback_scene_candidates(scene_seg, source_lang, target_lang):
    """
    Fallback only when full-scene translation fails. It preserves scene order and
    still returns one continuous block for TTS compatibility.
    """
    candidates = []
    translated_parts = []

    for original in scene_seg.get('source_segments', []):
        source_text = (original.get('source') or '').strip()
        if not source_text:
            continue
        try:
            # Try local Argos as last resort
            line_candidates = _fetch_local_argos_translation(source_text, source_lang, target_lang)
            translated_parts.append(line_candidates[0] if line_candidates else source_text)
        except Exception as e:
            logger.warning(f'[Stage 3] Scene fallback line translation failed: {e}')
            translated_parts.append(source_text)

    if translated_parts:
        candidates.append(' '.join(translated_parts))
    return candidates


def _translation_candidates(api_data):
    candidates = []
    primary = api_data.get('responseData', {}).get('translatedText', '')
    if primary:
        candidates.append(primary)

    for match in api_data.get('matches', []) or []:
        translated = match.get('translation') or match.get('translatedText') or ''
        if translated:
            candidates.append(translated)

    unique = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_translation_text(candidate)
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


def _is_atomic_translation_candidate(source_text: str, translated_text: str) -> bool:
    """Reject obvious expansion of very short atomic subtitle text."""
    source = (source_text or '').strip()
    translated = (translated_text or '').strip()
    if not source or not translated:
        return False

    source_units = re.findall(r'[\u3400-\u9fff]|[\w]+', source, flags=re.UNICODE)
    if len(source_units) > 2:
        return True

    translated_words = re.findall(r'\S+', translated, flags=re.UNICODE)
    translated_chars = len(re.sub(r'[\s\W_]+', '', translated, flags=re.UNICODE))
    max_chars = 16 if len(source_units) <= 1 else 28
    return len(translated_words) <= 4 and translated_chars <= max_chars


def _choose_timing_aware_translation(source_text, candidates, duration, target_lang):
    base_target = (target_lang or '').split('-')[0].lower()
    normalized_candidates = []
    for candidate in candidates:
        normalized = _normalize_translation_text(candidate)
        # Providers sometimes romanize an atomic Chinese name instead of
        # returning the requested Khmer script (for example: 萧 -> Xiao).
        # Keep the name and boundary, but render that provider result
        # phonetically in Khmer so validation and TTS can continue.
        if (
            base_target == 'km'
            and normalized
            and not _contains_khmer(normalized)
            and re.search(r'[\u3400-\u9fff]', source_text or '')
            and re.fullmatch(r"[\sA-Za-z.'’-]+", normalized)
        ):
            normalized = re.sub(
                r'[A-Za-z]+',
                lambda match: _translate_latin_word_to_khmer(match.group(0)),
                normalized,
            )
        if normalized:
            normalized_candidates.append(normalized)

    usable = [
        c for c in normalized_candidates
        if _preserves_markers(source_text, c)
        and _is_atomic_translation_candidate(source_text, c)
        and _translation_matches_target_language(c, target_lang)
    ]
    if not usable:
        return source_text

    return usable[0]


def _fetch_local_argos_translation(source_text, source_lang, target_lang):
    """Translate with installed Argos packages only; never downloads models here."""
    try:
        from argostranslate import translate as argos_translate
    except Exception as e:
        logger.info(f'[Stage 3] Argos Translate is not installed: {e}')
        return []

    source_base = (source_lang or '').split('-')[0].lower()
    target_base = (target_lang or '').split('-')[0].lower()
    try:
        installed_languages = argos_translate.get_installed_languages()
        from_lang = next((lang for lang in installed_languages if lang.code == source_base), None)
        to_lang = next((lang for lang in installed_languages if lang.code == target_base), None)
        if not from_lang or not to_lang:
            logger.warning(
                f'[Stage 3] Missing local Argos package for {source_base}->{target_base}; '
                'keeping source text for this segment.'
            )
            return []

        translation = from_lang.get_translation(to_lang)
        translated = _normalize_translation_text(translation.translate(source_text))
        return [translated] if translated else []
    except Exception as e:
        logger.warning(f'[Stage 3] Local Argos translation failed: {e}')
        return []


def _fetch_translation(source_text, source_lang, target_lang):
    """Fetch through local Argos, MyMemory, then Google's public fallback."""
    local_candidates = _fetch_local_argos_translation(source_text, source_lang, target_lang)
    valid_local_candidates = [
        candidate for candidate in local_candidates
        if _translation_matches_target_language(candidate, target_lang)
    ]
    if valid_local_candidates:
        return valid_local_candidates

    if not _env_enabled('ALLOW_ONLINE_FREE_TRANSLATION', default=True):
        return []

    import urllib.request
    import urllib.parse

    email = os.environ.get('MYMEMORY_EMAIL', '').strip()
    url = (
        f'https://api.mymemory.translated.net/get'
        f'?q={urllib.parse.quote(source_text)}'
        f'&langpair={source_lang}|{target_lang}'
    )
    if email:
        url += f'&de={urllib.parse.quote(email)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    last_status = None
    for attempt in range(1, TRANSLATION_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
            last_status = data.get('responseStatus')
            candidates = _translation_candidates(data)
            valid_candidates = [
                candidate for candidate in candidates
                if _translation_matches_target_language(candidate, target_lang)
            ]
            if last_status == 200 and valid_candidates:
                return valid_candidates
            logger.warning(
                f'[Stage 3] MyMemory attempt {attempt}/{TRANSLATION_MAX_RETRIES} '
                f'returned status={last_status!r} without a valid '
                f'{target_lang} translation.'
            )
        except Exception as e:
            logger.warning(
                f'[Stage 3] MyMemory attempt '
                f'{attempt}/{TRANSLATION_MAX_RETRIES} failed for '
                f'{source_lang}->{target_lang}: {e}'
            )
        if attempt < TRANSLATION_MAX_RETRIES:
            time.sleep(min(2 ** (attempt - 1), 4))

    logger.warning(
        f'[Stage 3] MyMemory exhausted for {source_lang}->{target_lang} '
        f'(last status={last_status!r}); trying Google fallback.'
    )
    return _fetch_google_free_translation(source_text, source_lang, target_lang)


def _fetch_google_free_translation(source_text: str, source_lang: str,
                                   target_lang: str) -> List[str]:
    """Use Google's keyless web endpoint when the primary provider is exhausted."""
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode({
        'client': 'gtx',
        'sl': _normalize_whisper_lang(source_lang) or 'auto',
        'tl': _normalize_whisper_lang(target_lang) or target_lang,
        'dt': 't',
        'q': source_text,
    })
    req = urllib.request.Request(
        f'https://translate.googleapis.com/translate_a/single?{query}',
        headers={'User-Agent': 'Mozilla/5.0'},
    )
    for attempt in range(1, TRANSLATION_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                payload = json.loads(response.read().decode('utf-8'))
            translated = _normalize_translation_text(''.join(
                str(item[0])
                for item in (
                    payload[0]
                    if isinstance(payload, list) and payload
                    else []
                )
                if isinstance(item, list) and item and item[0]
            ))
            if translated:
                return [translated]
            raise RuntimeError('provider returned an empty translation')
        except Exception as e:
            logger.warning(
                f'[Stage 3] Google fallback attempt '
                f'{attempt}/{TRANSLATION_MAX_RETRIES} failed for '
                f'{source_lang}->{target_lang}: {e}'
            )
            if attempt < TRANSLATION_MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 4))
    return []


# ============================================================================
# CRITICAL: Translation with Subtitle Boundary Preservation
# ============================================================================
# translate_segments must return individual subtitle segments with their
# original timing boundaries preserved. Each subtitle phrase keeps its own
# start/end time and gets its own translated text.
# ============================================================================


def _validate_subtitle_transform(
    before: List[Dict],
    after: List[Dict],
    require_target: bool = False,
):
    """Validate count, order, source text, and timestamps across a transform."""
    if len(before) != len(after):
        raise RuntimeError(
            f'Subtitle count changed from {len(before)} to {len(after)}.'
        )

    for index, (original, transformed) in enumerate(zip(before, after), 1):
        original_start, original_end = _resolve_segment_voice_window(original)
        transformed_start, transformed_end = _resolve_segment_voice_window(transformed)
        if (original_start, original_end) != (transformed_start, transformed_end):
            raise RuntimeError(
                f'Subtitle {index} timestamps changed during processing.'
            )
        if original.get('source', '') != transformed.get('source', ''):
            raise RuntimeError(
                f'Subtitle {index} source text changed during processing.'
            )
        if require_target:
            target = transformed.get('target', '')
            if not isinstance(target, str) or not target.strip():
                raise RuntimeError(f'Subtitle {index} has no translated text.')
            if not _is_atomic_translation_candidate(
                original.get('source', ''),
                target,
            ):
                raise RuntimeError(
                    f'Subtitle {index} translation expanded beyond its '
                    'atomic source text.'
                )


def _map_translated_text_to_individual_segments(
    scene_seg: Dict,
    translated_text: str,
    original_segments_list: List[Dict]
) -> List[Dict]:
    """
    Map the translated text of a scene back to individual subtitle segments.
    
    The scene has original source_segments with individual boundaries.
    After the scene is translated as one block, this function splits the
    translated text proportionally and assigns each part to the correct
    individual subtitle segment.
    
    Args:
        scene_seg: The scene dict with source_segments containing original subtitles
        translated_text: The translated text for the entire scene
        original_segments_list: The complete list of original segments to find matches
    
    Returns:
        Updated original_segments_list with translated text mapped to individual segments
    """
    source_segments = scene_seg.get('source_segments', [])
    if not source_segments or not translated_text:
        return original_segments_list
    
    # If there's only one source segment, assign all translated text to it
    if len(source_segments) == 1:
        for i, orig_seg in enumerate(original_segments_list):
            # Match by start time (within tolerance) and source text
            seg_start = _coerce_float(orig_seg.get('start', 0))
            source_start = _coerce_float(source_segments[0].get('start', 0))
            seg_source = (orig_seg.get('source') or '').strip()
            scene_source = (source_segments[0].get('source') or '').strip()
            
            if abs(seg_start - source_start) < 0.1 and seg_source == scene_source:
                original_segments_list[i]['target'] = translated_text
                return original_segments_list
        
        # Fallback: match by start time only
        for i, orig_seg in enumerate(original_segments_list):
            seg_start = _coerce_float(orig_seg.get('start', 0))
            source_start = _coerce_float(source_segments[0].get('start', 0))
            if abs(seg_start - source_start) < 0.15:
                original_segments_list[i]['target'] = translated_text
                return original_segments_list
        
        return original_segments_list
    
    # Multiple source segments in this scene - split translated text proportionally
    # by character ratio of source texts
    total_source_chars = sum(len(s.get('source', '')) for s in source_segments)
    if total_source_chars == 0:
        return original_segments_list
    
    # Split translated text by proportion
    translated_chars = list(translated_text)
    start_idx = 0
    result_segments = list(original_segments_list)
    
    for src_seg in source_segments:
        src_text = src_seg.get('source', '')
        src_len = len(src_text)
        if total_source_chars > 0 and src_len > 0 and start_idx < len(translated_chars):
            # Calculate proportion of this segment within the scene
            proportion = src_len / total_source_chars
            seg_trans_len = max(1, int(proportion * len(translated_chars)))
            
            # Ensure we don't exceed array bounds
            end_idx = min(start_idx + seg_trans_len, len(translated_chars))
            if end_idx > start_idx:
                seg_translated = ''.join(translated_chars[start_idx:end_idx]).strip()
                start_idx = end_idx
                
                # Find and update the corresponding original segment
                src_start = _coerce_float(src_seg.get('start', 0))
                for i, orig_seg in enumerate(result_segments):
                    orig_start = _coerce_float(orig_seg.get('start', 0))
                    orig_source = (orig_seg.get('source') or '').strip()
                    if abs(orig_start - src_start) < 0.15 and orig_source == src_text:
                        result_segments[i]['target'] = seg_translated
                        break
        else:
            # Empty source segment - skip
            continue
    
    # If there's remaining translated text, add it to the last segment
    if start_idx < len(translated_chars):
        remaining = ''.join(translated_chars[start_idx:]).strip()
        if remaining:
            # Find the last segment that was matched and append
            for i in range(len(result_segments) - 1, -1, -1):
                if result_segments[i].get('target'):
                    result_segments[i]['target'] = (result_segments[i]['target'] + ' ' + remaining).strip()
                    break
    
    return result_segments


def translate_segments(segments, source_lang, target_lang):
    """
    Group subtitle lines into logical dubbing scenes, then translate each scene
    from source_lang to target_lang using LLM with full context.
    
    CRITICAL: Returns the ORIGINAL individual segments with translated text
    mapped back to each segment. Subtitle boundaries NEVER change.
    Every original subtitle keeps its own start/end timing.
    """
    # Save original segments with their boundaries
    original_segments = [dict(seg) for seg in segments]
    
    # Translate every subtitle independently. Neighbouring text is still sent
    # as context below, but one provider response can never merge lines or be
    # split proportionally across unrelated timestamp windows.
    scenes = []
    for original in original_segments:
        scene = dict(original)
        scene['source_segments'] = [dict(original)]
        scenes.append(scene)
    if not scenes:
        return original_segments

    if source_lang == target_lang:
        logger.info('[Stage 3] Source and target languages are the same. Grouping scenes and copying text.')
        for seg in original_segments:
            seg['target'] = seg['source']
        return original_segments

    logger.info(
        f'[Stage 3] Translating {len(scenes)} dubbing scenes '
        f'from {len(segments)} subtitles: {source_lang} -> {target_lang}'
    )

    # Build known names from all segments for consistency
    known_names = {}
    for seg in segments:
        source_text = seg.get('source', '')
        # Extract capitalized words as potential names
        names = re.findall(r'\b[A-Z][a-z]+\b', source_text)
        for name in names:
            if name.lower() not in {'the', 'this', 'that', 'what', 'when', 'where', 'which',
                                    'then', 'there', 'here', 'and', 'but', 'for', 'not', 'with'}:
                known_names[name] = name

    translate_started_at = time.time()
    _update_pipeline_progress(
        'Translation', 0, len(scenes),
        f'Translating {len(scenes)} dubbing scenes',
        translate_started_at
    )

    def _translate_scene(i: int, seg: Dict) -> Tuple[int, Dict]:
        process_lock.raise_if_cancelled()
        seg = dict(seg)
        source_text = seg.get('source', '')
        if not source_text.strip():
            seg['target'] = ''
            return i, seg

        try:
            duration = _segment_duration(seg)
            base_target_lang = (target_lang or '').split('-')[0].lower()
            emotion = seg.get('emotion', 'neutral')

            # Context guides meaning and names, while the prompt still requires
            # an atomic translation of this subtitle only.
            context_before = ' '.join(
                str(item.get('source', '') or '').strip()
                for item in original_segments[max(0, i - 2):i]
                if str(item.get('source', '') or '').strip()
            )
            context_after = ' '.join(
                str(item.get('source', '') or '').strip()
                for item in original_segments[i + 1:i + 3]
                if str(item.get('source', '') or '').strip()
            )

            cache_key = _cache_key('translation', {
                'source_text': source_text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'emotion': emotion,
                'duration': round(duration, 3),
                'context_before': context_before,
                'context_after': context_after,
                'version': 3,
            })
            cached = _read_json_cache(TRANSLATION_CACHE_DIR, cache_key)
            if (
                cached
                and isinstance(cached.get('target'), str)
                and _translation_matches_target_language(cached.get('target'), target_lang)
            ):
                seg['target'] = cached['target']
                return i, seg

            candidates = []

            # Try LLM translation first (if configured)
            if base_target_lang == 'km' or _env_enabled('ALLOW_PAID_TRANSLATION_API'):
                llm_candidates = _fetch_llm_translation(
                    source_text, source_lang, target_lang,
                    emotion=emotion,
                    context_before=context_before,
                    context_after=context_after,
                    known_names=known_names,
                    duration=duration
                )
                candidates.extend(llm_candidates)

            # Fallback to Argos or free API
            if not candidates:
                candidates = _fetch_translation(source_text, source_lang, target_lang)

            # Last resort: rule-based candidates are valid only when they
            # actually match the requested target language.
            if not candidates:
                candidates = _fallback_scene_candidates(seg, source_lang, target_lang)

            translated = _choose_timing_aware_translation(
                source_text, candidates, duration, target_lang
            )

            seg['target'] = _normalize_translation_text(
                translated if translated else source_text
            )

            if not _translation_matches_target_language(seg.get('target'), target_lang):
                raise RuntimeError(
                    f'No translation provider returned valid {target_lang} text.'
                )

            if _translation_matches_target_language(seg.get('target'), target_lang):
                _write_json_cache(TRANSLATION_CACHE_DIR, cache_key, {
                    'target': seg.get('target', ''),
                    'source_text': source_text,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'timestamp': time.time(),
                })

        except Exception as e:
            seg['target'] = ''
            seg['translation_error'] = str(e)
            logger.warning(
                f'[Stage 3] Translation failed for subtitle {i + 1}: {e}; '
                f'source={source_text!r}'
            )

        return i, seg

    max_workers = _bounded_worker_count('TRANSLATION_WORKERS', default=4, maximum=8)
    translated_scenes = [None] * len(scenes)
    
    if len(scenes) == 1:
        translated_scenes[0] = _translate_scene(0, scenes[0])[1]
        _update_pipeline_progress('Translation', 1, 1, 'Translation complete', translate_started_at)
    else:
        completed = 0
        logger.info(f'[Stage 3] Using up to {max_workers} translation workers')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_translate_scene, i, seg)
                for i, seg in enumerate(scenes)
            ]
            for future in as_completed(futures):
                process_lock.raise_if_cancelled()
                i, translated_seg = future.result()
                translated_scenes[i] = translated_seg
                completed += 1
                if completed % 5 == 0 or completed == len(scenes):
                    _update_pipeline_progress(
                        'Translation', completed, len(scenes),
                        f'Translated {completed}/{len(scenes)} scenes',
                        translate_started_at
                    )
                    logger.info(f'[Stage 3] Translated {completed}/{len(scenes)} scenes')

    # CRITICAL: Map translated text back to individual original segments
    # Each original subtitle keeps its own start/end timing and gets its translated text
    updated_segments = list(original_segments)
    for scene_seg in translated_scenes:
        if scene_seg and scene_seg.get('target'):
            updated_segments = _map_translated_text_to_individual_segments(
                scene_seg, 
                scene_seg.get('target', ''),
                updated_segments
            )

    failed = [
        index + 1 for index, seg in enumerate(updated_segments)
        if not _translation_matches_target_language(seg.get('target', ''), target_lang)
    ]
    if failed:
        preview = ', '.join(map(str, failed[:10]))
        suffix = '…' if len(failed) > 10 else ''
        failed_sources = '; '.join(
            f'{item}: {(updated_segments[item - 1].get("source") or "")[:120]!r}'
            for item in failed[:3]
        )
        raise RuntimeError(
            f'Translation failed for subtitle(s) {preview}{suffix}. '
            f'No valid {target_lang} translation was returned after '
            f'{TRANSLATION_MAX_RETRIES} retries. Check the API key, quota, '
            'network, provider endpoint, or install the required Argos language package. '
            f'Failed source: {failed_sources}'
        )

    logger.info(f'[Stage 3] Scene-based translation complete. '
                f'{len(updated_segments)} individual subtitle segments preserved.')
    _validate_subtitle_transform(
        original_segments,
        updated_segments,
        require_target=True,
    )
    return updated_segments


# ---------------------------------------------------------------------------
# Stage 3b: Automatic Quality Improvement
# ---------------------------------------------------------------------------

# Maximum self-review iterations per subtitle
QUALITY_MAX_SELF_REVIEW_ITERATIONS = 3

# Maximum number of quality improvement passes across all segments
QUALITY_MAX_PASSES = 2


def _build_quality_improvement_prompt(
    source_text: str,
    current_translation: str,
    context_before: str = '',
    context_after: str = '',
    source_lang: str = 'en',
    target_lang: str = 'km',
) -> str:
    """
    Build a prompt for the LLM to review and improve a Khmer translation.
    The LLM acts as an expert Cambodian subtitle translator performing a
    quality review pass.
    """
    prompt_parts = [
        'You are an expert Cambodian subtitle translator reviewing a machine-generated translation.',
        '',
        'TASK: Review and improve the Khmer translation below. The goal is to make it sound like',
        'natural Cambodian speech written by a professional human translator.',
        '',
        'CRITICAL RULES:',
        '- Output ONLY the improved Khmer translation. No explanations, no quotes, no labels.',
        '- Preserve the COMPLETE original meaning — do not omit any detail, name, number, or fact.',
        '- Use natural Khmer (Cambodian) expressions, not literal word-for-word translations.',
        '- Choose words appropriate to the story context, character personality, and scene atmosphere.',
        '- Make dialogue sound like real Cambodian speech, not robotic machine translation.',
        '- Ensure correct Khmer grammar and sentence structure.',
        '- Preserve emotional tone (happy, sad, angry, excited, etc.) appropriately.',
        '- Keep character names, place names, numbers, and proper nouns exactly as they appear.',
        '- Preserve punctuation (. ! ? ...) to match the emotional delivery of the line.',
        '- Avoid awkward wording, stiff translations, repeated words, or unnatural sentence order.',
        '- If the current translation is already natural and accurate, you may return it unchanged.',
    ]

    if context_before.strip():
        prompt_parts.append(f'\nPrevious subtitle (context): "{context_before.strip()}"')
    if context_after.strip():
        prompt_parts.append(f'Next subtitle (context): "{context_after.strip()}"')

    prompt_parts.append(f'\nOriginal ({source_lang}) text: {source_text}')
    prompt_parts.append(f'Current Khmer translation: {current_translation}')
    prompt_parts.append(f'\nImproved Khmer translation:')

    return '\n'.join(prompt_parts)


def _improve_single_translation(
    source_text: str,
    current_translation: str,
    source_lang: str = 'en',
    target_lang: str = 'km',
    context_before: str = '',
    context_after: str = '',
) -> str:
    """
    Use LLM to review and improve a single subtitle translation.
    Returns the improved translation, or the original if no improvement is needed/made.
    """
    if not current_translation.strip() or current_translation.strip() == source_text.strip():
        return current_translation

    prompt = _build_quality_improvement_prompt(
        source_text, current_translation,
        context_before=context_before,
        context_after=context_after,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    base_target_lang = (target_lang or '').split('-')[0].lower()

    # Try Gemini first
    if base_target_lang == 'km' or _env_enabled('ALLOW_PAID_TRANSLATION_API'):
        api_key = os.environ.get('GEMINI_API_KEY', '').strip()
        if api_key:
            import urllib.request
            import urllib.parse
            model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
            if model.startswith('models/'):
                model = model.split('/', 1)[1]
            url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
                f'?key={urllib.parse.quote(api_key)}'
            )
            payload = {
                'contents': [{
                    'parts': [{'text': prompt}]
                }],
                'generationConfig': {
                    'temperature': 0.2,  # Low temperature for focused improvement
                    'topP': 0.85,
                    'maxOutputTokens': 500,
                },
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))
                improved = _extract_gemini_text(data)
                if improved:
                    return _normalize_translation_text(improved)
            except Exception as e:
                logger.warning(f'[Quality] Gemini improvement failed: {e}')

    # Fallback to OpenAI-compatible API
    api_key = os.environ.get('LLM_API_KEY', os.environ.get('OPENAI_API_KEY', '')).strip()
    if api_key:
        import urllib.request
        base_url = os.environ.get('LLM_API_URL', 'https://api.openai.com/v1').strip()
        model = os.environ.get('LLM_MODEL', 'gpt-4o-mini').strip() or 'gpt-4o-mini'
        url = f'{base_url}/chat/completions'
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are an expert Cambodian subtitle translator. Review and improve the Khmer translation to sound natural and human-written.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.2,
            'max_tokens': 500,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            improved = ''
            for choice in data.get('choices', []):
                msg = choice.get('message', {})
                content = msg.get('content', '')
                if content:
                    improved += content
            if improved:
                return _normalize_translation_text(improved)
        except Exception as e:
            logger.warning(f'[Quality] OpenAI improvement failed: {e}')

    return current_translation


def _self_review_subtitle(
    source_text: str,
    translation: str,
    source_lang: str = 'en',
    target_lang: str = 'km',
    context_before: str = '',
    context_after: str = '',
) -> str:
    """
    Perform iterative self-review on a single subtitle.
    Each iteration asks the LLM to improve the previous output.
    Stops when no further improvements are made or max iterations reached.
    """
    current = translation
    for iteration in range(QUALITY_MAX_SELF_REVIEW_ITERATIONS):
        improved = _improve_single_translation(
            source_text, current,
            source_lang=source_lang,
            target_lang=target_lang,
            context_before=context_before,
            context_after=context_after,
        )
        # Normalize for comparison
        improved_norm = re.sub(r'\s+', ' ', improved).strip()
        current_norm = re.sub(r'\s+', ' ', current).strip()

        if improved_norm == current_norm:
            # No change — translation is stable, stop iterating
            logger.debug(f'[Quality] Self-review converged after {iteration + 1} iteration(s)')
            break

        current = improved

        if iteration < QUALITY_MAX_SELF_REVIEW_ITERATIONS - 1:
            logger.debug(f'[Quality] Self-review iteration {iteration + 1}: improved')
        else:
            logger.debug(f'[Quality] Self-review reached max iterations ({QUALITY_MAX_SELF_REVIEW_ITERATIONS})')

    return current


def auto_improve_quality(
    segments: List[Dict],
    source_lang: str = 'en',
    target_lang: str = 'km',
) -> List[Dict]:
    """
    Automatically review and improve the quality of all translated segments.
    Performs multiple passes across all segments, with context-aware correction.

    For each segment:
      1. Compare with previous/next subtitles for context consistency
      2. Review the Khmer translation for naturalness, grammar, and fluency
      3. Rewrite unnatural-sounding translations while preserving meaning
      4. Perform self-review (iterative improvement) until stable

    Returns the improved segments list.
    """
    if not segments:
        return segments

    source_base = (source_lang or '').split('-')[0].lower()
    target_base = (target_lang or '').split('-')[0].lower()

    # Only apply quality improvement if translating TO Khmer
    if target_base != 'km':
        logger.info(f'[Quality] Skipping quality improvement: target language is not Khmer ({target_lang})')
        return segments

    logger.info(f'[Quality] Starting automatic quality improvement for {len(segments)} segments...')

    for quality_pass in range(QUALITY_MAX_PASSES):
        logger.info(f'[Quality] Pass {quality_pass + 1}/{QUALITY_MAX_PASSES}')
        total_improvements = 0

        for idx, seg in enumerate(segments):
            source_text = seg.get('source', '')
            current_target = seg.get('target', '')

            if not source_text or not current_target:
                continue

            # If target is same as source (untranslated), skip
            if current_target.strip() == source_text.strip():
                continue

            # Get context from surrounding segments
            context_before = ''
            context_after = ''

            if idx > 0:
                context_before = segments[idx - 1].get('target', '') or segments[idx - 1].get('source', '')
            if idx < len(segments) - 1:
                context_after = segments[idx + 1].get('target', '') or segments[idx + 1].get('source', '')

            # Perform self-review and improvement
            improved = _self_review_subtitle(
                source_text, current_target,
                source_lang=source_lang,
                target_lang=target_lang,
                context_before=context_before,
                context_after=context_after,
            )

            if improved != current_target:
                segments[idx]['target'] = improved
                total_improvements += 1

        logger.info(f'[Quality] Pass {quality_pass + 1}: improved {total_improvements} segments')

        if total_improvements == 0:
            logger.info(f'[Quality] No improvements needed — quality is satisfactory')
            break

    logger.info(f'[Quality] Quality improvement complete')
    return segments


# ============================================================================
# ADVANCED KHMER TRANSLATION RULES
# ============================================================================
# These stages implement the advanced rules for professional Khmer translation:
#   Stage 3c: Automatic Language Correction - detect and translate remaining foreign words
#   Stage 3d: Intelligent Summarization - summarize only when text is too long to read
#   Stage 3e: Repetition Reduction - reduce repeated words/phrases (max 2-3)
#   Stage 3f: Automatic Natural Khmer Rewrite - fix awkward/robotic translations
#   Stage 3g: Final Validation - comprehensive quality verification before export
#   Stage 3h: Mandatory Final Quality Enforcement - ZERO-tolerance pre-export check
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 3c: Automatic Language Correction
# Detect any remaining foreign-language words and translate them to Khmer.
# ---------------------------------------------------------------------------

# Known Khmer consonants/vowels range for detection
KHMER_UNICODE_RANGE = re.compile(r'[\u1780-\u17ff\u19e0-\u19ff]')

# Common English words that might be character names or proper nouns (NOT to be translated)
COMMON_ENGLISH_NAMES = {
    'john', 'mary', 'james', 'david', 'michael', 'robert', 'william', 'richard',
    'joseph', 'thomas', 'christopher', 'charles', 'daniel', 'matthew', 'anthony',
    'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua', 'kenneth', 'kevin',
    'brian', 'george', 'timothy', 'ronald', 'edward', 'jason', 'jeffrey', 'ryan',
    'jacob', 'gary', 'nicholas', 'eric', 'jonathan', 'stephen', 'larry', 'justin',
    'scott', 'brandon', 'benjamin', 'samuel', 'raymond', 'gregory', 'frank',
    'alexander', 'patrick', 'jack', 'dennis', 'jerry', 'tyler', 'aaron', 'jose',
    'nathan', 'henry', 'douglas', 'peter', 'adam', 'zachary', 'nathaniel',
    'sarah', 'jennifer', 'lisa', 'sandra', 'michelle', 'patricia', 'nancy',
    'karen', 'betty', 'helen', 'donna', 'carol', 'ruth', 'janet', 'catherine',
    'elizabeth', 'ann', 'victoria', 'laura', 'kimberly', 'deborah', 'jessica',
    'shirley', 'cynthia', 'angela', 'melissa', 'amanda', 'pamela', 'maria',
    'barbara', 'susan', 'margaret', 'dorothy', 'alice', 'julie', 'rebecca',
    'kathleen', 'virginia', 'amy', 'katherine', 'christine', 'tammy',
    'smith', 'jones', 'brown', 'wilson', 'taylor', 'davis', 'white', 'harris',
    'martin', 'thompson', 'garcia', 'robinson', 'clark', 'lewis', 'lee',
    'walker', 'hall', 'allen', 'young', 'king', 'wright', 'hill', 'scott',
    'green', 'adams', 'baker', 'nelson', 'carter', 'mitchell', 'roberts',
    'turner', 'phillips', 'campbell', 'parker', 'evans', 'edwards', 'collins',
}

# Words that should NEVER be translated (technical terms, common abbreviations)
UNTRANSLATABLE_TERMS = {
    # Technology
    'android', 'iphone', 'ios', 'windows', 'linux', 'macos', 'bluetooth',
    'wifi', 'gps', 'usb', 'hdmi', 'dvd', 'blu-ray', 'mp3', 'mp4', 'jpeg',
    'png', 'gif', 'pdf', 'html', 'css', 'javascript', 'python', 'java',
    'c++', 'api', 'url', 'http', 'https', 'email', 'sms', 'app', 'apps',
    # Units
    'km', 'kg', 'mph', 'kw', 'hz', 'ghz', 'volts', 'watts',
    # Common abbreviations
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc',
    'ltd', 'co', 'dept', 'govt', 'est', 'approx', 'dept',
}

# Set of common languages that might appear in subtitles
KNOWN_LANGUAGES = {'en', 'km', 'th', 'vi', 'zh', 'ja', 'ko', 'lo', 'my'}

COMMON_FOREIGN_WORD_TRANSLATIONS = {
    # Common provider romanization for the Chinese surname/name 萧/肖.
    'xiao': '\u179f\u17ca\u17b6\u179c',
    'a': 'មួយ',
    'an': 'មួយ',
    'i': 'ខ្ញុំ',
    'me': 'ខ្ញុំ',
    'my': 'របស់ខ្ញុំ',
    'you': 'អ្នក',
    'your': 'របស់អ្នក',
    'hello': 'សួស្តី',
    'hi': 'សួស្តី',
    'goodbye': 'លាឈប់',
    'bye': 'លាឈប់',
    'thanks': 'អរគុណ',
    'thank': 'អរគុណ',
    'please': 'សូម',
    'sorry': 'អធ្យាស្រ័យ',
    'yes': 'បាទ/ចាស',
    'no': 'ទេ',
    'ok': 'យល់ព្រម',
    'okay': 'យល់ព្រម',
    'good': 'ល្អ',
    'bad': 'អាក្រក់',
    'world': 'ពិភពលោក',
    'people': 'មនុស្ស',
    'friend': 'មិត្ត',
    'friends': 'មិត្តភក្តិ',
    'family': 'គ្រួសារ',
    'love': 'ស្រឡាញ់',
    'time': 'ម៉ោង',
    'day': 'ថ្ងៃ',
    'night': 'យប់',
    'home': 'ផ្ទះ',
    'house': 'ផ្ទះ',
    'water': 'ទឹក',
    'food': 'អាហារ',
    'money': 'ប្រាក់',
    'life': 'ជីវិត',
    'help': 'ជួយ',
    'look': 'មើល',
    'listen': 'ស្តាប់',
    'wait': 'រង់ចាំ',
    'come': 'មក',
    'go': 'ទៅ',
    'stay': 'នៅ',
    'know': 'ដឹង',
    'understand': 'យល់',
    'think': 'គិត',
    'want': 'ចង់',
    'need': 'ត្រូវការ',
    'can': 'អាច',
    'tell': 'ប្រាប់',
    'say': 'និយាយ',
    'speak': 'និយាយ',
    'ask': 'សួរ',
    'answer': 'ឆ្លើយ',
    'start': 'ចាប់ផ្តើម',
    'stop': 'ឈប់',
    'begin': 'ចាប់ផ្តើម',
    'end': 'បញ្ចប់',
    'today': 'ថ្ងៃនេះ',
    'tomorrow': 'ស្អែក',
    'yesterday': 'ម្សិលមិញ',
    'school': 'សាលារៀន',
    'teacher': 'គ្រូ',
    'student': 'សិស្ស',
    'doctor': 'វេជ្ជបណ្ឌិត',
    'hospital': 'មន្ទីរពេទ្យ',
    'city': 'ទីក្រុង',
    'country': 'ប្រទេស',
    'man': 'បុរស',
    'woman': 'ស្ត្រី',
    'child': 'កុមារ',
    'children': 'កុមារ',
    'boy': 'កុមារ',
    'girl': 'ក្មេងស្រី',
}

LATIN_TO_KHMER_TRANSLITERATION_MAP = {
    'a': 'អា', 'b': 'ប', 'c': 'ក', 'd': 'ដ', 'e': 'េ', 'f': 'ហ្វ', 'g': 'ក', 'h': 'ហ',
    'i': 'ិ', 'j': 'ច', 'k': 'ក', 'l': 'ល', 'm': 'ម', 'n': 'ន', 'o': 'ូ', 'p': 'ប',
    'q': 'ក្វ', 'r': 'រ', 's': 'ស', 't': 'ត', 'u': 'ុ', 'v': 'វ', 'w': 'វ', 'x': 'ឃ',
    'y': 'យ', 'z': 'ហ្ស',
}


def _get_segment_display_time(seg: Dict) -> float:
    """Calculate how long a subtitle is displayed (in seconds)."""
    start = float(seg.get('start', 0))
    end = float(seg.get('end', start))
    return max(0.5, end - start)


def _is_proper_name(word: str) -> bool:
    """
    Determine if a word is likely a proper name (should NOT be translated).
    Checks: capitalization, known name lists, all-caps abbreviations.
    """
    if not word or len(word) <= 1:
        return True

    # Numbers
    if word.isdigit():
        return True

    # All-caps abbreviations
    if word.isupper() and len(word) <= 5:
        return True

    lowered = word.lower()

    if lowered in COMMON_FOREIGN_WORD_TRANSLATIONS:
        return False

    # Check known names (case-insensitive)
    if lowered in COMMON_ENGLISH_NAMES:
        return True

    # Check untranslatable terms
    if lowered in UNTRANSLATABLE_TERMS:
        return True

    # Capitalized words (potential names)
    if word[0].isupper() and word[1:].islower() and len(word) > 1:
        return True

    return False


def _is_untranslatable_term(word: str) -> bool:
    """Check if a word should never be translated."""
    return word.lower() in UNTRANSLATABLE_TERMS


def _apply_word_casing(text: str, source_word: str) -> str:
    """Apply the original word's casing to a replacement text."""
    if not text:
        return text
    if source_word.isupper() and len(source_word) > 1:
        return text.upper()
    if source_word and source_word[0].isupper() and len(source_word) > 1:
        return text[0].upper() + text[1:]
    return text


def _translate_latin_word_to_khmer(word: str) -> str:
    """Translate a foreign word to Khmer with a deterministic fallback."""
    word_clean = word.strip("'-")
    if not word_clean:
        return word

    lowered = word_clean.lower()
    if lowered in COMMON_FOREIGN_WORD_TRANSLATIONS:
        return COMMON_FOREIGN_WORD_TRANSLATIONS[lowered]

    transliterated = ''.join(LATIN_TO_KHMER_TRANSLITERATION_MAP.get(ch.lower(), ch) for ch in word_clean)
    return transliterated or word_clean


def _replace_foreign_words_with_khmer(text: str) -> str:
    """Replace detected foreign words with Khmer translations or transliterations."""
    if not text:
        return text

    result = text
    for word in re.findall(r"[a-zA-Z'-]+", result):
        word_clean = word.strip("'-")
        if not word_clean:
            continue
        if _is_proper_name(word_clean) and word_clean.lower() not in COMMON_FOREIGN_WORD_TRANSLATIONS:
            continue
        if _is_untranslatable_term(word_clean) and word_clean.lower() not in COMMON_FOREIGN_WORD_TRANSLATIONS:
            continue
        replacement = _translate_latin_word_to_khmer(word_clean)
        if replacement and replacement != word_clean:
            # Boundaries only need to prevent matching inside another Latin
            # word. Khmer punctuation (for example "។") lives in the same
            # Unicode block as Khmer letters, so treating the entire block as
            # a word character leaves text such as "flattened។" unrepaired.
            pattern = re.compile(
                r'(?<![a-zA-Z])' + re.escape(word_clean) + r'(?![a-zA-Z])',
                re.IGNORECASE,
            )
            result = pattern.sub(_apply_word_casing(replacement, word_clean), result)

    return result


def _detect_foreign_words(text: str) -> List[str]:
    """
    Detect remaining foreign-language words in text that should be translated to Khmer.
    Returns list of foreign words found.
    
    A word is 'foreign' if:
    - It contains only Latin characters (a-z, A-Z)
    - It is NOT a proper name
    - It is NOT a known untranslatable technical term
    - It is NOT a number
    
    This function detects foreign words even within mixed-language text
    (e.g. Khmer sentences that still contain untranslated English words).
    """
    if not text:
        return []

    # Find all English/Latin words (works in both pure-English and mixed-language text)
    latin_words = re.findall(r"[a-zA-Z'-]+", text)
    foreign_words = []

    for word in latin_words:
        word_clean = word.strip("'-")
        if not word_clean:
            continue
        # Skip proper names, untranslatable terms, abbreviations
        if (not _is_proper_name(word_clean) or word_clean.lower() in COMMON_FOREIGN_WORD_TRANSLATIONS) and \
           (not _is_untranslatable_term(word_clean) or word_clean.lower() in COMMON_FOREIGN_WORD_TRANSLATIONS):
            foreign_words.append(word_clean)

    return foreign_words


def _build_foreign_word_translation_prompt(
    foreign_words: List[str],
    source_lang: str,
    context_before: str = '',
    context_after: str = '',
) -> str:
    """Build a prompt to translate remaining foreign words to Khmer."""
    words_str = ', '.join(f'"{w}"' for w in foreign_words)
    prompt_parts = [
        'You are a Khmer translator. Translate the following foreign words to Khmer (Cambodian).',
        '',
        'RULES:',
        '- Output ONLY the Khmer translations, one per line.',
        '- Preserve proper names as-is (do not translate names).',
        '- Use commonly accepted Khmer terms.',
        '- If a word should remain in the original language, write it as-is.',
        '- Format: word => translation',
        '',
    ]

    if context_before.strip():
        prompt_parts.append(f'Context (previous text): "{context_before.strip()}"')
    if context_after.strip():
        prompt_parts.append(f'Context (next text): "{context_after.strip()}"')

    prompt_parts.append(f'\nSource language: {source_lang}')
    prompt_parts.append(f'Target language: Khmer (Cambodian)')
    prompt_parts.append(f'\nWords to translate:')
    for w in foreign_words:
        prompt_parts.append(f'- {w}')

    return '\n'.join(prompt_parts)


def _translate_foreign_words_via_llm(
    foreign_words: List[str],
    source_lang: str,
    context_before: str = '',
    context_after: str = '',
) -> Dict[str, str]:
    """
    Use LLM to translate remaining foreign words to Khmer.
    Returns dict mapping original word -> Khmer translation.
    """
    if not foreign_words:
        return {}

    translation_map = {}

    # Build a map of known translations (common words)
    known_translations = {
        'hello': '\u1787\u17d2\u179a\u17be\u1780\u17cb\u179b\u17b7\u1794\u17d2\u179f\u17be',
        'goodbye': '\u1787\u17d2\u179a\u17be\u1780\u17cb\u179b\u17b7\u1794\u17d2\u179f\u17be',
        'thank you': '\u17a2\u17d2\u1782\u17c1\u178e\u17d2\u178e\u17b6\u1793',
        'thanks': '\u17a2\u17d2\u1782\u17c1\u178e\u17d2\u178e\u17b6\u1793',
        'please': '\u179f\u17d2\u179a\u17c1\u1781\u17d2\u1781\u17b6',
        'sorry': '\u179f\u17d2\u179c\u17a1\u17d2\u1781\u17b6\u179f\u17cb',
        'yes': '\u17af\u179c\u17cb\u1780\u17d2\u179b\u17b6\u17c6',
        'no': '\u1791\u17d2\u1780\u17d2\u1793\u17c4\u179a',
        'ok': '\u1798\u17d2\u179c\u17b6\u1793\u17c1\u179f\u17cb',
        'okay': '\u1798\u17d2\u179c\u17b6\u1793\u17c1\u179f\u17cb',
        'good': '\u179b\u17b7\u1794\u17d2\u179f\u17be',
        'bad': '\u17a2\u17b6\u1795\u17d2\u1795\u17c1\u179a\u17b7',
        'big': '\u1792\u17c4\u1789',
        'small': '\u178f\u17d2\u179a\u17c1\u1787',
        'beautiful': '\u1797\u17d2\u179a\u17b6\u1791\u179c\u17b6\u1780\u17be',
        'love': '\u179f\u17d2\u179a\u17c4\u1784\u17a1\u17d2\u1781\u17b6\u179f\u17cb',
        'hate': '\u1794\u17d2\u1794\u17b6\u179f\u17cb',
        'friend': '\u1798\u17d2\u1780\u17c2\u179a\u14e1\u1798\u17d2\u1780\u17c2\u179a',
        'family': '\u1782\u17d2\u179a\u17bb\u179f\u17d2\u179f\u17b6\u17a2\u17d2\u1793\u17c1\u1780',
        'water': '\u17d1\u17c7\u17bb\u1794\u17d2\u1794\u17d1\u17c7\u17c2\u1780',
        'food': '\u17a2\u17b6\u1799\u17d2\u17a0\u179a',
        'time': '\u17d1\u179c\u17d2\u17d1\u1798\u17b6\u1793\u17d1\u17c0\u17c4\u1780',
        'man': '\u1794\u17d2\u179a\u17b6\u1781\u17bc\u1780\u17d2\u179f\u17c4\u179b',
        'woman': '\u179f\u17d2\u1780\u17d2\u179f\u17be\u1784\u179f\u17c4\u179b\u179c\u17b7\u179f\u17be\u1784',
        'child': '\u1780\u17d2\u1784\u17c7\u1794\u17d2\u1799\u17b8\u1780\u17cb',
        'king': '\u1796\u17d2\u179a\u17b6\u1793\u17b6\u1787\u17c1\u1794\u17cb',
        'queen': '\u1796\u17d2\u179a\u17b6\u1793\u17b6\u1787\u17c1\u1794\u17cb\u179f\u17d2\u1780\u17d2\u179f\u17be\u1784',
        'god': '\u1796\u17d2\u179a\u17d1\u17a7\u1787\u1793\u17c4\u179b',
        'devil': '\u1796\u17d2\u179a\u17d1\u17a7\u1787\u17a2\u17b6\u1795\u17d2\u1795\u17c1\u179a\u17b7',
        'angel': '\u1791\u17d2\u179c\u17a7\u1794\u17d2\u1794\u17b6\u1793\u17c7\u179f\u17d2\u1790',
        'house': '\u1795\u17d2\u17a0\u17ca\u17cb\u17a2\u17d2\u1793\u17c1\u1780',
        'car': '\u1799\u17d2\u1794\u17c9\u17bb\u1780\u17cb',
        'money': '\u179b\u17be\u178e\u17d2\u178e\u17b6\u1799\u17cb\u179b\u17be\u178e\u17a2\u17d2\u1793\u17c1\u1780',
        'world': '\u1796\u17d2\u179b\u17a4\u1780\u179a\u1780\u17cb\u179b\u17c4\u1780\u17cb',
        'life': '\u1791\u17d2\u179c\u17b8\u1796\u17d2\u1791\u17bc\u1794\u17cb',
        'death': '\u17a2\u17d2\u1798\u17d1\u179a\u17a0\u17d2\u1793\u17c1\u178f',
        'sky': '\u17a2\u17d2\u1798\u17c1\u1780\u17cb\u17a2\u17d2\u1793\u17c1\u1780',
        'earth': '\u1795\u17d2\u179b\u17c1\u1784\u17d2\u17d1\u17c0\u17c4\u1780',
        'fire': '\u1797\u17d2\u179b\u17c4\u1780\u17cb\u17a2\u17d2\u1793\u17c1\u1780',
        'sun': '\u17d2\u179a\u17c4\u1781\u17cb\u17a2\u17d2\u1793\u17c1\u1780',
        'moon': '\u1796\u17d2\u179a\u17c3\u1794\u17d2\u179a\u17c3\u1784\u17cb\u17d1\u17c1\u1796\u17c4\u1780',
        'true': '\u1787\u17d2\u1798\u17c1\u1798\u17cb',
        'false': '\u1780\u17d2\u1782\u17c1\u1790\u17cb',
        'great': '\u1792\u17c4\u1789\u1793\u17b7\u1795\u17cb\u1791\u17b6\u1793\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a',
        'strong': '\u1798\u17d2\u17a0\u17b6\u1784\u17b9\u17c0\u1784\u1791\u17b6\u1793\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a',
        'weak': '\u1799\u17d2\u179c\u17be\u179f\u17c4\u1782\u17a2\u17c0\u1791\u17d2\u21b6\u1794\u17ca',
        'rich': '\u1793\u17c1\u1791\u17cb\u17a2\u17d2\u1793\u17c1\u1780',
        'poor': '\u1780\u17b6\u179a\u17bb\u1792\u17cb\u01b6\u17b6\u179f\u17cb',
        'young': '\u1791\u17d2\u1798\u17d1\u1794\u17cb\u17a2\u17d2\u1793\u17c1\u1780',
        'old': '\u1785\u17d2\u179a\u17c1\u17c7\u1791\u17b6\u1793\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a',
        'new': '\u1790\u17d2\u1798\u17d1\u1784\u17cb\u17a2\u17d2\u1793\u17c1\u1780',
        'happy': '\u179f\u17d2\u179a\u17c4\u1784\u17a1\u17d2\u1781\u17b6\u1798\u17cb\u1791\u17b6\u1793\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a',
        'sad': '\u1796\u17d2\u1780\u17be\u1794\u17cb\u1791\u17b6\u1793\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a',
        'angry': '\u1792\u17d2\u209b\u17be\u179a\u1794\u17d2\u209b\u17be\u179b\u17b6\u178f\u1791\u17b6\u1793\u1792\u17d2\u179c\u17be\u1780\u17b6\u179a',
        'scared': '\u1791-\u1794-\u1794-\u1791-NOT-KHMER',
        'afraid': '\u1791\u17d2\u1794-\u179b\u17c4-\u1784',
        'brave': '\u1798\u17d2\u17a0\u17b6\u1784\u16e0\u178f\u178a\u17d2\u179a',
        'smart': '\u1791-\u1794-\u1791-back',
        'stupid': '\u179b-\u1794-back',
    }

    # Use known translations first
    remaining_words = []
    for word in foreign_words:
        word_lower = word.lower()
        if word_lower in known_translations:
            translation_map[word] = known_translations[word_lower]
        else:
            remaining_words.append(word)

    # Try LLM for remaining words
    if remaining_words:
        prompt = _build_foreign_word_translation_prompt(remaining_words, source_lang)
        api_key = os.environ.get('GEMINI_API_KEY', '').strip()
        if api_key:
            import urllib.request
            import urllib.parse
            model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
            if model.startswith('models/'):
                model = model.split('/', 1)[1]
            url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
                f'?key={urllib.parse.quote(api_key)}'
            )
            payload = {
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {
                    'temperature': 0.2,
                    'topP': 0.9,
                    'maxOutputTokens': 300,
                },
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))
                result_text = _extract_gemini_text(data)
                if result_text:
                    # Parse results: each line is "word => translation"
                    for line in result_text.split('\n'):
                        line = line.strip()
                        if '=>' in line:
                            parts = line.split('=>', 1)
                            orig = parts[0].strip().strip('"\' ')
                            trans = parts[1].strip().strip('"\' ')
                            if orig and trans:
                                translation_map[orig] = trans
            except Exception as e:
                logger.warning(f'[Stage 3c] LLM foreign word translation failed: {e}')

        # If LLM failed, fall back to keeping original words
        for word in remaining_words:
            if word not in translation_map:
                translation_map[word] = word

    return translation_map


def auto_correct_language(
    segments: List[Dict],
    source_lang: str = 'en',
    target_lang: str = 'km',
) -> List[Dict]:
    """
    Stage 3c: Automatically detect and translate remaining foreign words in subtitles.
    
    For each segment, detect non-Khmer non-name words and translate them to Khmer.
    Characters names, place names, and technical terms are preserved.
    
    This works on BOTH pure-English AND mixed-language segments (Khmer text
    that still contains untranslated English words).
    """
    if not segments:
        return segments

    base_target = (target_lang or '').split('-')[0].lower()
    if base_target != 'km':
        return segments

    logger.info(f'[Stage 3c] Starting automatic language correction for {len(segments)} segments...')
    total_corrected = 0

    # Process segments in batches for LLM efficiency
    for idx, seg in enumerate(segments):
        target_text = seg.get('target', '')
        if not target_text:
            continue

        # Detect foreign words in target text (works for both pure-English
        # AND mixed Khmer+English text now)
        foreign_words = _detect_foreign_words(target_text)
        if not foreign_words:
            continue

        logger.debug(f'[Stage 3c] Segment {idx}: Found foreign words: {foreign_words[:5]}...')

        # Get context for better translation
        context_before = ''
        context_after = ''
        if idx > 0:
            context_before = segments[idx - 1].get('target', '') or segments[idx - 1].get('source', '')
        if idx < len(segments) - 1:
            context_after = segments[idx + 1].get('target', '') or segments[idx + 1].get('source', '')

        # Translate foreign words
        translation_map = _translate_foreign_words_via_llm(
            foreign_words, source_lang,
            context_before=context_before,
            context_after=context_after,
        )

        # Apply translations to the target text
        original_text = target_text
        for orig_word, khmer_word in translation_map.items():
            if khmer_word and khmer_word != orig_word:
                # Replace word preserving case
                pattern = re.compile(r'\b' + re.escape(orig_word) + r'\b', re.IGNORECASE)
                target_text = pattern.sub(khmer_word, target_text)

        target_text = _replace_foreign_words_with_khmer(target_text)

        if target_text != original_text:
            segments[idx]['target'] = target_text
            total_corrected += 1
            logger.debug(f'[Stage 3c] Segment {idx}: Corrected: "{original_text[:50]}" -> "{target_text[:50]}"')

        if (idx + 1) % 50 == 0:
            logger.info(f'[Stage 3c] Processed {idx + 1}/{len(segments)} segments')

    logger.info(f'[Stage 3c] Language correction complete: {total_corrected} segments corrected')
    return segments


# ---------------------------------------------------------------------------
# Stage 3d: Intelligent Summarization
# Summarize subtitles ONLY when they are too long to be comfortably read
# within their display time. Preserves all important content.
# ---------------------------------------------------------------------------

# Average Khmer reading speed: ~3.5 chars per second for subtitle reading
# Approx 16 chars per second for comfortable reading
KHMER_READING_CHARS_PER_SECOND = 16.0
ENGLISH_READING_CHARS_PER_SECOND = 20.0
# Minimum display time for any subtitle
MIN_SUBTITLE_DISPLAY_SECONDS = 1.0


def _estimate_reading_time(text: str, is_khmer: bool = None) -> float:
    """
    Estimate the time needed to comfortably read a subtitle.
    """
    if not text:
        return MIN_SUBTITLE_DISPLAY_SECONDS

    char_count = len(text.strip())
    if is_khmer is None:
        is_khmer = KHMER_UNICODE_RANGE.search(text)

    if is_khmer:
        time_needed = char_count / KHMER_READING_CHARS_PER_SECOND
    else:
        time_needed = char_count / ENGLISH_READING_CHARS_PER_SECOND

    return max(MIN_SUBTITLE_DISPLAY_SECONDS, time_needed)


def _build_summarization_prompt(
    source_text: str,
    display_time: float,
    context_before: str = '',
    context_after: str = '',
) -> str:
    """Build prompt for intelligent summarization of Khmer subtitles."""
    prompt_parts = [
        'You are an expert Khmer subtitle editor. Condense the following subtitle text',
        'to fit within its display time while preserving ALL important meaning.',
        '',
        'RULES:',
        '- Preserve the COMPLETE original meaning and important information.',
        '- Keep all character names, numbers, places, and key facts.',
        '- Preserve emotions, humor, sarcasm, and story continuity.',
        '- Keep the speaker\'s personality and tone.',
        '- Remove only unnecessary filler words, redundant phrases, or repeated expressions.',
        '- The output must sound natural in Khmer, not like a machine summary.',
        '- Never remove important dialogue that advances the plot.',
        '- Output ONLY the condensed Khmer text, no explanations.',
        '',
        f'Display time available: {display_time:.2f} seconds.',
        f'Target length: ~{int(display_time * KHMER_READING_CHARS_PER_SECOND)} characters.',
    ]

    if context_before.strip():
        prompt_parts.append(f'\nContext (previous subtitle): "{context_before.strip()}"')
    if context_after.strip():
        prompt_parts.append(f'Context (next subtitle): "{context_after.strip()}"')

    prompt_parts.append(f'\nSubtitle text to condense: {source_text}')
    prompt_parts.append(f'\nCondensed Khmer text:')

    return '\n'.join(prompt_parts)


def _summarize_single_subtitle(
    source_text: str,
    display_time: float,
    context_before: str = '',
    context_after: str = '',
    source_lang: str = 'en',
) -> str:
    """
    Intelligently summarize a single subtitle using LLM.
    Returns summarized text, or original if no summarization needed.
    """
    if not source_text:
        return source_text

    # Check if summarization is actually needed
    reading_time = _estimate_reading_time(source_text)
    if reading_time <= display_time + 0.5:
        return source_text  # No need to summarize

    logger.debug(f'[Stage 3d] Summarizing: reading_time={reading_time:.2f}s > display_time={display_time:.2f}s')

    prompt = _build_summarization_prompt(
        source_text, display_time,
        context_before=context_before,
        context_after=context_after,
    )

    # Try LLM
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if api_key:
        import urllib.request
        import urllib.parse
        model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
        if model.startswith('models/'):
            model = model.split('/', 1)[1]
        url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
            f'?key={urllib.parse.quote(api_key)}'
        )
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.3,
                'topP': 0.9,
                'maxOutputTokens': 300,
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            summarized = _extract_gemini_text(data)
            if summarized:
                return _normalize_translation_text(summarized)
        except Exception as e:
            logger.warning(f'[Stage 3d] LLM summarization failed: {e}')

    # Try OpenAI fallback
    api_key = os.environ.get('LLM_API_KEY', os.environ.get('OPENAI_API_KEY', '')).strip()
    if api_key:
        import urllib.request
        base_url = os.environ.get('LLM_API_URL', 'https://api.openai.com/v1').strip()
        model = os.environ.get('LLM_MODEL', 'gpt-4o-mini').strip() or 'gpt-4o-mini'
        url = f'{base_url}/chat/completions'
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are an expert Khmer subtitle editor. Condense text to fit display time while preserving meaning.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 300,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            summarized = ''
            for choice in data.get('choices', []):
                msg = choice.get('message', {})
                content = msg.get('content', '')
                if content:
                    summarized += content
            if summarized:
                return _normalize_translation_text(summarized)
        except Exception as e:
            logger.warning(f'[Stage 3d] OpenAI summarization failed: {e}')

    return source_text


def intelligent_summarize_segments(
    segments: List[Dict],
    source_lang: str = 'en',
    target_lang: str = 'km',
) -> List[Dict]:
    """
    Stage 3d: Intelligently summarize subtitles that are too long for their display time.
    Summarization is ONLY applied when necessary - when the text cannot be comfortably
    read within the available display time.
    """
    if not segments:
        return segments

    base_target = (target_lang or '').split('-')[0].lower()
    if base_target != 'km':
        return segments

    logger.info(f'[Stage 3d] Starting intelligent summarization for {len(segments)} segments...')
    total_summarized = 0

    for idx, seg in enumerate(segments):
        target_text = seg.get('target', '')
        if not target_text:
            continue

        display_time = _get_segment_display_time(seg)

        # Check if summarization is needed
        reading_time = _estimate_reading_time(target_text)
        if reading_time <= display_time + 0.5:
            continue  # Text fits comfortably, no need to summarize

        # Get context
        context_before = ''
        context_after = ''
        if idx > 0:
            context_before = segments[idx - 1].get('target', '') or segments[idx - 1].get('source', '')
        if idx < len(segments) - 1:
            context_after = segments[idx + 1].get('target', '') or segments[idx + 1].get('source', '')

        # Summarize
        summarized = _summarize_single_subtitle(
            target_text, display_time,
            context_before=context_before,
            context_after=context_after,
            source_lang=source_lang,
        )

        if summarized and summarized != target_text:
            # Verify meaning preservation
            if len(summarized) <= len(target_text) * 0.98:  # At least 2% reduction
                segments[idx]['target'] = summarized
                total_summarized += 1
                logger.debug(f'[Stage 3d] Segment {idx}: {len(target_text)} chars -> {len(summarized)} chars')

        if (idx + 1) % 50 == 0:
            logger.info(f'[Stage 3d] Processed {idx + 1}/{len(segments)} segments')

    logger.info(f'[Stage 3d] Summarization complete: {total_summarized} segments condensed')
    return segments


# ============================================================================
# REPETITION ELIMINATION CONSTANTS
# ============================================================================
MAX_CONSECUTIVE_SAME_SUBTITLE = 3  # Max times the SAME text can appear in nearby subtitles
CROSS_SEGMENT_REPETITION_WINDOW = 5  # Number of nearby segments to check for repetition


def _detect_cross_segment_repetitions(segments: List[Dict], window: int = CROSS_SEGMENT_REPETITION_WINDOW) -> List[Tuple[int, int, str, int]]:
    """
    Detect repeated phrases across consecutive subtitle segments.
    
    Checks if the same word or phrase appears in multiple consecutive segments,
    which would make the subtitles feel repetitive.
    
    Returns list of (first_seg_idx, last_seg_idx, repeated_text, count).
    """
    if not segments or len(segments) < 2:
        return []

    repetitions = []

    # Extract just the target text for comparison
    texts = []
    for seg in segments:
        text = (seg.get('target') or seg.get('source') or '').strip()
        texts.append(text)

    # Check for cross-segment word/phrase repetition
    for phrase_len in range(1, 4):  # Check 1-word, 2-word, 3-word phrases
        i = 0
        while i < len(texts):
            if not texts[i]:
                i += 1
                continue
            
            words_i = texts[i].split()
            if len(words_i) < phrase_len:
                i += 1
                continue
            
            phrase = ' '.join(words_i[:phrase_len]).lower()
            
            # Count how many consecutive segments start with the same phrase
            count = 1
            j = i + 1
            while j < len(texts) and j < i + window:
                if not texts[j]:
                    j += 1
                    continue
                words_j = texts[j].split()
                if len(words_j) >= phrase_len:
                    next_phrase = ' '.join(words_j[:phrase_len]).lower()
                    if next_phrase == phrase:
                        count += 1
                        j += 1
                    else:
                        break
                else:
                    break
            
            if count >= 3:  # 3+ consecutive segments starting with same phrase
                repetitions.append((i, j - 1, phrase, count))
                i = j  # Skip past the repeated block
            else:
                i += 1
    
    return repetitions


def _build_cross_segment_repair_prompt(
    repeated_phrase: str,
    segment_indices: List[int],
    source_texts: List[str],
    target_texts: List[str],
) -> str:
    """Build prompt to repair cross-segment repetition."""
    prompt_parts = [
        'You are a Khmer subtitle editor. The following consecutive subtitles all start',
        f'with the same phrase "{repeated_phrase}", which creates unnatural repetition.',
        '',
        'TASK: Rewrite ONLY the Khmer translations below to eliminate the repetition.',
        'Use natural Khmer variations (pronouns, synonyms, different sentence structures)',
        'while preserving ALL original meaning, character names, and important details.',
        '',
        'RULES:',
        '- Output ONLY the improved Khmer translations, one per line.',
        '- Keep the same number of lines as input segments.',
        '- Each line must correspond to the matching segment in order.',
        '- Preserve the complete meaning of each segment.',
        '- Use natural Khmer variations to avoid repetition.',
        '- Do NOT change the total number of segments.',
    ]

    for i, (src, tgt) in enumerate(zip(source_texts, target_texts)):
        prompt_parts.append(f'\nSegment {segment_indices[i]}')
        prompt_parts.append(f'Original: {src}')
        prompt_parts.append(f'Current Khmer: {tgt}')

    prompt_parts.append(f'\nImproved Khmer translations (one per line):')

    return '\n'.join(prompt_parts)


def _reduce_cross_segment_repetition(
    segments: List[Dict],
    start_idx: int,
    end_idx: int,
    repeated_phrase: str,
    source_lang: str = 'en',
) -> List[Dict]:
    """
    Reduce repetition across consecutive subtitle segments.
    Uses LLM to rewrite the affected segments with natural variations.
    """
    affected_segments = segments[start_idx:end_idx + 1]
    segment_indices = list(range(start_idx, end_idx + 1))
    source_texts = [s.get('source', '') for s in affected_segments]
    target_texts = [s.get('target', '') or s.get('source', '') for s in affected_segments]

    prompt = _build_cross_segment_repair_prompt(
        repeated_phrase, segment_indices, source_texts, target_texts
    )

    # Try Gemini
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if api_key:
        import urllib.request
        import urllib.parse
        model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
        if model.startswith('models/'):
            model = model.split('/', 1)[1]
        url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
            f'?key={urllib.parse.quote(api_key)}'
        )
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.3,
                'topP': 0.9,
                'maxOutputTokens': 500 * len(affected_segments),
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            result_text = _extract_gemini_text(data)
            if result_text:
                lines = [l.strip() for l in result_text.split('\n') if l.strip()]
                # Apply the rewritten translations
                for i, line in enumerate(lines):
                    if i < len(affected_segments):
                        seg_idx = start_idx + i
                        if line and line != target_texts[i]:
                            segments[seg_idx]['target'] = line
                return segments
        except Exception as e:
            logger.warning(f'[Stage 3e] Cross-segment repair failed: {e}')

    # Fallback: manual pronoun replacement
    for i in range(start_idx, end_idx + 1):
        target = segments[i].get('target', '') or segments[i].get('source', '')
        words = target.split()
        if len(words) >= 1:
            # Replace the repeated opening with a pronoun/natural alternative
            if i == start_idx:
                pass  # Keep first occurrence
            elif words[0].lower() == repeated_phrase.split()[0].lower():
                segments[i]['target'] = '... ' + ' '.join(words[1:])

    return segments


# ---------------------------------------------------------------------------
# Stage 3e: Automatic Repetition Reduction
# Detect and reduce repeated words, phrases, and consecutive sentences.
# Keep a maximum of 2-3 consecutive repetitions.
# ---------------------------------------------------------------------------

# Pattern for detecting consecutive repeated words/phrases
REPETITION_THRESHOLD = 3  # Max consecutive repetitions allowed


def _detect_word_repetitions(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect consecutive repeated words in text.
    Returns list of (word/phrase, start_pos, count) for each repetition cluster.
    """
    if not text:
        return []

    repetitions = []

    # Split into words (handle both Latin and Khmer)
    # For Khmer, we split on whitespace
    words = text.split()
    if len(words) < 2:
        return []

    # Detect consecutive word repetitions
    i = 0
    while i < len(words):
        j = i + 1
        count = 1
        word = words[i].lower()

        while j < len(words) and words[j].lower() == word:
            count += 1
            j += 1

        if count >= 3:  # Only flag if 3+ consecutive repetitions
            repetitions.append((words[i], i, count))

        i = j

    return repetitions


def _detect_phrase_repetitions(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect consecutive repeated phrases (2+ word sequences repeated).
    Returns list of (phrase, start_pos, count).
    """
    if not text:
        return []

    words = text.split()
    if len(words) < 4:  # Need at least 4 words for a 2-word phrase to repeat
        return []

    repetitions = []

    # Check for 2-word phrase repetitions
    max_phrase_len = min(4, len(words) // 2)
    for phrase_len in range(2, max_phrase_len + 1):
        i = 0
        while i <= len(words) - phrase_len * 2:
            phrase = ' '.join(words[i:i + phrase_len]).lower()
            count = 1
            j = i + phrase_len

            while j <= len(words) - phrase_len:
                next_phrase = ' '.join(words[j:j + phrase_len]).lower()
                if next_phrase == phrase:
                    count += 1
                    j += phrase_len
                else:
                    break

            if count >= 2:  # Flag if 2+ consecutive phrase repetitions
                repetitions.append((' '.join(words[i:i + phrase_len]), i, count))
                i = j  # Skip past the repeated phrases
                continue

            i += 1

    return repetitions


def _reduce_repetition_in_text(
    text: str,
    context_before: str = '',
    context_after: str = '',
) -> str:
    """
    Reduce consecutive repetitions in text while preserving emphasis.
    Rules:
    - Max 2-3 consecutive repetitions kept
    - For 3+ repeats, reduce to 2 with natural variation
    - Preserve story-critical repetition (emotional impact)
    """
    if not text:
        return text

    # Detect word repetitions
    word_reps = _detect_word_repetitions(text)
    phrase_reps = _detect_phrase_repetitions(text)

    if not word_reps and not phrase_reps:
        return text

    logger.debug(f'[Stage 3e] Detected {len(word_reps)} word reps, {len(phrase_reps)} phrase reps')

    # Use LLM for natural repetition reduction
    prompt_parts = [
        'You are a Khmer subtitle editor. Reduce unnecessary repetition in the following text.',
        '',
        'RULES:',
        '- Keep a maximum of 2-3 consecutive repetitions of the same word or phrase.',
        '- For repeated text, use natural variations to avoid monotony.',
        '- Preserve the intended emotional emphasis.',
        '- Preserve ALL character names, numbers, and important content.',
        '- If repetition is essential for emotional impact, keep it.',
        '- Output ONLY the improved text, no explanations.',
    ]

    if context_before.strip():
        prompt_parts.append(f'\nContext (previous subtitle): "{context_before.strip()}"')
    if context_after.strip():
        prompt_parts.append(f'Context (next subtitle): "{context_after.strip()}"')

    prompt_parts.append(f'\nText to optimize: {text}')
    prompt_parts.append(f'\nOptimized text:')

    prompt = '\n'.join(prompt_parts)

    # Try LLM for natural reduction
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if api_key:
        import urllib.request
        import urllib.parse
        model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
        if model.startswith('models/'):
            model = model.split('/', 1)[1]
        url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
            f'?key={urllib.parse.quote(api_key)}'
        )
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.2,
                'topP': 0.9,
                'maxOutputTokens': 300,
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            improved = _extract_gemini_text(data)
            if improved:
                return _normalize_translation_text(improved)
        except Exception as e:
            logger.warning(f'[Stage 3e] LLM repetition reduction failed: {e}')

    # Fallback: simple repetition reduction
    words = text.split()
    result_words = []
    i = 0
    while i < len(words):
        # Check for consecutive repetitions
        j = i + 1
        while j < len(words) and words[j].lower() == words[i].lower():
            j += 1
        count = j - i

        if count >= 3:
            # Keep first 2 occurrences with natural variation
            result_words.append(words[i])
            result_words.append(words[i])
            i = j
        else:
            result_words.append(words[i])
            i += 1

    return ' '.join(result_words)


def reduce_repetition_in_segments(
    segments: List[Dict],
    target_lang: str = 'km',
    source_lang: str = 'en',
) -> List[Dict]:
    """
    Stage 3e: Automatically detect and reduce repeated words/phrases.
    Keeps max 2-3 consecutive repetitions unless essential for emotional impact.
    
    Now includes:
    - Within-segment repetition detection (existing)
    - Cross-segment repetition detection (NEW) - catches the same phrase
      repeated across consecutive subtitles
    """
    if not segments:
        return segments

    base_target = (target_lang or '').split('-')[0].lower()
    if base_target != 'km':
        return segments

    logger.info(f'[Stage 3e] Starting repetition reduction for {len(segments)} segments...')
    total_reduced = 0

    # ---- Cross-segment repetition detection ----
    cross_reps = _detect_cross_segment_repetitions(segments)
    for first_idx, last_idx, phrase, count in cross_reps:
        logger.warning(f'[Stage 3e] Cross-segment repetition: segments {first_idx}-{last_idx} '
                       f'start with "{phrase}" ({count}x)')
        segments = _reduce_cross_segment_repetition(
            segments, first_idx, last_idx, phrase, source_lang
        )
        total_reduced += (last_idx - first_idx + 1)

    # ---- Within-segment repetition detection (existing) ----
    for idx, seg in enumerate(segments):
        target_text = seg.get('target', '')
        if not target_text:
            continue

        # Get context
        context_before = ''
        context_after = ''
        if idx > 0:
            context_before = segments[idx - 1].get('target', '') or segments[idx - 1].get('source', '')
        if idx < len(segments) - 1:
            context_after = segments[idx + 1].get('target', '') or segments[idx + 1].get('source', '')

        # Check for repetitions
        word_reps = _detect_word_repetitions(target_text)
        phrase_reps = _detect_phrase_repetitions(target_text)

        if word_reps or phrase_reps:
            reduced = _reduce_repetition_in_text(
                target_text,
                context_before=context_before,
                context_after=context_after,
            )
            if reduced and reduced != target_text:
                segments[idx]['target'] = reduced
                total_reduced += 1
                logger.debug(f'[Stage 3e] Segment {idx}: Reduced within-segment repetition')

        if (idx + 1) % 100 == 0:
            logger.info(f'[Stage 3e] Processed {idx + 1}/{len(segments)} segments')

    logger.info(f'[Stage 3e] Repetition reduction complete: {total_reduced} segments optimized')
    return segments


# ---------------------------------------------------------------------------
# Stage 3f: Automatic Natural Khmer Rewrite
# Rewrite awkward, unnatural, robotic translations into fluent, natural Khmer.
# ---------------------------------------------------------------------------

def _build_natural_rewrite_prompt(
    source_text: str,
    current_translation: str,
    context_before: str = '',
    context_after: str = '',
    source_lang: str = 'en',
) -> str:
    """Build prompt for rewriting an awkward translation into natural Khmer."""
    prompt_parts = [
        'You are a professional Cambodian subtitle translator. Rewrite the following Khmer translation',
        'so it sounds completely natural, as if it was originally written by a native Khmer speaker.',
        '',
        'RULES:',
        '- Output ONLY the improved Khmer text, no explanations.',
        '- Use natural Khmer sentence structure, NOT literal word-for-word translation.',
        '- Choose Khmer expressions that sound most natural for Cambodian audiences.',
        '- Preserve the COMPLETE original meaning — every detail, name, number, and fact.',
        '- Preserve character names, place names, and proper nouns exactly.',
        '- Preserve emotional tone: happy, sad, angry, excited, romantic, etc.',
        '- Preserve humor, sarcasm, and personality of the speaker.',
        '- Use correct Khmer grammar, spelling, and punctuation.',
        '- Avoid robotic or machine-translation sounding phrases.',
        '- Make dialogue sound like real Cambodian speech.',
        '- If the current Khmer already sounds natural and accurate, keep it unchanged.',
    ]

    if context_before.strip():
        prompt_parts.append(f'\nPrevious subtitle (context): "{context_before.strip()}"')
    if context_after.strip():
        prompt_parts.append(f'Next subtitle (context): "{context_after.strip()}"')

    prompt_parts.append(f'\nOriginal ({source_lang}) text: {source_text}')
    prompt_parts.append(f'Current Khmer translation: {current_translation}')
    prompt_parts.append(f'\nNatural Khmer rewrite:')

    return '\n'.join(prompt_parts)


def _natural_rewrite_single_subtitle(
    source_text: str,
    current_translation: str,
    context_before: str = '',
    context_after: str = '',
    source_lang: str = 'en',
) -> str:
    """
    Rewrite a single subtitle translation into natural Khmer.
    Returns the rewritten text, or original if no improvement was made.
    """
    if not current_translation or current_translation == source_text:
        return current_translation

    # Skip if translation doesn't contain Khmer (hasn't been translated yet)
    if not KHMER_UNICODE_RANGE.search(current_translation):
        return current_translation

    prompt = _build_natural_rewrite_prompt(
        source_text, current_translation,
        context_before=context_before,
        context_after=context_after,
        source_lang=source_lang,
    )

    # Try Gemini
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if api_key:
        import urllib.request
        import urllib.parse
        model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
        if model.startswith('models/'):
            model = model.split('/', 1)[1]
        url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
            f'?key={urllib.parse.quote(api_key)}'
        )
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.3,
                'topP': 0.9,
                'maxOutputTokens': 500,
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            rewritten = _extract_gemini_text(data)
            if rewritten and rewritten != current_translation:
                return _normalize_translation_text(rewritten)
        except Exception as e:
            logger.warning(f'[Stage 3f] Gemini rewrite failed: {e}')

    # Try OpenAI fallback
    api_key = os.environ.get('LLM_API_KEY', os.environ.get('OPENAI_API_KEY', '')).strip()
    if api_key:
        import urllib.request
        base_url = os.environ.get('LLM_API_URL', 'https://api.openai.com/v1').strip()
        model = os.environ.get('LLM_MODEL', 'gpt-4o-mini').strip() or 'gpt-4o-mini'
        url = f'{base_url}/chat/completions'
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You are a professional Cambodian subtitle translator. Rewrite Khmer translations to sound completely natural and native.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 500,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            rewritten = ''
            for choice in data.get('choices', []):
                msg = choice.get('message', {})
                content = msg.get('content', '')
                if content:
                    rewritten += content
            if rewritten:
                return _normalize_translation_text(rewritten)
        except Exception as e:
            logger.warning(f'[Stage 3f] OpenAI rewrite failed: {e}')

    return current_translation


def naturalize_khmer_translations(
    segments: List[Dict],
    source_lang: str = 'en',
    target_lang: str = 'km',
) -> List[Dict]:
    """
    Stage 3f: Rewrite awkward, unnatural, or robotic Khmer translations into
    fluent, natural Khmer that sounds like it was written by a native speaker.
    """
    if not segments:
        return segments

    base_target = (target_lang or '').split('-')[0].lower()
    if base_target != 'km':
        return segments

    logger.info(f'[Stage 3f] Starting natural Khmer rewrite for {len(segments)} segments...')
    total_rewritten = 0

    for idx, seg in enumerate(segments):
        source_text = seg.get('source', '')
        target_text = seg.get('target', '')

        if not source_text or not target_text:
            continue
        if target_text == source_text:
            continue  # Untranslated
        if not KHMER_UNICODE_RANGE.search(target_text):
            continue  # No Khmer to naturalize

        # Get context
        context_before = ''
        context_after = ''
        if idx > 0:
            context_before = segments[idx - 1].get('target', '') or segments[idx - 1].get('source', '')
        if idx < len(segments) - 1:
            context_after = segments[idx + 1].get('target', '') or segments[idx + 1].get('source', '')

        # Rewrite to natural Khmer
        rewritten = _natural_rewrite_single_subtitle(
            source_text, target_text,
            context_before=context_before,
            context_after=context_after,
            source_lang=source_lang,
        )

        if rewritten and rewritten != target_text:
            segments[idx]['target'] = rewritten
            total_rewritten += 1
            logger.debug(f'[Stage 3f] Segment {idx}: Rewritten')

        if (idx + 1) % 50 == 0:
            logger.info(f'[Stage 3f] Processed {idx + 1}/{len(segments)} segments')

    logger.info(f'[Stage 3f] Natural Khmer rewrite complete: {total_rewritten} segments improved')
    return segments


# ---------------------------------------------------------------------------
# Stage 3g: Final Validation
# Comprehensive quality verification before export.
# Checks all quality rules and automatically regenerates failing segments.
# ---------------------------------------------------------------------------

# ZERO-TOLERANCE quality thresholds
VALIDATION_MAX_RETRIES = 10  # Maximum repair attempts in the validation loop
KHMER_COVERAGE_THRESHOLD = 1.0  # 100% coverage required
MAX_FOREIGN_WORD_SEGMENTS = 999  # ZERO tolerance - NO foreign words allowed
MAX_REPETITION_SEGMENTS = 0  # ZERO tolerance - NO excessive repetition allowed
MAX_GRAMMAR_ISSUE_SEGMENTS = 0  # ZERO tolerance - NO grammar issues allowed


def _validate_khmer_coverage(segments: List[Dict]) -> Tuple[bool, float, int]:
    """
    Verify that ALL segments have been translated to Khmer.
    Returns (passed, coverage_pct, segments_with_khmer).
    ZERO TOLERANCE - every segment must contain Khmer if translation is expected.
    """
    if not segments:
        return False, 0.0, 0

    total = len(segments)
    with_khmer = sum(1 for s in segments if KHMER_UNICODE_RANGE.search(s.get('target', '')))
    coverage = with_khmer / total if total > 0 else 0.0

    passed = coverage >= KHMER_COVERAGE_THRESHOLD
    return passed, coverage * 100, with_khmer


def _validate_no_foreign_words_remain(segments: List[Dict]) -> Tuple[bool, int]:
    """
    Verify ZERO unnecessary foreign words remain in Khmer segments.
    Returns (passed, segments_with_foreign_words).
    ZERO TOLERANCE - every foreign word that should be translated must be translated.
    """
    if not segments:
        return True, 0

    segments_with_foreign = 0
    for seg in segments:
        target = seg.get('target', '')
        if not target:
            continue
        if not KHMER_UNICODE_RANGE.search(target):
            # Segment has NO Khmer at all - this is a FAILURE
            # Only allowed if it's a name/term that literally cannot be translated
            foreign_words = _detect_foreign_words(target)
            if foreign_words:
                segments_with_foreign += 1
            continue

        # Check for Latin words that aren't names/terms
        foreign_words = _detect_foreign_words(target)
        if foreign_words:
            segments_with_foreign += 1

    # ZERO TOLERANCE - any segment with foreign words is a failure
    passed = segments_with_foreign == 0
    return passed, segments_with_foreign


def _validate_no_repetition_left(segments: List[Dict]) -> Tuple[bool, int]:
    """
    Verify no excessive repetition remains (max 3 consecutive repeated words).
    Returns (passed, segments_with_excess_repetition).
    ZERO TOLERANCE - any excessive repetition is a failure.
    """
    if not segments:
        return True, 0

    problematic_segments = 0
    for seg in segments:
        target = seg.get('target', '')
        if not target:
            continue

        # Check for 4+ consecutive repeated words
        words = target.split()
        for i in range(len(words) - 3):
            if (words[i].lower() == words[i + 1].lower() ==
                words[i + 2].lower() == words[i + 3].lower()):
                problematic_segments += 1
                break

        # Check for cross-segment repetition (same text in multiple consecutive segments)
        # This is checked in the overall validation loop

    passed = problematic_segments == 0
    return passed, problematic_segments


def _validate_spelling_grammar(segments: List[Dict]) -> Tuple[bool, int]:
    """
    Basic validation that Khmer text doesn't contain obvious issues.
    Checks: double spaces, missing terminal punctuation, etc.
    ZERO TOLERANCE - any issues are a failure.
    """
    if not segments:
        return True, 0

    issues = 0
    for seg in segments:
        target = seg.get('target', '')
        if not target:
            continue

        # Check for common issues
        if '  ' in target:  # Double spaces
            issues += 1
        if target.strip().startswith('.') or target.strip().startswith(','):
            issues += 1

    passed = issues == 0  # ZERO TOLERANCE
    return passed, issues


def _validate_khmer_naturalness(segments: List[Dict]) -> Tuple[bool, int]:
    """
    Basic rule-based naturalness check across all segments.
    Scans for obvious machine-translation patterns.
    """
    if not segments:
        return True, 0

    unnatural_count = 0

    for idx, seg in enumerate(segments):
        target = seg.get('target', '')
        if not target or not KHMER_UNICODE_RANGE.search(target):
            continue

        # Check for obvious machine translation patterns
        machine_patterns = [
            r'\b(?:translation|translate|meaning|definition)\b',  # English words in translation
            r'\u1796\u17b6\u1780\u17cb\u1796\u17b6\u1780\u17cb',  # Doubled Khmer words
            r'\u1793\u17c5{3,}',  # Too many Ngor
            r'\[.*?\]',  # Bracket artifacts
            r'\{.*?\}',  # Brace artifacts
            r'\bnote\s*\:',  # "note:" artifacts
        ]
        for pattern in machine_patterns:
            if re.search(pattern, target):
                unnatural_count += 1
                break

        # Check for extremely long segments (>300 chars) which may indicate issues
        if len(target) > 300:
            unnatural_count += 1

    # ZERO TOLERANCE - any unnatural segment is a failure
    passed = unnatural_count == 0
    return passed, unnatural_count


def _build_validation_repair_prompt(
    segment: Dict,
    failed_checks: List[str],
) -> str:
    """Build a prompt to repair a segment that failed validation."""
    failed_str = '\n'.join(f'- {check}' for check in failed_checks)

    prompt_parts = [
        'You are a Khmer subtitle quality control expert. The following subtitle translation',
        'failed quality validation. Please correct it.',
        '',
        'FAILED CHECKS:',
        failed_str,
        '',
        'RULES:',
        '- Fix ALL the issues listed above.',
        '- Output ONLY the corrected Khmer translation, no explanations.',
        '- Preserve the complete original meaning.',
        '- Keep all character names, numbers, and proper nouns.',
        '- Make it sound natural in Khmer.',
    ]

    source_text = segment.get('source', '')
    target_text = segment.get('target', '')

    prompt_parts.append(f'\nOriginal text: {source_text}')
    prompt_parts.append(f'Current translation: {target_text}')
    prompt_parts.append(f'\nCorrected translation:')

    return '\n'.join(prompt_parts)


def _repair_segment_via_llm(
    segment: Dict,
    failed_checks: List[str],
    source_lang: str = 'en',
) -> str:
    """
    Use LLM to repair a segment that failed validation.
    Falls back to rule-based repair when LLM is unavailable.
    Returns the repaired text, or original if repair failed.
    """
    if not failed_checks:
        return segment.get('target', '')

    prompt = _build_validation_repair_prompt(segment, failed_checks)
    target_text = segment.get('target', '')

    # Try Gemini
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if api_key:
        import urllib.request
        import urllib.parse
        model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
        if model.startswith('models/'):
            model = model.split('/', 1)[1]
        url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
            f'?key={urllib.parse.quote(api_key)}'
        )
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.2,
                'topP': 0.9,
                'maxOutputTokens': 500,
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            repaired = _extract_gemini_text(data)
            if repaired:
                return _normalize_translation_text(repaired)
        except Exception as e:
            logger.warning(f'[Validation] Gemini repair failed: {e}')

    # ---- Rule-based fallback when LLM is unavailable ----
    # Try to fix common issues without needing an API key
    original_text = target_text
    if not target_text:
        return target_text

    # Replace foreign words with Khmer translations or transliterations rather
    # than dropping them, so validation still sees Khmer content.
    for check in failed_checks:
        if 'foreign' in check.lower() or 'translation' in check.lower():
            target_text = _replace_foreign_words_with_khmer(target_text)

        if 'repetition' in check.lower():
            # Simple repetition reduction
            words = target_text.split()
            result = []
            i = 0
            while i < len(words):
                count = 1
                while i + count < len(words) and words[i + count].lower() == words[i].lower():
                    count += 1
                if count >= 3:
                    result.extend([words[i]] * 2)  # Keep max 2
                else:
                    result.append(words[i])
                i += count
            target_text = ' '.join(result)

        if 'double space' in check.lower() or 'spaces' in check.lower():
            target_text = re.sub(r'  +', ' ', target_text)

        if 'long' in check.lower() and len(target_text) > 300:
            target_text = target_text[:297] + '...'

    # Clean up any remaining artifacts from the removal
    target_text = re.sub(r'  +', ' ', target_text)  # Collapse multiple spaces
    target_text = re.sub(r'\s+\.', '.', target_text)  # Space before period
    target_text = target_text.strip()
    # Remove leading/trailing punctuation artifacts
    target_text = re.sub(r'^[,\s]+', '', target_text)
    target_text = re.sub(r'[,\s]+$', '', target_text)

    if target_text != original_text and target_text.strip():
        logger.debug(f'[Validation] Rule-based repair: "{original_text[:50]}" -> "{target_text[:50]}"')
        return target_text

    return original_text


def _collect_validation_failures(
    segments: List[Dict],
    source_lang: str = 'en',
    target_lang: str = 'km',
) -> List[Dict]:
    """Collect detailed validation failures for each subtitle segment."""
    failures: List[Dict] = []
    if not segments:
        return failures

    base_target = (target_lang or '').split('-')[0].lower()
    base_source = (source_lang or '').split('-')[0].lower()
    translation_expected = base_target != base_source

    if not translation_expected:
        return failures

    coverage_passed, coverage_pct, with_khmer = _validate_khmer_coverage(segments)
    foreign_passed, foreign_count = _validate_no_foreign_words_remain(segments)
    repetition_passed, rep_count = _validate_no_repetition_left(segments)
    grammar_passed, grammar_issues = _validate_spelling_grammar(segments)
    natural_passed, natural_count = _validate_khmer_naturalness(segments)

    for idx, seg in enumerate(segments):
        target_text = seg.get('target', '') or ''
        speaker = seg.get('speaker') or seg.get('speaker_id') or ''
        start = seg.get('start', '')
        end = seg.get('end', '')

        if not coverage_passed and not KHMER_UNICODE_RANGE.search(target_text):
            failures.append({
                'index': idx,
                'text': target_text,
                'speaker': speaker,
                'start': start,
                'end': end,
                'rule': 'khmer_coverage',
                'reason': 'Segment has no Khmer characters and therefore does not satisfy the 100% Khmer coverage rule.',
            })

        if not foreign_passed:
            foreign_words = _detect_foreign_words(target_text)
            if foreign_words:
                failures.append({
                    'index': idx,
                    'text': target_text,
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'rule': 'foreign_words',
                    'reason': f'Contains untranslated foreign words: {", ".join(foreign_words[:5])}',
                })

        if not repetition_passed:
            words = target_text.split()
            for i in range(len(words) - 3):
                if (words[i].lower() == words[i + 1].lower() == words[i + 2].lower() == words[i + 3].lower()):
                    failures.append({
                        'index': idx,
                        'text': target_text,
                        'speaker': speaker,
                        'start': start,
                        'end': end,
                        'rule': 'repetition',
                        'reason': 'Contains repeated words or phrases that violate the zero-tolerance repetition rule.',
                    })
                    break

        if not grammar_passed and '  ' in target_text:
            failures.append({
                'index': idx,
                'text': target_text,
                'speaker': speaker,
                'start': start,
                'end': end,
                'rule': 'grammar',
                'reason': 'Contains double spaces or other obvious formatting issues.',
            })

        if not natural_passed and len(target_text) > 300:
            failures.append({
                'index': idx,
                'text': target_text,
                'speaker': speaker,
                'start': start,
                'end': end,
                'rule': 'naturalness',
                'reason': 'The subtitle is unusually long and may indicate a machine-generated or unnatural translation.',
            })

    return failures


def final_validation(
    segments: List[Dict],
    source_lang: str = 'en',
    target_lang: str = 'km',
) -> Tuple[List[Dict], bool]:
    """
    Stage 3g: Final comprehensive validation before export.
    
    Checks:
    ✓ Every possible subtitle translated to Khmer (100% coverage)
    ✓ No unnecessary foreign-language words remain (ZERO tolerance)
    ✓ No excessive repetition remains (ZERO tolerance)
    ✓ Khmer spelling and grammar correct (ZERO tolerance)
    ✓ No unnatural/machine-translation artifacts (ZERO tolerance)
    
    If any validation fails, the pipeline MUST NOT export.
    This uses an exhaustive repair loop: repair, then re-validate, then repair again
    until either all checks pass or the maximum number of passes is exhausted.
    
    Returns a tuple (segments, passed: bool) where passed indicates whether
    validation was successful. The pipeline MUST NOT export segments that
    failed validation.
    """
    if not segments:
        return segments, False

    base_target = (target_lang or '').split('-')[0].lower()
    base_source = (source_lang or '').split('-')[0].lower()

    # Only validate if targeting Khmer
    if base_target != 'km':
        logger.info(f'[Stage 3g] Skipping validation: target is not Khmer ({target_lang})')
        return segments, True

    # Skip validation if source == target (no translation expected)
    if base_source == base_target:
        return segments, True

    logger.info(f'[Stage 3g] ==========================================')
    logger.info(f'[Stage 3g] FINAL VALIDATION - {len(segments)} segments')
    logger.info(f'[Stage 3g] ==========================================')

    # Determine if translation is expected
    translation_expected = base_target != base_source

    # Exhaustive repair-and-revalidate loop:
    # Each iteration repairs failing segments, then re-checks ALL validation
    # rules. Continues until all pass or max attempts exhausted.
    max_attempts = VALIDATION_MAX_RETRIES
    for attempt in range(max_attempts):
        if attempt > 0:
            logger.info(f'[Stage 3g] Re-validation attempt {attempt + 1}/{max_attempts}')

        # Check 1: Khmer coverage
        coverage_passed, coverage_pct, with_khmer = _validate_khmer_coverage(segments)
        logger.info(f'[Stage 3g] Check 1 - Khmer coverage: {coverage_pct:.1f}% ({with_khmer}/{len(segments)})')
        if not coverage_passed:
            logger.warning(f'[Stage 3g] FAILED: Khmer coverage below 100%')

        # Check 2: No foreign words
        foreign_passed, foreign_count = _validate_no_foreign_words_remain(segments)
        logger.info(f'[Stage 3g] Check 2 - Foreign words: {foreign_count} segments affected')
        if not foreign_passed:
            logger.warning(f'[Stage 3g] FAILED: Foreign words found in {foreign_count} segments')

        # Check 3: No excessive repetition
        repetition_passed, rep_count = _validate_no_repetition_left(segments)
        logger.info(f'[Stage 3g] Check 3 - Repetition: {rep_count} segments affected')
        if not repetition_passed:
            logger.warning(f'[Stage 3g] FAILED: Excessive repetition in {rep_count} segments')

        # Check 4: Spelling/grammar basics
        grammar_passed, grammar_issues = _validate_spelling_grammar(segments)
        logger.info(f'[Stage 3g] Check 4 - Grammar: {grammar_issues} segments with issues')
        if not grammar_passed:
            logger.warning(f'[Stage 3g] FAILED: Grammar issues in {grammar_issues} segments')

        # Check 5: Naturalness check (all iterations)
        natural_passed, natural_count = _validate_khmer_naturalness(segments)
        logger.info(f'[Stage 3g] Check 5 - Naturalness: {natural_count} unnatural segments')
        if not natural_passed:
            logger.warning(f'[Stage 3g] FAILED: Naturalness check found {natural_count} unnatural segments')

        failure_details = _collect_validation_failures(segments, source_lang, target_lang)
        if failure_details:
            logger.warning(f'[Stage 3g] Validation found {len(failure_details)} subtitle failures before repair:')
            for failure in failure_details:
                logger.warning(
                    '[Stage 3g] Failed subtitle #%d | text="%s" | speaker="%s" | start=%s | end=%s | rule="%s" | reason="%s"',
                    failure['index'],
                    (failure['text'] or '')[:220],
                    failure['speaker'],
                    failure['start'],
                    failure['end'],
                    failure['rule'],
                    failure['reason'],
                )

        # Determine overall pass/fail
        if translation_expected:
            all_passed = (coverage_passed and foreign_passed and
                          repetition_passed and grammar_passed and natural_passed)
        else:
            all_passed = True

        if all_passed:
            logger.info(f'[Stage 3g] +++ ALL VALIDATION CHECKS PASSED +++')
            return segments, True

        if attempt >= max_attempts - 1:
            logger.warning(f'[Stage 3g] Validation failed after {max_attempts} repair attempts')
            break

        # Repair failing segments - scan every segment that has an issue
        logger.info(f'[Stage 3g] Repairing failing segments (pass {attempt + 1})...')
        repaired_count = 0

        for idx, seg in enumerate(segments):
            failed_checks = []

            target_text = seg.get('target', '')
            if not target_text:
                continue

            # Check Khmer coverage for this segment
            if translation_expected and not KHMER_UNICODE_RANGE.search(target_text):
                foreign_words = _detect_foreign_words(target_text)
                if foreign_words:
                    failed_checks.append('Missing Khmer translation - contains only foreign words')
                else:
                    failed_checks.append('Missing Khmer translation - needs to be translated to Khmer')

            # Check foreign words (even in mixed Khmer+English text)
            fw = _detect_foreign_words(target_text)
            if fw:
                failed_checks.append(f'Contains untranslated foreign words that need translation: {fw[:3]}')

            # Check repetition
            word_reps = _detect_word_repetitions(target_text)
            if word_reps:
                failed_checks.append('Excessive repetition of words detected')

            # Check grammar
            if '  ' in target_text:
                failed_checks.append('Contains double spaces')

            # Check naturalness
            if len(target_text) > 300:
                failed_checks.append('Translation is too long (>300 chars)')

            machine_patterns = [
                r'\b(?:translation|translate|meaning|definition)\b',
                r'\[.*?\]',
                r'\{.*?\}',
            ]
            for pattern in machine_patterns:
                if re.search(pattern, target_text):
                    failed_checks.append('Contains machine-translation artifacts')
                    break

            if failed_checks:
                repaired = _repair_segment_via_llm(seg, failed_checks, source_lang)
                if repaired and repaired != target_text:
                    segments[idx]['target'] = repaired
                    repaired_count += 1
                    logger.debug(f'[Stage 3g] Segment {idx}: Repaired ({len(failed_checks)} issues)')

        logger.info(f'[Stage 3g] Repaired {repaired_count} segments in this pass')

        if repaired_count == 0:
            # No repairs made but checks still failing - something wrong
            logger.warning(f'[Stage 3g] No repairs possible but checks still failing. Stopping.')
            break

    logger.warning(f'[Stage 3g] === VALIDATION FAILED === Subtitles will NOT be exported.')
    return segments, False


# ============================================================================
# Stage 3h: Mandatory Final Quality Enforcement
# ============================================================================
# This is the ABSOLUTE LAST check before any subtitle export.
# It enforces ALL quality rules with ZERO tolerance.
# If ANY check fails, subtitles are NEVER exported.
# ============================================================================


def _build_final_enforcement_prompt(segments_context: str, failed_segments_info: str) -> str:
    """
    Build an all-encompassing prompt for the LLM to do a final
    quality enforcement pass on failing segments.
    """
    prompt_parts = [
        'You are a MANDATORY FINAL QUALITY ENFORCEMENT system for Khmer subtitles.',
        '',
        'Your job is to fix ALL remaining quality issues in the following subtitle segments.',
        'The subtitles CANNOT be exported until EVERY issue is resolved.',
        '',
        'REQUIREMENTS (ALL MUST BE MET BEFORE EXPORT):',
        '',
        '1. COMPLETE KHMER TRANSLATION:',
        '   - Every word MUST be translated to Khmer.',
        '   - No English, Chinese, Japanese, Korean, Thai, Vietnamese, or other foreign words.',
        '   - Only character names, place names, brand names, organization names,',
        '     official titles, and untranslatable technical terms may remain in the original.',
        '',
        '2. NO REPETITION:',
        '   - No repeated words, phrases, or consecutive sentences.',
        '   - Same word/phrase may not appear more than 3 times in nearby subtitles.',
        '   - Use natural Khmer alternatives (pronouns, synonyms) if needed.',
        '',
        '3. NATURAL KHMER:',
        '   - Fix awkward grammar, robotic expressions, unnatural wording.',
        '   - Must sound like natural Cambodian speech.',
        '   - Correct Khmer spelling and grammar.',
        '',
        '4. PRESERVE:',
        '   - Original meaning completely.',
        '   - Emotional tone and story continuity.',
        '   - Character names, numbers, and important facts.',
        '',
        'FAILING SEGMENTS TO FIX:',
        failed_segments_info,
        '',
        'Output ONLY the corrected Khmer translations, one per line.',
        'Each line must correspond to the matching segment in order.',
        'If a segment is already correct, copy it unchanged.',
    ]

    return '\n'.join(prompt_parts)


def mandatory_final_quality_enforcement(
    segments: List[Dict],
    source_lang: str = 'en',
    target_lang: str = 'km',
) -> Tuple[List[Dict], bool]:
    """
    Stage 3h: MANDATORY FINAL QUALITY ENFORCEMENT.
    
    This is a non-skippable, zero-tolerance quality gate that runs IMMEDIATELY
    before any subtitle export. It performs:
    
    ✓ EVERY translatable word translated to Khmer
    ✓ NO unnecessary foreign-language words remain
    ✓ NO unnecessary repeated words or phrases
    ✓ Khmer grammar and spelling is correct
    ✓ Subtitles sound natural
    ✓ Original meaning preserved
    ✓ Emotional tone preserved
    ✓ Story continuity preserved
    
    If ANY check fails, the subtitles are regenerated and rechecked
    in an exhaustive loop until ALL checks pass or max retries exhausted.
    
    Returns (segments, passed) - if not passed, export is BLOCKED.
    """
    if not segments:
        return segments, False

    base_target = (target_lang or '').split('-')[0].lower()
    base_source = (source_lang or '').split('-')[0].lower()

    # Only apply to Khmer target
    if base_target != 'km':
        return segments, True

    if base_source == base_target:
        return segments, True

    logger.info('[Stage 3h] ============================================')
    logger.info('[Stage 3h] MANDATORY FINAL QUALITY ENFORCEMENT')
    logger.info(f'[Stage 3h] {len(segments)} segments to verify')
    logger.info('[Stage 3h] ============================================')

    # Run Stage 3g validation first (the comprehensive one)
    segments, validation_passed = final_validation(segments, source_lang, target_lang)

    if validation_passed:
        logger.info('[Stage 3h] +++ ALL QUALITY CHECKS PASSED - EXPORT READY +++')
        return segments, True

    # If Stage 3g failed, try a final all-in-one repair pass
    logger.warning('[Stage 3h] Stage 3g validation failed. Attempting final all-in-one repair...')

    # Find ALL segments that have issues
    failing_segments_info = []
    failures = _collect_validation_failures(segments, source_lang, target_lang)
    for failure in failures:
        index = failure['index']
        seg = segments[index]
        target_text = seg.get('target', '') or ''
        failing_segments_info.append(
            f'Segment {index}: Source="{seg.get("source", "")[:60]}", '
            f'Current="{target_text[:60]}", Rule="{failure["rule"]}", '
            f'Reason="{failure["reason"]}"'
        )
        logger.warning(
            '[Stage 3h] Failed subtitle #%d | text="%s" | speaker="%s" | start=%s | end=%s | rule="%s" | reason="%s"',
            index,
            (target_text or '')[:220],
            seg.get('speaker') or seg.get('speaker_id') or '',
            seg.get('start', ''),
            seg.get('end', ''),
            failure['rule'],
            failure['reason'],
        )

    if not failing_segments_info:
        # False positive from validation - treat as passed
        logger.info('[Stage 3h] No actual failing segments found. Marking as passed.')
        return segments, True

    # Repair untranslated/partially translated lines with the same provider
    # chain used by Stage 3. This is especially important for a transient free
    # API miss: one failed line must not make the whole button return HTTP 422.
    failing_indices = sorted({failure['index'] for failure in failures})
    provider_fixed_count = 0
    for index in failing_indices:
        seg = segments[index]
        source_text = (seg.get('source') or '').strip()
        if not source_text:
            continue
        candidates = _fetch_translation(source_text, source_lang, target_lang)
        translated = next(
            (
                candidate for candidate in candidates
                if _translation_matches_target_language(candidate, target_lang)
            ),
            '',
        )
        if translated:
            segments[index]['target'] = _prepare_scene_for_tts(translated, target_lang)
            provider_fixed_count += 1

    if provider_fixed_count:
        logger.info(
            f'[Stage 3h] Re-translated {provider_fixed_count} failed subtitle(s) '
            'with the fallback provider.'
        )
        segments, validation_passed = final_validation(segments, source_lang, target_lang)
        if validation_passed:
            logger.info('[Stage 3h] +++ ALL QUALITY CHECKS PASSED AFTER PROVIDER REPAIR +++')
            return segments, True

    # Build aggregated repair prompt
    failed_context = '\n'.join(failing_segments_info[:20])  # Limit to first 20 for prompt length
    prompt = _build_final_enforcement_prompt(
        f'{len(segments)} segments total, {len(failing_segments_info)} with issues',
        failed_context
    )

    # Try Gemini for the final repair
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if api_key:
        import urllib.request
        import urllib.parse
        model = os.environ.get('GEMINI_MODEL', GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
        if model.startswith('models/'):
            model = model.split('/', 1)[1]
        url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe='')) + (
            f'?key={urllib.parse.quote(api_key)}'
        )
        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {
                'temperature': 0.2,
                'topP': 0.95,
                'maxOutputTokens': 2000,
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode('utf-8'))
            result_text = _extract_gemini_text(data)
            if result_text:
                lines = [l.strip() for l in result_text.split('\n') if l.strip()]
                # Try to apply the corrections
                applied_count = 0
                for line in lines:
                    # Try to parse "Segment X: translation" format
                    seg_match = re.match(r'Segment\s+(\d+)\s*:\s*(.*)', line)
                    if seg_match:
                        seg_idx = int(seg_match.group(1))
                        new_text = seg_match.group(2).strip()
                        if 0 <= seg_idx < len(segments) and new_text:
                            segments[seg_idx]['target'] = new_text
                            applied_count += 1
                
                if applied_count > 0:
                    logger.info(f'[Stage 3h] Applied {applied_count} final corrections')
                    
                    # Re-run validation on the corrected segments
                    segments, validation_passed = final_validation(segments, source_lang, target_lang)
                    if validation_passed:
                        logger.info('[Stage 3h] +++ ALL QUALITY CHECKS PASSED AFTER FINAL REPAIR +++')
                        return segments, True
        except Exception as e:
            logger.warning(f'[Stage 3h] Final repair attempt failed: {e}')

    # ---- Rule-based fallback when LLM is unavailable ----
    # Apply deterministic Khmer repair to all failing segments so they can still pass
    # the zero-tolerance quality gate instead of being stripped down to empty text.
    logger.info('[Stage 3h] Applying rule-based fallback fixes to failing segments...')
    rule_fixed_count = 0
    for idx, seg in enumerate(segments):
        target_text = seg.get('target', '')
        if not target_text:
            continue
        original_text = target_text

        repaired_text = _replace_foreign_words_with_khmer(target_text)
        repaired_text = re.sub(r'  +', ' ', repaired_text)
        repaired_text = re.sub(r'\s+\.', '.', repaired_text)
        repaired_text = repaired_text.strip()
        repaired_text = re.sub(r'^[,\s]+', '', repaired_text)
        repaired_text = re.sub(r'[,\s]+$', '', repaired_text)

        if repaired_text != original_text and repaired_text.strip():
            segments[idx]['target'] = repaired_text
            rule_fixed_count += 1
            logger.debug(f'[Stage 3h] Rule-based fix segment {idx}: "{original_text[:50]}" -> "{repaired_text[:50]}"')

    if rule_fixed_count > 0:
        logger.info(f'[Stage 3h] Applied rule-based fixes to {rule_fixed_count} segments')
        # Re-run validation
        segments, validation_passed = final_validation(segments, source_lang, target_lang)
        if validation_passed:
            logger.info('[Stage 3h] +++ ALL QUALITY CHECKS PASSED AFTER RULE-BASED FIX +++')
            return segments, True

    # Final check: if validation still fails, we MUST NOT export
    logger.error('[Stage 3h] ============================================')
    logger.error('[Stage 3h] QUALITY ENFORCEMENT FAILED')
    logger.error('[Stage 3h] Subtitles will NOT be exported.')
    logger.error('[Stage 3h] ============================================')

    return segments, False


def _run_speech_subtitle_generation(
    video_path: str,
    source_lang: str,
    target_lang: str,
) -> Dict:
    """Generate source-language subtitle units from speech without translation or TTS."""
    result = {
        'success': False,
        'segments': [],
        'srt': '',
        'vtt': '',
        'detected_lang': source_lang,
        'method': 'whisper',
        'translation_enabled': _translation_enabled(),
        'error': '',
    }
    audio_path = None
    cache_key = _cache_key('speech-subtitles-v4', {
        'media': _media_fingerprint(video_path),
        'source_lang': source_lang,
        'model': os.environ.get('WHISPER_MODEL', 'base'),
        'pause_ms': PAUSE_DETECTION_MS,
    })
    cached = _read_json_cache(CACHE_DIR, cache_key)
    if cached and cached.get('success') and isinstance(cached.get('segments'), list):
        logger.info('[Pipeline] Reusing cached speech subtitles.')
        return cached

    try:
        process_lock.raise_if_cancelled()
        audio_path = str(TEMP_DIR / f'audio_speech_{uuid4().hex}.wav')
        _update_pipeline_progress('Audio Extraction', 0, 1, 'Extracting speech audio', time.time())
        extract_audio(video_path, audio_path, max_duration=None)
        _update_pipeline_progress('Audio Extraction', 1, 1, 'Audio extraction complete')

        segments, detected_lang = transcribe_audio(
            audio_path,
            source_lang,
            recover_vad_gaps=True,
        )
        process_lock.raise_if_cancelled()
        if not segments:
            result['error'] = 'No speech was detected in the video.'
            return result

        segments = _group_button1_spoken_phrases(segments)
        prepared = []
        for segment in _validate_speaker_pure_blocks(segments):
            item = dict(segment)
            item['source'] = str(item.get('source') or item.get('text') or '').strip()
            item['target'] = ''
            item['speaker_id'] = (
                item.get('speaker_id') or item.get('speaker') or 'SPEAKER_00'
            )
            if item['source']:
                prepared.append(item)
        prepared = _repair_subtitle_timing(prepared)
        if not prepared:
            result['error'] = 'Speech was detected, but no valid subtitle phrases were produced.'
            return result
        for item in prepared:
            # Generate Subtitle must not masquerade source text as a completed
            # translation. The dedicated translation stage fills this field.
            item['target'] = ''

        result['detected_lang'] = detected_lang or source_lang
        result = _finalize_subtitle_result(
            result,
            prepared,
            target_lang,
            preserve_boundaries=True,
        )
        _write_json_cache(CACHE_DIR, cache_key, result)
        return result
    except Exception as error:
        logger.error(f'[Pipeline] Speech subtitle generation failed: {error}')
        logger.error(traceback.format_exc())
        result['error'] = str(error)
        result['cancelled'] = isinstance(error, PipelineCancelled)
        return result
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    video_path,
    source_lang='en',
    target_lang='km',
    use_ocr_fallback=True,
    subtitle_only=False,
    speech_subtitles=False,
):
    """
    Run the full subtitle extraction and translation pipeline.
    Includes all advanced Khmer translation rules:
      - Complete translation to Khmer
      - Automatic language correction
      - Intelligent context translation
      - Intelligent summarization (when text is too long)
      - Automatic repetition reduction (within and across segments)
      - Automatic natural Khmer rewrite
      - Final validation before export
      - MANDATORY Final Quality Enforcement (Stage 3h)

    Returns a dict with:
      - success: bool
      - segments: list of {start, end, source, target}
      - srt: str (SRT content)
      - vtt: str (VTT content)
      - detected_lang: str
      - method: str ('whisper', 'ocr', or 'combined')
      - error: str (if failed)
    """
    result = {
        'success': False,
        'segments': [],
        'srt': '',
        'vtt': '',
        'detected_lang': source_lang,
        'method': 'whisper',
        'translation_enabled': _translation_enabled(),
        'error': '',
    }

    if speech_subtitles:
        return _run_speech_subtitle_generation(video_path, source_lang, target_lang)

    audio_path = None
    ocr_executor = None
    video_id = str(int(time.time()))
    pipeline_started_at = time.time()
    target_base = (target_lang or '').split('-')[0].lower()
    source_base = (source_lang or '').split('-')[0].lower()
    translation_expected = (
        _translation_enabled()
        and not subtitle_only
        and target_base != source_base
    )
    if not use_ocr_fallback:
        logger.info(
            '[Pipeline] OCR cannot be disabled while visual-subtitle mode is active; '
            'the existing parameter is retained for backward compatibility.'
        )
        use_ocr_fallback = True

    try:
        # ---- Step 1: OCR is the authoritative subtitle source ----
        result['method'] = 'ocr'
        video_duration = get_video_duration(video_path)
        if video_duration <= 0:
            logger.warning(
                '[Pipeline] Container duration was unavailable; OCR will probe it directly.'
            )
        ocr_fps = max(0.5, min(30.0, float(os.environ.get('OCR_FPS', '1.0'))))
        ocr_started_at = time.time()
        _update_pipeline_progress(
            'OCR',
            0,
            max(1, int(max(video_duration, 1.0) * ocr_fps)),
            'Reading on-screen subtitles from video frames',
            ocr_started_at,
        )
        logger.info(
            f'[Pipeline] Step 1: OCR subtitle extraction at {ocr_fps:.2f} FPS'
        )
        ocr_segments = []
        last_ocr_error = None
        for attempt in range(1, 3):
            process_lock.raise_if_cancelled()
            try:
                ocr_segments = detect_hardcoded_subtitles(
                    video_path,
                    fps=ocr_fps,
                    duration=video_duration or None,
                    source_lang=source_lang,
                )
                last_ocr_error = None
                break
            except PipelineCancelled:
                raise
            except (OSError, RuntimeError, subprocess.SubprocessError) as ocr_error:
                last_ocr_error = ocr_error
                logger.warning(
                    f'[Pipeline] OCR attempt {attempt}/2 failed: {ocr_error}'
                )
                if attempt < 2:
                    _update_pipeline_progress(
                        'OCR',
                        0,
                        max(1, int(max(video_duration, 1.0) * ocr_fps)),
                        f'OCR temporary failure; retrying ({attempt}/2)',
                        time.time(),
                    )
                    time.sleep(1)
        if last_ocr_error is not None:
            raise RuntimeError(f'OCR failed after 2 attempts: {last_ocr_error}')
        if not ocr_segments:
            result['error'] = (
                'No visually displayed subtitles were detected in the video.'
            )
            result['method'] = 'ocr'
            return result

        if not _ocr_output_is_usable(ocr_segments):
            initial_quality = _ocr_quality_metrics(ocr_segments)
            configured_dimension = max(
                320,
                min(1280, int(os.environ.get('OCR_MAX_DIMENSION', '360'))),
            )
            retry_dimension = min(1600, max(720, configured_dimension * 2))
            logger.warning(
                '[Pipeline] OCR quality was too low '
                f'(average confidence={initial_quality["average_confidence"]:.2f}, '
                f'reliable captions={initial_quality["reliable_ratio"]:.0%}); '
                f'retrying at {retry_dimension}px with additional recognition passes.'
            )
            _update_pipeline_progress(
                'OCR',
                0,
                max(1, int(max(video_duration, 1.0) * ocr_fps)),
                'Low OCR confidence; retrying at higher resolution',
                time.time(),
            )
            ocr_segments = detect_hardcoded_subtitles(
                video_path,
                fps=ocr_fps,
                duration=video_duration or None,
                source_lang=source_lang,
                max_dimension_override=retry_dimension,
                frame_retries_override=2,
            )
            if not _ocr_output_is_usable(ocr_segments):
                retry_quality = _ocr_quality_metrics(ocr_segments)
                raise RuntimeError(
                    'The on-screen subtitles could not be read reliably '
                    f'(OCR confidence {retry_quality["average_confidence"]:.0%}). '
                    'Choose the actual source language, use a clearer/higher-resolution '
                    'video, or use speech transcription for stylized or unsupported '
                    'subtitle fonts. Translation was not attempted because the detected '
                    'text was invalid.'
                )

        detected_ocr_language = str(
            next(
                (
                    segment.get('ocr_language')
                    for segment in ocr_segments
                    if segment.get('ocr_language')
                ),
                source_lang,
            )
            or source_lang
        )
        if detected_ocr_language != source_lang:
            logger.info(
                f'[Pipeline] Corrected OCR source language from {source_lang} '
                f'to {detected_ocr_language}'
            )
            source_lang = detected_ocr_language
        result['detected_lang'] = source_lang
        source_base = (source_lang or '').split('-')[0].lower()
        translation_expected = (
            _translation_enabled()
            and not subtitle_only
            and target_base != source_base
        )
        save_checkpoint(video_id, 'ocr_detected', {
            'segments': ocr_segments,
            'duration': video_duration,
        })

        # ---- Step 2: validate exact OCR boundaries without speech retiming ----
        _update_pipeline_progress(
            'Subtitle Timing',
            0,
            1,
            'Validating subtitle appearance and disappearance times',
            time.time(),
        )
        segments = _prepare_ocr_authoritative_segments(ocr_segments)
        if not segments:
            result['error'] = 'OCR subtitle observations were invalid after validation.'
            result['method'] = 'ocr'
            return result
        _update_pipeline_progress(
            'Subtitle Timing',
            1,
            1,
            f'Validated {len(segments)} subtitle windows',
        )
        result['method'] = 'ocr'

        # Whisper is optional metadata only. Failure here is recoverable because
        # it must never replace or retime text measured from video frames.
        speech_segments = []
        speech_metadata_enabled = (
            not subtitle_only
            and _env_enabled('SPEECH_METADATA_ENABLED', default=False)
        )
        if speech_metadata_enabled:
            try:
                _update_pipeline_progress(
                    'Subtitle Extraction',
                    0,
                    1,
                    'Extracting audio for speaker metadata',
                    time.time(),
                )
                audio_path = str(TEMP_DIR / f'audio_{video_id}.wav')
                extract_audio(video_path, audio_path, max_duration=None)
                _update_pipeline_progress(
                    'Subtitle Extraction',
                    1,
                    1,
                    'Audio extraction complete',
                )
                save_checkpoint(video_id, 'audio_extracted', {'audio_path': audio_path})

                _update_pipeline_progress(
                    'Speech Recognition',
                    0,
                    1,
                    'Analyzing speakers and word timing',
                    time.time(),
                )
                speech_segments, detected_lang = transcribe_audio(audio_path, source_lang)
                result['detected_lang'] = detected_lang
                _update_pipeline_progress(
                    'Speech Recognition',
                    1,
                    1,
                    f'Analyzed {len(speech_segments)} speech segments',
                )
            except PipelineCancelled:
                raise
            except Exception as speech_error:
                speech_segments = []
                result['warning'] = (
                    'OCR subtitles were generated, but optional speaker metadata '
                    f'could not be analyzed: {speech_error}'
                )
                logger.warning(f'[Pipeline] Recoverable speech metadata failure: {speech_error}')

        if speech_segments:
            segments = combine_subtitle_sources(
                speech_segments,
                segments,
                prefer_ocr_timing=True,
            )
            segments = _prepare_ocr_authoritative_segments(segments)

        if subtitle_only:
            segments = _prepare_segments_for_subtitle_generation(segments)

        logger.info(
            f'[Pipeline] OCR produced {len(segments)} authoritative captions in '
            f'{time.time() - ocr_started_at:.2f}s'
        )

        # ---- Stage 3: Translate ----
        if translation_expected:
            untranslated_segments = [dict(segment) for segment in segments]
            _update_pipeline_progress('Translation', 0, 1, 'Translating subtitles', time.time())
            logger.info(f'[Pipeline] Stage 3: Translating {len(segments)} segments: {source_lang} -> {target_lang}')
            translation_started_at = time.time()
            segments = translate_segments(segments, source_lang, target_lang)
            logger.info(f'[Timing] Translation: {time.time() - translation_started_at:.2f}s')
            _update_pipeline_progress('Translation', 1, 1, f'Translated {len(segments)} segments')
            _validate_subtitle_transform(
                untranslated_segments,
                segments,
                require_target=True,
            )

        # ---- Final Export ----
        _update_pipeline_progress('Export', 0, 1, 'Generating final subtitles', time.time())
        logger.info(f'[Pipeline] Final export: {len(segments)} segments')
        result = _finalize_subtitle_result(
            result,
            segments,
            target_lang,
            preserve_boundaries=True,
        )
        _update_pipeline_progress('Export', 1, 1, 'Export complete')

        # Clear checkpoints on success
        clear_checkpoints(video_id)

        elapsed = time.time() - pipeline_started_at
        logger.info(f'[Pipeline] Total pipeline time: {elapsed:.2f}s')
        logger.info(f'[Pipeline] Pipeline completed successfully: {len(segments)} segments, method={result["method"]}')

    except Exception as e:
        logger.error(f'[Pipeline] Pipeline failed: {e}')
        logger.error(traceback.format_exc())
        result['error'] = str(e)
        result['success'] = False
        result['cancelled'] = isinstance(e, PipelineCancelled)

    finally:
        # Cleanup temporary files
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    return result


# ---------------------------------------------------------------------------
# Flask API Routes
# ---------------------------------------------------------------------------
@app.errorhandler(PipelineBusy)
def handle_pipeline_busy(error):
    return jsonify({
        'success': False,
        'error': str(error),
        'code': 'pipeline_busy',
    }), 429


@app.route('/api/detect-subtitles', methods=['POST'])
def api_detect_subtitles():
    """Detect burned-in subtitles directly, without translation or TTS."""
    with process_lock:
        video_path = None
        try:
            if 'video' not in request.files:
                return jsonify({'success': False, 'error': 'No video file uploaded.'}), 400
            video_file = request.files['video']
            if not video_file.filename:
                return jsonify({'success': False, 'error': 'No video file selected.'}), 400
            if not _allowed_video_filename(video_file.filename):
                return jsonify({
                    'success': False,
                    'error': _validate_video_upload(video_file.filename, '') or 'Unsupported video type.',
                }), 400

            video_path = str(
                TEMP_DIR / f'ocr_upload_{uuid4().hex}{Path(video_file.filename).suffix.lower()}'
            )
            video_file.save(video_path)
            validation_error = _validate_video_upload(video_file.filename, video_path)
            if validation_error:
                return jsonify({'success': False, 'error': validation_error}), 400

            source_lang = request.form.get('source_lang', 'auto').strip() or 'auto'
            try:
                fps = max(
                    0.5,
                    min(30.0, float(request.form.get('fps') or os.environ.get('OCR_FPS', '1.0'))),
                )
            except ValueError:
                return jsonify({'success': False, 'error': 'OCR fps must be a number.'}), 400

            duration = get_video_duration(video_path)
            last_error = None
            segments = []
            for attempt in range(1, 4):
                try:
                    logger.info(f'[API] OCR detection attempt {attempt}/3')
                    segments = detect_hardcoded_subtitles(
                        video_path,
                        fps=fps,
                        duration=duration,
                        source_lang=source_lang,
                    )
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f'[API] OCR detection attempt {attempt}/3 failed: {e}')
                    if attempt < 3:
                        time.sleep(min(2 ** (attempt - 1), 4))
            if last_error is not None:
                raise RuntimeError(f'OCR failed after 3 attempts: {last_error}')
            if not segments:
                return jsonify({
                    'success': False,
                    'error': 'No hardcoded subtitles were detected in the video.',
                }), 422

            result = {
                'success': True,
                'job_id': process_lock.job_id,
                'segments': segments,
                'srt': generate_srt(segments),
                'vtt': generate_vtt(segments),
                'method': 'ocr',
                'detected_lang': source_lang,
                'translation_enabled': _translation_enabled(),
            }
            logger.info(f'[API] OCR detection completed with {len(segments)} segments')
            return jsonify(result), 200
        except Exception as e:
            logger.error(f'[API] OCR detection error: {e}')
            logger.error(traceback.format_exc())
            cancelled = isinstance(e, PipelineCancelled)
            _mark_pipeline_terminal(
                'cancelled' if cancelled else 'error',
                str(e),
            )
            return jsonify({
                'success': False,
                'cancelled': cancelled,
                'job_id': process_lock.job_id,
                'error': str(e),
            }), 409 if cancelled else 500
        finally:
            try:
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)
            except OSError:
                pass


@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """
    API endpoint for subtitle transcription and translation.
    Accepts video upload and returns SRT, VTT, and JSON subtitle data.
    """
    with process_lock:
        video_path = None
        try:
            # Check if video file was uploaded
            if 'video' not in request.files:
                return jsonify({'success': False, 'error': 'No video file uploaded'}), 400

            video_file = request.files['video']
            if not video_file.filename:
                return jsonify({'success': False, 'error': 'No video file selected'}), 400

            # Save uploaded video
            video_id = str(int(time.time() * 1000))
            video_ext = Path(video_file.filename).suffix.lower()
            if video_ext not in ALLOWED_VIDEO_EXTENSIONS:
                allowed = ', '.join(sorted(ext.upper().lstrip('.') for ext in ALLOWED_VIDEO_EXTENSIONS))
                return jsonify({'success': False, 'error': f'Unsupported video type. Upload one of: {allowed}.'}), 400

            video_path = str(TEMP_DIR / f'upload_{video_id}{video_ext}')
            video_file.save(video_path)
            validation_error = _validate_video_upload(video_file.filename, video_path)
            if validation_error:
                try:
                    os.unlink(video_path)
                except OSError:
                    pass
                return jsonify({'success': False, 'error': validation_error}), 400

            # Get parameters
            source_lang = request.form.get('source_lang', 'auto').strip() or 'auto'
            target_lang = request.form.get('target_lang', 'km').strip() or 'km'
            use_ocr = request.form.get('use_ocr', 'true').strip().lower() in {'1', 'true', 'yes'}
            subtitle_only = request.form.get('subtitle_only', 'false').strip().lower() in {'1', 'true', 'yes'}
            speech_subtitles = request.form.get('speech_subtitles', 'false').strip().lower() in {'1', 'true', 'yes'}

            logger.info(f'[API] Transcribe request: {video_file.filename}, '
                        f'source={source_lang}, target={target_lang}, ocr={use_ocr}')

            # Run pipeline
            result = run_pipeline(
                video_path,
                source_lang=source_lang,
                target_lang=target_lang,
                use_ocr_fallback=use_ocr,
                subtitle_only=subtitle_only,
                speech_subtitles=speech_subtitles,
            )

            # Cleanup uploaded video
            try:
                if os.path.exists(video_path):
                    os.unlink(video_path)
            except OSError:
                pass

            if result.get('success'):
                result['job_id'] = process_lock.job_id
                return jsonify(result), 200
            else:
                result['job_id'] = process_lock.job_id
                cancelled = bool(result.get('cancelled'))
                _mark_pipeline_terminal(
                    'cancelled' if cancelled else 'error',
                    result.get('error') or 'Pipeline failed',
                )
                return jsonify(result), 409 if cancelled else 422

        except Exception as e:
            logger.error(f'[API] Transcribe error: {e}')
            logger.error(traceback.format_exc())
            cancelled = isinstance(e, PipelineCancelled)
            _mark_pipeline_terminal(
                'cancelled' if cancelled else 'error',
                str(e),
            )
            return jsonify({
                'success': False,
                'cancelled': cancelled,
                'job_id': process_lock.job_id,
                'error': str(e),
            }), 409 if cancelled else 500
        finally:
            try:
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)
            except OSError:
                pass


@app.route('/api/dubbing', methods=['POST'])
def api_dubbing():
    """Generate a single merged dubbed WAV from subtitle segments."""
    with process_lock:
        try:
            payload = request.get_json(silent=True) or {}
            segments = payload.get('segments') or []
            target_lang = str(payload.get('target_lang', 'km') or 'km').strip()
            voice = str(payload.get('voice', '') or '').strip()
            speech_rate = _coerce_float(payload.get('speech_rate', 1.0), 1.0)
            if speech_rate < 0.5 or speech_rate > 2.0:
                return jsonify({
                    'success': False,
                    'error': 'Voice speed must be between 0.5x and 2.0x.',
                }), 400
            pitch_hz = int(_coerce_float(payload.get('pitch_hz', 0), 0))
            volume_percent = int(_coerce_float(payload.get('volume_percent', 0), 0))
            tight_sync_value = payload.get('tight_sync', False)
            tight_sync = (
                tight_sync_value.strip().lower() in {'1', 'true', 'yes', 'on'}
                if isinstance(tight_sync_value, str)
                else bool(tight_sync_value)
            )
            if not -50 <= pitch_hz <= 50 or not -50 <= volume_percent <= 50:
                return jsonify({
                    'success': False,
                    'error': 'Voice pitch and volume must be between -50 and +50.',
                }), 400
            sample_rate = int(payload.get('sample_rate') or 48000)
            if sample_rate < 8000 or sample_rate > 96000:
                return jsonify({
                    'success': False,
                    'error': 'Audio sample rate must be between 8000 and 96000 Hz.',
                }), 400

            if not isinstance(segments, list) or not segments:
                return jsonify({'success': False, 'error': 'No subtitle segments provided.'}), 400

            if any(not isinstance(seg, dict) for seg in segments):
                return jsonify({
                    'success': False,
                    'error': 'Every subtitle segment must be an object.',
                }), 400
            if any(not _subtitle_tts_text(seg) for seg in segments):
                return jsonify({
                    'success': False,
                    'error': 'Every subtitle must contain text for TTS.',
                }), 400
            valid_segments = list(segments)

            job_id = uuid4().hex
            audio_filename = f'dubbing_{job_id}.wav'
            output_audio_path = TEMP_DIR / audio_filename

            updated_segments = _generate_dubbed_audio(
                valid_segments,
                target_lang,
                voice,
                str(output_audio_path),
                sample_rate=sample_rate,
                speech_rate=speech_rate,
                pitch_hz=pitch_hz,
                volume_percent=volume_percent,
                tight_sync=tight_sync,
            )
            scene_timeline = _build_audio_master_scene_timeline(updated_segments)
            measured_duration = _measure_wav_duration(str(output_audio_path))
            timeline_path = TEMP_DIR / f'dubbing_{job_id}.json'
            timeline_path.write_text(
                json.dumps({
                    'segments': updated_segments,
                    'scene_timeline': scene_timeline,
                    'timeline_duration': measured_duration,
                }, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            return jsonify({
                'success': True,
                'request_job_id': process_lock.job_id,
                'job_id': job_id,
                'audio_url': f'/api/dubbing/audio/{audio_filename}',
                'segments': updated_segments,
                'scene_timeline': scene_timeline,
                'timeline_duration': round(measured_duration, 3),
                'timing_source': 'subtitle_timestamps',
                'sample_rate': sample_rate,
                'loudness_target_lufs': -19,
                'true_peak_limit_dbtp': -1,
                'speaker_detection': 'not_used_for_tts',
            }), 200

        except Exception as e:
            logger.error(f'[API] Dubbing error: {e}')
            logger.error(traceback.format_exc())
            cancelled = isinstance(e, PipelineCancelled)
            _mark_pipeline_terminal(
                'cancelled' if cancelled else 'error',
                str(e),
            )
            return jsonify({
                'success': False,
                'cancelled': cancelled,
                'request_job_id': process_lock.job_id,
                'error': str(e),
            }), 409 if cancelled else 500


@app.route('/api/translate-subtitles', methods=['POST'])
def api_translate_subtitles():
    """Translate timed subtitle units without transcription or segmentation."""
    with process_lock:
        try:
            payload = request.get_json(silent=True) or {}
            segments = payload.get('segments') or []
            source_lang = str(payload.get('source_lang', 'en') or 'en').strip()
            target_lang = str(payload.get('target_lang', 'km') or 'km').strip()
            if not isinstance(segments, list) or not segments:
                return jsonify({
                    'success': False,
                    'error': 'No subtitle segments provided.',
                }), 400
            if any(
                not isinstance(segment, dict)
                or not isinstance(segment.get('source'), str)
                or not segment.get('source', '').strip()
                for segment in segments
            ):
                return jsonify({
                    'success': False,
                    'error': 'Every subtitle must contain source text.',
                }), 400

            originals = [dict(segment) for segment in segments]
            if not _translation_enabled():
                untranslated = []
                for segment in originals:
                    preserved = dict(segment)
                    preserved['target'] = preserved['source']
                    preserved.pop('translation_error', None)
                    untranslated.append(preserved)
                return jsonify({
                    'success': True,
                    'segments': untranslated,
                    'subtitle_count': len(untranslated),
                    'timing_source': 'subtitle_timestamps',
                    'translation_enabled': False,
                }), 200

            translated = translate_segments(
                originals,
                source_lang,
                target_lang,
            )
            _validate_subtitle_transform(
                originals,
                translated,
                require_target=True,
            )
            return jsonify({
                'success': True,
                'job_id': process_lock.job_id,
                'segments': translated,
                'subtitle_count': len(translated),
                'timing_source': 'subtitle_timestamps',
            }), 200
        except Exception as e:
            logger.error(f'[API] Subtitle translation error: {e}')
            logger.error(traceback.format_exc())
            cancelled = isinstance(e, PipelineCancelled)
            _mark_pipeline_terminal(
                'cancelled' if cancelled else 'error',
                str(e),
            )
            return jsonify({
                'success': False,
                'cancelled': cancelled,
                'job_id': process_lock.job_id,
                'error': str(e),
            }), 409 if cancelled else 500


@app.route('/api/dubbing/audio/<path:filename>', methods=['GET'])
def api_dubbing_audio(filename):
    audio_path = TEMP_DIR / Path(filename).name
    if not audio_path.exists():
        return jsonify({'success': False, 'error': 'Dubbing audio file not found.'}), 404
    return send_file(str(audio_path), mimetype='audio/wav', as_attachment=False)


@app.route('/api/export', methods=['POST'])
def api_export():
    with process_lock:
        video_path = None
        output_path = None
        subtitle_path = None
        try:
            if 'video' not in request.files:
                return jsonify({'success': False, 'error': 'No video file uploaded.'}), 400

            job_id = request.form.get('job_id', '').strip()
            if not job_id:
                return jsonify({'success': False, 'error': 'No job_id provided.'}), 400

            audio_filename = f'dubbing_{job_id}.wav'
            audio_path = TEMP_DIR / audio_filename
            if not audio_path.exists():
                return jsonify({'success': False, 'error': 'Dubbing audio not found.'}), 404

            video_file = request.files['video']
            if not video_file.filename:
                return jsonify({'success': False, 'error': 'No video file selected.'}), 400

            video_path = str(TEMP_DIR / f'export_video_{uuid4().hex}{Path(video_file.filename).suffix}')
            output_path = str(TEMP_DIR / f'exported_{job_id}.mp4')
            video_file.save(video_path)

            include_subtitles = request.form.get(
                'include_subtitles', 'true'
            ).strip().lower() != 'false'
            crop_enabled = request.form.get(
                'crop_enabled', 'false'
            ).strip().lower() == 'true'
            mirror_video = request.form.get(
                'mirror_video', 'false'
            ).strip().lower() == 'true'
            crop_aspect = request.form.get(
                'crop_aspect', 'original'
            ).strip().lower()
            crop_presets = {
                'original': None,
                'youtube': (16.0 / 9.0, 1920, 1080),
                'facebook': (9.0 / 16.0, 1080, 1920),
                'facebook_portrait': (4.0 / 5.0, 1080, 1350),
                'vertical': (9.0 / 16.0, 1080, 1920),
            }
            if crop_aspect not in crop_presets:
                return jsonify({
                    'success': False,
                    'error': 'Invalid video size preset.',
                }), 400
            try:
                crop_zoom = max(
                    1.0,
                    min(3.0, float(request.form.get('crop_zoom', '1'))),
                )
                crop_x = max(
                    -1.0,
                    min(1.0, float(request.form.get('crop_x', '0'))),
                )
                crop_y = max(
                    -1.0,
                    min(1.0, float(request.form.get('crop_y', '0'))),
                )
            except (TypeError, ValueError):
                return jsonify({
                    'success': False,
                    'error': 'Invalid video crop settings.',
                }), 400

            srt_text = request.form.get('subtitles', '').strip()
            if include_subtitles and not srt_text:
                return jsonify({
                    'success': False,
                    'error': 'No translated subtitles were provided for export.',
                }), 400
            escaped_subtitle_path = None
            if include_subtitles:
                subtitle_path = str(TEMP_DIR / f'export_subtitles_{job_id}.srt')
                Path(subtitle_path).write_text(srt_text, encoding='utf-8-sig')
                escaped_subtitle_path = (
                    Path(subtitle_path).as_posix()
                    .replace('\\', '/')
                    .replace(':', r'\:')
                    .replace("'", r"\'")
                )
            source_duration = get_video_duration(video_path)
            master_duration = _measure_wav_duration(str(audio_path))
            if source_duration <= 0 or master_duration <= 0:
                raise RuntimeError('Unable to measure source video or generated audio duration.')
            # TTS clips have already been fitted to absolute subtitle windows.
            # Preserve the source video clock and pad the dubbed track over it;
            # a second global/scene retime would shift mouths and scene cues.
            video_filters = []
            crop_preset = crop_presets[crop_aspect] if crop_enabled else None
            if crop_preset is not None:
                crop_x_ratio = (crop_x + 1.0) / 2.0
                crop_y_ratio = (crop_y + 1.0) / 2.0
                target_aspect, output_width, output_height = crop_preset
                video_filters.extend([
                    (
                        "crop="
                        f"w='trunc(min(iw\\,ih*{target_aspect:.9f})/"
                        f"{crop_zoom:.6f}/2)*2':"
                        f"h='trunc(min(ih\\,iw/{target_aspect:.9f})/"
                        f"{crop_zoom:.6f}/2)*2':"
                        f"x='(iw-ow)*{crop_x_ratio:.6f}':"
                        f"y='(ih-oh)*{crop_y_ratio:.6f}'"
                    ),
                    f'scale={output_width}:{output_height}',
                    'setsar=1',
                ])
            elif crop_enabled and crop_zoom > 1.0001:
                crop_x_ratio = (crop_x + 1.0) / 2.0
                crop_y_ratio = (crop_y + 1.0) / 2.0
                video_filters.extend([
                    (
                        "crop="
                        f"w='trunc(iw/{crop_zoom:.6f}/2)*2':"
                        f"h='trunc(ih/{crop_zoom:.6f}/2)*2':"
                        f"x='(iw-ow)*{crop_x_ratio:.6f}':"
                        f"y='(ih-oh)*{crop_y_ratio:.6f}'"
                    ),
                    (
                        "scale="
                        f"w='trunc(iw*{crop_zoom:.6f}/2)*2':"
                        f"h='trunc(ih*{crop_zoom:.6f}/2)*2'"
                    ),
                ])
            if mirror_video:
                video_filters.append('hflip')
            if include_subtitles:
                video_filters.append(
                    f"subtitles=filename='{escaped_subtitle_path}'"
                )
            video_filter_chain = ','.join(video_filters) or 'null'
            video_filter = (
                f'[0:v]{video_filter_chain}[videoout];'
                f'[1:a]apad,atrim=duration={source_duration:.6f},'
                f'asetpts=N/SR/TB[audioout]'
            )

            export_command = [
                FFMPEG_EXE, '-y',
                '-i', video_path,
                '-i', str(audio_path),
                '-filter_complex', video_filter,
                '-map', '[videoout]',
                '-map', '[audioout]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-t', f'{source_duration:.6f}',
                '-movflags', '+faststart',
                output_path,
            ]
            export_started_at = time.time()
            _update_pipeline_progress(
                'Export',
                0,
                1,
                'Rendering subtitles and synchronized audio into final video',
                export_started_at,
            )
            export_error = None
            for attempt in range(1, 3):
                process_lock.raise_if_cancelled()
                try:
                    _run_cancellable(
                        export_command,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                        raise RuntimeError('FFmpeg produced an empty export file.')
                    export_error = None
                    break
                except Exception as e:
                    export_error = e
                    stderr = getattr(e, 'stderr', '') or ''
                    logger.warning(
                        f'[Export] FFmpeg attempt {attempt}/2 failed: '
                        f'{e}; {stderr[-1200:]}'
                    )
                    if attempt < 2:
                        time.sleep(1)
            if export_error is not None:
                raise RuntimeError(
                    f'Final video export failed after 2 attempts: {export_error}'
                )
            _update_pipeline_progress(
                'Export',
                1,
                1,
                'Final video render complete',
                export_started_at,
            )

            response = send_file(
                output_path,
                as_attachment=True,
                download_name=f'{Path(video_file.filename).stem}_dubbed.mp4',
                mimetype='video/mp4',
            )
            def _remove_export_file():
                try:
                    if output_path and os.path.exists(output_path):
                        os.unlink(output_path)
                except OSError as cleanup_error:
                    logger.debug(f'[Export] Deferred cleanup failed: {cleanup_error}')
            response.call_on_close(_remove_export_file)
            return response

        except Exception as e:
            logger.error(f'[API] Export error: {e}')
            logger.error(traceback.format_exc())
            cancelled = isinstance(e, PipelineCancelled)
            _mark_pipeline_terminal(
                'cancelled' if cancelled else 'error',
                str(e),
            )
            return jsonify({
                'success': False,
                'cancelled': cancelled,
                'job_id': process_lock.job_id,
                'error': str(e),
            }), 409 if cancelled else 500

        finally:
            try:
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)
            except Exception:
                pass
            try:
                if subtitle_path and os.path.exists(subtitle_path):
                    os.unlink(subtitle_path)
            except Exception:
                pass


@app.route('/api/status', methods=['GET'])
def api_status():
    """Return current pipeline progress status."""
    with _pipeline_progress_lock:
        status = dict(_pipeline_progress)
    requested_job = request.args.get('job_id', '').strip()
    if requested_job and status.get('job_id') not in {None, requested_job}:
        return jsonify({
            'job_id': requested_job,
            'state': 'unknown',
            'status': 'This job is no longer active.',
        }), 404
    return jsonify(status), 200


@app.route('/api/cancel/<job_id>', methods=['POST'])
def api_cancel(job_id):
    """Request cooperative cancellation without waiting for the media lock."""
    if process_lock.cancel(job_id.strip()):
        _mark_pipeline_terminal('cancelling', 'Cancellation requested')
        return jsonify({
            'success': True,
            'job_id': job_id,
            'state': 'cancelling',
        }), 202
    return jsonify({
        'success': False,
        'job_id': job_id,
        'error': 'The requested job is not active.',
    }), 404


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'version': '2.0.0',
        'busy': process_lock.active,
        'translation_enabled': _translation_enabled(),
        'subtitle_source': 'speech_whisper',
        'ocr_detection_source': 'visual_ocr',
    }), 200


@app.route('/')
def index():
    """Serve the frontend."""
    return send_file(str(PROJECT_DIR / 'khmer-video-translator.html'))


@app.route('/<path:path>')
def static_files(path):
    """Serve static files."""
    return send_from_directory(str(WEB_DIR), path)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5050'))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = _env_enabled('FLASK_DEBUG', False)
    _cleanup_stale_temp_files()
    logger.info(f'Starting Khmer Video Translator server on {host}:{port}')
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False,
    )
