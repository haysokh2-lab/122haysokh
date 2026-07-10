# Khmer Video Translator

Flask-based video localization studio with Whisper subtitles, translation, synchronized neural voice, and FFmpeg export.

## Structure

- `khmer-video-translator.html` is the browser UI.
- `subtitle_pipeline.py` is the active Flask API and media pipeline.
- `tests/test_subtitle_timing.py` covers timing, segmentation, translation validation, and dubbing timeline rules.
- `backend/` and the topic folders contain an earlier Node architecture draft and reference notes; they are not used by the active runtime.

## Requirements

- Python 3.10+
- Packages from `requirements.txt`
- FFmpeg on `PATH`, in `ffmpeg/bin`, or supplied by `imageio-ffmpeg`
- At least one translation provider: an installed Argos language pair, `GEMINI_API_KEY`, `OPENAI_API_KEY`, or online MyMemory access
- Internet access for Edge neural TTS

## Environment

Key variables include `GEMINI_API_KEY`, `OPENAI_API_KEY`, `WHISPER_MODEL`,
`WHISPER_DEVICE`, `FFMPEG_EXE`, and optional `APP_MAX_VIDEO_SECONDS` (zero or
unset means unlimited). Translation is enabled by default; set
`TRANSLATION_ENABLED=false` to temporarily bypass it without removing the
installed translation stage.
OCR defaults to a CPU-safe 1 FPS pass over the central 30%-84% of each frame,
excluding the UI-heavy outer bands. Set `OCR_SCAN_FULL_FRAME=true` for captions
at the extreme top/bottom, `OCR_MAX_DIMENSION` to trade speed for small-text
accuracy, and `OCR_FPS` for timing density. CUDA is selected automatically when
available (`OCR_GPU=auto`); set `OCR_GPU=false` to force CPU.
Completed OCR and every finished batch are cached by video content, so a retry
resumes instead of restarting and repeated generation is nearly immediate.
Optional Whisper speaker metadata is off in the fast profile; set
`SPEECH_METADATA_ENABLED=true` when speaker labels are more important than
latency.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
npm run dev
```

Open `http://localhost:5050`.

On Windows, `start-translator.cmd` can also be double-clicked. Keep its window
open while using the application, then open `http://127.0.0.1:5050`.

## Capabilities

- Frame-timed visual OCR is the sole displayed subtitle text/timing source.
- Word timestamps and optional pyannote diarization enrich OCR captions with speaker metadata only.
- Per-subtitle translation is enabled and can be temporarily disabled by configuration.
- One neural TTS clip per subtitle, fitted into the subtitle's measured time window.
- Absolute subtitle/video timing is preserved during preview and export; dubbed audio is padded over the unchanged source timeline.
- MP4 export with dubbed AAC audio and burned-in translated subtitles.

## Workflow

1. **Load Video** selects and previews a local video.
2. **Detect Subtitle** reads burned-in captions directly with frame-timed OCR.
3. **Generate Subtitle** uses visual OCR for displayed text/timing and speech analysis only for speaker metadata.
4. **Translate** translates OCR subtitle blocks without changing their boundaries.
5. **Generate Voice** creates and aligns one neural clip per subtitle.
6. **Preview** plays the video, subtitles, and generated audio on one absolute clock.
7. **Export Final Video** burns subtitles and muxes the padded dubbed track into MP4.
8. **Settings** focuses the existing language and processing controls.

Long-running stages report server-measured progress and ETA. **Cancel
processing** stops OCR/FFmpeg cooperatively; a second request receives an
immediate busy response instead of waiting behind the active job.

## Notes

Translation and TTS are real provider calls. Invalid keys, exhausted quota, a
blocked network, or a missing Argos language package are reported as errors;
the pipeline does not silently claim that unchanged source text is translated.

Automatic character identity requires `pyannote.audio` plus
`HUGGINGFACE_TOKEN`/`PYANNOTE_AUTH_TOKEN`. Without diarization, unlabeled speech
is safely marked `SPEAKER_00`; the application does not invent speaker changes.
The repository synchronizes existing visual scenes but does not contain a
facial-animation/Wav2Lip model, so it cannot synthesize new mouth, face, or body
motion.

EasyOCR recognizes Chinese, English, Japanese, Korean, and the other language
models exposed by the source-language selector. The installed EasyOCR release
does not include a Khmer recognition model; Khmer is fully supported as a
translation and neural-voice target, while Khmer burned-in source captions
require an external Khmer-capable OCR model.
