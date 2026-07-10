---
name: subtitle-timing-and-speaker-fix
description: Repair subtitle segmentation, speaker separation, subtitle timing, and TTS synchronization in an existing video localization pipeline without creating duplicate modules.
applyTo:
  - subtitle_pipeline.py
  - tests/**/*.py
---

# Subtitle Timing and Speaker Separation Repair

Use this skill when an existing subtitle/TTS pipeline is producing subtitles that:
- merge multiple speakers into one subtitle,
- split the same spoken phrase incorrectly,
- drift out of sync with the actual speech,
- or start TTS audio before or after the subtitle appears.

## Goal

Ensure that every subtitle corresponds to exactly one spoken phrase from exactly one speaker, and that synthesized voice playback starts exactly when the subtitle appears.

## Workflow

1. Inspect the existing subtitle pipeline end to end.
   - Review transcription, speaker detection, segmentation, timestamp generation, subtitle export, TTS synthesis, audio alignment, and final muxing.
   - Identify the stage that introduces speaker mixing, timing drift, or audio delay.

2. Fix the existing implementation in place.
   - Prefer modifying the current functions and classes over introducing new modules.
   - Reuse the existing pipeline structure and helper functions whenever possible.
   - Do not create duplicate implementations unless the current code cannot be safely extended.

3. Preserve one-speaker-per-subtitle behavior.
   - Split subtitles immediately when a speaker change is detected.
   - Never merge dialogue from two speakers into one subtitle block.
   - Keep natural sentence boundaries whenever possible.
   - Preserve speaker order exactly as spoken.

4. Anchor subtitle timing to speech evidence.
   - Use word-level timestamps from Whisper as the primary timing reference when available.
   - Ensure subtitle start and end times match the actual spoken phrase boundaries.
   - Remove accumulated timing drift by recalculating segment boundaries from the speech data instead of relying on accumulated offsets.

5. Synchronize TTS audio to the subtitle window.
   - Ensure voice playback begins at the same instant the subtitle appears.
   - Ensure voice playback does not start before the subtitle appears or continue after the subtitle disappears unless the timing explicitly requires it.
   - Adjust synthesized audio timing rather than delaying subtitle appearance.
   - Remove unnecessary silence before speech and prevent overlapping dialogue.

6. Validate the result.
   - Verify that each subtitle contains only one speaker.
   - Verify that subtitle timing follows the speech boundaries.
   - Verify that TTS playback stays aligned with the subtitle window.
   - Run the relevant tests and compile the pipeline to confirm no regressions.

## Quality Criteria

A fix is complete only when all of the following are true:
- every subtitle contains one speaker and one phrase,
- speaker changes create a new subtitle boundary immediately,
- subtitle timing is anchored to the actual speech timestamps,
- synthesized voice begins when the subtitle appears,
- no cumulative timing drift remains across the video,
- and the existing project still runs without introducing new errors.

## Constraints

- Modify existing files only.
- Do not add duplicate modules, duplicate helper files, or alternate implementations unless absolutely necessary.
- Preserve existing architecture and entry points.
- Keep the changes focused on subtitle segmentation, speaker separation, timing accuracy, and TTS synchronization.

## Example prompts

- Repair the subtitle pipeline so each subtitle contains one speaker only.
- Fix the synchronization so TTS audio starts exactly when the subtitle appears.
- Remove subtitle timing drift and make the timestamps match Whisper word boundaries.
- Refine the segmentation logic so rapid speaker alternation creates separate subtitles.
