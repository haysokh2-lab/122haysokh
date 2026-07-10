#!/usr/bin/env python3
"""Test the pipeline using Flask's test client."""
import sys
import os
import json

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create a test Flask app and run the pipeline function directly
from subtitle_pipeline import app, run_pipeline

video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_test_video.mp4')
if not os.path.exists(video_path):
    print(f'Video not found: {video_path}')
    sys.exit(1)

print(f'Testing pipeline with video: {video_path}')
print(f'File size: {os.path.getsize(video_path)} bytes')

# Run pipeline directly
result = run_pipeline(video_path, source_lang='en', target_lang='km', use_ocr_fallback=True)

print(f'\nSuccess: {result.get("success")}')
print(f'Method: {result.get("method")}')
print(f'Detected lang: {result.get("detected_lang")}')
print(f'Error: {result.get("error")}')
segments = result.get('segments', [])
print(f'Segments: {len(segments)}')

if segments:
    print('\nSample segments:')
    for s in segments[:5]:
        print(f'  [{s["start"]:.2f}-{s["end"]:.2f}] {s["source"][:80]}')

    # Test SRT and VTT
    if result.get('srt'):
        srt_lines = result['srt'].split('\n')
        print(f'\nSRT preview ({len(srt_lines)} lines):')
        print('\n'.join(srt_lines[:15]))

    if result.get('vtt'):
        vtt_lines = result['vtt'].split('\n')
        print(f'\nVTT preview ({len(vtt_lines)} lines):')
        print('\n'.join(vtt_lines[:15]))
else:
    print('\nNo segments found.')
    if result.get('error'):
        print(f'Error: {result["error"]}')