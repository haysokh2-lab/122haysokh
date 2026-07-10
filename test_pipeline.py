#!/usr/bin/env python3
"""Quick test for the subtitle pipeline."""
import json
import urllib.request
import os

# Read the sample video
video_path = 'sample_test_video.mp4'
if not os.path.exists(video_path):
    print(f'Video not found: {video_path}')
    exit(1)

with open(video_path, 'rb') as f:
    video_data = f.read()

boundary = '----TestBoundary123'

# Build multipart form data
body_parts = []
body_parts.append('--' + boundary)
body_parts.append('Content-Disposition: form-data; name="video"; filename="test.mp4"')
body_parts.append('Content-Type: video/mp4')
body_parts.append('')
body_parts.append('')  # will be replaced with binary

# Calculate parts before binary
before_binary = '\r\n'.join(body_parts).encode('utf-8') + b'\r\n'
after_binary = '\r\n--' + boundary + '--\r\n'

body_data = before_binary + video_data + after_binary.encode('utf-8')

req = urllib.request.Request(
    'http://localhost:5050/api/transcribe',
    data=body_data,
    headers={
        'Content-Type': 'multipart/form-data; boundary=' + boundary,
    }
)

print('Sending request to pipeline server...')
try:
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read().decode())
    print('Success:', result.get('success'))
    print('Method:', result.get('method'))
    print('Detected lang:', result.get('detected_lang'))
    segments = result.get('segments', [])
    print('Segments:', len(segments))
    if segments:
        for s in segments[:3]:
            print(f'  [{s["start"]:.2f}-{s["end"]:.2f}] {s["source"][:60]}')
    if result.get('srt'):
        print('\nSRT Preview (first 200 chars):')
        print(result['srt'][:200])
    if result.get('vtt'):
        print('\nVTT Preview (first 200 chars):')
        print(result['vtt'][:200])
except urllib.error.HTTPError as e:
    print('HTTP Error:', e.code, e.reason)
    print('Response:', e.read().decode()[:500])
except Exception as e:
    print('Error:', e)