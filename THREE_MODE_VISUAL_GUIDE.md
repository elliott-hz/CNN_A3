# Three-Mode Web Interface - Quick Visual Guide

## Header Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  🐕 Dog Emotion Recognition System                                  │
│                                                                      │
│  [📷 Upload Image] [🎬 Upload Video] [📹 Live Stream]              │
│     (Active)                                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Mode 1: Upload Image Mode (Default)

```
┌─────────────────────────────────────────────────────────────────────┐
│  📷 Click or drag & drop an image                                   │
│     Supports JPEG, PNG (max 10MB, auto-resized to 640px)           │
└─────────────────────────────────────────────────────────────────────┘

After upload:
┌─────────────────────────────────────────────────────────────────────┐
│  [Annotated Image with Bounding Boxes]                              │
│  ┌──────────┐                                                       │
│  │ 😊 Happy │  Dog #1                                               │
│  │   🐕     │                                                       │
│  └──────────┘                                                       │
│                                                                      │
│  Detection Cards:                                                     │
│  ┌─────────────────────────────────┐                                │
│  │ Dog #1  😊 Happy                │                                │
│  │ Detection Confidence: 95.2%     │                                │
│  │ Emotion Confidence: 87.3%       │                                │
│  │ Probabilities:                  │                                │
│  │   Happy    ████████████ 87.3%   │                                │
│  │   Relaxed  ██ 8.5%              │                                │
│  └─────────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Mode 2: Upload Video Mode (NEW!)

### Before Upload
```
┌─────────────────────────────────────────────────────────────────────┐
│  🎬 Click or drag & drop a video                                    │
│     Supports MP4, WebM, AVI (max 50MB)                             │
└─────────────────────────────────────────────────────────────────────┘
```

### After Upload
```
┌─────────────────────────────────────────────────────────────────────┐
│  ✅ Video loaded successfully!                                      │
│  📁 dog_video.mp4 • 12.45 MB                                        │
│  [🔄 Change Video]                                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  [Video Player]                                                      │
│  ┌───────────────────────────────────────────┐                      │
│  │                                           │                      │
│  │         🎥 Playing Video...               │                      │
│  │                                           │                      │
│  └───────────────────────────────────────────┐                      │
│  [⏸️ Pause]          ● Analysis Active        │                      │
└─────────────────────────────────────────────────────────────────────┘

Detection Results (updated every 3 seconds):
┌─────────────────────────────────────────────────────────────────────┐
│  🎥 Video Analysis Active                                            │
│  Frames are being analyzed every 3 seconds.                         │
│  Detections will appear here.                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Mode 3: Live Stream Mode

```
┌─────────────────────────────────────────────────────────────────────┐
│  [Live Camera Feed]                                                  │
│  ┌───────────────────────────────────────────┐                      │
│  │ 🔴 LIVE STREAM                            │                      │
│  │                                           │                      │
│  │         📹 Camera Active                  │                      │
│  │                                           │                      │
│  │  📹 Camera active • Real-time video feed  │                      │
│  │  Future: Emotion detection will be added  │                      │
│  └───────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Mode Switching Flow

```
User clicks mode button
        ↓
App clears all current state
        ↓
New mode component renders
        ↓
User interacts with new mode
```

## File Size Limits

| Mode | Format | Max Size | Processing |
|------|--------|----------|------------|
| 📷 Image | JPEG, PNG | 10MB | Immediate |
| 🎬 Video | MP4, WebM, AVI | 50MB | Every 3s |
| 📹 Live | Camera Feed | N/A | Future |

## Color Coding for Emotions

| Emotion | Emoji | Color | Hex Code |
|---------|-------|-------|----------|
| Happy | 😊 | Green | #4CAF50 |
| Angry | 😠 | Red | #f44336 |
| Relaxed | 😌 | Blue | #2196F3 |
| Frown | 😟 | Orange | #FF9800 |
| Alert | 👀 | Purple | #9C27B0 |

## Responsive Design

### Desktop (>768px)
- Three buttons in header row
- Full-width content area
- Multi-column detection cards

### Mobile (≤768px)
- Buttons wrap to multiple rows
- Single-column layout
- Stacked detection cards

---

**Tip**: The interface automatically manages state when switching modes, ensuring a clean experience every time!
