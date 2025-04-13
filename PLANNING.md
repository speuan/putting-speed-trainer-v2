# PLANNING.md

## Project Name
Golf Putting Speed Trainer (PWA)

## Goal
Create a browser-based (PWA) tool that allows users to analyze the speed and trajectory of golf putts using their mobile camera. The app will record, process, and track a golf ball to compute speed and motion metrics.

---

## Scope

### Core Features
- Static camera setup assumed (no camera motion).
- Golf ball detection via custom trained lightweight model.
- Use HTML5 video and canvas for frame capture.
- Real-time ball tracking and motion analysis.
- Calibrate pixel-to-real-world distance ratio.
- PWA capabilities: offline support, mobile-friendly.

### Future Extensions
- Stroke path visualization.
- Putt consistency scoring.
- AI coaching feedback.
- Upload/share results.

---

## Architecture

### Frontend (PWA)
- **HTML/CSS/JS** frontend using Web APIs (`getUserMedia`, `Canvas`)
- UI for camera controls, recording, and playback
- Optional TensorFlow.js for object detection

### ML Component
- **Custom TensorFlow.js object detection model**
  - Trained to recognize a golf ball.
  - Small custom MobileNet/SSD/YOLO model, trained externally (e.g., Roboflow or TensorFlow Model Maker)
  - Converts frame-by-frame detections to movement paths.
- Speed estimation using frame displacement and known dimensions (calibration)




### Storage
- In-memory array for recorded frames.
- Optional: use IndexedDB for persistence.

---

## Technology Stack

| Component | Tech |
| --------- | ---- |
| Frontend  | Vanilla JS, HTML5, CSS |
| Detection | TensorFlow.js (custom-trained model) |
| UI        | Responsive, mobile-first layout |
| Hosting   | GitHub Pages, Netlify, or Firebase Hosting |
| PWA       | Service Workers, Web App Manifest (later) |

---

## Key Design Considerations
- Model size must be small for fast browser performance.
- Must work offline once loaded (PWA requirement).
- Frame rate and resolution affect detection accuracy vs performance.

## Naming Conventions

### Files and Folders
- Use **lowercase** letters with **hyphens** (`-`) to separate words.
  - Example: `camera-controller.js`, `speed-calculation.js`

### Variables and Functions
- Use **camelCase** for variables and function names.
  - Example: `captureFrame()`, `calculateBallSpeed()`

### Classes
- Use **PascalCase** for class names.
  - Example: `BallTracker`, `SpeedEstimator`

### Constants
- Use **UPPER_SNAKE_CASE** for constants.
  - Example: `PIXEL_TO_CM_RATIO`, `FRAME_CAPTURE_INTERVAL`

### Model and AI Files
- Model files should be prefixed with `model-` and use hyphens.
  - Example: `model-golf-ball-detection.json`

---
