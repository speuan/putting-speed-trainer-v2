# TASK.md

## 📍 Initial Setup

- [x] Build base PWA with video and canvas capture
- [x] Add UI buttons for recording, stopping, and playback
- [x] Implement OpenCV motion detection for prototyping

---

## 🎯 Ball Detection Plan (TensorFlow.js)

- [x] Capture 100+ frames with golf ball (different positions/angles)
- [x] Use Roboflow or LabelImg to label ball in images
- [x] Train a small custom object detection model (MobileNet SSD)
- [x] Export model in TensorFlow.js format
- [x] Integrate TensorFlow.js into frontend and load model
- [x] Detect ball location and draw overlays
- [ ] Measure frame-to-frame displacement
- [ ] Calibrate pixel to cm ratio for speed calculation

---

## 🧠 Speed Estimation Logic

- [ ] Implement `getBallCenterFromModelOutput()` utility
- [ ] Track ball position per frame
- [ ] Compute speed = distance moved / time between frames
- [ ] Display speed overlay in canvas

---

## 🧪 Debugging + Utilities

- [ ] Add debug mode toggle in UI
- [ ] Add FPS counter
- [ ] Add bounding box overlay on detection
- [ ] Add calibration mode (tap to measure known distance)

---

## 🛠️ PWA Setup (later)

- [x] Add Web App Manifest
- [x] Add Service Worker for offline caching
- [ ] Add install prompt support

---

## 📦 Stretch Goals

- [ ] Store session history with timestamps
- [ ] Compare multiple putts side-by-side
- [ ] Analyze putting consistency over time