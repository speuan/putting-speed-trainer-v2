'use strict';

// Get DOM elements
const videoElement = document.getElementById('videoFeed');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d', { willReadFrequently: true });
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const playbackButton = document.getElementById('playbackButton');

// Add these at the top with other DOM elements
const debugPanel = document.createElement('div');
debugPanel.style.cssText = `
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 150px;
    background: rgba(0, 0, 0, 0.8);
    color: #fff;
    font-family: monospace;
    font-size: 12px;
    padding: 10px;
    overflow-y: auto;
    z-index: 1000;
`;
document.body.appendChild(debugPanel);

let stream = null;
let isRecording = false;
let animationFrameId = null;
let model = null; // TensorFlow.js model
let isModelLoaded = false;
let lastDetections = []; // Store last N detections for smoothing
const MAX_DETECTION_HISTORY = 5; // Number of frames to keep for smoothing
const MIN_CONFIDENCE = 0.001; // Minimum confidence to consider a detection
const IOU_THRESHOLD = 0.2; // Intersection over Union threshold for clustering

// Add debug logging function
function debugLog(message, type = 'info') {
    const colors = {
        info: '#fff',
        error: '#ff4444',
        success: '#44ff44',
        warning: '#ffff44'
    };
    
    const entry = document.createElement('div');
    entry.style.color = colors[type] || colors.info;
    entry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
    debugPanel.insertBefore(entry, debugPanel.firstChild);
    
    // Keep only last 20 messages
    while (debugPanel.children.length > 20) {
        debugPanel.removeChild(debugPanel.lastChild);
    }
    
    // Also log to console
    console.log(message);
}

// --- Model Loading ---

async function loadModel() {
    try {
        debugLog('Loading model...', 'info');
        debugLog(`TensorFlow.js version: ${tf.version.tfjs}`, 'info');
        debugLog(`Backend: ${tf.getBackend()}`, 'info');
        
        const modelUrl = './my_model_web_model/model.json';
        debugLog(`Attempting to load model from: ${modelUrl}`, 'info');
        
        try {
            const response = await fetch(modelUrl);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const modelJson = await response.json();
            debugLog('Model JSON loaded successfully', 'success');
        } catch (fetchError) {
            debugLog(`Error checking model.json: ${fetchError}`, 'error');
            throw new Error('Could not access model.json file');
        }
        
        model = await tf.loadGraphModel(modelUrl, {
            onProgress: (fraction) => {
                debugLog(`Model loading progress: ${(fraction * 100).toFixed(1)}%`, 'info');
            }
        });
        
        debugLog('Testing model...', 'info');
        const dummyTensor = tf.zeros([1, 640, 640, 3]);
        const testResult = await model.predict(dummyTensor);
        debugLog(`Model input shape: ${dummyTensor.shape}`, 'info');
        debugLog(`Model output shape: ${testResult.shape}`, 'info');
        dummyTensor.dispose();
        testResult.dispose();
        
        isModelLoaded = true;
        debugLog('Model loaded and tested successfully', 'success');
        
        recordButton.disabled = false;
    } catch (error) {
        debugLog(`Error loading model: ${error.message}`, 'error');
        debugLog(`Error stack: ${error.stack}`, 'error');
        alert('Failed to load ball detection model. Check debug panel for details.');
        throw error;
    }
}

// --- Ball Detection ---

async function detectBall(imageData) {
    if (!model) {
        debugLog('Model not loaded yet', 'warning');
        return null;
    }
    
    try {
        debugLog(`Input image size: ${imageData.width}x${imageData.height}`, 'info');
        
        const tensor = tf.tidy(() => {
            const imageTensor = tf.browser.fromPixels(imageData);
            const normalized = tf.div(tf.cast(imageTensor, 'float32'), 255);
            const resized = tf.image.resizeBilinear(normalized, [640, 640]);
            const batched = resized.expandDims(0);
            return batched;
        });
        
        const predictions = await model.predict(tensor);
        tensor.dispose();
        
        const arrayPreds = await predictions.array();
        predictions.dispose();
        
        const detections = processDetections(arrayPreds[0]);
        
        if (detections) {
            const [x, y, w, h, conf] = detections;
            debugLog(`Detection found: x=${x.toFixed(4)}, y=${y.toFixed(4)}, w=${w.toFixed(4)}, h=${h.toFixed(4)}, conf=${(conf * 100).toFixed(2)}%`, 'success');
        } else {
            debugLog('No detections above threshold', 'warning');
        }
        
        return detections;
    } catch (error) {
        debugLog(`Error during detection: ${error.message}`, 'error');
        return null;
    }
}

function processDetections(predictions) {
    if (!predictions || predictions.length !== 5) {
        console.log('Invalid predictions format:', predictions);
        return null;
    }
    
    // Get all detections above minimum confidence
    const detections = [];
    for (let i = 0; i < predictions[0].length; i++) {
        const confidence = predictions[4][i];
        if (confidence > MIN_CONFIDENCE) {
            detections.push({
                x: predictions[0][i] / 640, // Normalize coordinates
                y: predictions[1][i] / 640,
                w: predictions[2][i] / 640,
                h: predictions[3][i] / 640,
                confidence: confidence
            });
        }
    }
    
    if (detections.length === 0) return null;
    
    // Cluster overlapping detections
    const clusters = clusterDetections(detections);
    
    // Get the cluster with highest confidence
    let bestCluster = clusters[0];
    for (let i = 1; i < clusters.length; i++) {
        if (clusters[i].confidence > bestCluster.confidence) {
            bestCluster = clusters[i];
        }
    }
    
    return [
        bestCluster.x,
        bestCluster.y,
        bestCluster.w,
        bestCluster.h,
        bestCluster.confidence
    ];
}

function clusterDetections(detections) {
    const clusters = [];
    
    for (const detection of detections) {
        let added = false;
        
        for (const cluster of clusters) {
            if (calculateIoU(detection, cluster) > IOU_THRESHOLD) {
                // Merge detection into cluster with weighted average
                const totalWeight = cluster.confidence + detection.confidence;
                cluster.x = (cluster.x * cluster.confidence + detection.x * detection.confidence) / totalWeight;
                cluster.y = (cluster.y * cluster.confidence + detection.y * detection.confidence) / totalWeight;
                cluster.w = (cluster.w * cluster.confidence + detection.w * detection.confidence) / totalWeight;
                cluster.h = (cluster.h * cluster.confidence + detection.h * detection.confidence) / totalWeight;
                cluster.confidence = Math.max(cluster.confidence, detection.confidence);
                added = true;
                break;
            }
        }
        
        if (!added) {
            clusters.push({...detection});
        }
    }
    
    return clusters;
}

function calculateIoU(box1, box2) {
    // Convert from center format to corner format
    const box1Left = box1.x - box1.w/2;
    const box1Right = box1.x + box1.w/2;
    const box1Top = box1.y - box1.h/2;
    const box1Bottom = box1.y + box1.h/2;
    
    const box2Left = box2.x - box2.w/2;
    const box2Right = box2.x + box2.w/2;
    const box2Top = box2.y - box2.h/2;
    const box2Bottom = box2.y + box2.h/2;
    
    // Calculate intersection
    const intersectionLeft = Math.max(box1Left, box2Left);
    const intersectionRight = Math.min(box1Right, box2Right);
    const intersectionTop = Math.max(box1Top, box2Top);
    const intersectionBottom = Math.min(box1Bottom, box2Bottom);
    
    if (intersectionRight < intersectionLeft || intersectionBottom < intersectionTop) {
        return 0;
    }
    
    const intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);
    const box1Area = box1.w * box1.h;
    const box2Area = box2.w * box2.h;
    
    return intersectionArea / (box1Area + box2Area - intersectionArea);
}

function smoothDetections(detections) {
    if (detections.length === 0) return null;
    
    // Calculate weighted average based on confidence
    let totalWeight = 0;
    let smoothedX = 0;
    let smoothedY = 0;
    let smoothedW = 0;
    let smoothedH = 0;
    let maxConfidence = 0;
    
    for (const detection of detections) {
        const [x, y, w, h, confidence] = detection;
        const weight = confidence;
        totalWeight += weight;
        
        smoothedX += x * weight;
        smoothedY += y * weight;
        smoothedW += w * weight;
        smoothedH += h * weight;
        maxConfidence = Math.max(maxConfidence, confidence);
    }
    
    return [
        smoothedX / totalWeight,
        smoothedY / totalWeight,
        smoothedW / totalWeight,
        smoothedH / totalWeight,
        maxConfidence
    ];
}

function drawDetections(detection) {
    if (!canvasElement || !canvasCtx || !detection) {
        return;
    }
    
    // Extract values from detection (these are normalized 0-1)
    const [x, y, w, h, confidence] = detection;
    
    // Convert normalized coordinates to canvas coordinates
    const centerX = x * canvasElement.width;
    const centerY = y * canvasElement.height;
    const boxWidth = w * canvasElement.width;
    const boxHeight = h * canvasElement.height;
    
    // Calculate top-left corner for drawing
    const drawX = centerX - (boxWidth / 2);
    const drawY = centerY - (boxHeight / 2);
    
    try {
        // Draw thin bounding box
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = '#00FF00';
        canvasCtx.strokeRect(drawX, drawY, boxWidth, boxHeight);
        
        // Draw small confidence score
        const text = `${(confidence * 100).toFixed(1)}%`;
        canvasCtx.font = '14px Arial';
        
        // Text background
        const padding = 4;
        const textMetrics = canvasCtx.measureText(text);
        canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        canvasCtx.fillRect(
            drawX, 
            drawY - 20, 
            textMetrics.width + padding * 2, 
            18
        );
        
        // Text
        canvasCtx.fillStyle = '#00FF00';
        canvasCtx.fillText(text, drawX + padding, drawY - 6);
        
    } catch (error) {
        debugLog(`Error during drawing: ${error.message}`, 'error');
    }
}

// --- Camera Setup ---

async function setupCamera() {
    try {
        const constraints = {
            video: {
                facingMode: 'environment',
                width: { ideal: 640 },
                height: { ideal: 640 }
            },
            audio: false
        };
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;

        // Wait for video metadata and then start everything
        videoElement.onloadedmetadata = () => {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            console.log(`Camera setup complete. Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
            
            // Load the model after camera is ready
            loadModel().then(() => {
                startVideoProcessing();
            });
        };

    } catch (err) {
        console.error("Error accessing camera:", err);
        alert(`Could not access camera: ${err.name} - ${err.message}`);
        if (err.name === "OverconstrainedError" || err.name === "NotFoundError") {
            console.log("Falling back to default camera...");
            try {
                const fallbackConstraints = { 
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 640 }
                    },
                    audio: false 
                };
                stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
                videoElement.srcObject = stream;
                videoElement.onloadedmetadata = () => {
                    canvasElement.width = videoElement.videoWidth;
                    canvasElement.height = videoElement.videoHeight;
                    console.log(`Fallback camera setup complete. Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
                    loadModel().then(() => {
                        startVideoProcessing();
                    });
                };
            } catch (fallbackErr) {
                console.error("Error accessing fallback camera:", fallbackErr);
                alert(`Could not access fallback camera: ${fallbackErr.name} - ${fallbackErr.message}`);
            }
        }
    }
}

// --- Video Processing & Drawing ---

async function drawVideoFrame() {
    if (!videoElement.paused && !videoElement.ended) {
        // Update canvas dimensions if needed
        if (canvasElement.width !== videoElement.videoWidth || canvasElement.height !== videoElement.videoHeight) {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            debugLog(`Updated canvas dimensions to ${videoElement.videoWidth}x${videoElement.videoHeight}`, 'info');
        }

        // Create a temporary canvas for compositing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvasElement.width;
        tempCanvas.height = canvasElement.height;
        const tempCtx = tempCanvas.getContext('2d');

        // Draw the current video frame to temp canvas
        tempCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        // Get frame data for detection from the temp canvas
        let detections = null;
        if (isRecording && isModelLoaded) {
            try {
                const imageData = tempCtx.getImageData(0, 0, canvasElement.width, canvasElement.height);
                detections = await detectBall(imageData);
            } catch (error) {
                debugLog(`Error processing frame: ${error.message}`, 'error');
            }
        }

        // Now draw everything to the main canvas
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // Draw the video frame
        canvasCtx.drawImage(tempCanvas, 0, 0);

        // If we have detections, draw them last
        if (detections) {
            drawDetections(detections);
            
            // Log coordinates for debugging
            const [x, y, w, h, conf] = detections;
            const screenX = x * canvasElement.width;
            const screenY = y * canvasElement.height;
            debugLog(`Screen coordinates: (${screenX.toFixed(1)}, ${screenY.toFixed(1)})`, 'info');
        }

        // Request the next frame
        animationFrameId = requestAnimationFrame(drawVideoFrame);
    }
}

function startVideoProcessing() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId); // Stop previous loop if any
    }
    drawVideoFrame(); // Start the drawing loop
}

function stopVideoProcessing() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    // Optionally clear the canvas when stopped
    // canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

// --- Control Logic ---

recordButton.addEventListener('click', () => {
    if (!stream) {
        alert("Camera not ready yet.");
        return;
    }
    isRecording = true;
    console.log("Recording started...");

    // Update button states
    recordButton.disabled = true;
    stopButton.disabled = false;
    playbackButton.disabled = true; // Disable playback during recording
});

stopButton.addEventListener('click', () => {
    isRecording = false;
    console.log("Recording stopped.");

    // Update button states
    recordButton.disabled = false;
    stopButton.disabled = true;
    playbackButton.disabled = false; // Enable playback after stopping (if frames were recorded)
});

playbackButton.addEventListener('click', () => {
    // --- Placeholder for Playback ---
    console.log("Playback button clicked (not implemented yet).");
    alert("Playback functionality is not yet implemented.");
    // TODO: Implement playback logic, e.g., drawing recordedFrames onto the canvas
    // --- End Placeholder ---
});

// --- Initialization ---

window.addEventListener('load', () => {
    // Disable record button until model is loaded
    recordButton.disabled = true;
    
    if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
        setupCamera();
    } else {
        alert("getUserMedia API is not supported in your browser.");
        console.error("getUserMedia not supported.");
    }
});

// --- Cleanup on page unload ---
window.addEventListener('beforeunload', () => {
    stopVideoProcessing();
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    // Clean up TensorFlow memory
    if (model) {
        model.dispose();
    }
});