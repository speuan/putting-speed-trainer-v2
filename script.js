'use strict';

// Get DOM elements
const videoElement = document.getElementById('videoFeed');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d', { willReadFrequently: true });
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const playbackButton = document.getElementById('playbackButton');

let stream = null;
let isRecording = false;
let animationFrameId = null;
let model = null; // TensorFlow.js model
let isModelLoaded = false;
let lastDetections = []; // Store last N detections for smoothing
const MAX_DETECTION_HISTORY = 5; // Number of frames to keep for smoothing
const MIN_CONFIDENCE = 0.001; // Minimum confidence to consider a detection
const IOU_THRESHOLD = 0.2; // Intersection over Union threshold for clustering

// --- Model Loading ---

async function loadModel() {
    try {
        // Show loading indicator (you might want to add a UI element for this)
        console.log('Loading model...');
        console.log('TensorFlow.js version:', tf.version.tfjs);
        console.log('Backend:', tf.getBackend());
        
        // Load the model from your exported files
        const modelUrl = './my_model_web_model/model.json';
        console.log('Attempting to load model from:', modelUrl);
        
        // First check if the model file exists
        try {
            const response = await fetch(modelUrl);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const modelJson = await response.json();
            console.log('Model JSON loaded:', modelJson);
        } catch (fetchError) {
            console.error('Error checking model.json:', fetchError);
            throw new Error('Could not access model.json file');
        }
        
        // Load the model
        model = await tf.loadGraphModel(modelUrl, {
            onProgress: (fraction) => {
                console.log(`Model loading progress: ${(fraction * 100).toFixed(1)}%`);
            }
        });
        
        // Test the model with a dummy tensor to ensure it's working
        console.log('Testing model...');
        const dummyTensor = tf.zeros([1, 640, 640, 3]);
        const testResult = await model.predict(dummyTensor);
        console.log('Model input shape:', dummyTensor.shape);
        console.log('Model output shape:', testResult.shape);
        dummyTensor.dispose();
        testResult.dispose();
        
        isModelLoaded = true;
        console.log('Model loaded and tested successfully');
        
        // Enable UI elements that depend on the model
        recordButton.disabled = false;
    } catch (error) {
        console.error('Detailed error loading model:', error);
        console.error('Error stack:', error.stack);
        alert('Failed to load ball detection model. Check console for details.');
        throw error;
    }
}

// --- Ball Detection ---

async function detectBall(imageData) {
    if (!model) {
        console.log('Model not loaded yet');
        return null;
    }
    
    try {
        console.log('Input image size:', imageData.width, 'x', imageData.height);
        
        // Convert the image data to a tensor and preprocess
        const tensor = tf.tidy(() => {
            const imageTensor = tf.browser.fromPixels(imageData);
            console.log('Original tensor shape:', imageTensor.shape);
            
            // Normalize the image (convert to float32 and scale to [0,1])
            const normalized = tf.div(tf.cast(imageTensor, 'float32'), 255);
            console.log('Normalized tensor shape:', normalized.shape);
            
            // Ensure the image is 640x640
            const resized = tf.image.resizeBilinear(normalized, [640, 640]);
            console.log('Resized tensor shape:', resized.shape);
            
            const batched = resized.expandDims(0);
            console.log('Final input tensor shape:', batched.shape);
            return batched;
        });
        
        // Run inference
        console.log('Running model prediction...');
        const predictions = await model.predict(tensor);
        console.log('Raw prediction output shape:', predictions.shape);
        
        // Clean up tensor memory
        tensor.dispose();
        
        // Process predictions
        const arrayPreds = await predictions.array();
        predictions.dispose();
        
        console.log('Prediction array structure:', {
            length: arrayPreds.length,
            firstDimLength: arrayPreds[0].length,
            sampleValues: arrayPreds[0].map(arr => arr.slice(0, 3))
        });
        
        // Process the predictions to get the highest confidence detection
        const detections = processDetections(arrayPreds[0]);
        console.log('Processed detections:', detections);
        
        return detections;
    } catch (error) {
        console.error('Error during detection:', error);
        console.error('Error stack:', error.stack);
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
    if (!detection) return;
    
    // Extract values from detection (these are now normalized 0-1)
    const [x, y, w, h, confidence] = detection;
    
    // Log the normalized values
    console.log('Drawing normalized detection:', { x, y, w, h, confidence });
    
    // Convert normalized coordinates to canvas coordinates
    const boxX = x * canvasElement.width;
    const boxY = y * canvasElement.height;
    const boxWidth = w * canvasElement.width;
    const boxHeight = h * canvasElement.height;
    
    // Log the canvas coordinates
    console.log('Drawing canvas coordinates:', { boxX, boxY, boxWidth, boxHeight });
    
    // Draw debug points at corners to verify coordinate system
    const corners = [
        [boxX, boxY],                           // Top-left
        [boxX + boxWidth, boxY],                // Top-right
        [boxX + boxWidth, boxY + boxHeight],    // Bottom-right
        [boxX, boxY + boxHeight]                // Bottom-left
    ];
    
    // Draw the corners as small circles
    canvasCtx.fillStyle = '#FF0000';
    corners.forEach(([cx, cy]) => {
        canvasCtx.beginPath();
        canvasCtx.arc(cx, cy, 3, 0, 2 * Math.PI);
        canvasCtx.fill();
    });
    
    // Draw bounding box
    canvasCtx.strokeStyle = '#00FF00';
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeRect(boxX, boxY, boxWidth, boxHeight);
    
    // Draw confidence score with more precision since values are very small
    canvasCtx.fillStyle = '#00FF00';
    canvasCtx.font = '16px Arial';
    canvasCtx.fillText(`Ball: ${(confidence * 100).toFixed(4)}%`, boxX, boxY - 5);
    
    // Draw crosshair at center
    const centerX = boxX + boxWidth/2;
    const centerY = boxY + boxHeight/2;
    
    canvasCtx.beginPath();
    canvasCtx.moveTo(centerX - 10, centerY);
    canvasCtx.lineTo(centerX + 10, centerY);
    canvasCtx.moveTo(centerX, centerY - 10);
    canvasCtx.lineTo(centerX, centerY + 10);
    canvasCtx.stroke();
    
    // Draw coordinate values for debugging
    canvasCtx.fillStyle = '#FF0000';
    canvasCtx.font = '12px Arial';
    canvasCtx.fillText(`x: ${boxX.toFixed(1)}, y: ${boxY.toFixed(1)}`, boxX, boxY - 20);
    canvasCtx.fillText(`w: ${boxWidth.toFixed(1)}, h: ${boxHeight.toFixed(1)}`, boxX, boxY - 35);
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
        // Draw the current video frame onto the canvas
        canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        if (isRecording && isModelLoaded) {
            try {
                // Get the frame data
                const imageData = canvasCtx.getImageData(0, 0, canvasElement.width, canvasElement.height);
                
                // Detect ball in the frame
                const detections = await detectBall(imageData);
                
                if (detections) {
                    // Draw detection results
                    drawDetections(detections);
                }
            } catch (error) {
                console.error('Error processing frame:', error);
            }
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