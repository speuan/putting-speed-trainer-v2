import { createDebugPanel, debugLog } from './utils/debug.js';
import { loadModel, detectBall, MODEL_INPUT_SIZE, PROCESS_EVERY_N_FRAMES } from './detection/ballDetection.js';

// Get DOM elements
const videoElement = document.getElementById('videoFeed');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d', { willReadFrequently: true });
const recordButton = document.getElementById('recordButton');
const stopButton = document.getElementById('stopButton');
const playbackButton = document.getElementById('playbackButton');

// Initialize debug panel
const debugPanel = createDebugPanel();

// State variables
let stream = null;
let isRecording = false;
let animationFrameId = null;
let frameCount = 0;

// Initialize the application
async function init() {
    try {
        // Load the model
        await loadModel(debugPanel);
        recordButton.disabled = false;
    } catch (error) {
        alert('Failed to initialize the application. Check debug panel for details.');
    }
}

// Start recording
async function startRecording() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: MODEL_INPUT_SIZE,
                height: MODEL_INPUT_SIZE,
                facingMode: 'environment'
            }
        });
        
        videoElement.srcObject = stream;
        isRecording = true;
        recordButton.disabled = true;
        stopButton.disabled = false;
        
        // Start processing frames
        processFrame();
    } catch (error) {
        debugLog(debugPanel, `Error accessing camera: ${error.message}`, 'error');
        alert('Failed to access camera. Check debug panel for details.');
    }
}

// Process video frames
function processFrame() {
    if (!isRecording) return;
    
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        // Draw the video frame to the canvas
        canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // Only process every Nth frame
        if (frameCount % PROCESS_EVERY_N_FRAMES === 0) {
            detectBall(debugPanel, canvasElement)
                .then(detections => {
                    if (detections) {
                        // Draw detection box
                        const [x, y, w, h, confidence] = detections;
                        const boxX = x * canvasElement.width;
                        const boxY = y * canvasElement.height;
                        const boxW = w * canvasElement.width;
                        const boxH = h * canvasElement.height;
                        
                        canvasCtx.strokeStyle = '#00FF00';
                        canvasCtx.lineWidth = 2;
                        canvasCtx.strokeRect(
                            boxX - boxW/2,
                            boxY - boxH/2,
                            boxW,
                            boxH
                        );
                        
                        // Draw confidence score
                        canvasCtx.fillStyle = '#00FF00';
                        canvasCtx.font = '16px Arial';
                        canvasCtx.fillText(
                            `${(confidence * 100).toFixed(1)}%`,
                            boxX - boxW/2,
                            boxY - boxH/2 - 5
                        );
                    }
                })
                .catch(error => {
                    debugLog(debugPanel, `Error in detection: ${error.message}`, 'error');
                });
        }
        frameCount++;
    }
    
    animationFrameId = requestAnimationFrame(processFrame);
}

// Stop recording
function stopRecording() {
    isRecording = false;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    videoElement.srcObject = null;
    recordButton.disabled = false;
    stopButton.disabled = true;
    playbackButton.disabled = false;
}

// Event listeners
recordButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);

// Initialize the application
init(); 