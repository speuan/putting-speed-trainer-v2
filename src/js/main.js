import { debugLog } from './utils/debug.js';
import { loadModel, detectObjects, drawDetections, MODEL_INPUT_SIZE } from './detection/objectDetection.js';

// Get DOM elements
const videoElement = document.getElementById('videoFeed');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d', { willReadFrequently: true });
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');

// State variables
let stream = null;
let isProcessing = false;
let animationFrameId = null;
let frameCount = 0;
const PROCESS_EVERY_N_FRAMES = 3; // Process every 3rd frame for better performance

// Initialize the application
async function init() {
    try {
        // Load the model
        await loadModel();
        startButton.disabled = false;
    } catch (error) {
        alert('Failed to initialize the application. Check debug panel for details.');
    }
}

// Start camera and processing
async function startCamera() {
    console.log('Starting camera...');
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('Camera API not supported');
            throw new Error('Camera API is not supported in your browser');
        }

        console.log('Camera API supported');

        // Request camera with specific constraints
        const constraints = {
            video: true,  // Simplified constraints first
            audio: false
        };
        
        console.log('Requesting camera with constraints:', constraints);
        
        // Stop any existing stream
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log('Got camera stream:', stream);
        
        videoElement.srcObject = stream;
        console.log('Set video source');
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                console.log('Video metadata loaded');
                videoElement.play()
                    .then(() => {
                        console.log('Video playback started');
                        resolve();
                    })
                    .catch(err => {
                        console.error('Video playback failed:', err);
                        throw err;
                    });
            };
        });
        
        // Set canvas size to match video
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        console.log(`Canvas size set to ${canvasElement.width}x${canvasElement.height}`);
        
        // Update button states
        startButton.disabled = true;
        stopButton.disabled = false;
        
        // Start processing frames
        isProcessing = true;
        processFrame();
        
        debugLog('Camera started successfully', 'success');
    } catch (error) {
        console.error('Camera start error:', error);
        debugLog(`Error accessing camera: ${error.message}`, 'error');
        alert(`Camera error: ${error.message}. Please check console for details.`);
    }
}

// Stop camera and processing
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    isProcessing = false;
    startButton.disabled = false;
    stopButton.disabled = true;
    
    // Clear the canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    debugLog('Camera stopped', 'info');
}

// Process each frame
async function processFrame() {
    if (!isProcessing) return;
    
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        // Only process every Nth frame
        if (frameCount % PROCESS_EVERY_N_FRAMES === 0) {
            try {
                // Draw the current frame to the canvas
                canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                
                // Get the frame data
                const imageData = canvasCtx.getImageData(0, 0, canvasElement.width, canvasElement.height);
                
                // Detect objects
                const detections = await detectObjects(imageData);
                
                if (detections && detections.length > 0) {
                    // Draw detections
                    drawDetections(canvasCtx, detections, canvasElement.width, canvasElement.height);
                }
            } catch (error) {
                debugLog(`Error processing frame: ${error.message}`, 'error');
            }
        }
        frameCount++;
    }
    
    // Request next frame
    animationFrameId = requestAnimationFrame(processFrame);
}

// Event listeners
startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);

// Initialize when the page loads
window.addEventListener('DOMContentLoaded', async () => {
    try {
        // Initialize the model first
        await loadModel();
        startButton.disabled = false;
        debugLog('Application initialized successfully', 'success');
    } catch (error) {
        debugLog(`Failed to initialize: ${error.message}`, 'error');
        alert('Failed to initialize the application. Please check the debug panel for details.');
    }
});

// Cleanup when the page unloads
window.addEventListener('beforeunload', () => {
    stopCamera();
}); 