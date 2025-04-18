import { debugLog } from './utils/debug.js';
import { loadModel, detectObjects, drawDetections, MODEL_INPUT_SIZE } from './detection/objectDetection.js';

// Get DOM elements
const videoElement = document.getElementById('videoFeed');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement?.getContext('2d', { willReadFrequently: true });
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');

// State management
const state = {
    isModelLoaded: false,
    isCameraReady: false,
    isProcessing: false,
    stream: null,
    error: null
};

// State variables
let animationFrameId = null;
let frameCount = 0;
const PROCESS_EVERY_N_FRAMES = 3; // Process every 3rd frame for better performance

// Verify all required elements exist
function verifyElements() {
    const elements = {
        video: videoElement,
        canvas: canvasElement,
        canvasContext: canvasCtx,
        startButton: startButton,
        stopButton: stopButton
    };

    for (const [name, element] of Object.entries(elements)) {
        if (!element) {
            throw new Error(`Required element ${name} not found`);
        }
    }
}

// Initialize the application
async function init() {
    console.log('Initializing application...');
    try {
        // Verify DOM elements
        verifyElements();
        console.log('All required elements found');

        // Check for required APIs
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('Camera API is not supported in your browser');
        }
        console.log('Camera API is supported');

        // Load the model in the background
        loadModel()
            .then(() => {
                state.isModelLoaded = true;
                console.log('Model loaded successfully');
                debugLog('Model loaded successfully', 'success');
                updateButtonStates();
            })
            .catch(error => {
                console.error('Model loading error:', error);
                debugLog(`Model loading failed: ${error.message}`, 'error');
                state.error = error;
                updateButtonStates();
            });

        // Enable start button immediately - model will load in background
        updateButtonStates();
        debugLog('Application initialized successfully', 'success');
    } catch (error) {
        console.error('Initialization error:', error);
        debugLog(`Failed to initialize: ${error.message}`, 'error');
        state.error = error;
        updateButtonStates();
    }
}

function updateButtonStates() {
    // Start button enabled if camera API available and no processing
    startButton.disabled = state.isProcessing || !navigator.mediaDevices?.getUserMedia || !!state.error;
    // Stop button enabled only when processing
    stopButton.disabled = !state.isProcessing;
}

// Start camera and processing
async function startCamera() {
    console.log('Starting camera...');
    try {
        // Request camera with specific constraints
        const constraints = {
            video: {
                facingMode: 'environment',  // Try environment camera first
                width: { ideal: MODEL_INPUT_SIZE },
                height: { ideal: MODEL_INPUT_SIZE }
            },
            audio: false
        };
        
        console.log('Requesting camera with constraints:', constraints);
        
        // Stop any existing stream
        if (state.stream) {
            state.stream.getTracks().forEach(track => track.stop());
        }
        
        state.stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log('Got camera stream:', state.stream);
        
        videoElement.srcObject = state.stream;
        console.log('Set video source');
        
        // Wait for video to be ready
        await new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error('Video loading timed out'));
            }, 10000); // 10 second timeout

            videoElement.onloadedmetadata = () => {
                console.log('Video metadata loaded');
                videoElement.play()
                    .then(() => {
                        clearTimeout(timeoutId);
                        console.log('Video playback started');
                        resolve();
                    })
                    .catch(err => {
                        clearTimeout(timeoutId);
                        console.error('Video playback failed:', err);
                        reject(err);
                    });
            };
        });
        
        // Set canvas size to match video
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        console.log(`Canvas size set to ${canvasElement.width}x${canvasElement.height}`);
        
        // Update state and buttons
        state.isCameraReady = true;
        state.isProcessing = true;
        updateButtonStates();
        
        // Start processing frames
        processFrame();
        
        debugLog('Camera started successfully', 'success');
    } catch (error) {
        console.error('Camera start error:', error);
        debugLog(`Error accessing camera: ${error.message}`, 'error');
        state.error = error;
        updateButtonStates();
        
        // Try fallback to any available camera
        if (error.name === 'OverconstrainedError' || error.name === 'NotFoundError') {
            try {
                const fallbackConstraints = {
                    video: true,
                    audio: false
                };
                
                state.stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
                videoElement.srcObject = state.stream;
                await videoElement.play();
                
                state.isCameraReady = true;
                state.isProcessing = true;
                updateButtonStates();
                
                processFrame();
                
                debugLog('Camera started with fallback settings', 'warning');
            } catch (fallbackError) {
                console.error('Fallback camera error:', fallbackError);
                debugLog(`Fallback camera failed: ${fallbackError.message}`, 'error');
                alert('Could not access any camera. Please check permissions and try again.');
            }
        } else {
            alert(`Camera error: ${error.message}. Please check console for details.`);
        }
    }
}

// Stop camera and processing
function stopCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
    
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    state.isProcessing = false;
    state.isCameraReady = false;
    updateButtonStates();
    
    // Clear the canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    debugLog('Camera stopped', 'info');
}

// Process each frame
async function processFrame() {
    if (!state.isProcessing) return;
    
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
window.addEventListener('DOMContentLoaded', init);

// Cleanup when the page unloads
window.addEventListener('beforeunload', () => {
    stopCamera();
}); 