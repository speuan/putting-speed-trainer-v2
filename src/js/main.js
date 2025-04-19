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

// Check if model files are accessible
async function checkModelFiles() {
    try {
        debugLog('Checking model files...', 'info');
        
        // Use absolute path for model URL to ensure proper shard URL resolution
        const baseUrl = window.location.href.substring(0, window.location.href.lastIndexOf('/') + 1);
        const modelUrl = new URL('./src/assets/my_model_web_model_2/model.json', baseUrl).href;
        
        debugLog(`Checking model at: ${modelUrl}`, 'info');
        
        // First check the model.json file
        try {
            const response = await fetch(modelUrl, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            debugLog('Model JSON is accessible', 'success');
        } catch (fetchError) {
            debugLog(`Error accessing model.json: ${fetchError}`, 'error');
            throw new Error('Could not access model.json file');
        }
        
        // Try to read model JSON to check if it's valid
        try {
            const jsonResponse = await fetch(modelUrl);
            const modelJson = await jsonResponse.json();
            
            // Verify we can access at least one of the shard files
            if (modelJson.weightsManifest && modelJson.weightsManifest.length > 0) {
                const shards = modelJson.weightsManifest[0].paths;
                if (shards && shards.length > 0) {
                    // Get model directory from model URL
                    const modelDir = modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1);
                    const firstShard = new URL(shards[0], modelDir).href;
                    
                    debugLog(`Checking first shard: ${firstShard}`, 'info');
                    
                    const shardResponse = await fetch(firstShard, { method: 'HEAD' });
                    if (!shardResponse.ok) {
                        throw new Error(`Shard not accessible: ${shardResponse.status}`);
                    }
                    debugLog('Model shard is accessible', 'success');
                }
            }
            
            debugLog('Model JSON structure validated', 'success');
            return true;
        } catch (jsonError) {
            debugLog(`Error validating model JSON: ${jsonError}`, 'error');
            throw jsonError;
        }
    } catch (error) {
        debugLog(`Model files check failed: ${error.message}`, 'error');
        return false;
    }
}

// Initialize the application
async function init() {
    console.log('Initializing application...');
    try {
        // Log device information
        console.log('User Agent:', navigator.userAgent);
        const isIOS = iOSDevice();
        if (isIOS) {
            console.log('iOS device detected - using compatible settings');
            debugLog('iOS device detected - using compatible settings', 'info');
        }

        // Wait for TensorFlow.js to be ready
        if (!window.tensorflowReady) {
            console.log('Waiting for TensorFlow.js to initialize...');
            await new Promise((resolve) => {
                const checkTf = setInterval(() => {
                    if (window.tensorflowReady) {
                        clearInterval(checkTf);
                        resolve();
                    }
                }, 100);
                
                // Timeout after 10 seconds
                setTimeout(() => {
                    clearInterval(checkTf);
                    throw new Error('TensorFlow.js initialization timeout');
                }, 10000);
            });
        }
        console.log('TensorFlow.js is ready');
        
        // Special handling for iOS devices
        if (isIOS) {
            try {
                // On iOS, allocate a smaller tensor to test GPU memory
                console.log('Testing WebGL memory allocation for iOS...');
                const smallDummy = tf.zeros([1, 224, 224, 3]);
                smallDummy.dispose();
                debugLog('WebGL memory test passed', 'success');
            } catch (error) {
                console.error('WebGL memory test failed:', error);
                debugLog('WebGL memory test failed - will use CPU backend', 'warning');
                
                // Force CPU backend if WebGL fails
                await tf.setBackend('cpu');
                await tf.ready();
                debugLog(`Fallback to CPU backend: ${tf.getBackend()}`, 'warning');
            }
        }

        // Verify DOM elements
        verifyElements();
        console.log('All required elements found');

        // Check for required APIs
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('Camera API is not supported in your browser');
        }
        console.log('Camera API is supported');
        
        // Verify model files are accessible before trying to load them
        const modelFilesOk = await checkModelFiles();
        if (!modelFilesOk) {
            debugLog('Model files check failed, using debug mode', 'warning');
            // Still continue initialization but mark the error
            state.error = new Error('Model files not accessible');
        }

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
        alert('Failed to initialize the application. Please check the console for details.');
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
    console.log('startCamera function called from main.js');
    
    try {
        // Check if camera is already running from the HTML button
        if (videoElement.srcObject && videoElement.srcObject.active) {
            console.log('Camera already active from HTML button');
            const stream = videoElement.srcObject;
            
            // Update state with existing stream
            state.stream = stream;
            
            // Set canvas size to match video
            if (videoElement.videoWidth) {
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                console.log(`Canvas size set to ${canvasElement.width}x${canvasElement.height}`);
            } else {
                // Wait for video to be ready
                await new Promise((resolve) => {
                    videoElement.onloadedmetadata = () => {
                        canvasElement.width = videoElement.videoWidth || MODEL_INPUT_SIZE;
                        canvasElement.height = videoElement.videoHeight || MODEL_INPUT_SIZE;
                        console.log(`Canvas size set to ${canvasElement.width}x${canvasElement.height} after metadata`);
                        resolve();
                    };
                    
                    // Timeout just in case metadata doesn't load
                    setTimeout(resolve, 1000);
                });
            }
            
            // Update state and buttons
            state.isCameraReady = true;
            state.isProcessing = true;
            updateButtonStates();
            
            // Start processing frames
            processFrame();
            
            debugLog('Camera already running, starting processing', 'success');
            return;
        }
        
        // Continue with normal camera initialization if not already running...
        // iOS Safari requires user interaction to start camera
        if (iOSDevice()) {
            debugLog('iOS device detected - using compatible settings', 'info');
        }
        
        // Request camera with iOS-compatible constraints
        const constraints = {
            video: {
                facingMode: 'environment',
                width: { ideal: MODEL_INPUT_SIZE },
                height: { ideal: MODEL_INPUT_SIZE }
            },
            audio: false
        };
        
        // Simplify constraints for iOS
        if (iOSDevice()) {
            constraints.video = {
                facingMode: 'environment'
            };
        }
        
        console.log('Requesting camera with constraints:', constraints);
        
        // Stop any existing stream
        if (state.stream) {
            state.stream.getTracks().forEach(track => track.stop());
        }
        
        // Reset video element
        videoElement.srcObject = null;
        videoElement.load();
        
        // Get new stream
        state.stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log('Got camera stream:', state.stream);
        
        // Set up video element with iOS-compatible settings
        videoElement.srcObject = state.stream;
        videoElement.style.display = 'block';
        videoElement.setAttribute('playsinline', 'true');
        videoElement.setAttribute('muted', 'true');
        videoElement.setAttribute('autoplay', 'true');
        console.log('Set video source with iOS attributes');
        
        // iOS Safari requires a different approach to starting video
        if (iOSDevice()) {
            console.log('Using iOS-specific video playback approach');
            videoElement.play().catch(error => {
                console.error('iOS play error:', error);
            });
        }
        
        // Wait for video to be ready
        await new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error('Video loading timed out'));
            }, 10000);

            const cleanup = () => {
                videoElement.removeEventListener('loadedmetadata', onMetadata);
                videoElement.removeEventListener('error', onError);
                clearTimeout(timeoutId);
            };

            const onError = (error) => {
                cleanup();
                console.error('Video error event:', error);
                reject(new Error(`Video error: ${error.message || 'Unknown error'}`));
            };

            const onMetadata = async () => {
                try {
                    console.log('Video metadata loaded');
                    if (!iOSDevice()) {
                        await videoElement.play();
                    }
                    console.log('Video playback started or attempted');
                    cleanup();
                    resolve();
                } catch (error) {
                    console.error('Play error in onMetadata:', error);
                    cleanup();
                    reject(error);
                }
            };

            videoElement.addEventListener('loadedmetadata', onMetadata);
            videoElement.addEventListener('error', onError);
        });
        
        // Double check video is actually playing
        if (videoElement.paused) {
            console.warn('Video is still paused after initialization');
            if (iOSDevice()) {
                console.log('Attempting playback again for iOS');
                try {
        await videoElement.play();
                } catch (error) {
                    console.error('Second play attempt failed:', error);
                }
            } else {
                throw new Error('Video failed to start playing');
            }
        }
        
        // Set canvas size to match video
        canvasElement.width = videoElement.videoWidth || MODEL_INPUT_SIZE;
        canvasElement.height = videoElement.videoHeight || MODEL_INPUT_SIZE;
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
            try {
            console.log('Attempting fallback camera approach');
            const fallbackConstraints = {
                video: true,
                audio: false
            };
            
            // Reset video element
            videoElement.srcObject = null;
            videoElement.load();
            
            state.stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
            videoElement.srcObject = state.stream;
            videoElement.style.display = 'block';
            videoElement.setAttribute('playsinline', 'true');
            videoElement.setAttribute('muted', 'true');
            videoElement.setAttribute('autoplay', 'true');
            
                await videoElement.play();
                
            canvasElement.width = videoElement.videoWidth || MODEL_INPUT_SIZE;
            canvasElement.height = videoElement.videoHeight || MODEL_INPUT_SIZE;
                
            state.isCameraReady = true;
            state.isProcessing = true;
            updateButtonStates();
                
                processFrame();
                
                debugLog('Camera started with fallback settings', 'warning');
            } catch (fallbackError) {
            console.error('Fallback camera error:', fallbackError);
            debugLog(`Fallback camera failed: ${fallbackError.message}`, 'error');
            alert('Could not access camera. Please ensure camera permissions are granted and try again.');
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
    
    const isIOS = iOSDevice();
    
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        // Only process every Nth frame (use more frames for iOS)
        const processEveryFrames = isIOS ? 6 : PROCESS_EVERY_N_FRAMES;
        
        if (frameCount % processEveryFrames === 0) {
            try {
                // Draw the current frame to the canvas
                canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                
                // Get the frame data
                const imageData = canvasCtx.getImageData(0, 0, canvasElement.width, canvasElement.height);
                
                // Special handling for iOS
                if (isIOS) {
                    // Use a smaller processing size for iOS
                    const scaledCanvas = document.createElement('canvas');
                    const scaledCtx = scaledCanvas.getContext('2d', { willReadFrequently: true });
                    
                    // Create a scaled-down version (half size) for iOS
                    const scaleFactor = 0.5;
                    const scaledWidth = Math.floor(canvasElement.width * scaleFactor);
                    const scaledHeight = Math.floor(canvasElement.height * scaleFactor);
                    
                    scaledCanvas.width = scaledWidth;
                    scaledCanvas.height = scaledHeight;
                    
                    // Draw scaled down image
                    scaledCtx.drawImage(videoElement, 0, 0, scaledWidth, scaledHeight);
                    const scaledImageData = scaledCtx.getImageData(0, 0, scaledWidth, scaledHeight);
                    
                    // Detect objects with the scaled-down data
                    const detections = await detectObjects(scaledImageData);
                    
                    // Clear canvas for fresh drawing
                    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

                    // Draw video frame
                    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                    
                    // Draw directly with original detections - coordinates are normalized
                    if (detections && detections.length > 0) {
                        debugLog(`Found ${detections.length} detections, drawing boxes`, 'success');
                        drawDetections(canvasCtx, detections, canvasElement.width, canvasElement.height);
                    } else {
                        debugLog('No detections in this frame', 'info');
                    }
                    
                    // Clean up
                    scaledCanvas.remove();
                } else {
                    // Normal detection for non-iOS
                    const detections = await detectObjects(imageData);
                    
                    if (detections && detections.length > 0) {
                        // Clear canvas first
                        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                        
                        // Redraw video frame
                        canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                        
                        // Draw detections
                        debugLog(`Drawing ${detections.length} detection boxes`, 'info');
                        drawDetections(canvasCtx, detections, canvasElement.width, canvasElement.height);
                    }
                }
            } catch (error) {
                console.error('Processing error:', error);
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

// Helper function to detect iOS devices
function iOSDevice() {
    return /iPad|iPhone|iPod/.test(navigator.userAgent) || 
           (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
} 