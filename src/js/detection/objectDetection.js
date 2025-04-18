import { debugLog } from '../utils/debug.js';

// Constants
export const MODEL_INPUT_SIZE = 640;
export const MIN_CONFIDENCE = 0.5;

let model = null;
let isModelLoaded = false;
let modelLoadingPromise = null;

// Custom model loader to handle weight files correctly
export async function loadModel() {
    // If model loading is already in progress, return the existing promise
    if (modelLoadingPromise) {
        return modelLoadingPromise;
    }
    
    // Create a new promise for model loading
    modelLoadingPromise = (async () => {
        try {
            debugLog('Loading model...', 'info');
            debugLog(`TensorFlow.js version: ${tf.version.tfjs}`, 'info');
            
            // Ensure WebGL backend for iOS
            const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || 
                          (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
            
            if (isIOS) {
                debugLog('iOS device detected - ensuring WebGL backend', 'info');
                try {
                    await tf.setBackend('webgl');
                    await tf.ready();
                } catch (backendError) {
                    debugLog(`WebGL backend error: ${backendError}`, 'error');
                    // Fall back to default backend if WebGL fails
                }
            }
            
            debugLog(`Using backend: ${tf.getBackend()}`, 'info');
            
            // STEP 1: First fetch and parse the model.json file ourselves
            const modelJsonUrl = './src/assets/my_model_web_model_2/model.json';
            debugLog(`Fetching model JSON from: ${modelJsonUrl}`, 'info');
            
            let modelJson;
            try {
                const response = await fetch(modelJsonUrl);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                modelJson = await response.json();
                debugLog('Model JSON loaded successfully', 'success');
            } catch (jsonError) {
                debugLog(`Error loading model JSON: ${jsonError}`, 'error');
                throw new Error(`Could not load model.json: ${jsonError.message}`);
            }
            
            // STEP 2: Modify the weights paths to use absolute URLs
            if (modelJson.weightsManifest && modelJson.weightsManifest.length > 0) {
                // Get the base URL for resolving relative paths
                const baseModelPath = './src/assets/my_model_web_model_2/';
                
                // Replace each relative path with an absolute one
                for (let i = 0; i < modelJson.weightsManifest.length; i++) {
                    const manifest = modelJson.weightsManifest[i];
                    if (manifest.paths) {
                        // Replace all paths with absolute URLs
                        modelJson.weightsManifest[i].paths = manifest.paths.map(path => {
                            return `${baseModelPath}${path}`;
                        });
                    }
                }
                
                debugLog('Modified weight paths to use absolute URLs', 'success');
            }
            
            // STEP 3: Load the model using the modified JSON
            try {
                debugLog('Loading model with modified paths...', 'info');
                // Convert model JSON to a blob
                const modelJsonBlob = new Blob([JSON.stringify(modelJson)], {type: 'application/json'});
                const modelJsonUrl = URL.createObjectURL(modelJsonBlob);
                
                // Load from the blob URL (all paths are now absolute)
                model = await tf.loadGraphModel(modelJsonUrl, {
                    onProgress: (fraction) => {
                        debugLog(`Model loading progress: ${(fraction * 100).toFixed(1)}%`, 'info');
                    }
                });
                
                // Clean up the blob URL
                URL.revokeObjectURL(modelJsonUrl);
                
                debugLog('Model loaded successfully', 'success');
            } catch (loadError) {
                debugLog(`Error loading model with modified paths: ${loadError}`, 'error');
                throw loadError;
            }
            
            // Test the model with a small tensor
            debugLog('Testing model...', 'info');
            const dummyTensor = tf.zeros([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]);
            let testResult;
            
            try {
                testResult = await model.predict(dummyTensor);
                debugLog(`Model input shape: ${dummyTensor.shape}`, 'info');
                debugLog(`Model output shape: ${testResult.shape}`, 'info');
            } catch (predictError) {
                debugLog(`Model test prediction failed: ${predictError}`, 'error');
                throw new Error(`Model test failed: ${predictError.message}`);
            } finally {
                // Clean up tensors
                if (dummyTensor) dummyTensor.dispose();
                if (testResult) testResult.dispose();
            }
            
            isModelLoaded = true;
            debugLog('Model loaded and tested successfully', 'success');
            return true;
        } catch (error) {
            debugLog(`Error loading model: ${error}`, 'error');
            model = null;
            isModelLoaded = false;
            throw error;
        } finally {
            // Reset the promise so future calls can retry
            modelLoadingPromise = null;
        }
    })();
    
    return modelLoadingPromise;
}

// Object detection function
export async function detectObjects(imageData) {
    if (!model) {
        debugLog('Model not loaded yet', 'warning');
        return [];
    }
    
    try {
        // Use tf.tidy to clean up intermediate tensors
        const tensor = tf.tidy(() => {
            // Convert image to tensor
            const imageTensor = tf.browser.fromPixels(imageData);
            
            // Normalize and resize
            const normalized = tf.div(tf.cast(imageTensor, 'float32'), 255);
            const resized = tf.image.resizeBilinear(normalized, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
            
            // Add batch dimension [1, width, height, channels]
            return resized.expandDims(0);
        });
        
        // Run inference
        const predictions = await model.predict(tensor);
        tensor.dispose();
        
        // Convert to regular array
        const arrayPreds = await predictions.array();
        predictions.dispose();
        
        // Process detections into a simpler format
        return processDetections(arrayPreds[0]);
    } catch (error) {
        debugLog(`Error during detection: ${error.message}`, 'error');
        console.error('Detection error:', error);
        return [];
    }
}

// Process the raw model output into usable detections
function processDetections(modelOutput) {
    if (!modelOutput || modelOutput.length !== 5) {
        return [];
    }
    
    const [boxes, scores, classes, valid_detections] = modelOutput;
    const detections = [];
    
    // Get number of valid detections (usually comes in the 4th element)
    const numDetections = valid_detections ? valid_detections[0] : boxes[0].length;
    
    for (let i = 0; i < numDetections; i++) {
        const confidence = scores[0][i];
        
        if (confidence >= MIN_CONFIDENCE) {
            // Model outputs [y1, x1, y2, x2] normalized coordinates
            const box = boxes[0][i];
            
            detections.push({
                box: {
                    y1: box[0],
                    x1: box[1],
                    y2: box[2],
                    x2: box[3]
                },
                class: Math.round(classes[0][i]),
                confidence: confidence
            });
        }
    }
    
    return detections;
}

// Draw detection boxes on canvas
export function drawDetections(ctx, detections, canvasWidth, canvasHeight) {
    if (!detections || detections.length === 0) return;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Set detection drawing styles
    ctx.lineWidth = 3;
    ctx.font = '16px sans-serif';
    ctx.textBaseline = 'bottom';
    
    detections.forEach(detection => {
        const { box, class: classId, confidence } = detection;
        
        // Convert normalized coordinates to canvas coordinates
        const x = box.x1 * canvasWidth;
        const y = box.y1 * canvasHeight;
        const width = (box.x2 - box.x1) * canvasWidth;
        const height = (box.y2 - box.y1) * canvasHeight;
        
        // Draw bounding box
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.stroke();
        
        // Draw label background
        const label = `Ball: ${Math.round(confidence * 100)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
        ctx.fillRect(x, y - 20, textWidth + 10, 20);
        
        // Draw text
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 5, y);
    });
}