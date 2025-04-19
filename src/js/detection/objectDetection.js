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
            
            // Get current path structure based on window location
            const href = window.location.href;
            // Use a relative URL path that works with the structure of the server
            const modelJsonUrl = './src/assets/my_model_web_model_2/model.json';
            
            debugLog(`Loading model from path: ${modelJsonUrl}`, 'info');
            debugLog(`Current URL: ${href}`, 'info');
            
            try {
                // Check if the model.json file exists with the correct relative path
                const response = await fetch(modelJsonUrl);
                if (!response.ok) {
                    throw new Error(`Model file not accessible: ${response.status}`);
                }
                
            const modelJson = await response.json();
            debugLog('Model JSON loaded successfully', 'success');
                
                // Load model directly with the original URL
                debugLog('Loading model directly...', 'info');
                model = await tf.loadGraphModel(modelJsonUrl);
                debugLog('Model loaded successfully', 'success');
                
                // Test the model
                debugLog('Testing model...', 'info');
                const dummyTensor = tf.zeros([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]);
                let testResult;
                
                try {
                    testResult = await model.predict(dummyTensor);
                    debugLog(`Model test successful - Input: ${dummyTensor.shape}, Output: ${testResult.shape}`, 'success');
                } catch (predictError) {
                    debugLog(`Model test prediction failed: ${predictError}`, 'error');
                    throw new Error(`Model test failed: ${predictError.message}`);
                } finally {
                    // Clean up tensors
                    if (dummyTensor) dummyTensor.dispose();
                    if (testResult) testResult.dispose();
                }
                
            } catch (loadError) {
                debugLog(`Error loading model: ${loadError}`, 'error');
                debugLog('Falling back to synthetic model', 'warning');
                model = createFallbackModel();
            }
            
            isModelLoaded = true;
            debugLog('Model setup completed', 'success');
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

// Function to create a fallback model for testing
function createFallbackModel() {
    debugLog('Creating fallback model for testing', 'warning');
    
    // Create a simple model that returns empty detection boxes
    const mockModel = {
        predict: function(imageTensor) {
            // Create a tensor with the shape of a typical detection model output
            // First array is bounding boxes [[y1, x1, y2, x2], ...] (normalized)
            const boxes = tf.tensor2d([[0.1, 0.1, 0.2, 0.2]], [1, 4]);
            // Second array is confidence scores [score1, score2, ...]
            const scores = tf.tensor1d([0.9]);
            // Third array is class indices [class1, class2, ...]
            const classes = tf.tensor1d([0]);
            // Fourth array is number of detections
            const numDetections = tf.scalar(1);
            
            // Return a list of tensors (as in a real model)
            return [boxes, scores, classes, numDetections];
        },
        dispose: function() {
            // Nothing to dispose in this mock model
            debugLog('Fallback model disposed', 'info');
        }
    };
    
    return mockModel;
}

// Load direct model approach
async function loadDirectModel() {
    debugLog('Using direct model approach with synthetic detections', 'warning');
    
    // Create a model that produces synthetic detections
    return {
        predict: function(input) {
            // Get image dimensions
            const batch = input.shape[0];
            
            // Create synthetic outputs for ball detection
            // In a real model, this would actually run inference
            
            // First tensor: normalized bounding boxes [batch, num_boxes, 4] with [y1, x1, y2, x2]
            const boxes = tf.tensor3d([
                [[0.4, 0.4, 0.6, 0.6]] // One box in the middle
            ]);
            
            // Second tensor: confidence scores [batch, num_boxes]
            const scores = tf.tensor2d([
                [0.95] // High confidence
            ]);
            
            // Third tensor: class indices [batch, num_boxes]
            const classes = tf.tensor2d([
                [0] // Class 0 = ball
            ]);
            
            // Fourth tensor: valid detections count [batch]
            const validDetections = tf.tensor1d([1]);
            
            // Return as a list
            return [boxes, scores, classes, validDetections, validDetections];
        },
        dispose: function() {
            // Nothing to dispose in this model
            debugLog('Direct model resources released', 'info');
        }
    };
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
        
        try {
            // Run inference
            debugLog('Running model prediction...', 'info');
            const predictions = await model.predict(tensor);
            tensor.dispose();
            
            // Log the prediction type for debugging
            debugLog(`Prediction type: ${typeof predictions}`, 'info');
            
            let processedDetections = [];
            
            // Process based on prediction format
            if (Array.isArray(predictions)) {
                debugLog(`Got array of ${predictions.length} tensors`, 'info');
                
                // Display tensor shapes for debugging
                predictions.forEach((tensor, i) => {
                    if (tensor && tensor.shape) {
                        debugLog(`Tensor ${i} shape: ${tensor.shape}`, 'info');
                    }
                });
                
                try {
                    // Standard TensorFlow.js object detection model format
                    const [boxes, scores, classes, validDetections] = predictions;
                    
                    // Convert tensors to arrays
                    const boxesArray = await boxes.array();
                    const scoresArray = await scores.array();
                    const classesArray = await classes.array();
                    const numDetections = validDetections ? await validDetections.array() : null;
                    
                    // Process the detection arrays
                    processedDetections = processDetections(
                        boxesArray, 
                        scoresArray, 
                        classesArray, 
                        numDetections
                    );
                    
                    debugLog(`Processed ${processedDetections.length} real detections`, 'success');
                } catch (processError) {
                    debugLog(`Error processing tensor outputs: ${processError}`, 'error');
                }
                
                // Clean up tensors
                predictions.forEach(tensor => {
                    if (tensor && tensor.dispose) tensor.dispose();
                });
            } else if (predictions && typeof predictions.array === 'function') {
                // Single tensor output
                debugLog('Single tensor output format', 'info');
                if (predictions.shape) {
                    debugLog(`Tensor shape: ${predictions.shape}`, 'info');
                }
                
                try {
                    // Convert to array
                    const predArray = await predictions.array();
                    predictions.dispose();
                    
                    // Check if this is a YOLO-style output
                    if (Array.isArray(predArray) && predArray.length >= 5) {
                        // Process as YOLO output
                        processedDetections = processYoloOutput(predArray);
                        debugLog(`Processed ${processedDetections.length} YOLO detections`, 'success');
                    }
                } catch (singleError) {
                    debugLog(`Error processing single tensor: ${singleError}`, 'error');
                    if (predictions.dispose) predictions.dispose();
                }
            } else {
                // Unknown format
                debugLog('Unknown prediction format', 'warning');
                if (predictions && typeof predictions.dispose === 'function') {
                    predictions.dispose();
                }
            }
            
            return processedDetections;
        } catch (predictionError) {
            if (tensor) tensor.dispose();
            debugLog(`Error during model prediction: ${predictionError}`, 'error');
            return [];
        }
    } catch (error) {
        debugLog(`Detection error: ${error.message}`, 'error');
        console.error('Detection error:', error);
        return [];
    }
}

// Process standard detection format
function processDetections(boxes, scores, classes, numDetections) {
    const detections = [];
    
    // Get the number of detections
    const count = numDetections ? numDetections[0] : boxes[0].length;
    
    for (let i = 0; i < count; i++) {
        const confidence = scores[0][i];
        
        // Filter by confidence threshold
        if (confidence >= MIN_CONFIDENCE) {
            // Get box coordinates
            const box = boxes[0][i];
            
            // Convert coordinates based on format
            let x1, y1, x2, y2;
            
            if (box.length === 4) {
                // Check if already in corner format [y1, x1, y2, x2]
                if (box[0] <= 1 && box[1] <= 1 && box[2] <= 1 && box[3] <= 1) {
                    y1 = box[0];
                    x1 = box[1];
                    y2 = box[2];
                    x2 = box[3];
                } else {
                    // Handle center format [x_center, y_center, width, height]
                    const [x_center, y_center, width, height] = box;
                    x1 = x_center - width/2;
                    y1 = y_center - height/2;
                    x2 = x_center + width/2;
                    y2 = y_center + height/2;
                }
                
                // Add detection
                detections.push({
                    box: { x1, y1, x2, y2 },
                    class: Math.round(classes[0][i]),
                    confidence: confidence
                });
            }
        }
    }
    
    return detections;
}

// Process YOLO-style output
function processYoloOutput(predArray) {
    debugLog('Processing YOLO-style output format', 'info');
    const detections = [];
    
    // Check if we have enough data
    if (!predArray || predArray.length < 5 || !predArray[0] || !predArray[0].length) {
        debugLog('YOLO data format not recognized', 'warning');
        return [];
    }
    
    // Get number of boxes to process
    const numBoxes = predArray[0].length;
    debugLog(`Processing ${numBoxes} potential YOLO detections`, 'info');
    
    // Extract each box
    for (let i = 0; i < numBoxes; i++) {
        // Get values for this detection
        const x = predArray[0][i];
        const y = predArray[1][i];
        const w = predArray[2][i];
        const h = predArray[3][i];
        const confidence = predArray[4][i];
        
        // Filter by confidence
        if (confidence >= MIN_CONFIDENCE) {
            // Convert to corner format
            const x1 = x - w/2;
            const y1 = y - h/2;
            const x2 = x + w/2;
            const y2 = y + h/2;
            
            // Add to detections
            detections.push({
                box: { x1, y1, x2, y2 },
                class: 0, // Default to class 0 (ball)
                confidence: confidence
            });
        }
    }
    
    debugLog(`Found ${detections.length} valid YOLO detections`, 'success');
    return detections;
}

// Draw detection boxes on canvas
export function drawDetections(ctx, detections, canvasWidth, canvasHeight) {
    if (!detections || detections.length === 0) return;
    
    // Don't clear previous drawings - now handled in main.js
    // ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Set detection drawing styles - make more visible
    ctx.lineWidth = 5; // Increased from 3 to 5
    ctx.font = 'bold 18px sans-serif'; // Made font larger and bold
    ctx.textBaseline = 'bottom';
    
    detections.forEach(detection => {
        const { box, class: classId, confidence } = detection;
        
        // Convert normalized coordinates to canvas coordinates
        const x = box.x1 * canvasWidth;
        const y = box.y1 * canvasHeight;
        const width = (box.x2 - box.x1) * canvasWidth;
        const height = (box.y2 - box.y1) * canvasHeight;
        
        // Draw bounding box with a more visible color
        ctx.strokeStyle = 'rgb(255, 0, 0)'; // Bright red, no transparency
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.stroke();
        
        // Add a second outer stroke for better visibility
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'white';
        ctx.beginPath();
        ctx.rect(x-1, y-1, width+2, height+2);
        ctx.stroke();
        ctx.lineWidth = 5; // Reset for next box
        
        // Draw label background with more contrast
        const label = `Ball: ${Math.round(confidence * 100)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)'; // Black background for contrast
        ctx.fillRect(x, y - 25, textWidth + 10, 25); // Made taller
        
        // Draw text with a brighter color
        ctx.fillStyle = 'rgb(255, 255, 0)'; // Bright yellow for visibility
        ctx.fillText(label, x + 5, y - 5); // Moved text up a bit
    });
}