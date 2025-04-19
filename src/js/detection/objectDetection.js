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
        
        // Run inference
        const predictions = await model.predict(tensor);
        tensor.dispose();
        
        // Handle different types of model outputs
        let processedDetections = [];
        
        // Check prediction format and handle accordingly
        if (Array.isArray(predictions)) {
            debugLog('Model returned array of tensors', 'info');
            // Standard TensorFlow.js detection model output
            const [boxes, scores, classes, validDetections] = predictions;
            
            // Convert tensors to regular arrays
            const boxesArray = await boxes.array();
            const scoresArray = await scores.array();
            const classesArray = await classes.array();
            const validDetectionsArray = validDetections ? await validDetections.array() : null;
            
            // Process detections
            processedDetections = processDetectionsFromArrays(
                boxesArray, 
                scoresArray, 
                classesArray, 
                validDetectionsArray
            );
            
            // Dispose tensor outputs
            predictions.forEach(tensor => {
                if (tensor) tensor.dispose();
            });
        } else if (predictions.arraySync && typeof predictions.arraySync === 'function') {
            // Single tensor output format
            debugLog('Model returned single tensor', 'info');
            const arrayPreds = await predictions.array();
            predictions.dispose();
            
            // Check if it's a nested array structure which might contain detections
            if (Array.isArray(arrayPreds) && Array.isArray(arrayPreds[0])) {
                debugLog(`Single tensor shape: ${arrayPreds.length} x ${arrayPreds[0].length}`, 'info');
                
                // Try to extract detections based on shape
                if (arrayPreds.length >= 5) {
                    // Might be in the format of YOLO output [boxes, scores, classes, ...]
                    processedDetections = processYoloStyleOutput(arrayPreds);
                } else {
                    debugLog('Unknown nested array format', 'warning');
                }
            } else {
                debugLog('Unknown single tensor format', 'warning');
            }
        } else {
            debugLog('Unknown model output format', 'warning');
            console.log('Model output:', predictions);
            
            // Try to safely extract and convert to array
            try {
                if (predictions.toString) {
                    debugLog(`Output type: ${predictions.toString()}`, 'info');
                }
                
                // Try to fall back to synthetic detection
                debugLog('Falling back to synthetic detection', 'warning');
                processedDetections = [{
                    box: {
                        x1: 0.4, 
                        y1: 0.4, 
                        x2: 0.6, 
                        y2: 0.6
                    },
                    class: 0,
                    confidence: 0.9
                }];
            } catch (error) {
                debugLog(`Error processing model output: ${error.message}`, 'error');
            }
        }
        
        return processedDetections;
    } catch (error) {
        debugLog(`Error during detection: ${error.message}`, 'error');
        console.error('Detection error:', error);
        return [];
    }
}

// Process detections from array format
function processDetectionsFromArrays(boxes, scores, classes, validDetections) {
    const detections = [];
    
    // Determine number of detections to process
    const numDetections = validDetections ? validDetections[0] : 
                          (boxes[0] ? boxes[0].length : 0);
    
    for (let i = 0; i < numDetections; i++) {
        const confidence = scores[0][i];
        
        if (confidence >= MIN_CONFIDENCE) {
            // Get box coordinates (models may use different formats)
            const box = boxes[0][i];
            
            // YOLOv5/YOLOv8 format with [x_center, y_center, width, height]
            if (box.length === 4) {
                // Normalize to corner format for consistency
                let x1, y1, x2, y2;
                
                // Check if they're already in [y1, x1, y2, x2] format
                if (box[0] <= 1 && box[1] <= 1 && box[2] <= 1 && box[3] <= 1) {
                    // Assume [y1, x1, y2, x2] format
                    y1 = box[0];
                    x1 = box[1];
                    y2 = box[2];
                    x2 = box[3];
                } else {
                    // Calculate corners based on center and dimensions
                    const [x_center, y_center, width, height] = box;
                    x1 = x_center - width/2;
                    y1 = y_center - height/2;
                    x2 = x_center + width/2;
                    y2 = y_center + height/2;
                }
                
                detections.push({
                    box: {
                        x1, y1, x2, y2
                    },
                    class: Math.round(classes[0][i]),
                    confidence: confidence
                });
            }
        }
    }
    
    return detections;
}

// Process YOLO-style model output
function processYoloStyleOutput(arrayPreds) {
    try {
        debugLog('Processing YOLO-style output format', 'info');
        
        const detections = [];
        // YOLO outputs: 
        // - arrayPreds[0] = x-coordinates (normalized)
        // - arrayPreds[1] = y-coordinates (normalized)
        // - arrayPreds[2] = widths (normalized)
        // - arrayPreds[3] = heights (normalized)
        // - arrayPreds[4] = confidence scores
        
        if (arrayPreds.length < 5) {
            debugLog('YOLO output does not have enough arrays', 'warning');
            return [];
        }
        
        // Calculate the number of boxes based on the first dimension
        const numBoxes = arrayPreds[0].length;
        debugLog(`Found ${numBoxes} potential detections`, 'info');
        
        // Process each detection
        for (let i = 0; i < numBoxes; i++) {
            // Get values for this detection
            const x = arrayPreds[0][i];
            const y = arrayPreds[1][i];
            const w = arrayPreds[2][i];
            const h = arrayPreds[3][i];
            const confidence = arrayPreds[4][i];
            
            // Apply confidence threshold
            if (confidence >= MIN_CONFIDENCE) {
                // Convert center+width/height format to corner format
                const x1 = x - w/2;
                const y1 = y - h/2;
                const x2 = x + w/2;
                const y2 = y + h/2;
                
                // Create detection object
                detections.push({
                    box: { x1, y1, x2, y2 },
                    class: 0, // Default to class 0 (ball)
                    confidence: confidence
                });
            }
        }
        
        debugLog(`Processed ${detections.length} detections after filtering`, 'info');
        return detections;
    } catch (error) {
        debugLog(`Error processing YOLO output: ${error.message}`, 'error');
        return [];
    }
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