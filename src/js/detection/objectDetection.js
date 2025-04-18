import { debugLog } from '../utils/debug.js';

// Constants
export const MODEL_INPUT_SIZE = 640;
export const MIN_CONFIDENCE = 0.5;

let model = null;
let isModelLoaded = false;

export async function loadModel() {
    try {
        debugLog('Loading model...', 'info');
        debugLog(`TensorFlow.js version: ${tf.version.tfjs}`, 'info');
        debugLog(`Backend: ${tf.getBackend()}`, 'info');
        
        const modelUrl = './src/assets/my_model_web_model 2/model.json';
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
        
        model = await tf.loadGraphModel(modelUrl);
        
        // Test the model with a dummy tensor
        debugLog('Testing model...', 'info');
        const dummyTensor = tf.zeros([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]);
        const testResult = await model.predict(dummyTensor);
        debugLog(`Model input shape: ${dummyTensor.shape}`, 'info');
        debugLog(`Model output shape: ${testResult.shape}`, 'info');
        dummyTensor.dispose();
        testResult.dispose();
        
        isModelLoaded = true;
        debugLog('Model loaded and tested successfully', 'success');
        return true;
    } catch (error) {
        debugLog(`Error loading model: ${error.message}`, 'error');
        debugLog(`Error stack: ${error.stack}`, 'error');
        throw error;
    }
}

export async function detectObjects(imageData) {
    if (!model || !isModelLoaded) {
        debugLog('Model not loaded yet', 'warning');
        return null;
    }
    
    try {
        // Process image using tf.tidy to automatically clean up tensors
        const tensor = tf.tidy(() => {
            const imageTensor = tf.browser.fromPixels(imageData);
            const normalized = tf.div(tf.cast(imageTensor, 'float32'), 255);
            const resized = tf.image.resizeBilinear(normalized, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
            return resized.expandDims(0);
        });
        
        const predictions = await model.predict(tensor);
        tensor.dispose();
        
        const arrayPreds = await predictions.array();
        predictions.dispose();
        
        // Process detections
        const detections = processDetections(arrayPreds[0]);
        return detections;
    } catch (error) {
        debugLog(`Error during detection: ${error.message}`, 'error');
        return null;
    }
}

function processDetections(predictions) {
    const detections = [];
    
    // Each prediction row contains [x, y, width, height, confidence, ...class_scores]
    for (const pred of predictions) {
        const [x, y, width, height, confidence, ...classScores] = pred;
        
        if (confidence > MIN_CONFIDENCE) {
            detections.push({
                x: x,
                y: y,
                width: width,
                height: height,
                confidence: confidence,
                class: classScores.indexOf(Math.max(...classScores))
            });
        }
    }
    
    return detections;
}

export function drawDetections(ctx, detections, canvasWidth, canvasHeight) {
    // Clear previous drawings
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Draw each detection
    for (const det of detections) {
        // Convert normalized coordinates to canvas coordinates
        const x = det.x * canvasWidth;
        const y = det.y * canvasHeight;
        const width = det.width * canvasWidth;
        const height = det.height * canvasHeight;
        
        // Draw bounding box
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - width/2, y - height/2, width, height);
        
        // Draw label
        ctx.fillStyle = '#00ff00';
        ctx.font = '16px Arial';
        ctx.fillText(
            `${det.class === 0 ? 'Ball' : 'Coin'} ${(det.confidence * 100).toFixed(1)}%`,
            x - width/2,
            y - height/2 - 5
        );
    }
} 