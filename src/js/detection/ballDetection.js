import { debugLog } from '../utils/debug.js';

// Constants
export const MODEL_INPUT_SIZE = 640;
export const MIN_CONFIDENCE = 0.7;
export const IOU_THRESHOLD = 0.2;
export const PROCESS_EVERY_N_FRAMES = 3;

let model = null;
let isModelLoaded = false;
let lastProcessedDetections = null;

export async function loadModel(debugPanel) {
    try {
        debugLog(debugPanel, 'Loading model...', 'info');
        debugLog(debugPanel, `TensorFlow.js version: ${tf.version.tfjs}`, 'info');
        debugLog(debugPanel, `Backend: ${tf.getBackend()}`, 'info');
        
        const modelUrl = './src/assets/models/model.json';
        debugLog(debugPanel, `Attempting to load model from: ${modelUrl}`, 'info');
        
        try {
            const response = await fetch(modelUrl);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const modelJson = await response.json();
            debugLog(debugPanel, 'Model JSON loaded successfully', 'success');
        } catch (fetchError) {
            debugLog(debugPanel, `Error checking model.json: ${fetchError}`, 'error');
            throw new Error('Could not access model.json file');
        }
        
        model = await tf.loadGraphModel(modelUrl, {
            onProgress: (fraction) => {
                debugLog(debugPanel, `Model loading progress: ${(fraction * 100).toFixed(1)}%`, 'info');
            }
        });
        
        debugLog(debugPanel, 'Testing model...', 'info');
        const dummyTensor = tf.zeros([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]);
        const testResult = await model.predict(dummyTensor);
        debugLog(debugPanel, `Model input shape: ${dummyTensor.shape}`, 'info');
        debugLog(debugPanel, `Model output shape: ${testResult.shape}`, 'info');
        dummyTensor.dispose();
        testResult.dispose();
        
        isModelLoaded = true;
        debugLog(debugPanel, 'Model loaded and tested successfully', 'success');
        
        return true;
    } catch (error) {
        debugLog(debugPanel, `Error loading model: ${error.message}`, 'error');
        debugLog(debugPanel, `Error stack: ${error.stack}`, 'error');
        throw error;
    }
}

export async function detectBall(debugPanel, imageData) {
    if (!model) {
        debugLog(debugPanel, 'Model not loaded yet', 'warning');
        return null;
    }
    
    try {
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
        
        const detections = processDetections(arrayPreds[0], MODEL_INPUT_SIZE);
        
        if (detections) {
            lastProcessedDetections = detections;
        }
        
        return detections;
    } catch (error) {
        debugLog(debugPanel, `Error during detection: ${error.message}`, 'error');
        return null;
    }
}

function processDetections(predictions, inputSize) {
    if (!predictions || predictions.length !== 5) {
        return null;
    }
    
    const detections = [];
    for (let i = 0; i < predictions[0].length; i++) {
        const confidence = predictions[4][i];
        if (confidence > MIN_CONFIDENCE) {
            detections.push({
                x: predictions[0][i] / inputSize,
                y: predictions[1][i] / inputSize,
                w: predictions[2][i] / inputSize,
                h: predictions[3][i] / inputSize,
                confidence: confidence
            });
        }
    }
    
    if (detections.length === 0) {
        lastProcessedDetections = null;
        return null;
    }
    
    const clusters = clusterDetections(detections);
    
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
    const box1Left = box1.x - box1.w/2;
    const box1Right = box1.x + box1.w/2;
    const box1Top = box1.y - box1.h/2;
    const box1Bottom = box1.y + box1.h/2;
    
    const box2Left = box2.x - box2.w/2;
    const box2Right = box2.x + box2.w/2;
    const box2Top = box2.y - box2.h/2;
    const box2Bottom = box2.y + box2.h/2;
    
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