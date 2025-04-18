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
        
        const modelUrl = 'src/assets/my_model_web_model_2/model.json';
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
// ... existing code ...
    } catch (error) {
        debugLog(`Error loading model: ${error}`, 'error');
        throw error;
    }
}