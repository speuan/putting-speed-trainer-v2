// Debug panel setup and logging utilities
export function createDebugPanel() {
    const debugPanel = document.createElement('div');
    debugPanel.style.cssText = `
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 150px;
        background: rgba(0, 0, 0, 0.8);
        color: #fff;
        font-family: monospace;
        font-size: 12px;
        padding: 10px;
        overflow-y: auto;
        z-index: 1000;
    `;
    document.body.appendChild(debugPanel);
    return debugPanel;
}

// Debug logging utility
export function debugLog(message, type = 'info') {
    // Try to find the debug panel
    const debugPanel = document.getElementById('debugPanel');
    if (!debugPanel) {
        console.log(`${type.toUpperCase()}: ${message}`);
        return;
    }
    
    // Color-coded message types
    const colors = {
        info: '#fff',
        error: '#ff4444',
        success: '#44ff44',
        warning: '#ffff44'
    };
    
    // Create and add entry to debug panel
    const entry = document.createElement('div');
    entry.style.color = colors[type] || colors.info;
    entry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
    debugPanel.insertBefore(entry, debugPanel.firstChild);
    
    // Keep only last 50 messages (increased for iOS debugging)
    while (debugPanel.children.length > 50) {
        debugPanel.removeChild(debugPanel.lastChild);
    }
    
    // Also log to console with type prefix
    console.log(`${type.toUpperCase()}: ${message}`);
    
    // Add special handling for errors to make sure they're visible
    if (type === 'error') {
        console.error(message);
        
        // On iOS, ensure errors are visible by logging stack traces
        if (/iPad|iPhone|iPod/.test(navigator.userAgent) || 
            (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)) {
            console.error('iOS Error Details:', new Error(message).stack);
        }
    }
}

// Log object utility (for complex structures)
export function debugLogObject(label, obj, type = 'info') {
    try {
        // Convert object to string with pretty formatting
        const objString = JSON.stringify(obj, null, 2);
        debugLog(`${label}: ${objString}`, type);
    } catch (error) {
        debugLog(`${label}: [Unable to stringify object: ${error.message}]`, 'error');
    }
}

// Performance timing utility
export function debugTime(label) {
    console.time(label);
    return () => {
        console.timeEnd(label);
    };
}

// Export iOS detection utility for use elsewhere
export function isIOS() {
    return /iPad|iPhone|iPod/.test(navigator.userAgent) || 
           (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
} 