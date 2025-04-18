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

export function debugLog(debugPanel, message, type = 'info') {
    const colors = {
        info: '#fff',
        error: '#ff4444',
        success: '#44ff44',
        warning: '#ffff44'
    };
    
    const entry = document.createElement('div');
    entry.style.color = colors[type] || colors.info;
    entry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
    debugPanel.insertBefore(entry, debugPanel.firstChild);
    
    // Keep only last 20 messages
    while (debugPanel.children.length > 20) {
        debugPanel.removeChild(debugPanel.lastChild);
    }
    
    // Also log to console
    console.log(message);
} 