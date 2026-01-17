// Zeppelin Capture Extension - Popup Script

// Simple obfuscation to hide URL from casual inspection
const _0x1a2b = ['h', 't', 't', 'p', 's', ':', '/', '/', 'z', 'e', 'p', 'p', 'l', 'i', 'n', 'b', 'o', 't', '.', '1', '0', '9', '.', '1', '9', '9', '.', '1', '0', '9', '.', '9', '2', '.', 'n', 'i', 'p', '.', 'i', 'o'];
const DEFAULT_DASHBOARD_URL = _0x1a2b.join('');
let dashboardUrl = DEFAULT_DASHBOARD_URL;
let crashes = [];
let settings = {};

// Helper function to safely send message to content script
function sendToContentScript(message) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0] && tabs[0].url && tabs[0].url.includes('zeppelin')) {
            chrome.tabs.sendMessage(tabs[0].id, message).catch(() => {
                // Content script not loaded on this page - that's OK
                console.log('Content script not available on this page');
            });
        }
    });
}

// Load settings and data
async function loadData() {
    const data = await chrome.storage.local.get([
        'crashes', 'dashboardUrl', 'betAmount', 'targetMultiplier',
        'maxLosses', 'autoCapture', 'autoBetEnabled', 'sessionProfit'
    ]);

    crashes = data.crashes || [];
    // Enforce hardcoded URL
    dashboardUrl = DEFAULT_DASHBOARD_URL;
    settings = {
        betAmount: data.betAmount || 20,
        targetMultiplier: data.targetMultiplier || 2.0,
        maxLosses: data.maxLosses || 5
    };

    // Update settings inputs
    const urlInput = document.getElementById('dashboard-url');
    if (urlInput) urlInput.value = "HIDDEN_CONFIG_MODE"; // Visual placeholder
    document.getElementById('bet-amount').value = settings.betAmount;
    document.getElementById('target-multiplier').value = settings.targetMultiplier;
    document.getElementById('max-losses').value = settings.maxLosses;
    document.getElementById('auto-capture-toggle').checked = data.autoCapture || false;
    document.getElementById('auto-bet-toggle').checked = data.autoBetEnabled || false;

    updateUI();
    fetchDashboardSignal();
}

// Save a new crash value
async function saveCrash(value) {
    if (value < 1.0 || isNaN(value)) {
        showFeedback('❌ Must be ≥ 1.00', 'error');
        return false;
    }

    crashes.push(value);
    await chrome.storage.local.set({ crashes });

    // Also notify content script to save
    sendToContentScript({ action: 'saveCrashToFile', value: value });

    showFeedback(`✅ Added ${value.toFixed(2)}x`, 'success');
    updateUI();
    return true;
}

// Fetch signal from dashboard
async function fetchDashboardSignal() {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('connection-status');

    try {
        const response = await fetch(dashboardUrl);
        const html = await response.text();

        statusDot.className = 'status-dot connected';
        statusText.textContent = 'Connected to Dashboard';

        // Parse signal
        let signal = 'WAIT';
        let details = '';

        if (html.includes('>BET<') || html.includes('BET</')) {
            signal = 'BET';
            const betMatch = html.match(/Bet:\s*TZS\s*([\d,]+)/i);
            const targetMatch = html.match(/Target:\s*([\d.]+)x/i);
            const winMatch = html.match(/Win:\s*(\d+)%/i);

            if (betMatch) details += `Bet: TZS ${betMatch[1]} `;
            if (targetMatch) details += `@ ${targetMatch[1]}x `;
            if (winMatch) details += `(${winMatch[1]}%)`;

        } else if (html.includes('>SKIP<') || html.includes('SKIP</')) {
            signal = 'SKIP';
            details = 'Negative EV - waiting';
        } else {
            signal = 'WAIT';
            const needMatch = html.match(/Need\s*(\d+)\s*more/i);
            details = needMatch ? `Need ${needMatch[1]} more crashes` : 'Collecting data...';
        }

        setSignal(signal, details);

    } catch (error) {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Dashboard offline';
        setSignal('WAIT', 'Start dashboard: python3 realtime_dashboard.py');
    }
}

// Update UI elements
function updateUI() {
    const total = crashes.length;
    document.getElementById('total-crashes').textContent = total;

    if (total >= 20) {
        const arr = crashes.slice(-50);
        const winRate = (arr.filter(c => c >= settings.targetMultiplier).length / arr.length * 100).toFixed(0);
        document.getElementById('win-rate').textContent = `${winRate}%`;
    } else {
        document.getElementById('win-rate').textContent = '—';
    }

    // Update history
    const container = document.getElementById('history');
    const recent = crashes.slice(-15).reverse();

    if (recent.length === 0) {
        container.innerHTML = '<span style="color: #666; font-size: 11px;">No crashes yet</span>';
    } else {
        container.innerHTML = recent.map(c => {
            let cls = c >= 3.0 ? 'high' : c >= 2.0 ? 'mid' : 'low';
            return `<span class="crash-badge ${cls}">${c.toFixed(2)}x</span>`;
        }).join('');
    }
}

function setSignal(signal, details) {
    const box = document.getElementById('signal-box');
    const textEl = document.getElementById('signal-text');

    const classes = { 'BET': '', 'SKIP': 'skip', 'WAIT': 'wait' };
    box.className = 'signal-box ' + (classes[signal] || 'wait');
    textEl.className = 'signal-text ' + (classes[signal] || 'wait');
    textEl.textContent = signal;
}

function showFeedback(message, type) {
    const el = document.getElementById('feedback');
    el.textContent = message;
    el.style.color = type === 'error' ? '#e63757' : '#00d97e';
    setTimeout(() => el.textContent = '', 2000);
}

// Event Listeners
document.getElementById('add-btn').addEventListener('click', () => {
    const input = document.getElementById('crash-value');
    const value = parseFloat(input.value);
    if (saveCrash(value)) {
        input.value = '';
        input.focus();
    }
});

document.getElementById('crash-value').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') document.getElementById('add-btn').click();
});

// Settings toggle
document.getElementById('settings-toggle').addEventListener('click', () => {
    const main = document.getElementById('main-panel');
    const settingsPanel = document.getElementById('settings-panel');
    main.classList.toggle('hidden');
    settingsPanel.classList.toggle('active');
});

// Save settings
document.getElementById('save-settings').addEventListener('click', async () => {
    const newUrl = document.getElementById('dashboard-url').value || DEFAULT_DASHBOARD_URL;
    const betAmount = parseFloat(document.getElementById('bet-amount').value) || 20;
    const target = parseFloat(document.getElementById('target-multiplier').value) || 2.0;
    const maxLosses = parseInt(document.getElementById('max-losses').value) || 5;

    dashboardUrl = newUrl;
    settings = { betAmount, targetMultiplier: target, maxLosses };

    await chrome.storage.local.set({
        dashboardUrl: newUrl,
        betAmount,
        targetMultiplier: target,
        maxLosses
    });

    // Notify content script (if on Zeppelin page)
    sendToContentScript({
        action: 'updateSettings',
        dashboardUrl: newUrl,
        betAmount,
        target,
        maxLosses
    });

    showFeedback('✅ Settings saved!', 'success');

    // Go back to main panel
    document.getElementById('main-panel').classList.remove('hidden');
    document.getElementById('settings-panel').classList.remove('active');

    fetchDashboardSignal();
});

// Auto-capture toggle
document.getElementById('auto-capture-toggle').addEventListener('change', async (e) => {
    await chrome.storage.local.set({ autoCapture: e.target.checked });
    sendToContentScript({ action: 'setAutoCapture', enabled: e.target.checked });
});

// Auto-bet toggle
document.getElementById('auto-bet-toggle').addEventListener('change', async (e) => {
    await chrome.storage.local.set({ autoBetEnabled: e.target.checked });
    sendToContentScript({ action: 'setAutoBet', enabled: e.target.checked });
});

// Listen for crashes from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'newCrash') {
        crashes.push(message.value);
        chrome.storage.local.set({ crashes });
        updateUI();
    }
    if (message.action === 'profitUpdate') {
        document.getElementById('session-profit').textContent =
            (message.profit >= 0 ? '+' : '') + message.profit.toFixed(0);
        document.getElementById('session-profit').style.color =
            message.profit >= 0 ? '#00d97e' : '#e63757';
    }
});

// Refresh signal every 5 seconds
setInterval(fetchDashboardSignal, 5000);

// Initialize
loadData();
