// Zeppelin Capture - Background Service Worker
// Handles CORS-free fetch to dashboard and syncs crashes

// Default dashboard URL
let dashboardUrl = 'http://localhost:8050';

// Load saved URL
chrome.storage.local.get(['dashboardUrl'], (data) => {
    if (data.dashboardUrl) {
        dashboardUrl = data.dashboardUrl;
    }
});

// Listen for storage changes
chrome.storage.onChanged.addListener((changes) => {
    if (changes.dashboardUrl) {
        dashboardUrl = changes.dashboardUrl.newValue;
        console.log('📡 Dashboard URL updated:', dashboardUrl);
    }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

    // Fetch signal from dashboard
    if (message.action === 'fetchDashboard') {
        const url = message.url || dashboardUrl;

        fetch(`${url}/api/signal`)
            .then(response => response.json())
            .then(data => {
                let signal = data.signal || 'WAIT';
                let betAmount = 20;
                let targetMultiplier = 2.0;

                if (signal === 'BET' && data.bet) {
                    betAmount = data.bet;
                }
                if (data.target) {
                    targetMultiplier = data.target;
                }

                console.log(`📡 Signal from dashboard: ${signal}`);
                sendResponse({
                    success: true,
                    signal,
                    betAmount,
                    targetMultiplier,
                    prob: data.prob || 0
                });
            })
            .catch(error => {
                console.log('⚠️ Dashboard fetch failed:', error.message);
                sendResponse({ success: false, error: error.message });
            });

        return true; // Keep message channel open for async response
    }

    // Send crash to dashboard
    if (message.action === 'sendCrashToDashboard') {
        const url = message.url || dashboardUrl;
        const value = message.value;

        fetch(`${url}/api/crash`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value: value })
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`✅ Crash ${value}x sent to dashboard (total: ${data.total})`);
                }
                sendResponse(data);
            })
            .catch(error => {
                console.log('⚠️ Failed to send crash to dashboard:', error.message);
                sendResponse({ success: false, error: error.message });
            });

        return true;
    }

    // Handle new crash from content script
    if (message.action === 'newCrash') {
        const value = message.value;

        // Save to local storage
        chrome.storage.local.get(['crashes'], (data) => {
            const crashes = data.crashes || [];
            crashes.push(value);

            // Keep only last 200
            if (crashes.length > 200) {
                crashes.splice(0, crashes.length - 200);
            }

            chrome.storage.local.set({ crashes });
        });

        // Also send to dashboard API
        fetch(`${dashboardUrl}/api/crash`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value: value })
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log(`📊 Crash ${value}x synced to dashboard`);
                }
            })
            .catch(error => {
                console.log('⚠️ Could not sync crash to dashboard:', error.message);
            });
    }

    // Send betting behavior to dashboard (CORS-free)
    if (message.action === 'sendBettingBehavior') {
        const url = message.url || dashboardUrl;
        const data = message.data;

        fetch(`${url}/api/betting-behavior`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    console.log(`📊 Betting behavior sent: ${data.totalBettors} bettors, ${data.poolSize} pool`);
                }
                sendResponse(result);
            })
            .catch(error => {
                console.log('⚠️ Could not send betting behavior:', error.message);
                sendResponse({ success: false, error: error.message });
            });

        return true; // Keep channel open for async
    }

    // Send round timing data to dashboard (CORS-free)
    if (message.action === 'sendRoundData') {
        const url = message.url || dashboardUrl;
        const data = message.data;

        fetch(`${url}/api/round-data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    console.log(`⏱️ Round timing data sent: ${data.durationMs}ms, ${data.crashValue}x`);
                }
                sendResponse(result);
            })
            .catch(error => {
                console.log('⚠️ Could not send round timing:', error.message);
                sendResponse({ success: false, error: error.message });
            });

        return true;
    }
});

console.log('🚀 Zeppelin Background Service Worker loaded');
