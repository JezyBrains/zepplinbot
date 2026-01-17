// Zeppelin Auto-Bet Content Script
// Automatically places bets based on analyzer signals

// Utility to check if extension context is still valid
function isContextValid() {
    try {
        return !!(chrome && chrome.runtime && chrome.runtime.id);
    } catch (e) {
        return false;
    }
}

let autoBetEnabled = false;
let autoCapture = false;
let lastCrash = null;
let currentSignal = 'WAIT';
let betAmount = 20; // Default bet in TZS
let targetMultiplier = 2.0;
let isWaitingForRound = false;
let consecutiveLosses = 0;
let maxLosses = 5; // Safety: stop after 5 consecutive losses
let sessionProfit = 0;
let velocitySafetyEnabled = true;
let velocityEmergencyThreshold = 1.5; // Trigger pull if speed > 1.5x average


// ============ ROUND TIMING DATA COLLECTION ============
let currentRoundTiming = {
    roundId: null,
    startTime: null,
    endTime: null,
    bettorsAtStart: 0,
    stakeAtStart: 0,
    currentBettors: 0,
    currentCashedOut: 0,
    totalWinTZS: 0,      // Total TZS leaving the pool
    velocityMetrics: [], // {coef, time_ms, delta_ms}
    lastCoefValue: 0,
    lastCoefTime: 0,
    seeds: { key: null, hash: null }, // Provably Fair seeds
    cashoutEvents: [],   // {time_ms, cashed_count, total_won, at_multiplier}
    phase: 'BETTING'     // BETTING, RUNNING, CRASHED
};



let lastPhase = 'BETTING';
let roundTimingInterval = null;

// ============ EXPLOIT CONFIGURATION ============
let latencyOffset = 0.05; // Cash out 0.05x early to compensate for network latency
let sheepThreshold = 0.40; // If 40% of players cash out in a single poll, trigger escape
let baitStatus = false; // Flag for detected bait round


// Dashboard URL - configurable via settings
let DASHBOARD_URL = 'http://localhost:8050';

// Load settings from storage
if (isContextValid()) {
    chrome.storage.local.get(['autoBetEnabled', 'autoCapture', 'betAmount', 'maxLosses', 'dashboardUrl', 'targetMultiplier'], (data) => {
        if (chrome.runtime.lastError) return;
        autoBetEnabled = data.autoBetEnabled || false;
        autoCapture = data.autoCapture || false;
        betAmount = data.betAmount || 20;
        maxLosses = data.maxLosses || 5;
        DASHBOARD_URL = data.dashboardUrl || 'http://localhost:8050';
        targetMultiplier = data.targetMultiplier || 2.0;

        if (autoCapture) startObserver();
        if (autoBetEnabled) startAutoBet();

        console.log(`🚀 Zeppelin Bot: Dashboard URL = ${DASHBOARD_URL}`);
    });
}

// Listen for popup messages
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    switch (msg.action) {
        case 'setAutoCapture':
            autoCapture = msg.enabled;
            if (autoCapture) startObserver();
            else stopObserver();
            showNotification(autoCapture ? '📸 Auto-capture ON' : '📸 Auto-capture OFF');
            break;

        case 'setAutoBet':
            autoBetEnabled = msg.enabled;
            if (autoBetEnabled) {
                startAutoBet();
                showNotification('🤖 Auto-bet ENABLED', 'success');
            } else {
                showNotification('🤖 Auto-bet DISABLED', 'info');
            }
            break;

        case 'updateSettings':
            betAmount = msg.betAmount || msg.bet || betAmount;
            maxLosses = msg.maxLosses || maxLosses;
            targetMultiplier = msg.target || msg.targetMultiplier || targetMultiplier;
            if (msg.signal) {
                currentSignal = msg.signal;
                console.log(`⚡ Real-time Signal update: ${currentSignal}`);
            }
            if (msg.dashboardUrl) {
                DASHBOARD_URL = msg.dashboardUrl;
            }
            // Update UI/Notification only if it's a manual update or important signal change
            if (msg.action === 'updateSettings' && !msg.signal) {
                showNotification('⚙️ Settings updated', 'info');
            }
            break;

        case 'saveCrashToFile':
            // Save crash value to dashboard's data file
            saveCrashToCSV(msg.value);
            break;

        case 'getStatus':
            sendResponse({
                autoBetEnabled,
                autoCapture,
                currentSignal,
                consecutiveLosses,
                sessionProfit,
                betAmount,
                dashboardUrl: DASHBOARD_URL
            });
            break;
    }
});

// Save crash to the dashboard's CSV file via fetch
async function saveCrashToCSV(value) {
    if (!isContextValid()) return;
    try {
        // Method 1: Try to add via dashboard's input mechanism
        const formData = new FormData();
        formData.append('crash_value', value);

        // Store in extension storage for sync
        try {
            chrome.storage.local.get(['crashes'], (data) => {
                try {
                    if (chrome.runtime.lastError) return;
                    const crashes = data.crashes || [];
                    if (!crashes.includes(value)) {
                        crashes.push(value);
                        chrome.storage.local.set({ crashes });
                    }
                    console.log(`💾 Saved crash ${value}x to storage`);

                    // Notify popup to refresh
                    try {
                        chrome.runtime.sendMessage({
                            action: 'newCrash',
                            value: value,
                            total: crashes.length
                        }).catch(() => { });
                    } catch (e) { }
                } catch (e) { }
            });
        } catch (e) { }

    } catch (error) {
        console.log('⚠️ Could not save to CSV:', error);
    }
}

// ============ AUTO-CAPTURE ============

let observer = null;
let lastHistoryHash = '';
let checkInterval = null;
let lastBettingData = null;
let currentRoundInfo = { roundId: null, lastCoefficient: null, reliabilityCode: null };

// ============ ROUND INFO EXTRACTION ============
function extractRoundInfo() {
    const roundData = {
        roundId: null,
        lastRoundId: null,
        lastCoefficient: null,
        coefficients: [],
        timestamp: Date.now()
    };

    // Generate persistent Local Game ID
    // Format: "S{SessionTimestamp}-R{Counter}"
    // This allows unique tracking even across page reloads if we store the session
    const timestamp = Date.now();

    // Get session ID from storage or create new
    if (isContextValid()) {
        try {
            chrome.storage.local.get(['sessionId', 'roundCounter'], (data) => {
                try {
                    if (chrome.runtime.lastError) return;
                    let sessionId = data.sessionId;
                    if (!sessionId) {
                        sessionId = Math.floor(Date.now() / 1000); // Unix timestamp
                        chrome.storage.local.set({ sessionId, roundCounter: 0 });
                    }

                    let counter = data.roundCounter || 0;
                    // ... rest of logic handled in next chunk or unchanged

                    if (roundData.coefficients.length > counter) {
                        counter = roundData.coefficients.length;
                        chrome.storage.local.set({ roundCounter: counter });
                    }

                    // Use official ID if scraped, otherwise use Local ID
                    roundData.roundId = `${sessionId}-${counter}`;
                    currentRoundInfo.roundId = roundData.roundId;
                } catch (e) { }
            });
        } catch (e) {
            // Context invalidated
        }
    }

    try {
        // Primary method: Get coefficients from game history links (confirmed selector)
        const historyLinks = document.querySelectorAll('.game-history__link');

        if (historyLinks.length > 0) {
            historyLinks.forEach((el, idx) => {
                const text = el.textContent.trim();
                const match = text.match(/([\d.]+)x?/);
                if (match && idx < 20) {
                    const coef = parseFloat(match[1]);
                    if (coef >= 1.0 && coef < 10000) {
                        roundData.coefficients.push(coef);
                    }
                }
            });
        }

        if (roundData.coefficients.length > 0) {
            roundData.lastCoefficient = roundData.coefficients[0];
            currentRoundInfo.lastCoefficient = roundData.lastCoefficient;
        }

    } catch (e) {
        console.log('⚠️ Error extracting round info:', e);
    }

    return roundData;
}

// ============ BETTING BEHAVIOR EXTRACTION ============
function extractBettingBehavior() {
    const data = {
        timestamp: Date.now(),
        totalBettors: 0,
        totalStaked: 0,
        activeBets: [],
        cashedOut: [],
        stillBetting: [],
        poolSize: 0,
        onlinePlayers: 0,
        balance: 0,
        totalWonTZS: 0, // Sum of all 'won' columns in the table
        // Round context
        roundId: currentRoundInfo.roundId,
        lastCoefficient: currentRoundInfo.lastCoefficient
    };


    try {
        // Zeppelin: balance is in .header__user-balance > span.yellowColor containing "5564.67 TSh"
        const balanceSpan = document.querySelector('.header__user-balance .yellowColor');
        if (balanceSpan) {
            // Text format: " 5564.67 TSh "
            let text = balanceSpan.textContent.trim();
            // Remove currency suffix and any non-numeric chars except dots
            text = text.replace(/[^\d.]/g, '');
            const balance = parseFloat(text);
            if (!isNaN(balance) && balance > 0) {
                data.balance = balance;
                console.log(`💰 Balance: ${data.balance} TSH`);
            }
        } else {
            // Fallback for desktop or other layouts
            const balanceSelectors = [
                '.balance', '.user-balance', '[class*="balance"]',
                '.header-balance-container', '.wallet-balance'
            ];
            for (const selector of balanceSelectors) {
                const el = document.querySelector(selector);
                if (el) {
                    const text = el.textContent.replace(/[,\s]/g, '');
                    const match = text.match(/[\d.]+/);
                    if (match) {
                        data.balance = parseFloat(match[0]);
                        console.log(`💰 Balance found: ${data.balance}`);
                        break;
                    }
                }
            }
        }

        // Get online player count from .chat__online-players
        const onlineEl = document.querySelector('.chat__online-players');
        if (onlineEl) {
            const match = onlineEl.textContent.match(/\d+/);
            if (match) data.onlinePlayers = parseInt(match[0]);
        } else {
            // Fallback: search for text containing "online"
            const onlineSpan = Array.from(document.querySelectorAll('span, div')).find(el =>
                el.textContent && el.textContent.toLowerCase().includes('online (')
            );
            if (onlineSpan) {
                const match = onlineSpan.textContent.match(/\d+/);
                if (match) data.onlinePlayers = parseInt(match[0]);
            }
        }

        // Get all bets from table.table-theme (second table usually has bets)
        const tables = document.querySelectorAll('table.table-theme');
        const betTable = tables[1] || tables[0]; // Second table has bet rows

        if (betTable) {
            const betRows = betTable.querySelectorAll('tr');
            betRows.forEach(row => {
                const bet = {
                    user: '',
                    amount: 0,
                    cashedOut: false,
                    cashoutMultiplier: null,
                    won: 0
                };

                // User from td.td-size__users
                const userEl = row.querySelector('.td-size__users, td:first-child');
                if (userEl) bet.user = userEl.textContent.trim();

                // Stake amount from second td.text-center
                const cells = row.querySelectorAll('td.text-center, td');
                if (cells.length > 1) {
                    const amountText = cells[1]?.textContent?.replace(/[,\s]/g, '') || '';
                    const amountMatch = amountText.match(/[\d.]+/);
                    if (amountMatch) bet.amount = parseFloat(amountMatch[0]);
                }

                // Cashout multiplier from td.td-size__coefficient
                const coefEl = row.querySelector('.td-size__coefficient');
                if (coefEl) {
                    const coefText = coefEl.textContent.trim();
                    if (coefText !== '--' && coefText.includes('x')) {
                        bet.cashedOut = true;
                        const coefMatch = coefText.match(/[\d.]+/);
                        if (coefMatch) bet.cashoutMultiplier = parseFloat(coefMatch[0]);
                    }
                }

                // Win amount from td.td-size__win
                const winEl = row.querySelector('.td-size__win');
                if (winEl) {
                    const winText = winEl.textContent.trim();
                    if (winText !== '--') {
                        const winMatch = winText.replace(/[,\s]/g, '').match(/[\d.]+/);
                        if (winMatch) bet.won = parseFloat(winMatch[0]);
                    }
                }

                if (bet.amount > 0) {
                    data.activeBets.push(bet);
                    data.totalStaked += bet.amount;
                    if (bet.cashedOut) {
                        data.cashedOut.push(bet);
                        data.totalWonTZS += bet.won || 0;
                    } else {
                        data.stillBetting.push(bet);
                    }
                }

            });
            data.totalBettors = data.activeBets.length;
        }

        // Set poolSize to the total staked amount for this round (User confirmed this preference)
        data.poolSize = data.totalStaked;

        // Add live coefficient for real-time monitoring on dashboard
        data.liveCoefficient = readLiveCoefficient();

        // Legacy: We could still track jackpot separately if needed, but for now we replace it.
        // const jackpotBoxes = document.querySelectorAll('.jackpot-box'); ...

    } catch (e) {
        console.log('⚠️ Error extracting betting data:', e);
    }

    // Log if data changed significantly
    if (!lastBettingData ||
        Math.abs(data.totalBettors - (lastBettingData.totalBettors || 0)) > 2 ||
        data.cashedOut.length !== (lastBettingData.cashedOut?.length || 0)) {
        console.log('📊 Betting behavior:', {
            bettors: data.totalBettors,
            staked: data.totalStaked,
            cashedOut: data.cashedOut.length,
            stillIn: data.stillBetting.length,
            pool: data.poolSize,
            online: data.onlinePlayers,
            liveCoef: data.liveCoefficient
        });
    }

    lastBettingData = data;
    return data;
}

// Send betting data to dashboard via background script (CORS-free)
function sendBettingDataToDashboard(data) {
    if (!isContextValid()) return;
    try {
        chrome.runtime.sendMessage({
            action: 'sendBettingBehavior',
            url: DASHBOARD_URL,
            data: data
        }, (response) => {
            try {
                if (chrome.runtime.lastError) return;
                // Response handling here if needed
            } catch (e) { }
        });
    } catch (e) {
        // Silently fail if context invalidated during call
    }
}

function startObserver() {
    if (observer) return;

    // Only use interval, not MutationObserver (to prevent too frequent checks)
    if (!checkInterval) {
        checkInterval = setInterval(() => {
            // Safety: Clear interval if extension context invalidated
            if (!isContextValid()) {
                console.log('🛑 Extension context invalidated. Stopping observer.');
                clearInterval(checkInterval);
                checkInterval = null;
                return;
            }

            checkForCrash();

            // Extract round info
            const roundInfo = extractRoundInfo();
            if (roundInfo.roundId && roundInfo.roundId !== lastBettingData?.roundId) {
                console.log(`🎲 Round ${roundInfo.roundId} | Last: ${roundInfo.lastCoefficient}x`);
            }

            // Extract betting behavior (includes round context)
            const bettingData = extractBettingBehavior();
            if (bettingData.totalBettors > 0 || bettingData.roundId) {
                sendBettingDataToDashboard(bettingData);
            }
        }, 200);  // Higher precision: Check every 200ms
    }

    // Start round timing tracker for behavioral analysis
    startRoundTimingTracker();

    console.log('🚀 Auto-capture started with round tracking and betting behavior');
}

function stopObserver() {
    if (checkInterval) {
        clearInterval(checkInterval);
        checkInterval = null;
    }
}

function checkForCrash() {
    // Zeppelin game uses .game-history__link for crash history values
    const historyItems = document.querySelectorAll('.game-history__link');

    if (historyItems.length === 0) return;

    // Create hash of top 3 items to detect real changes
    const topItems = Array.from(historyItems).slice(0, 3);
    const currentHash = topItems.map(el => (el.textContent || '').trim()).join('|');

    // Only process if history actually changed
    if (currentHash === lastHistoryHash) return;
    lastHistoryHash = currentHash;

    // First item is the most recent crash
    const firstItem = historyItems[0];
    const text = (firstItem.textContent || firstItem.innerText || '').trim();
    const match = text.match(/^(\d+\.?\d*)x$/i);

    if (match) {
        const value = parseFloat(match[1]);
        if (value >= 1.0 && value < 1000 && value !== lastCrash) {
            // Check against recent crashes in storage to prevent duplicates
            if (isContextValid()) {
                try {
                    chrome.storage.local.get(['crashes'], (data) => {
                        try {
                            if (chrome.runtime.lastError) return;
                            const crashes = data.crashes || [];
                            const recentCrashes = crashes.slice(-5);

                            // Don't add if this value is in the last 5 crashes
                            if (!recentCrashes.includes(value)) {
                                console.log(`🎯 New crash detected: ${value}x`);
                                captureCrash(value);
                            } else {
                                console.log(`⏭️ Skipping duplicate: ${value}x`);
                            }
                        } catch (e) { }
                    });
                } catch (e) { }
            }
        }
    }
}



function captureCrash(value) {
    if (value === lastCrash) return;

    const wasWin = value >= targetMultiplier;

    // Update stats
    if (autoBetEnabled && isWaitingForRound) {
        if (wasWin) {
            const profit = (betAmount * (targetMultiplier - 1));
            sessionProfit += profit;
            consecutiveLosses = 0;
            showNotification(`🎉 WIN! +${profit.toFixed(0)} TZS`, 'success');
        } else {
            sessionProfit -= betAmount;
            consecutiveLosses++;
            showNotification(`❌ Lost ${betAmount} TZS (${consecutiveLosses}/${maxLosses})`, 'warning');

            // Safety stop
            if (consecutiveLosses >= maxLosses) {
                autoBetEnabled = false;
                chrome.storage.local.set({ autoBetEnabled: false });
                showNotification(`⚠️ AUTO-BET STOPPED: ${maxLosses} consecutive losses`, 'error');
            }
        }
        isWaitingForRound = false;
    }

    lastCrash = value;
    console.log(`🎯 Crash: ${value}x`);

    // Save to storage with deduplication and limits
    try {
        chrome.storage.local.get(['crashes', 'lastCrashTime'], (data) => {
            try {
                if (chrome.runtime.lastError) return;
                let crashes = data.crashes || [];
                const now = Date.now();
                const lastTime = data.lastCrashTime || 0;

                // Only add if at least 5 seconds since last crash (prevents duplicates from page reload)
                if (now - lastTime > 5000) {
                    crashes.push(value);

                    // Keep only last 200 crashes to prevent unbounded growth
                    if (crashes.length > 200) {
                        crashes = crashes.slice(-200);
                    }

                    chrome.storage.local.set({ crashes, lastCrashTime: now });

                    // Update signal based on new data
                    updateSignal(crashes);

                    // Send to popup/background (with error handling)
                    try {
                        chrome.runtime.sendMessage({ action: 'newCrash', value, wasWin }).catch(() => {
                            // Popup not open - that's OK
                        });
                    } catch (e) {
                        // Ignore errors when popup/background not available
                    }
                }
            } catch (e) { }
        });
    } catch (e) { }
}

function updateSignal(crashes) {
    // First try to fetch from dashboard (preferred)
    fetchSignalFromDashboard();

    // Fallback to local calculation if dashboard unavailable
    if (currentSignal === 'WAIT' && crashes.length >= 20) {
        const recent = crashes.slice(-50);
        const p = recent.filter(c => c >= targetMultiplier).length / recent.length;
        const ev = (p * (targetMultiplier - 1)) - (1 - p);
        currentSignal = ev > 0 ? 'BET' : 'SKIP';
    }
}

// ============ FETCH SIGNAL FROM DASHBOARD ============
// Reads the BET/SKIP signal directly from your analyzer at localhost:8050
// Uses background script to avoid CORS issues

async function fetchSignalFromDashboard() {
    if (!isContextValid()) return;
    try {
        chrome.runtime.sendMessage(
            { action: 'fetchDashboard', url: DASHBOARD_URL },
            (response) => {
                try {
                    if (chrome.runtime.lastError) return;
                    if (response && response.success) {
                        currentSignal = response.signal;
                        if (response.betAmount > 0) betAmount = response.betAmount;
                        if (response.targetMultiplier > 0) targetMultiplier = response.targetMultiplier;
                        console.log(`📡 Dashboard: ${currentSignal} at ${targetMultiplier}x, ${betAmount} TZS`);
                    }
                } catch (e) { }
            }
        );
    } catch (error) {
        // Silently fail if context invalidated
    }
}

// Fetch signal from dashboard every 5 seconds (reduced frequency)
const signalInterval = setInterval(() => {
    if (!isContextValid()) {
        clearInterval(signalInterval);
        return;
    }
    fetchSignalFromDashboard();
}, 5000);

// ============ AUTO-BET ============

function startAutoBet() {
    console.log('🤖 Starting auto-bet...');

    // Check for betting opportunity every 2 seconds
    setInterval(checkAndBet, 2000);
}

function checkAndBet() {
    if (!autoBetEnabled) return;
    if (isWaitingForRound) {
        // Silently skip - waiting for result
        return;
    }
    if (consecutiveLosses >= maxLosses) {
        console.log(`🛑 Auto-bet paused: Hit ${maxLosses} loss limit`);
        return;
    }

    // Check if we should bet
    if (currentSignal !== 'BET') {
        console.log(`⏳ Signal is "${currentSignal}" - waiting for BET signal...`);
        return;
    }

    // Check if betting is available (look for bet button)
    const betButton = findBetButton();
    if (!betButton) return;

    // Set bet amount
    const betInput = findBetInput();
    if (betInput) {
        betInput.value = betAmount;
        betInput.dispatchEvent(new Event('input', { bubbles: true }));
        console.log(`💰 Set bet amount: ${betAmount}`);
    } else {
        console.log('⚠️ Bet input not found');
    }

    // Click bet button
    console.log(`🎰 Placing bet: ${betAmount} TZS targeting ${targetMultiplier}x`);
    betButton.click();

    isWaitingForRound = true;
    showNotification(`🎰 Bet placed: ${betAmount} TZS → ${targetMultiplier}x`, 'info');

    // Start monitoring the live multiplier to trigger cashout
    startCashoutMonitor(targetMultiplier);
}

// ============ LIVE CASHOUT MONITORING ============
let cashoutMonitorInterval = null;
let currentBetActive = false;

function startCashoutMonitor(target) {
    currentBetActive = true;
    console.log(`👀 Started high-precision cashout monitoring for target: ${target}x (Latency Comp: ${latencyOffset}x)`);

    // Check every 50ms for hyper-fast response
    cashoutMonitorInterval = setInterval(() => {
        if (!currentBetActive) {
            clearInterval(cashoutMonitorInterval);
            return;
        }

        const currentCoef = readLiveCoefficient();
        if (currentCoef > 0) {
            // Apply latency compensation: trigger slightly early to beat the network delay
            const adjustedTarget = target - latencyOffset;

            if (currentCoef >= adjustedTarget) {
                console.log(`🎯 TARGET REACHED (Adjusted: ${adjustedTarget})! Executing cashout at ${currentCoef}x`);
                executeCashout();
                stopCashoutMonitor();
            }
        }
    }, 50);
}

function stopCashoutMonitor() {
    currentBetActive = false;
    if (cashoutMonitorInterval) {
        clearInterval(cashoutMonitorInterval);
        cashoutMonitorInterval = null;
    }
    isWaitingForRound = false;
}

function readLiveCoefficient() {
    // Zeppelin: live coefficient is in .game-box__container-inner-text > b
    // When game is running: the span is visible (no hidden attribute)
    // When betting phase: the span has hidden="" attribute
    // When crashed: the span has class "redColor"

    const coefSpans = document.querySelectorAll('.game-box__container-inner-text');
    for (const span of coefSpans) {
        // Skip if hidden (betting phase)
        if (span.hasAttribute('hidden')) continue;

        // Get the b tag inside
        const bTag = span.querySelector('b');
        if (bTag) {
            const text = bTag.textContent.trim();  // " 6.54x " -> "6.54x"
            const match = text.match(/^([\d.]+)x$/);
            if (match) {
                const coef = parseFloat(match[1]);
                if (coef >= 1.0) {
                    // Check if crashed (has redColor class) - don't cash out on crashed value
                    if (!span.classList.contains('redColor')) {
                        return coef;
                    }
                }
            }
        }
    }
    return 0;
}

function executeCashout() {
    const cashoutBtn = findCashoutButton();
    if (cashoutBtn) {
        console.log('💵 Clicking cashout button!');
        cashoutBtn.click();
        showNotification(`💵 Cashed out!`, 'success');
        return true;
    }
    console.log('⚠️ Cashout button not found');
    return false;
}

function findCashoutButton() {
    // Zeppelin uses a single .bet-button that changes text:
    // During betting: "PLACE YOUR BET" | During round: "CASH OUT" with amount
    const betBtn = document.querySelector('.bet-button');
    if (betBtn && betBtn.offsetParent !== null && !betBtn.disabled) {
        const text = (betBtn.textContent || '').toLowerCase();
        // It's in cashout mode if text contains 'cash' or 'out' or shows a number with TZS
        if (text.includes('cash') || text.includes('out') || /\d+(\.\d+)?\s*(tzs|tsh)?/i.test(text)) {
            console.log(`✅ Found cashout button: .bet-button`);
            return betBtn;
        }
    }
    return null;
}

function findBetButton() {
    // Zeppelin uses .bet-button which shows "PLACE YOUR BET" during betting phase
    const betBtn = document.querySelector('.bet-button');
    if (betBtn && betBtn.offsetParent !== null && !betBtn.disabled) {
        const text = (betBtn.textContent || '').toLowerCase();
        // In betting mode, button says "PLACE YOUR BET"
        if (text.includes('place') || text.includes('bet')) {
            console.log(`✅ Found bet button: .bet-button`);
            return betBtn;
        }
    }
    console.log('⚠️ No bet button found (not in betting phase?)');
    return null;
}

// ============ ROUND TIMING TRACKING ============

function detectRoundPhase() {
    // Check for betting phase: loader visible with "PLACE YOUR BETS PLEASE"
    const loader = document.querySelector('.game-loader__main-container');
    if (loader && loader.offsetParent !== null) {
        return 'BETTING';
    }

    // Check coefficient span state
    const coefSpans = document.querySelectorAll('.game-box__container-inner-text');
    for (const span of coefSpans) {
        if (span.hasAttribute('hidden')) continue;

        if (span.classList.contains('redColor')) {
            return 'CRASHED';
        }

        // Visible span without redColor = running
        const bTag = span.querySelector('b');
        if (bTag && bTag.textContent.trim().match(/^\d+\.\d+x$/)) {
            return 'RUNNING';
        }
    }

    return 'BETTING';  // Default to betting if can't determine
}

function startRoundTimingTracker() {
    if (roundTimingInterval) return;

    roundTimingInterval = setInterval(() => {
        // Safety: Clear interval if context invalidated
        if (!isContextValid()) {
            clearInterval(roundTimingInterval);
            roundTimingInterval = null;
            return;
        }

        const currentPhase = detectRoundPhase();

        if (lastPhase === 'BETTING' && currentPhase === 'RUNNING') {
            const bettingData = extractBettingBehavior();
            currentRoundTiming = {
                roundId: `R${Date.now()}`,
                startTime: performance.now(), // High-precision float timestamp
                endTime: null,
                bettorsAtStart: bettingData.totalBettors || 0,
                stakeAtStart: bettingData.poolSize || 0,
                currentBettors: bettingData.totalBettors || 0,
                currentCashedOut: 0,
                totalWinTZS: 0,
                velocityMetrics: [],
                lastCoefValue: 1.0,
                lastCoefTime: performance.now(),
                cashoutEvents: [],
                phase: 'RUNNING'
            };
            console.log(`⏱️ Round started accurately! Bettors: ${currentRoundTiming.bettorsAtStart}, Stake: ${currentRoundTiming.stakeAtStart}`);
        }


        // During RUNNING phase: track cashouts and velocity
        if (currentPhase === 'RUNNING' && currentRoundTiming.startTime) {
            const bettingData = extractBettingBehavior();
            const cashedOutNow = bettingData.cashedOut?.length || 0;
            const currentCoef = readLiveCoefficient();
            const now = performance.now();

            // 🏎️ MULTIPLIER VELOCITY TRACKING & EMERGENCY PULL (Phase 10)
            if (currentCoef > currentRoundTiming.lastCoefValue) {
                const deltaMs = now - currentRoundTiming.lastCoefTime;

                // Emergency Pull Logic: Check for sudden acceleration (speed manipulation)
                if (velocitySafetyEnabled && currentBetActive && currentRoundTiming.velocityMetrics.length > 5) {
                    // Calculate average delta from last 5 steps
                    const lastVelocities = currentRoundTiming.velocityMetrics.slice(-5).map(v => v.delta_ms);
                    const avgDelta = lastVelocities.reduce((a, b) => a + b, 0) / lastVelocities.length;

                    // If current delta is much SMALLER than average (meaning it's GOING FASTER)
                    // Note: lower delta_ms = higher speed
                    if (deltaMs < avgDelta / velocityEmergencyThreshold) {
                        console.log(`⚠️ VELOCITY ANOMALY! Speed increased significantly (${Math.round(avgDelta)}ms -> ${Math.round(deltaMs)}ms). Emergency exit...`);
                        executeCashout();
                        stopCashoutMonitor();
                        showNotification('⚠️ Emergency Pull: Velocity Anomaly!', 'warning');
                    }
                }

                currentRoundTiming.velocityMetrics.push({
                    coef: currentCoef,
                    time_ms: Math.round(now - currentRoundTiming.startTime),
                    delta_ms: Math.round(deltaMs)
                });
                currentRoundTiming.lastCoefValue = currentCoef;
                currentRoundTiming.lastCoefTime = now;
            }

            // BEHAVIORAL EXPLOIT: Sheep Pattern Detection

            if (currentRoundTiming.bettorsAtStart > 10) {
                const suddenOut = cashedOutNow - currentRoundTiming.currentCashedOut;
                const outPercentage = suddenOut / currentRoundTiming.bettorsAtStart;

                if (outPercentage >= sheepThreshold && currentBetActive) {
                    console.log(`🐑 SHEEP DETECTED! ${Math.round(outPercentage * 100)}% players bailed! Emergency exit...`);
                    executeCashout();
                    stopCashoutMonitor();
                }
            }

            // Record cashout event if count changed
            if (cashedOutNow > currentRoundTiming.currentCashedOut) {
                const elapsedMs = now - currentRoundTiming.startTime;
                currentRoundTiming.cashoutEvents.push({
                    time_ms: Math.round(elapsedMs),
                    cashed_count: cashedOutNow,
                    total_won: bettingData.totalWonTZS,
                    at_multiplier: currentCoef
                });
                console.log(`📈 Accurate Cashout: ${cashedOutNow}/${currentRoundTiming.bettorsAtStart} at ${currentCoef}x (${Math.round(elapsedMs)}ms) | Won: ${bettingData.totalWonTZS} TZS`);
                currentRoundTiming.currentCashedOut = cashedOutNow;
                currentRoundTiming.totalWinTZS = bettingData.totalWonTZS;
            }
            currentRoundTiming.currentBettors = bettingData.totalBettors || currentRoundTiming.currentBettors;
        }


        if (lastPhase === 'RUNNING' && currentPhase === 'CRASHED' && currentRoundTiming.startTime) {
            currentRoundTiming.endTime = performance.now();
            currentRoundTiming.phase = 'CRASHED';

            const durationMs = Math.round(currentRoundTiming.endTime - currentRoundTiming.startTime);
            const finalCoef = readCrashedCoefficient();

            console.log(`💥 Round ended: ${finalCoef}x, Duration: ${durationMs}ms, Cashouts: ${currentRoundTiming.currentCashedOut}/${currentRoundTiming.bettorsAtStart}`);

            // Trigger seed harvesting after a short delay to let history update
            setTimeout(() => {
                harvestProvablyFairData().then(seeds => {
                    sendRoundTimingData({
                        ...currentRoundTiming,
                        durationMs: durationMs,
                        crashValue: finalCoef,
                        seeds: seeds,
                        cashoutRatio: currentRoundTiming.bettorsAtStart > 0
                            ? currentRoundTiming.currentCashedOut / currentRoundTiming.bettorsAtStart
                            : 0
                    });
                });
            }, 3000); // 3s delay for history to update
        }


        lastPhase = currentPhase;
    }, 50);
}

async function harvestProvablyFairData() {
    console.log('🔍 Harvesting Provably Fair seeds...');
    try {
        // 1. Find and click most recent history item
        const historyLinks = document.querySelectorAll('.game-history__link, .round-history-item');
        if (historyLinks.length > 0) {
            historyLinks[0].click(); // Click newest

            // 2. Wait for modal animation
            await new Promise(r => setTimeout(r, 800));

            // 3. Scrape Round Key and Reliability Code
            const containers = document.querySelectorAll('.jm-panel__container');
            const seeds = { key: 'N/A', hash: 'N/A' };

            if (containers.length >= 2) {
                // Usually first is Round Key, second is Hash
                seeds.key = containers[0].textContent.trim();
                seeds.hash = containers[1].textContent.trim();
            } else {
                // Fallback text-based search
                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                let node;
                while (node = walker.nextNode()) {
                    if (node.textContent.includes('Round key:')) {
                        seeds.key = node.parentElement.nextElementSibling?.textContent?.trim() || 'N/A';
                    }
                    if (node.textContent.includes('Reliability code:')) {
                        seeds.hash = node.parentElement.nextElementSibling?.textContent?.trim() || 'N/A';
                    }
                }
            }

            // 4. Close modal (usually by clicking overlay or close btn)
            const closeBtn = document.querySelector('.jm-modal__close-btn, .modal-close');
            if (closeBtn) closeBtn.click();
            else document.body.click(); // Click away

            console.log(`✅ Seeds harvested: ${seeds.key.substring(0, 10)}...`);
            return seeds;
        }
    } catch (e) {
        console.log('⚠️ Seed harvesting failed:', e);
    }
    return { key: 'N/A', hash: 'N/A' };
}


function readCrashedCoefficient() {
    const coefSpans = document.querySelectorAll('.game-box__container-inner-text.redColor');
    for (const span of coefSpans) {
        const bTag = span.querySelector('b');
        if (bTag) {
            const text = bTag.textContent.trim();
            const match = text.match(/^([\d.]+)x$/);
            if (match) return parseFloat(match[1]);
        }
    }
    return 0;
}

async function sendRoundTimingData(data) {
    if (!isContextValid()) return;
    try {
        // Send via background script to bypass CORS
        chrome.runtime.sendMessage({
            action: 'sendRoundData',
            data: data
        }, (response) => {
            if (chrome.runtime.lastError) return;
            if (response?.success) {
                console.log('📤 Round timing data sent to dashboard');
            }
        });
    } catch (e) {
        console.log('⚠️ Failed to send round timing:', e);
    }
}

function findBetInput() {
    const inputs = document.querySelectorAll('input[type="number"], input[type="text"]');
    for (const input of inputs) {
        const placeholder = (input.placeholder || '').toLowerCase();
        const nearby = input.closest('div')?.textContent?.toLowerCase() || '';
        if (placeholder.includes('bet') || nearby.includes('bet:') || nearby.includes('amount')) {
            return input;
        }
    }
    return inputs[0]; // First number input as fallback
}

function findCashoutInput() {
    // Try specific class selectors first
    const classSelectors = [
        '[class*="cashout"] input',
        '[class*="auto-cashout"] input',
        'input[class*="cashout"]',
        '.bet-control input:last-of-type',
        'input[placeholder*="cashout"]',
        'input[placeholder*="x"]'
    ];

    for (const selector of classSelectors) {
        try {
            const el = document.querySelector(selector);
            if (el && el.offsetParent !== null) {
                console.log(`✅ Found cashout input via: ${selector}`);
                return el;
            }
        } catch (e) { }
    }

    // Fallback: find inputs and match by label text
    const inputs = document.querySelectorAll('input[type="number"], input[type="text"]');
    for (const input of inputs) {
        const placeholder = (input.placeholder || '').toLowerCase();
        const parent = input.closest('div');
        const parentText = (parent?.textContent || '').toLowerCase();

        // Check for cashout-related text
        if (placeholder.includes('cashout') ||
            placeholder.includes('auto') ||
            placeholder.includes('x') ||
            parentText.includes('auto cashout') ||
            parentText.includes('автовыплата') ||  // Russian
            parentText.includes('cashout at')) {
            console.log(`✅ Found cashout input via text matching`);
            return input;
        }
    }

    // Last resort: second input in bet control area (usually bet amount is first, cashout is second)
    const allInputs = document.querySelectorAll('.bet-control input, .bet-panel input, [class*="bet"] input');
    if (allInputs.length >= 2) {
        console.log(`✅ Found cashout input as second input in bet area`);
        return allInputs[1];
    }

    console.log('⚠️ No cashout input found');
    return null;
}

// ============ UI ============

function showNotification(message, type = 'info') {
    const existing = document.getElementById('zeppelin-notification');
    if (existing) existing.remove();

    const colors = {
        success: '#00d97e',
        warning: '#ffc107',
        error: '#e63757',
        info: '#00b4d8'
    };

    const div = document.createElement('div');
    div.id = 'zeppelin-notification';
    div.innerHTML = message;
    div.style.cssText = `
    position: fixed;
    top: 10px;
    right: 10px;
    padding: 12px 20px;
    background: ${colors[type] || colors.info};
    color: ${type === 'warning' ? '#000' : '#fff'};
    border-radius: 8px;
    font-family: sans-serif;
    font-weight: 600;
    font-size: 14px;
    z-index: 999999;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    animation: slideIn 0.3s ease;
  `;

    document.body.appendChild(div);
    setTimeout(() => div.remove(), 3000);
}

// Add control panel to page with settings
// TACTICAL OS INJECTED PANEL
function addControlPanel() {
    if (document.getElementById('zeppelin-panel')) return;

    const panel = document.createElement('div');
    panel.id = 'zeppelin-panel';
    panel.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
      <span style="font-family: 'Courier New', monospace; font-weight: 700; color: #00F0FF; letter-spacing: 1px;">_ZEPPELIN_BOT</span>
      <button id="zep-settings-btn" style="
        background: none;
        border: none;
        color: rgba(255,255,255,0.4);
        cursor: pointer;
        font-size: 14px;
        padding: 2px 5px;
      ">⛭</button>
    </div>
    <div id="zep-main-view">
      <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.02); padding: 8px;">
        <div style="text-align: center;">
          <div style="font-size: 9px; color: rgba(255,255,255,0.4); text-transform: uppercase;">SIGNAL</div>
          <div id="zep-signal" style="font-family: 'Courier New', monospace; font-size: 16px; font-weight: 700; color: #fff;">WAIT</div>
        </div>
        <div style="text-align: center;">
          <div style="font-size: 9px; color: rgba(255,255,255,0.4); text-transform: uppercase;">PROFIT</div>
          <div id="zep-profit" style="font-family: 'Courier New', monospace; font-size: 16px; font-weight: 700; color: #fff;">0</div>
        </div>
        <div style="text-align: center;">
          <div style="font-size: 9px; color: rgba(255,255,255,0.4); text-transform: uppercase;">LOGS</div>
          <div id="zep-crashes" style="font-family: 'Courier New', monospace; font-size: 16px; font-weight: 700; color: #fff;">0</div>
        </div>
      </div>
      <div style="display: flex; gap: 5px;">
        <button id="zep-autobet-btn" style="
          flex: 1;
          padding: 10px;
          background: #080808;
          border: 1px solid #00F0FF;
          color: #00F0FF;
          font-family: 'Courier New', monospace;
          font-weight: 700;
          cursor: pointer;
          text-transform: uppercase;
          transition: all 0.2s;
        ">AUTO_EXECUTE: OFF</button>
      </div>
    </div>
    <div id="zep-settings-view" style="display: none;">
      <div style="margin-bottom: 10px;">
        <label style="font-family: 'Courier New', monospace; font-size: 9px; color: rgba(255,255,255,0.5); display: block; margin-bottom: 3px;">DASHBOARD_UPLINK</label>
        <input id="zep-url" type="text" value="${DASHBOARD_URL}" style="
          width: 100%;
          padding: 8px;
          background: #000;
          border: 1px solid rgba(255,255,255,0.2);
          color: #00F0FF;
          font-family: 'Courier New', monospace;
          font-size: 11px;
        ">
      </div>
      <div style="display: flex; gap: 8px; margin-bottom: 10px;">
        <div style="flex: 1;">
          <label style="font-family: 'Courier New', monospace; font-size: 9px; color: rgba(255,255,255,0.5); display: block; margin-bottom: 3px;">UNIT_SIZE</label>
          <input id="zep-bet" type="number" value="${betAmount}" style="
            width: 100%;
            padding: 8px;
            background: #000;
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            font-family: 'Courier New', monospace;
            font-size: 12px;
          ">
        </div>
        <div style="flex: 1;">
          <label style="font-family: 'Courier New', monospace; font-size: 9px; color: rgba(255,255,255,0.5); display: block; margin-bottom: 3px;">TARGET_X</label>
          <input id="zep-target" type="number" step="0.1" value="${targetMultiplier}" style="
            width: 100%;
            padding: 8px;
            background: #000;
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            font-family: 'Courier New', monospace;
            font-size: 12px;
          ">
        </div>
      </div>
      <div style="margin-bottom: 10px;">
        <label style="font-family: 'Courier New', monospace; font-size: 9px; color: rgba(255,255,255,0.5); display: block; margin-bottom: 3px;">SAFETY_STOP (LOSS)</label>
        <input id="zep-maxloss" type="number" value="${maxLosses}" style="
          width: 100%;
          padding: 8px;
          background: #000;
          border: 1px solid rgba(255,255,255,0.2);
          color: #fff;
          font-family: 'Courier New', monospace;
          font-size: 12px;
        ">
      </div>
      <button id="zep-save-btn" style="
        width: 100%;
        padding: 10px;
        background: #00F0FF;
        border: none;
        color: #000;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        cursor: pointer;
        margin-bottom: 8px;
      ">COMMIT_CHANGES</button>
      <button id="zep-clear-btn" style="
        width: 100%;
        padding: 10px;
        background: #FF3131;
        border: none;
        color: #fff;
        font-family: 'Courier New', monospace;
        font-weight: 700;
        cursor: pointer;
      ">PURGE_DATA</button>
    </div>
  `;
    panel.style.cssText = `
    position: fixed;
    bottom: 80px;
    right: 20px;
    width: 280px;
    background: #080808;
    color: white;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    font-family: 'Courier New', monospace;
    z-index: 9999;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
  `;
    document.body.appendChild(panel);

    // Settings toggle
    document.getElementById('zep-settings-btn').addEventListener('click', () => {
        const mainView = document.getElementById('zep-main-view');
        const settingsView = document.getElementById('zep-settings-view');
        const isSettings = settingsView.style.display !== 'none';

        mainView.style.display = isSettings ? 'block' : 'none';
        settingsView.style.display = isSettings ? 'none' : 'block';
        document.getElementById('zep-settings-btn').textContent = isSettings ? '⚙️' : '✕';
    });

    // Save settings
    document.getElementById('zep-save-btn').addEventListener('click', async () => {
        const newUrl = document.getElementById('zep-url').value;
        const newBet = parseFloat(document.getElementById('zep-bet').value);
        const newTarget = parseFloat(document.getElementById('zep-target').value);
        const newMaxLoss = parseInt(document.getElementById('zep-maxloss').value);

        DASHBOARD_URL = newUrl;
        betAmount = newBet;
        targetMultiplier = newTarget;
        maxLosses = newMaxLoss;

        if (isContextValid()) {
            try {
                chrome.storage.local.set({
                    dashboardUrl: newUrl,
                    betAmount: newBet,
                    targetMultiplier: newTarget,
                    maxLosses: newMaxLoss
                });
            } catch (e) { }
        }

        showNotification('✅ Settings saved!', 'success');

        // Go back to main view
        document.getElementById('zep-main-view').style.display = 'block';
        document.getElementById('zep-settings-view').style.display = 'none';
        document.getElementById('zep-settings-btn').textContent = '⚙️';
    });

    // Clear all data button
    document.getElementById('zep-clear-btn').addEventListener('click', async () => {
        if (confirm('Clear all crash data and reset counters?')) {
            if (isContextValid()) {
                try {
                    chrome.storage.local.set({
                        crashes: [],
                        sessionProfit: 0,
                        consecutiveLosses: 0,
                        lastCrashTime: 0
                    });
                } catch (e) { }
            }

            sessionProfit = 0;
            consecutiveLosses = 0;
            lastCrash = null;

            showNotification('🗑️ All data cleared!', 'success');
            updatePanel();

            // Go back to main view
            document.getElementById('zep-main-view').style.display = 'block';
            document.getElementById('zep-settings-view').style.display = 'none';
            document.getElementById('zep-settings-btn').textContent = '⚙️';
        }
    });

    document.getElementById('zep-autobet-btn').addEventListener('click', () => {
        autoBetEnabled = !autoBetEnabled;
        if (isContextValid()) {
            try {
                chrome.storage.local.set({ autoBetEnabled });
            } catch (e) { }
        }
        updatePanel();
        showNotification(autoBetEnabled ? '🤖 Auto-bet ON' : '🤖 Auto-bet OFF', autoBetEnabled ? 'success' : 'info');
    });

    // Update panel every second
    const panelInterval = setInterval(() => {
        if (!isContextValid()) {
            clearInterval(panelInterval);
            return;
        }
        updatePanel();
    }, 1000);
}

function updatePanel() {
    const signalEl = document.getElementById('zep-signal');
    const profitEl = document.getElementById('zep-profit');
    const crashesEl = document.getElementById('zep-crashes');
    const btnEl = document.getElementById('zep-autobet-btn');

    if (signalEl) {
        signalEl.textContent = currentSignal;
        signalEl.style.color = currentSignal === 'BET' ? '#00d97e' :
            currentSignal === 'SKIP' ? '#e63757' : '#ffc107';
    }

    if (profitEl) {
        profitEl.textContent = (sessionProfit >= 0 ? '+' : '') + sessionProfit.toFixed(0);
        profitEl.style.color = sessionProfit >= 0 ? '#00d97e' : '#e63757';
    }

    if (crashesEl) {
        if (isContextValid()) {
            try {
                chrome.storage.local.get(['crashes'], (data) => {
                    try {
                        if (chrome.runtime.lastError) return;
                        crashesEl.textContent = (data.crashes || []).length;
                    } catch (e) { }
                });
            } catch (e) { }
        }
    }

    if (btnEl) {
        btnEl.textContent = autoBetEnabled ? '🤖 Auto-Bet: ON' : '🤖 Auto-Bet: OFF';
        btnEl.style.background = autoBetEnabled ? '#00d97e' : '#333';
        btnEl.style.color = autoBetEnabled ? '#000' : '#00d97e';
    }
}

// Initialize
addControlPanel();
startObserver();
console.log('🚀 Zeppelin Bot loaded - Auto-capture active');
