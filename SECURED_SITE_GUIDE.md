# Guide: Collecting Data from Secured Gaming Site

Since your site (Zeppelin) has bot protection, here are **4 methods** to collect data:

## 🎯 Quick Start - Recommended Method

### Method 1: Interactive Mode (EASIEST)
This opens a browser where you can interact with the site, then capture data when ready.

```bash
python collect_data.py --mode interactive
```

**How it works:**
1. Browser opens with the gaming site
2. You complete any verification/CAPTCHA
3. Navigate to see the game results
4. Press Enter each time you want to capture the visible numbers
5. Type 'done' when finished
6. Data is automatically saved

---

## 📊 All Collection Methods

### Method 2: Manual Entry (NO BROWSER)
Type numbers manually - perfect if you're watching the game on another device.

```bash
python collect_data.py --mode manual
```

**How it works:**
1. No browser opens
2. You type each result as you see it
3. Type 'done' when finished
4. Commands: 'undo' to remove last entry

**Example:**
```
Enter result: 5
Enter result: 12
Enter result: 8
Enter result: done
```

---

### Method 3: Monitor Mode (AUTO-COLLECT)
Opens browser and automatically captures new numbers every few seconds.

```bash
python collect_data.py --mode monitor --duration 10 --interval 15
```

**Parameters:**
- `--duration 10`: Monitor for 10 minutes
- `--interval 15`: Check every 15 seconds

**How it works:**
1. Browser opens
2. You complete verification (first 30 seconds)
3. System automatically captures new numbers
4. Runs for specified duration

---

### Method 4: Auto Mode (ONE-TIME CAPTURE)
Opens browser, waits for you to verify, then captures all visible numbers once.

```bash
python collect_data.py --mode auto
```

**How it works:**
1. Browser opens
2. You have 30 seconds to complete verification
3. System captures all numbers on the page
4. Browser closes

---

## 🚀 Complete Workflow

### Step 1: Collect Data
```bash
# Use interactive mode (recommended)
python collect_data.py --mode interactive --output data/zeppelin_data.csv
```

### Step 2: Review Collected Data
```bash
# Check what was collected
cat data/zeppelin_data.csv
```

### Step 3: Make Predictions
```bash
# Predict next 5 outcomes
python main.py --data-file data/zeppelin_data.csv --steps 5 --visualize
```

### Step 4: View Results
- Console shows predictions
- `outputs/predictions.png` - visualization
- `outputs/predictions_interactive.html` - interactive chart

---

## 💡 Tips for Best Results

### For the Gaming Site:
1. **Collect at least 50-100 results** for accurate predictions
2. **Use interactive mode** - it's most reliable for secured sites
3. **Watch for patterns** - the system will find them automatically
4. **Collect during active gameplay** for real-time data

### Dealing with Security:
- ✅ Browser opens in **normal mode** (not headless) to avoid detection
- ✅ **Manual verification** time built in (30 seconds)
- ✅ **Human-like behavior** - you control the interaction
- ✅ **No automated clicking** that triggers bot detection

### Data Quality:
- Collect data when the game is showing results clearly
- If numbers aren't captured automatically, use manual entry
- Remove duplicates (system does this automatically)
- More data = better predictions

---

## 🔧 Troubleshooting

### "No numbers found"
**Solution 1:** Use manual entry mode
```bash
python collect_data.py --mode manual
```

**Solution 2:** Specify custom selector (if you know the HTML class)
```bash
python collect_data.py --mode interactive --selector ".game-result"
```

### "Browser doesn't open"
Install Chrome WebDriver:
```bash
# macOS
brew install chromedriver

# Or download from: https://chromedriver.chromium.org/
```

### "Session expired"
The URL has a session ID that may expire. Get a fresh URL:
1. Open the game in your browser
2. Copy the full URL
3. Update in config.yaml or use --url parameter

```bash
python collect_data.py --url "YOUR_NEW_URL" --mode interactive
```

---

## 📈 Example Session

```bash
$ python collect_data.py --mode interactive

INTERACTIVE SCRAPING SESSION
Browser is open. Instructions:
1. Interact with the page as needed
2. Press Enter to capture data
3. Type 'done' when finished

Press Enter to capture data: [ENTER]
Captured 3 numbers: [5, 12, 8]

Press Enter to capture data: [ENTER]
Captured 2 numbers: [15, 3]

Press Enter to capture data: done

Total data points collected: 5
✅ Saved 5 data points to data/collected_data.csv

Next steps:
1. Review collected data: data/collected_data.csv
2. Run prediction: python main.py --data-file data/collected_data.csv --steps 5 --visualize
```

---

## 🎲 Understanding the Predictions

After collecting data, the system will:
1. **Train multiple AI models** on your historical data
2. **Find patterns** in the number sequences
3. **Predict next outcomes** with confidence intervals
4. **Show which models** are most accurate for your data

**Example output:**
```
Ensemble Prediction: [7.2]
Confidence Interval (95%): [5.1, 9.3]

Individual Models:
  lstm      : 7.5
  xgboost   : 7.1
  prophet   : 7.0
  arima     : 7.3

Model Weights:
  lstm      : 0.35 (35% contribution)
  xgboost   : 0.30 (30% contribution)
```

This means the system predicts **7.2** as the next outcome, with 95% confidence it will be between **5.1 and 9.3**.

---

## 🔄 Continuous Collection

To build a large dataset over time:

```bash
# Day 1: Collect initial data
python collect_data.py --mode interactive --output data/zeppelin_data.csv

# Day 2: Add more data (appends to existing file)
python collect_data.py --mode interactive --output data/zeppelin_data.csv

# Day 3: Make predictions with all collected data
python main.py --data-file data/zeppelin_data.csv --steps 10 --visualize --evaluate
```

The more data you collect, the better the predictions become!

---

## ⚡ Quick Commands Reference

```bash
# Interactive collection (recommended)
python collect_data.py --mode interactive

# Manual typing
python collect_data.py --mode manual

# Auto-monitor for 15 minutes
python collect_data.py --mode monitor --duration 15

# Predict after collecting
python main.py --data-file data/collected_data.csv --steps 5 --visualize

# Evaluate model accuracy
python main.py --data-file data/collected_data.csv --evaluate
```

---

**Ready to start? Run this command:**
```bash
python collect_data.py --mode interactive
```
