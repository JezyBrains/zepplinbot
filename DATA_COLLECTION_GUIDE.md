# Easy Data Collection Guide

Three simple methods to collect crash data - choose what works best for you!

---

## 🚀 Method 1: Quick Collect (EASIEST)

**Best for:** Manual entry, no setup required

### How to use:
```bash
python3 quick_collect.py
```

### Steps:
1. Run the script
2. Copy crash values from Zeppelin
3. Paste them into terminal
4. Type `done` when finished
5. Data automatically added!

### Example:
```
📝 Paste crash values (type 'done' when finished):

2.05x
1.47x
4.13x
3.53x
done

✅ Added 4 crashes to dataset
```

**Advantages:**
- ✅ No setup needed
- ✅ Works immediately
- ✅ Can paste many values at once
- ✅ Handles any format (2.05x, 2.05, etc.)

---

## 🎯 Method 2: Auto Clipboard Monitor

**Best for:** Continuous collection while playing

### How to use:
```bash
pip3 install pyperclip  # One-time install
python3 auto_collect.py
```

### Steps:
1. Run the script
2. Play Zeppelin normally
3. Copy crash values (Ctrl+C / Cmd+C)
4. Script auto-detects and adds them!
5. Press Ctrl+C to stop

### Example:
```
🎯 Monitoring clipboard for crash values...

✅ Detected 3 crash(es): [2.05, 1.47, 4.13]
   Added to dataset (Total: 421 crashes)
```

**Advantages:**
- ✅ Automatic detection
- ✅ No manual entry
- ✅ Works in background
- ✅ Copy from anywhere

---

## 📹 Method 3: OBS Screen Capture (ADVANCED)

**Best for:** Fully automated collection

### Setup (one-time):
```bash
# Install OBS Studio
brew install obs

# Setup capture regions
python3 obs_capture.py --mode setup
```

### Steps:
1. Open OBS Studio
2. Add Browser Source with Zeppelin URL
3. Create Windowed Projector
4. Define ROI (Region of Interest) for crash numbers
5. Run monitoring:
```bash
python3 obs_capture.py --mode monitor --duration 30
```

**Advantages:**
- ✅ Fully automated
- ✅ No manual copying
- ✅ Can run for hours
- ✅ OCR extracts values

**Disadvantages:**
- ⚠️ Requires OBS setup
- ⚠️ Needs Tesseract OCR
- ⚠️ More complex

---

## 📊 Recommended Workflow

### For Quick Testing:
```bash
python3 quick_collect.py  # Add 20-30 crashes
python3 monitor_dashboard.py --bankroll 100
```

### For Regular Use:
```bash
python3 auto_collect.py  # Run while playing
# Copy crashes as you play
# Ctrl+C when done
python3 monitor_dashboard.py --bankroll 100
```

### For Automation:
```bash
python3 obs_capture.py --mode setup  # One-time
python3 obs_capture.py --mode monitor --duration 60
```

---

## 🎯 Data Format

All methods accept these formats:
- `2.05x` ✅
- `2.05X` ✅
- `2.05` ✅
- `2.05 x` ✅

Multiple values:
```
2.05x 1.47x 4.13x  # Space-separated ✅
2.05x
1.47x
4.13x              # Line-separated ✅
```

---

## 📈 After Collection

Check your data:
```bash
# View dataset
python3 -c "import pandas as pd; df=pd.read_csv('data/zeppelin_data.csv'); print(f'Total: {len(df)} crashes'); print(df.tail(10))"

# Run analysis
python3 monitor_dashboard.py --bankroll 100 --target 2.0

# Check Kelly signal
python3 kelly_dashboard.py --bankroll 100
```

---

## 🛠️ Troubleshooting

### Quick Collect Issues:
```bash
# If script doesn't run:
chmod +x quick_collect.py
python3 quick_collect.py
```

### Auto Clipboard Issues:
```bash
# If pyperclip not working:
pip3 install pyperclip

# On macOS, may need:
pip3 install pyobjc-framework-Cocoa
```

### OBS Issues:
```bash
# If Tesseract not found:
brew install tesseract

# If OpenCV issues:
pip3 install opencv-python
```

---

## 💡 Pro Tips

### 1. Batch Collection
Copy 20-50 crashes at once for faster collection:
```bash
python3 quick_collect.py
# Paste all crashes
# Type done
```

### 2. Order Matters
Script asks: "Is X the most recent crash?"
- Answer `y` if you copied newest-to-oldest
- Answer `n` if you copied oldest-to-newest

### 3. Validation
Always check last crash matches:
```bash
python3 -c "import pandas as pd; print(pd.read_csv('data/zeppelin_data.csv').tail(1))"
```

### 4. Backup Data
```bash
cp data/zeppelin_data.csv data/backup_$(date +%Y%m%d).csv
```

---

## 📊 Current Dataset Status

Check your progress:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/zeppelin_data.csv')
print(f'Total crashes: {len(df)}')
print(f'Target: 500 crashes')
print(f'Remaining: {500 - len(df)}')
print(f'Progress: {len(df)/500*100:.1f}%')
"
```

---

## 🎯 Recommended Method

**For most users: Method 1 (Quick Collect)**
- Fastest to start
- No dependencies
- Most reliable
- Easy to verify

**For power users: Method 2 (Auto Clipboard)**
- Convenient while playing
- Automatic detection
- Minimal effort

**For automation: Method 3 (OBS)**
- Fully hands-off
- Long-term collection
- Requires setup time

---

## 🚀 Quick Start

```bash
# Easiest way to get started:
python3 quick_collect.py

# Paste your crash values
# Type 'done'
# Run analysis
python3 monitor_dashboard.py --bankroll 100
```

That's it! 🎉
