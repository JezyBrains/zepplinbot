# Screen Capture Guide - Visual Data Collection

Perfect solution for secured gaming sites! Use OBS or screen capture to extract game results automatically.

## 🎯 How It Works

1. **Setup**: Define screen regions where numbers appear (one-time)
2. **Monitor**: System captures and reads those regions automatically
3. **Extract**: OCR (text recognition) pulls numbers from screenshots
4. **Predict**: Use collected data for predictions

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Vision Dependencies

```bash
pip3 install opencv-python pytesseract Pillow mss pyautogui
```

**Install Tesseract OCR:**
```bash
# macOS
brew install tesseract

# Or download from: https://github.com/tesseract-ocr/tesseract
```

### Step 2: Setup Screen Regions

```bash
python3 screen_collect.py --mode setup
```

**What happens:**
1. You'll be asked to name regions (e.g., "result", "multiplier")
2. Open the Zeppelin game in your browser
3. Click and drag to select where each number appears
4. Press 'c' to confirm each region
5. Configuration saves automatically

**Example:**
```
Enter region name: result
Enter region name: multiplier
Enter region name: done

[Screenshot appears - select the result number area]
✅ Region saved: result
   Test extraction: '5.2'
   Numbers found: [5.2]
```

### Step 3: Start Monitoring

```bash
python3 screen_collect.py --mode monitor --duration 15 --interval 5
```

- Monitors for 15 minutes
- Checks every 5 seconds
- Auto-saves to `data/screen_data.csv`

---

## 📊 All Collection Methods

### Method 1: Live Screen Monitoring (RECOMMENDED)

Automatically captures numbers from your screen while you watch the game.

```bash
# Setup regions first (one-time)
python3 screen_collect.py --mode setup

# Start monitoring
python3 screen_collect.py --mode monitor --duration 20 --interval 3
```

**Perfect for:**
- Real-time data collection
- Watching the game while collecting
- No manual typing needed

---

### Method 2: OBS Recording Processing

Record the game with OBS, then extract all numbers from the video.

```bash
# 1. Record game with OBS
# 2. Setup regions (if not done)
python3 screen_collect.py --mode setup

# 3. Process the recording
python3 screen_collect.py --mode obs --video ~/Videos/zeppelin_game.mp4
```

**Perfect for:**
- Batch processing multiple games
- Extracting historical data from recordings
- Processing long sessions

---

### Method 3: Live Preview (Testing)

See what the system is capturing in real-time.

```bash
python3 screen_collect.py --mode preview --region result
```

Shows a live window with:
- Captured region (zoomed 3x)
- Extracted value
- Press 'q' to quit

**Perfect for:**
- Testing your region setup
- Verifying OCR accuracy
- Adjusting region positions

---

## 🎮 Complete Workflow

### Workflow A: Live Monitoring

```bash
# 1. Setup (one-time)
python3 screen_collect.py --mode setup

# 2. Open Zeppelin game in browser

# 3. Start monitoring
python3 screen_collect.py --mode monitor --duration 30

# 4. Make predictions
python3 main.py --data-file data/screen_data.csv --steps 5 --visualize
```

### Workflow B: OBS Recording

```bash
# 1. Setup regions (one-time)
python3 screen_collect.py --mode setup

# 2. Record game with OBS (save as MP4)

# 3. Process recording
python3 screen_collect.py --mode obs --video zeppelin_session.mp4

# 4. Make predictions
python3 main.py --data-file data/screen_data.csv --steps 5 --visualize
```

---

## 🔧 Region Setup Tips

### Best Practices:

1. **Make regions tight** - Select just the number, not extra space
2. **High contrast** - Works best with clear, bold numbers
3. **Stable position** - Numbers should stay in same spot
4. **Test first** - Use preview mode to verify accuracy

### Common Regions for Zeppelin:

- **result** - The main outcome number
- **multiplier** - Current multiplier value
- **next** - Next predicted value (if shown)
- **history** - Recent results

### Example Setup Session:

```
Enter region name: result
[Select the main result number on screen]
✅ Region saved: result
   Coordinates: x=850, y=400, w=120, h=60
   Test extraction: '7.2'
   Numbers found: [7.2]

Enter region name: multiplier
[Select the multiplier display]
✅ Region saved: multiplier
   Test extraction: '1.5x'
   Numbers found: [1.5]

Enter region name: done
```

---

## 📹 OBS Setup Guide

### Recording Settings:

1. **Resolution**: 1920x1080 or higher
2. **FPS**: 30 fps minimum
3. **Format**: MP4 (H.264)
4. **Quality**: High (for clear text)

### OBS Scene Setup:

1. Add **Display Capture** or **Window Capture**
2. Focus on the game area
3. Make sure numbers are clearly visible
4. Start recording when game begins

### Processing Tips:

- System samples every 30 seconds by default
- Longer recordings = more data points
- Clear, stable video = better accuracy

---

## 🎯 Accuracy Tips

### For Best OCR Results:

1. **Screen Resolution**: Higher is better (1080p+)
2. **Font Size**: Larger numbers = better recognition
3. **Contrast**: Dark text on light background (or vice versa)
4. **Stability**: Avoid moving/scrolling during capture
5. **Lighting**: Consistent screen brightness

### If Numbers Aren't Detected:

```bash
# 1. Test with preview mode
python3 screen_collect.py --mode preview --region result

# 2. Adjust region if needed
python3 screen_collect.py --mode setup

# 3. Check Tesseract is installed
tesseract --version
```

---

## 🔄 Continuous Collection

### Build Dataset Over Time:

```bash
# Day 1: Setup and collect
python3 screen_collect.py --mode setup
python3 screen_collect.py --mode monitor --duration 20

# Day 2: Collect more (appends to existing data)
python3 screen_collect.py --mode monitor --duration 20

# Day 3: Make predictions
python3 main.py --data-file data/screen_data.csv --steps 10 --visualize --evaluate
```

### Automated Collection:

Create a script to run monitoring sessions:

```bash
#!/bin/bash
# collect_session.sh

echo "Starting data collection session..."
python3 screen_collect.py --mode monitor --duration 30 --interval 5
echo "Session complete!"
```

---

## 📊 Data Output

### What Gets Saved:

```csv
timestamp,value
2026-01-14 21:00:00,5.2
2026-01-14 21:00:05,7.8
2026-01-14 21:00:10,3.4
...
```

### Multiple Regions:

If you define multiple regions (result, multiplier), you can:
- Track different values separately
- Correlate patterns between them
- Use for advanced predictions

---

## 🐛 Troubleshooting

### "Tesseract not found"

```bash
# Install Tesseract
brew install tesseract

# Verify installation
tesseract --version
```

### "No numbers detected"

**Solutions:**
1. Enlarge the region (include more area)
2. Check contrast (numbers should be clear)
3. Use preview mode to test
4. Try different preprocessing in code

### "Region not capturing correctly"

```bash
# Redefine the region
python3 screen_collect.py --mode setup

# Test with preview
python3 screen_collect.py --mode preview --region result
```

### "Video processing too slow"

Adjust sample rate in code:
```python
# In screen_collect.py, change sample_rate
df = obs.process_obs_recording(args.video, sample_rate=60)  # Check every 60 seconds
```

---

## 🎨 Advanced Features

### Custom Preprocessing

Edit `src/screen_capture.py` to adjust OCR:

```python
# For white text on dark background
gray = cv2.bitwise_not(gray)

# For better noise reduction
gray = cv2.GaussianBlur(gray, (5, 5), 0)
```

### Multiple Monitor Support

```python
# Capture from specific monitor
analyzer.capture_screen(monitor_number=2)
```

### Scheduled Monitoring

Use cron (macOS/Linux):
```bash
# Run every hour
0 * * * * cd /path/to/prediction-system && python3 screen_collect.py --mode monitor --duration 10
```

---

## 📈 Example Session

```bash
$ python3 screen_collect.py --mode setup

SCREEN CAPTURE SETUP WIZARD
Make sure the game is visible on screen!

Enter region name: result
Enter region name: done

[Screenshot taken - select result area]
✅ Region saved: result
   Test extraction: '5.2'

Setup complete!

$ python3 screen_collect.py --mode monitor --duration 5

👁️  MONITOR MODE
Monitoring regions: ['result']

📊 21:00:05 - result: 5.2
📊 21:00:10 - result: 7.8
📊 21:00:15 - result: 3.4
📊 21:00:20 - result: 9.1
📊 21:00:25 - result: 2.7

✅ Saved 5 data points to data/screen_data.csv

$ python3 main.py --data-file data/screen_data.csv --steps 3 --visualize

Ensemble Prediction: [6.3, 4.8, 7.2]
```

---

## 🎯 Comparison: Manual vs Screen Capture

| Feature | Manual Entry | Screen Capture |
|---------|-------------|----------------|
| Setup Time | None | 5 minutes (one-time) |
| Speed | Slow | Fast (automatic) |
| Accuracy | 100% | 95%+ with good setup |
| Effort | High | Low |
| Best For | Small datasets | Large datasets |

---

## 🚀 Quick Commands

```bash
# Setup (one-time)
python3 screen_collect.py --mode setup

# Monitor for 20 minutes
python3 screen_collect.py --mode monitor --duration 20

# Preview/test
python3 screen_collect.py --mode preview --region result

# Process OBS video
python3 screen_collect.py --mode obs --video game.mp4

# Make predictions
python3 main.py --data-file data/screen_data.csv --steps 5 --visualize
```

---

## 💡 Pro Tips

1. **Start with manual** to understand the game patterns
2. **Use screen capture** once you're collecting regularly
3. **Combine methods** - manual for verification, screen for volume
4. **Test regions** with preview mode before long monitoring sessions
5. **Record with OBS** for backup and batch processing

---

**Ready to start? Install dependencies and run setup:**

```bash
brew install tesseract
pip3 install opencv-python pytesseract Pillow mss
python3 screen_collect.py --mode setup
```
