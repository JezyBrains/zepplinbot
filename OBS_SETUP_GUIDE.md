# OBS Windowed Projector Setup Guide

## 🎯 Complete Pipeline: OBS → OCR → Predictions

This method **bypasses bot detection** by treating the website as a video feed instead of interacting with HTML/DOM.

---

## 📋 Prerequisites

### 1. Install OBS Studio
```bash
# macOS
brew install --cask obs

# Or download from: https://obsproject.com/
```

### 2. Install Python Dependencies
```bash
pip3 install opencv-python pytesseract pyautogui Pillow numpy pandas
```

### 3. Install Tesseract OCR
```bash
# macOS
brew install tesseract

# Verify installation
tesseract --version
```

---

## 🚀 Complete Setup (5 Steps)

### Step 1: Configure OBS Studio

1. **Open OBS Studio**

2. **Add Browser Source:**
   - Click `+` in Sources panel
   - Select "Browser"
   - Name it "Zeppelin Game"

3. **Configure Browser Source:**
   - URL: Your Zeppelin game URL
   - Width: 1920
   - Height: 1080
   - FPS: 30

4. **Launch Windowed Projector:**
   - Right-click on "Zeppelin Game" source
   - Select **"Windowed Projector (Source)"**
   - A new window opens showing just the game

5. **Position the Projector:**
   - Move the projector window to a stable position
   - Make it large enough to see numbers clearly
   - Keep it visible (don't minimize)

---

### Step 2: Define ROI (Region of Interest)

```bash
python3 obs_capture.py --mode setup
```

**What happens:**
1. Script asks for region names (e.g., "result")
2. You press Enter when projector is ready
3. A screenshot appears
4. Click and drag to select where the result number appears
5. Press SPACE to confirm
6. Configuration saves to `obs_roi_config.json`

---

### Step 3: Test with Live Preview

```bash
python3 obs_capture.py --mode preview --region result
```

You'll see live capture with extracted values. Press 'q' to quit.

---

### Step 4: Start Continuous Monitoring

```bash
python3 obs_capture.py --mode monitor --duration 30 --interval 2
```

Monitors for 30 minutes, capturing every 2 seconds.

---

### Step 5: Make Predictions

```bash
python3 main.py --data-file data/obs_capture_data.csv --steps 5 --visualize
```

---

## 🎨 Image Preprocessing Pipeline

The system applies these steps to maximize OCR accuracy:

1. Grayscale Conversion
2. Upscaling (3x)
3. Gaussian Blur
4. Adaptive Thresholding
5. Morphological Operations

Result: Clean, high-contrast text that Tesseract can read accurately.

---

## 🎯 Why This Method Works

### Traditional Scraping (Blocked):
```
Browser → HTML/DOM → JavaScript → Anti-Bot Detection ❌
```

### OBS Projector Method (Works):
```
Website → OBS Video Feed → Screen Capture → OCR → Data ✅
```

**Key Advantages:**
1. No DOM interaction - Never touches HTML/JavaScript
2. Appears as video - Anti-bot can't detect
3. Human-like - Just watching a video feed
4. Stable capture - Windowed projector isolates source
5. No browser automation - No Selenium signatures

---

## 🐛 Troubleshooting

### "Tesseract not found"
```bash
brew install tesseract
```

### "No numbers detected"
1. Increase ROI size
2. Check text contrast
3. Use preview mode to debug

### "OBS Projector not found"
1. Make sure projector window is open
2. Use manual ROI selection

---

## 📊 Quick Commands

```bash
# Setup ROI regions
python3 obs_capture.py --mode setup

# Test with preview
python3 obs_capture.py --mode preview --region result

# Monitor for 30 minutes
python3 obs_capture.py --mode monitor --duration 30

# Make predictions
python3 main.py --data-file data/obs_capture_data.csv --steps 5 --visualize
```
