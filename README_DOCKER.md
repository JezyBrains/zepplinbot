# Zeppelin Tactical OS - Deployment Guide

This repository is container-ready for Dockploy and GitHub Actions.

## 🐳 Docker Deployment

To build and run the Tactical OS dashboard container:

```bash
# Build and start in background
docker-compose up -d --build

# View logs
docker-compose logs -f
```

The dashboard will be available at `http://localhost:8050`.

### Persistent Data
Data is persisted in the `./data` directory. The container maps this volume to `/app/data` to ensure your crash history and training models are saved across restarts.

## 🧩 Browser Extension Setup

The extension has been redesigned with the **Tactical OS (Obsidian Terminal)** aesthetic.

1. Go to `chrome://extensions/`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `browser-extension` folder in this repository.

### Extension Features
- **Tactical HUD**: Monospaced cyber-brutalist interface.
- **Auto-Capture**: Syncs crash data to the Dockerized dashboard.
- **Signal Relay**: Displays real-time `BET` / `WAIT` / `SKIP` signals from the server.
- **In-Page Overlay**: Injects a non-intrusive control panel directly into the betting site.
