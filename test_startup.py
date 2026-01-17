
import os
import sys
import shutil

# Simulate clean environment
if os.path.exists('data'):
    shutil.rmtree('data')

print("Starting simulation with NO data directory...")

try:
    import realtime_dashboard
    print("Application imported successfully.")
    
    # Simulate layout generation
    print("Simulating layout generation...")
    layout = realtime_dashboard.app.layout
    print("Layout generated successfully.")
    
    # Simulate create_dashboard explicitly to be sure
    print("Running create_dashboard()...")
    realtime_dashboard.create_dashboard()
    print("create_dashboard() passed.")
    
except Exception as e:
    print(f"\nCRASH DETECTED:\n{e}")
    import traceback
    traceback.print_exc()
