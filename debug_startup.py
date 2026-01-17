
import sys
import traceback

print("🔍 Simulating Gunicorn import...")
try:
    from realtime_dashboard import server
    print("✅ Import successful! Server object found.")
    print(f"Start type: {type(server)}")
except Exception as e:
    print("❌ ERROR DURING IMPORT:")
    traceback.print_exc()
    sys.exit(1)
