#!/usr/bin/env python3
"""
Launcher script for TinyML Predictive Maintenance Dashboard
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting TinyML Predictive Maintenance Dashboard...")
    print("📊 Dashboard will open in your default browser")
    print("🔧 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()