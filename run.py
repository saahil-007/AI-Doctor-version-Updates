import os
import sys
import threading
from app import app as main_app
from arena_backend import create_arena_app

def run_main_app():
    """Run the main Flask app on port 5000"""
    main_app.run(debug=False, host='0.0.0.0', port=5000)

def run_arena_app():
    """Run the arena Flask app on port 5001"""
    arena_app = create_arena_app()
    arena_app.run(debug=False, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    # Create threads for both apps
    main_thread = threading.Thread(target=run_main_app)
    arena_thread = threading.Thread(target=run_arena_app)
    
    # Start both apps
    main_thread.start()
    arena_thread.start()
    
    # Wait for both threads to complete
    main_thread.join()
    arena_thread.join()