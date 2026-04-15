#!/usr/bin/env python3
"""
Simple HTTP server to serve dashboard and images
Run this script to start the dashboard server
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to set correct MIME types"""
    
    def end_headers(self):
        # Add CORS headers to allow loading images
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def guess_type(self, path):
        """Set correct MIME type for PNG files"""
        if path.endswith('.png'):
            return 'image/png'
        return super().guess_type(path)

def main():
    # Change to the directory containing dashboard.html
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create results directory if it doesn't exist
    results_dir = script_dir / 'results'
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)
        print(f"[INFO] Created results directory: {results_dir}")
    
    # Start server
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print("=" * 60)
        print("ML DASHBOARD SERVER")
        print("=" * 60)
        print(f"Server running at: http://localhost:{PORT}")
        print(f"Dashboard URL: http://localhost:{PORT}/dashboard.html")
        print("\nMake sure your results are in the 'results' folder:")
        print("  - results/training_history.png")
        print("  - results/reconstructions.png")
        print("  - results/rbm_filters.png")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)
        
        # Open browser automatically
        webbrowser.open(f'http://localhost:{PORT}/dashboard.html')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n[INFO] Server stopped")
            httpd.shutdown()

if __name__ == "__main__":
    main()