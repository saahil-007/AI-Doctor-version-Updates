import os
from app import app
from http.server import BaseHTTPRequestHandler
import json

# Change to the project directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # For GET requests, return a simple response
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write('Flask API is running'.encode('utf-8'))
        return

    def do_POST(self):
        # For POST requests, forward to the Flask app
        # Get the request data
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b''
        
        # Strip the /api prefix from the path since Flask routes don't include it
        path = self.path
        if path.startswith('/api'):
            path = path[4:]  # Remove '/api' prefix
        
        # Create a test client to handle the request
        with app.test_client() as client:
            # Make the request to the Flask app
            response = client.open(
                path=path,
                method='POST',
                headers=dict(self.headers),
                data=post_data,
                query_string=self.path.split('?')[1] if '?' in self.path else ''
            )
            
            # Send the response back
            self.send_response(response.status_code)
            for key, value in response.headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.get_data())

# Keep the app import for local development
if __name__ == "__main__":
    app.run()