import os
from app import app

# Change to the project directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Vercel expects a 'handler' function for serverless functions
# This handler will process incoming requests using the Flask app's WSGI interface
def handler(event, context):
    # Extract request information from the event
    path = event.get('path', '/')
    
    # Strip the /api prefix from the path since Flask routes don't include it
    if path.startswith('/api'):
        path = path[4:]  # Remove '/api' prefix
    
    http_method = event.get('httpMethod', 'GET')
    headers = event.get('headers', {})
    query_string = event.get('queryStringParameters') or {}
    body = event.get('body', '')
    
    # Create a test client to handle the request
    with app.test_client() as client:
        # Make the request to the Flask app
        response = client.open(
            path=path,
            method=http_method,
            headers=headers,
            data=body,
            query_string=query_string
        )
        
        # Return the response in Vercel's expected format
        return {
            'statusCode': response.status_code,
            'headers': dict(response.headers),
            'body': response.get_data(as_text=True)
        }

# Keep the app import for local development
if __name__ == "__main__":
    app.run()