# Image Quality Checker API

A Flask-based REST API that performs quality checks on uploaded images. The API uses asynchronous processing and includes security features, rate limiting, and comprehensive logging.

## Features

- Image quality analysis (resolution, file size, aspect ratio)
- Multiple security checks for uploaded files
- Asynchronous processing using Celery
- Rate limiting to prevent abuse
- Comprehensive logging system
- Support for JPEG and PNG formats
- Real-time status checking for submitted tasks

## Requirements

- Python 3.8+
- Redis server (for Celery and rate limiting)
- System dependencies for python-magic

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd image-quality-checker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note for Windows users:
- The application uses python-magic-bin which includes necessary DLL files
- No additional libmagic installation is required
- Make sure to install Redis for Windows from https://github.com/microsoftarchive/redis/releases

Redis Installation for Windows:
1. Download Redis-x64-xxx.msi from https://github.com/microsoftarchive/redis/releases
2. Run the installer
3. Start Redis using one of these methods:
    
    Method 1 - Using the Redis CLI:
    ```bash
    redis-server
    ```
    
    Method 2 - Using the Windows Services:
    - Open Services (Win + R, type 'services.msc')
    - Find "Redis" in the list
    - Right-click and select "Start"
    
    Method 3 - Using Command Prompt as Administrator:
    ```bash
    net start Redis
    ```

To verify Redis is running:
```bash
redis-cli ping
```
You should see "PONG" as the response

Troubleshooting Redis:
1. If Redis service won't start, try:
   ```bash
   net stop Redis
   net start Redis
   ```

2. Or run Redis directly:
   ```bash
   redis-server
   ```

3. Verify Redis is running on port 6379:
   ```bash
   netstat -an | findstr 6379
   ```

3. Run the application:

Then, in a new terminal, start the Celery worker:
```bash
python -m celery -A app.celery worker -P solo --loglevel=info
```

If you encounter any issues:
1. Make sure Redis is running and responding with:
   ```bash
   redis-cli ping
   ```

2. Try clearing Redis cache:
   ```bash
   redis-cli
   > FLUSHALL
   > exit
   ```

3. Start the Flask application in a new terminal:
   ```bash
   python app.py
   ```