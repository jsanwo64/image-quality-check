from flask import Flask, request, jsonify
from PIL import Image
import io
from celery import Celery
import uuid
import magic
import filetype
import os
import logging
import logging.handlers
import time
from datetime import datetime
import json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
import cv2  # Add this import for OpenCV

# Configure logging
def setup_logging():
    """Configure logging settings"""
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        'logs/app.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    
    # Error log file handler
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/error.log',
        maxBytes=10485760,
        backupCount=5
    )
    error_handler.setFormatter(log_formatter)
    error_handler.setLevel(logging.ERROR)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    return root_logger

logger = setup_logging()

app = Flask(__name__)

# Create Celery subclass to properly integrate with Flask
class FlaskCelery(Celery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'app' in kwargs:
            self.init_app(kwargs['app'])

    def init_app(self, app):
        self.app = app
        self.config_from_object(app.config)

# Initialize Celery with Redis
celery = FlaskCelery(
    'app',
    backend='redis://localhost:6379/0',
    broker='redis://localhost:6379/0'
)

# Configure Celery
celery.conf.update({
    'broker_url': 'redis://localhost:6379/0',
    'result_backend': 'redis://localhost:6379/0',
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'enable_utc': True,
    'broker_transport_options': {'visibility_timeout': 3600},
    'worker_pool': 'solo'
})

# Initialize with Flask app
celery.init_app(app)

# Constants for image quality checks
MAX_FILE_SIZE_MB = 10  # 10 MB
MIN_WIDTH = 378
MIN_HEIGHT = 438
MAX_FILE_SIZE_KB = MAX_FILE_SIZE_MB * 1024  # Convert MB to KB

# Allowed image formats and their MIME types
ALLOWED_FORMATS = {
    'jpeg': 'image/jpeg',
    'jpg': 'image/jpeg',
    'png': 'image/png'
}

# Configure rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379/1",
    strategy="fixed-window", # or "moving-window"
)

# Rate limit exceeded handler
@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Rate limit exceeded for IP: {get_remote_address()}")
    return jsonify({
        "success": False,
        "error": "Rate limit exceeded. Please try again later.",
        "reset_time": str(e.reset_time),
        "retry_after": e.retry_after
    }), 429

def log_request_details(request_id, action, details):
    """Log request details with structured data"""
    log_data = {
        'request_id': request_id,
        'timestamp': datetime.utcnow().isoformat(),
        'action': action,
        'details': details
    }
    logger.info(json.dumps(log_data))

def is_safe_image(image_data, request_id):
    """
    Perform multiple security checks on the image
    Returns (is_safe, error_message)
    """
    try:
        # Check 1: Basic format validation using filetype instead of imghdr
        kind = filetype.guess(image_data)
        if not kind or kind.mime.split('/')[0] != 'image' or kind.extension not in ALLOWED_FORMATS:
            logger.warning(f"Request {request_id}: Invalid format detected: {kind.extension if kind else 'unknown'}")
            return False, f"Invalid image format. Allowed formats: {', '.join(ALLOWED_FORMATS.keys())}"

        # Check 2: MIME type validation using python-magic
        mime = magic.from_buffer(image_data, mime=True)
        if mime not in ALLOWED_FORMATS.values():
            logger.warning(f"Request {request_id}: Invalid MIME type detected: {mime}")
            return False, f"Invalid MIME type: {mime}"

        # Check 3: Try opening with PIL to verify image integrity
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                img.verify()
                
            # Check 4: Second pass to check for image corruption
            with Image.open(io.BytesIO(image_data)) as img:
                img.transpose(Image.FLIP_LEFT_RIGHT)
                
        except Exception as e:
            logger.error(f"Request {request_id}: Image integrity check failed: {str(e)}")
            return False, f"Image integrity check failed: {str(e)}"

        logger.info(f"Request {request_id}: Image passed all security checks")
        return True, None

    except Exception as e:
        logger.error(f"Request {request_id}: Security check failed: {str(e)}", exc_info=True)
        return False, f"Security check failed: {str(e)}"

# Add constants for blur detection
MIN_BLUR_VARIANCE = 100  # Adjust this threshold based on your needs

def check_blur(image_data):
    """
    Check image blurriness using Laplacian variance method
    Returns (is_sharp, variance)
    """
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        
        # Check if image is sharp enough
        is_sharp = variance >= MIN_BLUR_VARIANCE
        
        return is_sharp, variance
    except Exception as e:
        logger.error(f"Error checking blur: {str(e)}")
        return False, 0

@celery.task(name='app.check_image_quality')
def check_image_quality(image_data, request_id):
    """Analyze image quality metrics"""
    start_time = time.time()
    try:
        # Initialize quality issues list at the very beginning
        quality_issues = []

        # Perform security check before processing
        is_safe, error_message = is_safe_image(image_data, request_id)
        if not is_safe:
            logger.error(f"Request {request_id}: Security check failed in task: {error_message}")
            return {
                "success": False,
                "error": f"Security check failed: {error_message}"
            }

        # Open image using PIL
        img = Image.open(io.BytesIO(image_data))
        
        # Get image properties
        width, height = img.size
        format = img.format.lower()
        mode = img.mode
        file_size_kb = len(image_data) / 1024
        
        # Check for blurriness
        is_sharp, blur_variance = check_blur(image_data)
        blur_score = round(blur_variance, 2)  # Round to 2 decimal places
        
        if not is_sharp:
            quality_issues.append(f"Image is blurry (score: {blur_score}, minimum required: {MIN_BLUR_VARIANCE})")
        
        # Log image details with blur info
        log_request_details(request_id, 'image_analysis', {
            'width': width,
            'height': height,
            'format': format,
            'mode': mode,
            'file_size_kb': file_size_kb,
            'blur_variance': blur_variance
        })
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Additional quality checks
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            quality_issues.append(f"Low resolution: {width}x{height}. Minimum required: {MIN_WIDTH}x{MIN_HEIGHT}")
            
        if file_size_kb > MAX_FILE_SIZE_KB:
            quality_issues.append(f"File size too large: {file_size_kb:.2f}KB. Maximum allowed: {MAX_FILE_SIZE_KB}KB")
        
        if format not in ALLOWED_FORMATS:
            quality_issues.append(f"Invalid format: {format}. Allowed formats: {', '.join(ALLOWED_FORMATS.keys())}")
        
        result = {
            "success": True,
            "format": format,
            "mode": mode,
            "width": width,
            "height": height,
            "aspect_ratio": f"{aspect_ratio:.2f}",
            "file_size_kb": f"{file_size_kb:.2f}",
            "blur": {
                "is_blurry": not is_sharp,
                "score": blur_score,
                "threshold": MIN_BLUR_VARIANCE
            },
            "quality_issues": quality_issues,
            "passes_quality_check": len(quality_issues) == 0,
            "processing_time": f"{time.time() - start_time:.2f}s"
        }
        
        logger.info(f"Request {request_id}: Image analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Request {request_id}: Error processing image: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/check-image', methods=['POST'])
@limiter.limit("10 per minute", error_message="Too many image uploads. Please wait before trying again.")
def check_image():
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log request with IP address
    logger.info(f"Request {request_id}: New image check request received from IP: {get_remote_address()}")
    
    if 'image' not in request.files:
        logger.warning(f"Request {request_id}: No image file provided")
        return jsonify({
            "success": False,
            "error": "No image file provided"
        }), 400
        
    image_file = request.files['image']
    
    if image_file.filename == '':
        logger.warning(f"Request {request_id}: Empty filename provided")
        return jsonify({
            "success": False,
            "error": "No selected file"
        }), 400
    
    # Log file details
    log_request_details(request_id, 'file_received', {
        'filename': image_file.filename,
        'content_type': image_file.content_type
    })
    
    # Read the file data
    image_data = image_file.read()
    
    # Perform security checks
    is_safe, error_message = is_safe_image(image_data, request_id)
    if not is_safe:
        return jsonify({
            "success": False,
            "error": error_message
        }), 400
    
    # Check file size before processing
    file_size_kb = len(image_data) / 1024
    if file_size_kb > MAX_FILE_SIZE_KB:
        logger.warning(f"Request {request_id}: File size too large: {file_size_kb:.2f}KB")
        return jsonify({
            "success": False,
            "error": f"File size too large: {file_size_kb:.2f}KB. Maximum allowed: {MAX_FILE_SIZE_KB}KB"
        }), 400
    
    # Submit task to Celery
    task = check_image_quality.delay(image_data, request_id)
    
    processing_time = time.time() - start_time
    logger.info(f"Request {request_id}: Task submitted to Celery. Processing time: {processing_time:.2f}s")
    
    # Return task ID immediately
    return jsonify({
        "success": True,
        "task_id": task.id,
        "status": "Processing",
        "request_id": request_id
    })

@app.route('/check-status/<task_id>', methods=['GET'])
@limiter.limit("30 per minute", error_message="Too many status checks. Please wait before trying again.")
def check_status(task_id):
    """Check the status of an image processing task"""
    logger.info(f"Status check requested for task {task_id} from IP: {get_remote_address()}")
    
    task = check_image_quality.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'status': 'Processing',
            'result': None
        }
    elif task.state == 'SUCCESS':
        response = {
            'status': 'Completed',
            'result': task.get()
        }
    else:
        logger.error(f"Task {task_id} failed: {str(task.result)}")
        response = {
            'status': 'Failed',
            'result': str(task.result)
        }
    
    logger.info(f"Status for task {task_id}: {response['status']}")
    return jsonify(response)

# Add exempt routes or custom limits for specific IPs if needed
@limiter.request_filter
def ip_whitelist():
    return request.remote_addr == "127.0.0.1"

# Configure per-route limits
CUSTOM_LIMITS = {
    "check_image": {
        "default": ["10 per minute", "100 per hour", "500 per day"],
        "description": "Limits for image upload endpoint"
    },
    "check_status": {
        "default": ["30 per minute", "300 per hour"],
        "description": "Limits for status check endpoint"
    }
}

def update_rate_limits():
    """Update rate limits based on configuration"""
    for route, config in CUSTOM_LIMITS.items():
        if hasattr(app.view_functions.get(route), '_rate_limit_string'):
            app.view_functions[route]._rate_limit_string = config['default']

# Initialize custom rate limits
update_rate_limits()

# Add monitoring endpoint for rate limit status
@app.route('/rate-limit-status', methods=['GET'])
@limiter.exempt
def rate_limit_status():
    if request.remote_addr != "127.0.0.1":
        return jsonify({"error": "Access denied"}), 403
        
    return jsonify({
        "limits": CUSTOM_LIMITS,
        "current_usage": {
            "check_image": limiter.get_window_stats("check_image"),
            "check_status": limiter.get_window_stats("check_status")
        }
    })

# Add middleware to log rate limit hits
@app.before_request
def log_request_info():
    if request.endpoint in CUSTOM_LIMITS:
        remaining = getattr(request, 'view_rate_limit', None)
        if remaining:
            logger.info(
                f"Rate limit for {request.endpoint}: "
                f"{remaining.remaining}/{remaining.limit} remaining. "
                f"Reset in {remaining.reset_seconds} seconds"
            )

# Update error handlers to include rate limit information
@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(
        f"Rate limit exceeded for IP: {get_remote_address()}, "
        f"Endpoint: {request.endpoint}, "
        f"Reset in: {e.retry_after} seconds"
    )
    return jsonify({
        "success": False,
        "error": "Rate limit exceeded. Please try again later.",
        "reset_time": str(e.reset_time),
        "retry_after": e.retry_after,
        "limit": str(e.description)
    }), 429

if __name__ == '__main__':
    app.run(debug=True)
