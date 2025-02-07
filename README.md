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
2. Install dependencies:
3. Run the application:

