from flask import Flask, request, jsonify
from PIL import Image
import io
import imghdr

app = Flask(__name__)

def check_image_quality(image_data):
    """Analyze image quality metrics"""
    try:
        # Open image using PIL
        img = Image.open(io.BytesIO(image_data))
        
        # Get image properties
        width, height = img.size
        format = img.format
        mode = img.mode
        file_size = len(image_data) / 1024  # Size in KB
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Define quality thresholds
        min_width = 800
        min_height = 600
        min_file_size = 50  # KB
        max_file_size = 5000  # KB
        
        # Quality checks
        quality_issues = []
        
        if width < min_width or height < min_height:
            quality_issues.append(f"Low resolution: {width}x{height}. Minimum recommended: {min_width}x{min_height}")
            
        if file_size < min_file_size:
            quality_issues.append(f"File size too small: {file_size:.2f}KB. Minimum recommended: {min_file_size}KB")
            
        if file_size > max_file_size:
            quality_issues.append(f"File size too large: {file_size:.2f}KB. Maximum recommended: {max_file_size}KB")
        
        return {
            "success": True,
            "format": format,
            "mode": mode,
            "width": width,
            "height": height,
            "aspect_ratio": f"{aspect_ratio:.2f}",
            "file_size_kb": f"{file_size:.2f}",
            "quality_issues": quality_issues,
            "passes_quality_check": len(quality_issues) == 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/check-image', methods=['POST'])
def check_image():
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided"
        }), 400
        
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({
            "success": False,
            "error": "No selected file"
        }), 400
        
    # Check if the file is actually an image
    image_data = image_file.read()
    if not imghdr.what(None, image_data):
        return jsonify({
            "success": False,
            "error": "Invalid image file"
        }), 400
    
    # Process the image and return quality metrics
    result = check_image_quality(image_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
