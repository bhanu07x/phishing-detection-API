from flask import Flask, request, jsonify
import os
from phishing_detector import PhishingDetector  # Import your detector class

app = Flask(__name__)

# Initialize the phishing detector and load the trained model
detector = PhishingDetector()
detector.load_model(
    model_path='phishing_detector_model.pkl',
    scaler_path='phishing_detector_scaler.pkl'
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running"""
    return jsonify({"status": "healthy", "message": "Phishing detection API is running"}), 200

@app.route('/api/detect', methods=['POST'])
def detect_phishing():
    """
    Endpoint to detect if a URL is a phishing website
    
    Expected JSON input:
    {
        "url": "https://example.com"
    }
    
    Returns:
    {
        "url": "https://example.com",
        "is_phishing": true/false,
        "confidence": 0.95,
        "processing_time": 1.23
    }
    """
    try:
        # Get URL from request
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({"error": "Missing URL parameter"}), 400
            
        url = data['url']
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return jsonify({"error": "Invalid URL format. URL must start with http:// or https://"}), 400
        
        # Make prediction
        result = detector.predict(url)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-detect', methods=['POST'])
def batch_detect():
    """
    Endpoint to detect multiple URLs at once
    
    Expected JSON input:
    {
        "urls": ["https://example1.com", "https://example2.com"]
    }
    
    Returns:
    {
        "results": [
            {
                "url": "https://example1.com",
                "is_phishing": true,
                "confidence": 0.95,
                "processing_time": 1.23
            },
            {
                "url": "https://example2.com",
                "is_phishing": false,
                "confidence": 0.87,
                "processing_time": 0.95
            }
        ]
    }
    """
    try:
        # Get URLs from request
        data = request.get_json()
        
        if not data or 'urls' not in data:
            return jsonify({"error": "Missing URLs parameter"}), 400
            
        urls = data['urls']
        
        if not isinstance(urls, list):
            return jsonify({"error": "URLs parameter must be a list"}), 400
            
        # Process each URL
        results = []
        for url in urls:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                results.append({
                    "url": url,
                    "error": "Invalid URL format. URL must start with http:// or https://"
                })
                continue
                
            # Make prediction
            try:
                result = detector.predict(url)
                results.append(result)
            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e)
                })
        
        return jsonify({"results": results}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)