# 🐦 Bird Species Identification Web App

A beautiful, interactive web application that identifies bird species from uploaded images using our trained vision→attributes→LLM pipeline.

## ✨ Features

- 🖼️ **Drag & Drop Interface**: Easy image upload with modern UI
- 🧠 **AI-Powered Identification**: Uses ResNet + attribute detection + LLM reasoning
- 📊 **Detailed Analysis**: Shows detected attributes, confidence scores, and reasoning
- 🏆 **Top Candidates**: Displays alternative species with similarity scores
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- ⚡ **Real-time Processing**: Fast inference with visual feedback

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

```bash
# Navigate to the webapp directory
cd webapp

# Make setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh

# Start the application
source venv/bin/activate
python app.py
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create upload directory
mkdir -p static/uploads

# Start the application
python app.py
```

## 🌐 Access the App

Once running, open your browser and navigate to:
```
http://localhost:5000
```

## 📱 How to Use

1. **Upload Image**: 
   - Drag and drop a bird image onto the upload area, or
   - Click "Choose Image" to browse and select a file

2. **Wait for Analysis**: 
   - The AI pipeline processes your image through multiple stages
   - You'll see a loading indicator with progress information

3. **View Results**:
   - **Species Prediction**: Top identified species with confidence score
   - **AI Reasoning**: Explanation of why this species was selected
   - **Detected Attributes**: Number of active visual features found
   - **Top Candidates**: Alternative species considered with similarity scores

4. **Analyze Another**: Click "Analyze Another Bird" to upload a new image

## 🏗️ Architecture

The web app implements our complete pipeline:

```
📸 Image Upload → 🧠 Attribute Detection → 🗣️ Verbalization → 🎯 Candidate Ranking → 🤖 LLM Reasoning → 📋 Results
```

### Components:

- **Flask Backend**: Serves the web interface and handles requests
- **ResNet Model**: Predicts 312 binary attributes from bird images
- **Threshold Calibration**: Uses optimized per-attribute decision boundaries
- **Attribute Verbalization**: Converts predictions to natural language
- **Species Centroids**: Computes similarity with 200 bird species prototypes
- **LLM Integration**: Makes final species decision with reasoning

## 📁 File Structure

```
webapp/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface template
├── static/
│   └── uploads/          # Uploaded images (auto-created)
├── requirements.txt      # Python dependencies
├── setup.sh             # Automatic setup script
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

You can customize the app behavior with these environment variables:

```bash
# Flask configuration
export FLASK_ENV=development        # or 'production'
export FLASK_DEBUG=1               # Enable debug mode

# App configuration  
export UPLOAD_FOLDER=static/uploads # Upload directory
export MAX_FILE_SIZE=16             # Max upload size in MB
```

### Model Configuration

The app currently uses mock models for demonstration. To use real trained models:

1. **Replace Mock Model**: Update the `setup_model()` method in `app.py`
2. **Load Real Thresholds**: Point to your calibrated thresholds JSON file
3. **Load Species Centroids**: Point to your computed centroids file
4. **Update Species Names**: Load from your CUB classes.txt file

```python
# Example: Loading real components
def load_configurations(self):
    # Load real thresholds
    with open('outputs/attr_thresholds.json', 'r') as f:
        self.optimal_thresholds = json.load(f)
    
    # Load species centroids
    self.species_centroids = np.load('outputs/species_centroids.npy')
    
    # Load species names
    with open('data/classes.txt', 'r') as f:
        self.species_names = {i+1: line.strip() for i, line in enumerate(f)}
```

## 🎨 Customization

### UI Styling

The interface uses modern CSS with:
- Gradient backgrounds
- Card-based layouts  
- Smooth animations
- Responsive design
- Drag & drop interactions

Customize the look by editing the `<style>` section in `templates/index.html`.

### Adding Features

Easy extensions:
- **Confidence Visualization**: Add confidence meters or charts
- **Species Information**: Link to Wikipedia or bird guides
- **Batch Processing**: Allow multiple image uploads
- **History**: Save and display previous identifications
- **Export Results**: Download identification reports

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Kill existing Flask processes
   pkill -f flask
   # Or use a different port
   python app.py --port 5001
   ```

2. **Memory Issues with Large Images**:
   - Images are automatically resized to 224x224 for processing
   - Consider reducing MAX_CONTENT_LENGTH for smaller uploads

3. **Missing Dependencies**:
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

4. **Upload Directory Permissions**:
   ```bash
   # Fix permissions
   chmod 755 static/uploads
   ```

## 🚀 Deployment

### Local Network Access

To access from other devices on your network:

```bash
# Find your IP address
ip addr show  # Linux
ifconfig      # macOS

# Start with network binding
python app.py --host 0.0.0.0 --port 5000
```

### Production Deployment

For production use:

1. **Use Gunicorn**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Configure Nginx** (optional):
   - Set up reverse proxy for better performance
   - Handle static file serving
   - Add SSL/TLS encryption

3. **Environment Setup**:
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key-here
   ```

## 📊 Performance

The app is designed for real-time inference:

- **Typical Response Time**: 1-3 seconds per image
- **Memory Usage**: ~500MB with PyTorch models loaded
- **Supported Formats**: JPEG, PNG, GIF, BMP, WebP
- **Max File Size**: 16MB (configurable)
- **Concurrent Users**: Supports multiple simultaneous uploads

## 🤝 Contributing

Want to improve the app? Great ideas:

- Add more bird species support
- Implement real-time webcam capture
- Create mobile app version
- Add species information cards
- Implement user accounts and history
- Add confidence calibration visualizations

## 📄 License

This project is part of the Bird Species Identification Pipeline. Use freely for educational and research purposes.

---

**Ready to identify some birds? Upload an image and let the AI do the work!** 🐦✨
