"""
Lightweight demo version of the Bird Species Identification Web App

This version works without requiring PyTorch/heavy dependencies,
perfect for quick testing and demonstrations.
"""

import os
import json
import random
import numpy as np
from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bird-identification-demo-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class MockBirdPipeline:
    """Lightweight mock pipeline for demonstration."""
    
    def __init__(self):
        """Initialize with mock data."""
        logger.info("Initializing mock bird identification pipeline...")
        
        # Load species database
        self.species_names = {
            1: "Black-footed Albatross", 2: "Laysan Albatross", 3: "Sooty Albatross",
            4: "Groove-billed Ani", 5: "Crested Auklet", 6: "Least Auklet",
            7: "Parakeet Auklet", 8: "Rhinoceros Auklet", 9: "Brewer Blackbird",
            10: "Red-winged Blackbird", 11: "Rusty Blackbird", 12: "Yellow-headed Blackbird",
            13: "Bobolink", 14: "Indigo Bunting", 15: "Lazuli Bunting",
            16: "Painted Bunting", 17: "Northern Cardinal", 18: "Spotted Catbird",
            19: "Gray Catbird", 20: "Yellow-breasted Chat", 21: "Eastern Towhee",
            22: "Chuck-will's-widow", 23: "Brandt Cormorant", 24: "Red-faced Cormorant",
            25: "Pelagic Cormorant", 26: "Bronzed Cowbird", 27: "Shiny Cowbird",
            28: "Brown Creeper", 29: "American Crow", 30: "Fish Crow",
            31: "Black-billed Cuckoo", 32: "Mangrove Cuckoo", 33: "Yellow-billed Cuckoo",
            34: "Gray-crowned Rosy Finch", 35: "Purple Finch", 36: "Northern Flicker",
            37: "Acadian Flycatcher", 38: "Great Crested Flycatcher", 39: "Least Flycatcher",
            40: "Olive-sided Flycatcher", 41: "Scissor-tailed Flycatcher", 42: "Vermilion Flycatcher",
            43: "Yellow-bellied Flycatcher", 44: "Magnificent Frigatebird", 45: "Northern Fulmar",
            46: "Gadwall", 47: "American Goldfinch", 48: "European Goldfinch",
            49: "Boat-tailed Grackle", 50: "Common Grackle"
        }
        
        # Fill remaining species
        for i in range(51, 201):
            self.species_names[i] = f"Bird Species {i}"
        
        # Mock attribute groups with realistic names
        self.attribute_descriptions = [
            "sharp pointed bill", "hooked bill", "curved bill", "long bill", "short bill",
            "striped head pattern", "solid head color", "capped head", "black crown", "white crown",
            "blue wings", "brown wings", "black wings", "striped wing pattern", "solid wing color",
            "speckled upperparts", "solid upperparts", "blue back", "brown back", "gray back",
            "white underparts", "yellow underparts", "spotted breast", "streaked breast", "solid breast",
            "long tail", "short tail", "forked tail", "rounded tail", "pointed tail",
            "small size", "medium size", "large size"
        ]
        
        logger.info("Mock pipeline initialized successfully")
    
    def analyze_image_colors(self, image_path):
        """Analyze image to generate realistic mock attributes."""
        try:
            # Load image and analyze basic properties
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Calculate color statistics
            mean_colors = np.mean(image_array, axis=(0, 1))
            std_colors = np.std(image_array, axis=(0, 1))
            
            # Generate attributes based on image properties
            attributes = []
            
            # Bill attributes (based on image sharpness/edges)
            bill_attrs = ["sharp pointed bill", "hooked bill", "curved bill"]
            attributes.extend(random.sample(bill_attrs, k=random.randint(1, 2)))
            
            # Color-based attributes
            r, g, b = mean_colors
            if b > r and b > g:  # More blue
                attributes.extend(["blue wings", "blue back"])
            elif r > g and r > b:  # More red/warm
                attributes.extend(["brown wings", "brown back"])
            else:  # More green/neutral
                attributes.extend(["gray back", "brown wings"])
            
            # Size attributes (based on image dimensions)
            width, height = image.size
            if max(width, height) > 1000:
                attributes.append("large size")
            elif max(width, height) > 500:
                attributes.append("medium size")
            else:
                attributes.append("small size")
            
            # Add some random attributes
            extra_attrs = random.sample(self.attribute_descriptions, k=random.randint(3, 6))
            attributes.extend(extra_attrs)
            
            # Remove duplicates and limit
            attributes = list(set(attributes))[:8]
            
            return attributes
            
        except Exception as e:
            logger.warning(f"Error analyzing image: {e}")
            # Return random attributes as fallback
            return random.sample(self.attribute_descriptions, k=random.randint(4, 8))
    
    def identify_bird(self, image_path):
        """Mock bird identification pipeline."""
        try:
            # Analyze image for attributes
            detected_attributes = self.analyze_image_colors(image_path)
            
            # Generate mock candidates
            candidate_species = random.sample(list(self.species_names.keys()), k=5)
            candidates = []
            
            for i, species_id in enumerate(candidate_species):
                similarity = random.uniform(0.95 - i*0.05, 0.99 - i*0.05)  # Decreasing similarity
                candidates.append({
                    'species_id': species_id,
                    'species_name': self.species_names[species_id],
                    'similarity': similarity
                })
            
            # Select top candidate
            top_candidate = candidates[0]
            
            # Generate reasoning
            reasoning_templates = [
                f"Based on the observed {', '.join(detected_attributes[:3])}, this species shows the strongest match.",
                f"The combination of {detected_attributes[0]} and {detected_attributes[1] if len(detected_attributes) > 1 else 'other features'} is most consistent with this species.",
                f"Key distinguishing features including {detected_attributes[0]} align well with this species profile.",
                f"The visual characteristics, particularly {', '.join(detected_attributes[:2])}, support this identification."
            ]
            
            reasoning = random.choice(reasoning_templates)
            confidence = random.uniform(0.75, 0.95)
            
            # Format attribute description
            attr_description = f"Bird with {', '.join(detected_attributes)}"
            
            result = {
                'success': True,
                'attribute_stats': {
                    'total_active': len(detected_attributes),
                    'groups_covered': random.randint(4, 7),
                    'description': attr_description
                },
                'prediction': {
                    'predicted_species_id': top_candidate['species_id'],
                    'predicted_species_name': top_candidate['species_name'],
                    'confidence': round(confidence, 3),
                    'reasoning': reasoning,
                    'all_candidates': candidates
                },
                'processing_info': {
                    'model_type': 'Mock ResNet + Attribute Pipeline',
                    'num_attributes': 312,
                    'candidates_considered': len(candidates)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in mock bird identification: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

# Initialize mock pipeline
pipeline = MockBirdPipeline()

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(random.random() * 1000000))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run bird identification
        result = pipeline.identify_bird(filepath)
        
        # Add image URL for display
        if result['success']:
            result['image_url'] = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'pipeline_loaded': True,
        'demo_mode': True,
        'species_count': len(pipeline.species_names)
    })

def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    print("🐦 Starting Bird Species Identification Demo App...")
    print("📍 Open your browser to: http://localhost:5001")
    print("🎯 Demo Mode: Using mock AI for instant results")
    print("💡 Upload any bird image to see the identification pipeline in action!")
    print("-" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
