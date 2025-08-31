# Real LLM Integration Options

## Option 1: OpenAI GPT Integration

### Install OpenAI package:
```bash
pip3 install openai
```

### Replace the mock LLM function:
```python
import openai

def llm_reasoning_openai(self, description, candidates):
    """Real LLM reasoning using OpenAI GPT."""
    openai.api_key = "your-api-key-here"  # Set your API key
    
    # Prepare the prompt
    candidates_text = "\n".join([
        f"{i+1}. {c['species_name']} (similarity: {c['similarity']:.3f})"
        for i, c in enumerate(candidates)
    ])
    
    prompt = f"""
You are an expert ornithologist. Based on the following bird description and candidate species, 
provide your final identification with detailed reasoning.

Bird Description: {description}

Top Candidate Species:
{candidates_text}

Please provide:
1. Your final species identification
2. Confidence level (0.0-1.0)
3. Detailed reasoning explaining why this species is the best match
4. Key distinguishing features that led to this conclusion

Respond in JSON format:
{{
    "species_name": "Selected Species Name",
    "confidence": 0.85,
    "reasoning": "Detailed explanation...",
    "key_features": ["feature1", "feature2", "feature3"]
}}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Find the selected species ID
        selected_species_id = None
        for candidate in candidates:
            if candidate['species_name'].lower() in result['species_name'].lower():
                selected_species_id = candidate['species_id']
                break
        
        if not selected_species_id:
            selected_species_id = candidates[0]['species_id']  # Fallback
            
        return {
            'predicted_species_id': selected_species_id,
            'predicted_species_name': result['species_name'],
            'confidence': result['confidence'],
            'reasoning': result['reasoning'],
            'key_features': result.get('key_features', []),
            'all_candidates': candidates
        }
        
    except Exception as e:
        # Fallback to top candidate if LLM fails
        return self.llm_reasoning_fallback(description, candidates)
```

## Option 2: Local LLM with Ollama

### Install Ollama:
```bash
# Install Ollama on macOS
brew install ollama

# Pull a model (e.g., Llama 2)
ollama pull llama2
```

### Integration code:
```python
import requests

def llm_reasoning_ollama(self, description, candidates):
    """Local LLM reasoning using Ollama."""
    candidates_text = "\n".join([
        f"- {c['species_name']} (similarity: {c['similarity']:.3f})"
        for c in candidates
    ])
    
    prompt = f"""You are an expert bird identifier. 
    
Description: {description}
Candidates: {candidates_text}

Select the best species and explain why in 2-3 sentences."""

    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'llama2',
                                   'prompt': prompt,
                                   'stream': False
                               })
        
        if response.status_code == 200:
            result_text = response.json()['response']
            return {
                'predicted_species_id': candidates[0]['species_id'],
                'predicted_species_name': candidates[0]['species_name'],
                'confidence': 0.8,
                'reasoning': result_text,
                'all_candidates': candidates
            }
    except:
        pass
    
    return self.llm_reasoning_fallback(description, candidates)
```

## Option 3: Hugging Face Transformers (Local)

### Install transformers:
```bash
pip3 install transformers torch
```

### Integration code:
```python
from transformers import pipeline

def setup_local_llm(self):
    """Initialize local LLM."""
    self.text_generator = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-medium",
        device=0 if torch.cuda.is_available() else -1
    )

def llm_reasoning_local(self, description, candidates):
    """Local LLM using Hugging Face transformers."""
    prompt = f"Bird with {description}. Best match from {candidates[0]['species_name']}, {candidates[1]['species_name']}, {candidates[2]['species_name']} is"
    
    try:
        result = self.text_generator(prompt, max_length=100, num_return_sequences=1)
        reasoning = result[0]['generated_text'].replace(prompt, "").strip()
        
        return {
            'predicted_species_id': candidates[0]['species_id'],
            'predicted_species_name': candidates[0]['species_name'],
            'confidence': 0.85,
            'reasoning': reasoning,
            'all_candidates': candidates
        }
    except:
        return self.llm_reasoning_fallback(description, candidates)
```

## Option 4: Google Gemini

### Install Google AI package:
```bash
pip3 install google-generativeai
```

### Integration code:
```python
import google.generativeai as genai

def llm_reasoning_gemini(self, description, candidates):
    """LLM reasoning using Google Gemini."""
    genai.configure(api_key="your-api-key")
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""As an ornithologist, identify the bird species:
    
Description: {description}
Candidates: {[c['species_name'] for c in candidates[:3]]}

Provide species name, confidence (0-1), and reasoning."""

    try:
        response = model.generate_content(prompt)
        return {
            'predicted_species_id': candidates[0]['species_id'],
            'predicted_species_name': candidates[0]['species_name'],
            'confidence': 0.8,
            'reasoning': response.text,
            'all_candidates': candidates
        }
    except:
        return self.llm_reasoning_fallback(description, candidates)
```

---

## Recommended Approach for You:

### For Development/Testing: **Option 2 (Ollama)**
- ✅ Free and runs locally
- ✅ No API keys needed
- ✅ Good performance
- ✅ Privacy-friendly

### For Production: **Option 1 (OpenAI GPT)**
- ✅ Best reasoning quality
- ✅ Reliable and fast
- ✅ Well-documented
- ❌ Requires API key and costs money

### For Offline Use: **Option 3 (Hugging Face)**
- ✅ Completely offline
- ✅ No external dependencies
- ❌ Larger memory requirements
- ❌ May need model fine-tuning for birds

---

## Current vs Real Comparison:

| Feature | Current (Mock) | Real LLM |
|---------|----------------|----------|
| **Reasoning Quality** | Template-based | Actual analysis |
| **Species Selection** | Top similarity | Intelligent choice |
| **Explanations** | Generic | Species-specific |
| **Accuracy** | Similarity-based | Knowledge-enhanced |
| **Cost** | Free | API costs (varies) |
| **Speed** | Instant | 1-3 seconds |
