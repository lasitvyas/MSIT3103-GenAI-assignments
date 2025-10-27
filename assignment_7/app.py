"""
FastAPI Application for Generative AI Model Deployment
This app provides a REST API for text generation using GPT-2
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Generative AI API",
    description="REST API for text generation using GPT-2 model",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

# Request models
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt for generation")
    max_length: Optional[int] = Field(100, description="Maximum length of generated text", ge=10, le=500)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1, le=100)
    top_p: Optional[float] = Field(0.9, description="Top-p (nucleus) sampling parameter", ge=0.1, le=1.0)
    num_return_sequences: Optional[int] = Field(1, description="Number of sequences to generate", ge=1, le=5)

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Artificial intelligence is",
                "max_length": 100,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "num_return_sequences": 1
            }
        }

class TextGenerationResponse(BaseModel):
    generated_texts: List[str]
    prompt: str
    generation_time: float
    model_info: dict
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

# Startup event
@app.on_event("startup")
async def load_model():
    """Load the GPT-2 model and tokenizer on startup"""
    global model, tokenizer, device
    
    try:
        logger.info("Loading GPT-2 model and tokenizer...")
        start_time = time.time()
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.to(device)
        model.eval()
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Generative AI REST API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "docs": "/docs",
            "model_info": "/model-info"
        }
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API and model status"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        timestamp=datetime.now().isoformat()
    )

# Model info endpoint
@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "GPT-2",
        "model_type": "Causal Language Model",
        "parameters": "124M",
        "device": str(device),
        "framework": "PyTorch",
        "library": "Transformers (Hugging Face)",
        "capabilities": ["text_generation", "text_completion"],
        "max_length": 1024
    }

# Text generation endpoint
@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """
    Generate text based on the input prompt
    
    Parameters:
    - prompt: Input text to complete
    - max_length: Maximum length of generated text (10-500)
    - temperature: Controls randomness (0.1-2.0, higher = more random)
    - top_k: Limits vocabulary to top k tokens
    - top_p: Nucleus sampling threshold
    - num_return_sequences: Number of different completions to generate
    
    Returns:
    - Generated text(s) with metadata
    """
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Encode input
        input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                num_return_sequences=request.num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {len(generated_texts)} text(s) in {generation_time:.3f}s")
        
        return TextGenerationResponse(
            generated_texts=generated_texts,
            prompt=request.prompt,
            generation_time=round(generation_time, 3),
            model_info={
                "model": "GPT-2",
                "device": str(device),
                "parameters_used": {
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_k": request.top_k,
                    "top_p": request.top_p
                }
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Batch generation endpoint
@app.post("/generate-batch")
async def generate_batch(prompts: List[str], max_length: int = 100):
    """
    Generate text for multiple prompts in batch
    
    Parameters:
    - prompts: List of input prompts
    - max_length: Maximum length for each generation
    """
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(prompts) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 prompts per batch")
    
    try:
        results = []
        total_start = time.time()
        
        for prompt in prompts:
            request = TextGenerationRequest(
                prompt=prompt,
                max_length=max_length,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                num_return_sequences=1
            )
            response = await generate_text(request)
            results.append({
                "prompt": prompt,
                "generated_text": response.generated_texts[0],
                "generation_time": response.generation_time
            })
        
        total_time = time.time() - total_start
        
        return {
            "results": results,
            "total_prompts": len(prompts),
            "total_time": round(total_time, 3),
            "average_time_per_prompt": round(total_time / len(prompts), 3)
        }
        
    except Exception as e:
        logger.error(f"Error during batch generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
