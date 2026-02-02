                      
"""
Simple OpenAI-compatible API server using Transformers for models that vLLM doesn't support.
Usage: python tools/transformers_api_server.py --model <model_path> --port <port>
"""
import argparse
import json
import logging
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transformers API Server")

model = None
tokenizer = None
device = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    stream: bool = False

def load_model(model_path: str, device_map: str = "auto"):
    """Load model and tokenizer."""
    global model, tokenizer, device

    logger.info(f"Loading model from {model_path}...")

    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if hasattr(model, "eval"):
        model.eval()

    logger.info("Model loaded successfully")

def format_chat_messages(messages: List[ChatMessage]) -> str:
    """Format chat messages into a prompt string using model's chat template if available."""
                                                   
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        try:
                                                               
            chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            formatted = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            logger.warning(f"Failed to use chat template: {e}, falling back to simple formatting")

    formatted = ""
    for msg in messages:
        role = msg.role
        content = msg.content

        if role == "system":
            formatted += f"System: {content}\n\n"
        elif role == "user":
            formatted += f"User: {content}\n\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n\n"

    formatted += "Assistant:"
    return formatted

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{
            "id": model.config.name_or_path if model else "unknown",
            "object": "model",
            "created": 0,
            "owned_by": "transformers"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
                         
        prompt = format_chat_messages(request.messages)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens or 512,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        response = {
            "id": f"chatcmpl-{hash(prompt)}",
            "object": "chat.completion",
            "created": 0,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": outputs[0].shape[0] - inputs.input_ids.shape[1],
                "total_tokens": outputs[0].shape[0]
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

def main():
    parser = argparse.ArgumentParser(description="Transformers-based OpenAI API server")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--port", type=int, default=21111, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map for model loading")

    args = parser.parse_args()

    load_model(args.model, args.device_map)

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

