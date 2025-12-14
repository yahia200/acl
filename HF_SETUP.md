# Hugging Face Inference API Setup

The project now uses Hugging Face's Inference API instead of local model loading. This is much lighter and faster!

## Setup Steps

### 1. Get a Hugging Face Token (Free)

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Copy the token

### 2. Set the Token

Choose one method:

**Option A: config.txt (Recommended)**

Edit `config.txt` and replace `your_huggingface_token_here` with your actual token:
```
URI=neo4j://localhost:7687
USERNAME=neo4j
PASSWORD=12345678
HF_TOKEN=hf_your_actual_token_here
```

**Option B: Environment Variable**
```powershell
# PowerShell
$env:HF_TOKEN = "hf_your_token_here"
```

**Option C: Pass to LLMHandler**
```python
from component_3_llm_layer import LLMHandler

llm = LLMHandler(
    model_name="google/flan-t5-base",
    api_token="hf_your_token_here"
)
```

The system checks in this order: parameter → config.txt → environment variable

## Benefits

✅ **No torch installation needed** - Saves ~2GB disk space  
✅ **No model downloads** - No waiting for large model files  
✅ **Faster startup** - Ready in seconds  
✅ **Free tier available** - 1000 requests/day on most models  
✅ **Access to more models** - Use any HF model without local resources

## Supported Models

The free Inference API supports many models:

- `google/flan-t5-base` - Great for Q&A (recommended)
- `google/flan-t5-large` - Better quality, slower
- `mistralai/Mistral-7B-Instruct-v0.2` - Very powerful
- `meta-llama/Llama-2-7b-chat-hf` - Good conversational model
- Many more at https://huggingface.co/models

## Rate Limits

**Free Tier:**
- 1,000 requests per day (most models)
- Rate limited to prevent abuse
- Perfect for development and testing

**Pro Tier ($9/month):**
- Higher rate limits
- Priority access
- More compute resources

For this airline analysis project, the free tier is sufficient!
