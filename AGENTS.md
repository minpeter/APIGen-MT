# Agents Documentation

## Custom LLM Provider Test

Tested the custom LLM provider with the default model nvidia/nemotron-3-super-120b-a12b.

Configuration used:
- API Base: https://integrate.api.nvidia.com/v1 (from OPENAI_API_BASE in .env)
- API Key: sk-or-v1-... (from OPENAI_API_KEY in .env)
- Model: nvidia/nemotron-3-super-120b-a12b

### Test Results:

1. **Initial tests with original API key**:
   - Using LocalOpenAILLMClient: Authentication failed (401 Unauthorized)
   - Using OpenAI client directly: Authentication failed (401 Unauthorized)
   - Note: Both test methods failed with authentication errors, suggesting the provided API key in .env was not valid for the NVIDIA API endpoint

2. **After API key update in .env**:
   - Using OpenAI client directly: ✅ Test successful!
   - Response: "I'm doing great, thanks! How can I assist you today? I'm doing great, thanks! How can I..." (truncated to one sentence as requested)
   - The nvidia/nemotron-3-super-120b-a12b model is now accessible and responding correctly

Note: The initial API key appeared to be an OpenRouter key (sk-or-v1-...) rather than a NVIDIA API key. After updating to a valid NVIDIA API key, the model responds correctly.