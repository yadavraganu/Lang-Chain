# Messages
Messages are the fundamental unit of context for models in LangChain, acting as the structured input and output for chat-based interactions.

They represent the state of a conversation by wrapping content in a standardized format that works across all model providers. Each message object contains three primary components:

- Role: Identifies the message type, such as system (for instructions), user, or ai.
- Content: The actual data being communicated, which can include text, images, audio, or other files.
- Metadata: Optional fields for tracking response information, message IDs, or token usage.
### Basic Usage
You can interact with models by creating these message objects and passing them as a list to the model's invoke() method:
```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-4o")

# Define conversation history
messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hello, how are you?"),
    AIMessage("I'm doing well, how can I help you?")
]

# Pass the list to the model
response = model.invoke(messages)
```
```python
from langchain.chat_models import init_chat_model

# Initialize the model
model = init_chat_model("gpt-4o")

# Pass messages using dictionary format
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I create a REST API in Python?"},
    {"role": "assistant", "content": "To create a REST API, you can use frameworks like FastAPI or Flask..."}
]

# The model accepts this list directly
response = model.invoke(messages)
```
Use message prompts when managing multi-turn conversations, handling multimodal content, or providing system-level instructions. For simple, single-turn requests, a standard string prompt is usually sufficient.
