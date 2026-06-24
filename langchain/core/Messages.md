## Messages in LangChain

Messages are the fundamental unit of context for models in LangChain, acting as the structured input and output for chat-based interactions. They represent the state of a conversation by wrapping content in a standardized format that works across all model providers. 

Each message object contains three primary components:
- **Role**: Identifies the message type, such as system (for instructions), user, or ai.
- **Content**: The actual data being communicated, which can include text, images, audio, or other files.
- **Metadata**: Optional fields for tracking response information, message IDs, or token usage.

### 1. Basic Usage

You can interact with models by passing message objects or standardized dictionary formats as a list to the model's `invoke()` method:

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-4o")

# Option A: Using Message Objects
messages_obj = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hello, how are you?"),
    AIMessage("I'm doing well, how can I help you?")
]
response = model.invoke(messages_obj)

# Option B: Using Standardized Dictionary Format
messages_dict = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I create a REST API in Python?"},
    {"role": "assistant", "content": "To create a REST API, you can use frameworks like FastAPI..."}
]
response = model.invoke(messages_dict)

```

*> **Tip:** Use message prompts when managing multi-turn conversations, handling multimodal content, or providing system-level instructions. For simple, single-turn requests, a standard string prompt is usually sufficient.*

### 2. Message Types Reference

| Type | Role | Purpose |
| --- | --- | --- |
| **SystemMessage** | "system" | Set instructions, context, and behavior guidelines for the model |
| **HumanMessage** | "user" | User queries, questions, or input data |
| **AIMessage** | "assistant" | Model responses and generated output |
| **ToolMessage** | "tool" | Results returned from tool/function calls |
| **FunctionMessage** | "function" | Legacy function call responses (deprecated in favor of ToolMessage) |

#### Core Instantiation Examples

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# System Instruction
system_msg = SystemMessage(content="You are an expert Python developer.")

# User Input
user_msg = HumanMessage(content="How do I optimize a Python function?")

# AI Response (With a Tool Call)
ai_msg = AIMessage(content="I'll check the weather for you.", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "NYC"}}])

# Tool Response Execution
tool_msg = ToolMessage(content="Weather in NYC: Sunny, 72°F", tool_call_id="call_1")

```
### 3. Advanced Content & Metadata

#### Metadata Management

Add metadata to messages for tracking, logging, and context management:

```python
message = HumanMessage(
    content="What's the capital of France?",
    metadata={"message_id": "msg_12345", "user_id": "user_789"}
)
print(message.metadata["user_id"])  # Output: user_789

```

#### Multimodal Content

Messages can handle text, remote image URLs, or local base64-encoded images:

```python
import base64
from langchain.messages import HumanMessage

# Using Image URLs
msg_url = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "[https://example.com/image.jpg](https://example.com/image.jpg)"}}
])

# Using Base64 Local Images
with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

msg_b64 = HumanMessage(content=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
])

```

### 4. Message Serialization

LangChain messages provide built-in serialization utilities to prepare conversation histories for JSON persistence or storage.

```python
import json
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.messages import message_to_dict, messages_to_dict, messages_from_dict

message = HumanMessage(content="Hello!", metadata={"id": "123"})

# Convert single message or list to dictionaries
single_dict = message_to_dict(message)
messages = [SystemMessage("You are helpful"), HumanMessage("Hi")]
list_of_dicts = messages_to_dict(messages)

# JSON Persistence Workflow
messages_json = json.dumps(list_of_dicts)
restored_messages = messages_from_dict(json.loads(messages_json))

```
### 5. Built-in Message Operations

Instead of executing manual array-slicing or loop filtration routines, leverage LangChain's optimized, tokenizer-aware utilities.

#### Trimming (Context Windows)

Keep conversation history within exact token limits using the target model's actual tokenizer.

```python
from langchain_core.messages import trim_messages
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o")

trimmed_messages = trim_messages(
    messages,
    strategy="last",
    token_counter=model,       # Automatically handles token counting via model rules
    max_tokens=4096,
    start_on="human",          # Prevents loose, trailing tool outputs
    include_system=True,       # Always preserves the initial instruction block
)

```

#### Filtering

Sift through complex chat history collections instantly.

```python
from langchain_core.messages import filter_messages, HumanMessage, ToolMessage

# Extract only human input turns
user_messages = filter_messages(messages, include_types=[HumanMessage])

# Strip out structural tool responses for cleaner text generation pipelines
clean_history = filter_messages(messages, exclude_types=[ToolMessage])

```

#### Merging Sequential Turns

Combine sequential inputs sent consecutively by the exact same sender role.

```python
from langchain_core.messages import merge_messages

# Merges contiguous blocks of HumanMessages or AIMessages natively
merged_history = merge_messages(messages)

```

### 6. Advanced Custom Handlers

#### Native Searching, Archiving & Mutations

For specific, deep alterations to localized areas of your chat lists:

```python
def search_messages_by_content(messages, query, case_sensitive=False):
    """Find messages containing specific text."""
    query = query if case_sensitive else query.lower()
    return [
        (i, msg) for i, msg in enumerate(messages)
        if isinstance(msg.content, str) and (query in (msg.content if case_sensitive else msg.content.lower()))
    ]

def update_message_at_index(messages, index, new_content=None, metadata_updates=None):
    """Safely edit content or append metadata parameters on specific indexed positions."""
    if 0 <= index < len(messages):
        if new_content is not None:
            messages[index].content = new_content
        if metadata_updates:
            if not messages[index].metadata:
                messages[index].metadata = {}
            messages[index].metadata.update(metadata_updates)
        return messages
    raise IndexError(f"Message index {index} out of range.")

def summarize_old_history(messages, model_name="gpt-4o"):
    """Compile past messaging context using an LLM to save token real-estate."""
    from langchain.messages import SystemMessage, HumanMessage
    model = init_chat_model(model_name)
    
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    summary_prompt = [
        SystemMessage("You are a helpful assistant that summarizes conversations."),
        HumanMessage(f"Please summarize this conversation in 2-3 sentences:\n\n{conversation_text}")
    ]
    return model.invoke(summary_prompt).content

```
