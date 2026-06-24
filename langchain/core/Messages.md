# Messages
Messages are the fundamental unit of context for models in LangChain, acting as the structured input and output for chat-based interactions.

They represent the state of a conversation by wrapping content in a standardized format that works across all model providers. Each message object contains three primary components:

- Role: Identifies the message type, such as system (for instructions), user, or ai.
- Content: The actual data being communicated, which can include text, images, audio, or other files.
- Metadata: Optional fields for tracking response information, message IDs, or token usage.

## Basic Usage
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

## Message Types Reference

| Type | Role | Purpose |
|------|------|---------|
| **SystemMessage** | "system" | Set instructions, context, and behavior guidelines for the model |
| **HumanMessage** | "user" | User queries, questions, or input data |
| **AIMessage** | "assistant" | Model responses and generated output |
| **ToolMessage** | "tool" | Results returned from tool/function calls |
| **FunctionMessage** | "function" | Legacy function call responses (deprecated in favor of ToolMessage) |

## Working with Different Message Types

### SystemMessage
Provides context and instructions that shape the model's behavior throughout the conversation:

```python
from langchain.messages import SystemMessage

system_msg = SystemMessage(
    content="You are an expert Python developer. Always provide efficient and well-documented code."
)
```

### HumanMessage
Represents user input in the conversation:

```python
from langchain.messages import HumanMessage

user_msg = HumanMessage(content="How do I optimize a Python function?")
```

### AIMessage
Captures the model's response:

```python
from langchain.messages import AIMessage

ai_msg = AIMessage(content="You can optimize by using list comprehensions, caching, and profiling...")
```

### ToolMessage
Used when the model calls external tools or functions:

```python
from langchain.messages import ToolMessage, AIMessage

# Model decides to use a tool
ai_msg = AIMessage(content="I'll check the weather for you.", tool_calls=[...])

# Tool result comes back
tool_msg = ToolMessage(
    content="Weather in NYC: Sunny, 72°F",
    tool_call_id="weather_123"
)
```

## Metadata and Optional Fields

Add metadata to messages for tracking, logging, and context management:

```python
from langchain.messages import HumanMessage

message = HumanMessage(
    content="What's the capital of France?",
    metadata={
        "message_id": "msg_12345",
        "user_id": "user_789",
        "timestamp": "2024-06-24T10:30:00Z",
        "source": "web_chat"
    }
)

# Access metadata
print(message.metadata["user_id"])  # Output: user_789
```

## Multimodal Content

Messages can contain images, audio, and other media types:

```python
from langchain.messages import HumanMessage
from langchain_core.content import ImageContent

# Message with image
message = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
)

# Pass to model
response = model.invoke([message])
```

For base64-encoded images:

```python
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ]
)
```

## Message Serialization

### Converting to Dictionary
Convert messages to dictionaries for storage or transmission:

```python
from langchain.messages import HumanMessage

message = HumanMessage(content="Hello!", metadata={"id": "123"})
message_dict = {
    "type": message.type,
    "content": message.content,
    "metadata": message.metadata
}
```

### JSON Persistence
Store and retrieve messages from JSON:

```python
import json
from langchain.messages import messages_from_dict, messages_to_dict

# Convert messages to JSON
messages = [SystemMessage("You are helpful"), HumanMessage("Hi")]
messages_json = json.dumps(messages_to_dict(messages))

# Reconstruct from JSON
restored_messages = messages_from_dict(json.loads(messages_json))
```

## Tool/Function Messages in Workflows

Handle multi-turn conversations with tool calls:

```python
from langchain.messages import HumanMessage, AIMessage, ToolMessage

messages = [
    HumanMessage("What's the weather in New York and London?"),
    AIMessage(
        content="I'll get the weather for both cities.",
        tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "New York"}},
            {"id": "call_2", "name": "get_weather", "args": {"city": "London"}}
        ]
    ),
    ToolMessage(content="New York: 72°F, Sunny", tool_call_id="call_1"),
    ToolMessage(content="London: 65°F, Cloudy", tool_call_id="call_2"),
    # Model generates final response with both tool results
]

response = model.invoke(messages)
```

## Common Message Operations

### 1. Adding Messages

Add new messages to an existing conversation history:

```python
from langchain.messages import HumanMessage, AIMessage

# Start with existing messages
messages = [SystemMessage("You are helpful")]

# Add a user message
messages.append(HumanMessage("What is AI?"))

# Add an AI response
messages.append(AIMessage("AI is Artificial Intelligence..."))

# Add multiple messages at once
new_messages = [
    HumanMessage("Tell me more"),
    AIMessage("Here's more information...")
]
messages.extend(new_messages)
```

Helper function to safely add messages:

```python
def add_message(messages, content, role="user", metadata=None):
    """Safely add a message to the conversation"""
    from langchain.messages import HumanMessage, AIMessage, SystemMessage
    
    if not content:
        raise ValueError("Message content cannot be empty")
    
    message_classes = {
        "user": HumanMessage,
        "assistant": AIMessage,
        "system": SystemMessage
    }
    
    msg_class = message_classes.get(role)
    if not msg_class:
        raise ValueError(f"Unknown role: {role}")
    
    msg = msg_class(content=content)
    if metadata:
        msg.metadata = metadata
    
    messages.append(msg)
    return messages

# Usage
messages = []
messages = add_message(messages, "You are helpful", role="system")
messages = add_message(messages, "Hello!", role="user", metadata={"user_id": "123"})
```

### 2. Trimming/Windowing Messages

Reduce conversation length to stay within token limits:

```python
def trim_messages(messages, max_count=10, keep_system=True):
    """Keep only the most recent messages, optionally preserving system messages"""
    if keep_system and messages and messages[0].type == "system":
        system_msg = messages[0]
        other_msgs = messages[1:]
        
        if len(other_msgs) > max_count:
            return [system_msg] + other_msgs[-max_count:]
        return messages
    
    if len(messages) > max_count:
        return messages[-max_count:]
    return messages

# Usage
messages = [SystemMessage(...), HumanMessage(...), AIMessage(...), ...]
trimmed = trim_messages(messages, max_count=10, keep_system=True)
response = model.invoke(trimmed)
```

Trim by token count:

```python
from langchain.callbacks import get_openai_callback

def trim_by_tokens(messages, max_tokens=4096):
    """Trim messages to stay within token limit"""
    token_count = 0
    kept_messages = []
    
    # Keep system message first
    if messages and messages[0].type == "system":
        kept_messages.append(messages[0])
        token_count += len(messages[0].content.split())  # Rough estimate
    
    # Add recent messages until token limit
    for msg in reversed(messages[1:]):
        msg_tokens = len(msg.content.split()) if isinstance(msg.content, str) else 100
        if token_count + msg_tokens <= max_tokens:
            kept_messages.insert(0 if not kept_messages or kept_messages[0].type != "system" else 1, msg)
            token_count += msg_tokens
        else:
            break
    
    return kept_messages

# Usage
trimmed = trim_by_tokens(messages, max_tokens=4096)
```

### 3. Deleting Messages

Remove specific messages from conversation:

```python
def delete_message_by_index(messages, index):
    """Delete message at specific index"""
    if 0 <= index < len(messages):
        return messages[:index] + messages[index+1:]
    raise IndexError(f"Message index {index} out of range")

def delete_messages_by_type(messages, message_type):
    """Delete all messages of a specific type"""
    from langchain.messages import HumanMessage, AIMessage, ToolMessage
    
    return [msg for msg in messages if not isinstance(msg, message_type)]

def delete_old_messages(messages, keep_last_n=5):
    """Keep only the last N messages (useful for memory management)"""
    if len(messages) > keep_last_n:
        return messages[-keep_last_n:]
    return messages

# Usage examples
messages = trim_messages(messages, max_count=5)
messages = delete_message_by_index(messages, 2)  # Delete message at index 2
messages = delete_messages_by_type(messages, ToolMessage)  # Remove all tool messages
```

### 4. Summarizing Messages

Create summaries of long conversations:

```python
from langchain.chat_models import init_chat_model

def summarize_messages(messages, model_name="gpt-4o"):
    """Summarize a conversation using an LLM"""
    from langchain.messages import SystemMessage, HumanMessage
    
    model = init_chat_model(model_name)
    
    # Create a prompt to summarize
    conversation_text = "\n".join([
        f"{msg.type}: {msg.content}" for msg in messages
    ])
    
    summary_prompt = [
        SystemMessage("You are a helpful assistant that summarizes conversations."),
        HumanMessage(
            f"Please summarize this conversation in 2-3 sentences:\n\n{conversation_text}"
        )
    ]
    
    summary = model.invoke(summary_prompt)
    return summary.content

# Usage
messages = [HumanMessage(...), AIMessage(...), HumanMessage(...), AIMessage(...)]
summary = summarize_messages(messages)
print(f"Summary: {summary}")
```

Compress old messages into a summary:

```python
def compress_conversation(messages, summary_after_n=10, model_name="gpt-4o"):
    """Compress old messages into a summary"""
    from langchain.messages import SystemMessage, HumanMessage
    
    if len(messages) <= summary_after_n:
        return messages
    
    # Keep system message and recent messages
    system_msgs = [m for m in messages if m.type == "system"]
    conversation_msgs = [m for m in messages if m.type != "system"]
    
    if len(conversation_msgs) > summary_after_n:
        old_messages = conversation_msgs[:-summary_after_n]
        recent_messages = conversation_msgs[-summary_after_n:]
        
        # Summarize old messages
        model = init_chat_model(model_name)
        old_text = "\n".join([f"{m.type}: {m.content}" for m in old_messages])
        
        summary_prompt = [
            SystemMessage("Summarize this conversation concisely."),
            HumanMessage(old_text)
        ]
        
        summary = model.invoke(summary_prompt).content
        
        # Add summary as a system context
        from langchain.messages import SystemMessage
        summary_msg = SystemMessage(
            content=f"Previous conversation summary: {summary}"
        )
        
        return system_msgs + [summary_msg] + recent_messages
    
    return messages

# Usage
compressed = compress_conversation(messages, summary_after_n=10)
```

### 5. Filtering Messages

Extract specific message types or patterns:

```python
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

def filter_messages_by_type(messages, message_types):
    """Filter messages by type"""
    return [msg for msg in messages if type(msg) in message_types]

def filter_messages_by_role(messages, roles):
    """Filter messages by role"""
    return [msg for msg in messages if msg.type in roles]

def filter_messages_by_metadata(messages, key, value):
    """Filter messages by metadata"""
    return [
        msg for msg in messages
        if msg.metadata and msg.metadata.get(key) == value
    ]

# Usage
user_messages = filter_messages_by_type(messages, [HumanMessage])
ai_tool_messages = filter_messages_by_type(messages, [AIMessage, ToolMessage])
messages_from_user_123 = filter_messages_by_metadata(messages, "user_id", "123")
```

### 6. Searching/Finding Messages

Search for specific messages:

```python
def search_messages_by_content(messages, query, case_sensitive=False):
    """Find messages containing specific text"""
    if not case_sensitive:
        query = query.lower()
    
    results = []
    for i, msg in enumerate(messages):
        content = msg.content
        if isinstance(content, str):
            search_text = content if case_sensitive else content.lower()
            if query in search_text:
                results.append((i, msg))
    
    return results

def find_message_by_metadata(messages, key, value):
    """Find first message with specific metadata"""
    for msg in messages:
        if msg.metadata and msg.metadata.get(key) == value:
            return msg
    return None

def find_consecutive_exchanges(messages, min_length=1):
    """Find consecutive user-AI exchanges"""
    exchanges = []
    current_exchange = []
    
    for msg in messages:
        if msg.type in ["user", "assistant"]:
            current_exchange.append(msg)
            # Complete exchange: user + AI response
            if len(current_exchange) >= 2 and msg.type == "assistant":
                if len(current_exchange) >= min_length * 2:
                    exchanges.append(current_exchange)
                current_exchange = []
    
    return exchanges

# Usage
results = search_messages_by_content(messages, "API", case_sensitive=False)
for idx, msg in results:
    print(f"Found at index {idx}: {msg.content}")

weather_msg = find_message_by_metadata(messages, "topic", "weather")
exchanges = find_consecutive_exchanges(messages, min_length=1)
```

### 7. Updating/Editing Messages

Modify existing messages:

```python
def update_message_content(messages, index, new_content):
    """Update content of a message"""
    if 0 <= index < len(messages):
        msg = messages[index]
        msg.content = new_content
        return messages
    raise IndexError(f"Message index {index} out of range")

def update_message_metadata(messages, index, metadata_updates):
    """Update metadata of a message"""
    if 0 <= index < len(messages):
        msg = messages[index]
        if not msg.metadata:
            msg.metadata = {}
        msg.metadata.update(metadata_updates)
        return messages
    raise IndexError(f"Message index {index} out of range")

def merge_messages(messages):
    """Combine consecutive messages from same sender"""
    if not messages:
        return messages
    
    merged = []
    current_msg = messages[0]
    
    for msg in messages[1:]:
        if msg.type == current_msg.type:
            # Combine content
            if isinstance(current_msg.content, str) and isinstance(msg.content, str):
                current_msg.content += "\n" + msg.content
        else:
            merged.append(current_msg)
            current_msg = msg
    
    merged.append(current_msg)
    return merged

# Usage
messages = update_message_content(messages, 0, "Updated content")
messages = update_message_metadata(messages, 0, {"reviewed": True})
messages = merge_messages(messages)
```

### 8. Batch Operations

Perform operations on multiple messages:

```python
def batch_add_metadata(messages, metadata):
    """Add metadata to all messages"""
    for msg in messages:
        if not msg.metadata:
            msg.metadata = {}
        msg.metadata.update(metadata)
    return messages

def batch_tag_messages(messages, tag, indices=None):
    """Tag specific or all messages"""
    indices = indices or range(len(messages))
    
    for i in indices:
        if 0 <= i < len(messages):
            if not messages[i].metadata:
                messages[i].metadata = {}
            tags = messages[i].metadata.get("tags", [])
            if tag not in tags:
                tags.append(tag)
            messages[i].metadata["tags"] = tags
    
    return messages

def remove_messages_by_condition(messages, condition_func):
    """Remove messages matching a condition"""
    return [msg for msg in messages if not condition_func(msg)]

# Usage
messages = batch_add_metadata(messages, {"session_id": "sess_123"})
messages = batch_tag_messages(messages, "important", indices=[0, 2, 5])
messages = remove_messages_by_condition(messages, lambda m: m.type == "tool")
```

## Advanced Patterns

### Building Efficient Context Windows
Manage long conversations by removing older messages:

```python
def build_sliding_window(messages, window_size=10, keep_system=True):
    """Create a sliding window of messages"""
    if keep_system and messages and messages[0].type == "system":
        return [messages[0]] + messages[-window_size:]
    return messages[-window_size:]

# Usage
context_window = build_sliding_window(messages, window_size=10)
response = model.invoke(context_window)
```

### Message Filtering by Type
Extract specific message types:

```python
def get_messages_by_type(messages, message_type):
    """Filter messages by type"""
    return [msg for msg in messages if isinstance(msg, message_type)]

# Get all user messages
from langchain.messages import HumanMessage
user_messages = get_messages_by_type(messages, HumanMessage)
```

### Message Transformation
Transform or annotate messages before sending to model:

```python
def annotate_messages(messages):
    """Add timestamps and counters"""
    from datetime import datetime
    
    annotated = []
    for i, msg in enumerate(messages):
        if not msg.metadata:
            msg.metadata = {}
        msg.metadata.update({
            "index": i,
            "timestamp": datetime.now().isoformat()
        })
        annotated.append(msg)
    return annotated

# Apply transformation
processed_messages = annotate_messages(messages)
```

## Common Best Practices

1. **Always include a SystemMessage** at the start to set context and instructions
2. **Preserve order** - Maintain chronological order of messages for accurate context
3. **Limit conversation length** - Trim older messages to stay within token limits
4. **Use metadata wisely** - Track important context like user IDs and timestamps
5. **Validate content** - Ensure messages contain appropriate content before passing to models
6. **Handle errors gracefully** - Catch and log issues with malformed messages
7. **Monitor token usage** - Track tokens to avoid exceeding model limits
8. **Archive long conversations** - Save old messages for history but work with recent ones

## Error Handling

```python
from langchain.messages import HumanMessage, ValidationError

def safe_create_message(content, role="user"):
    """Safely create messages with validation"""
    try:
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")
        
        if role == "user":
            return HumanMessage(content=content)
        elif role == "system":
            from langchain.messages import SystemMessage
            return SystemMessage(content=content)
        else:
            raise ValueError(f"Unknown role: {role}")
    except (ValueError, ValidationError) as e:
        print(f"Error creating message: {e}")
        return None

def safe_invoke_with_messages(model, messages):
    """Safely invoke model with error handling"""
    try:
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        response = model.invoke(messages)
        return response
    except Exception as e:
        print(f"Error invoking model: {e}")
        return None

# Safe usage
msg = safe_create_message("Hello!", role="user")
if msg:
    response = safe_invoke_with_messages(model, [msg])
```

## Summary

Messages are the core building block for LangChain chat interactions. They provide:
- **Standardized communication** across all model providers
- **Flexibility** with support for multimodal content and metadata
- **Structure** for managing complex, multi-turn conversations
- **Rich operations** for adding, trimming, filtering, and summarizing conversations
- **Tools** for serialization, transformation, and optimization

Master message handling to build robust, scalable conversational AI applications.
