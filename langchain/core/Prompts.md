# Prompts
Prompts in LangChain are templated instructions used to structure and format inputs sent to language models.

They act as a bridge between your application logic and the LLM, allowing you to define a consistent structure while injecting dynamic data at runtime. By using prompt templates, you can create reusable patterns that ensure your prompts are robust and maintainable.

### Core Concepts
- Prompt Templates: These are the standard way to define prompts. They allow you to add placeholders (variables) like {topic} that get filled in with actual values when your application runs.
- Chat Prompts: The recommended format for modern applications. They structure prompts as a list of messages (such as system, user, or assistant roles), which is the native format for most current chat-based LLM APIs.
- Completion Prompts: An older, string-based format. These are generally maintained for backward compatibility and should be avoided for new projects unless required.

LangChain provides different types of prompt templates to handle varying levels of complexity, primarily divided into string-based templates and message-based (chat) templates.

### String PromptTemplates
These are used to format a single string and are ideal for simpler, linear inputs where you want to construct a prompt from a basic string pattern.
```python
from langchain_core.prompts import PromptTemplate

# A simple string template
template = "Tell me a joke about {topic}."
prompt = PromptTemplate.from_template(template)

print(prompt.invoke({"topic": "cats"}))
# Output: Tell me a joke about cats.
```
### ChatPromptTemplates
These templates format an array of messages rather than a single string. This is the recommended approach for modern chat models, as it allows you to explicitly define different roles (like system, user, or assistant) for each message in the conversation.
```python
from langchain_core.prompts import ChatPromptTemplate

# A chat template with role-based messages
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Tell me a joke about {topic}."),
])

print(chat_template.invoke({"topic": "cats"}))
```
### MessagesPlaceholder
When building more complex ChatPromptTemplate structures, you may need to insert a dynamic list of messages at a specific location, such as conversation history. MessagesPlaceholder allows you to slot in an array of messages directly.
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
])

# Pass an array of messages into the "history" variable
prompt_template.invoke({
    "history": [HumanMessage(content="Hello!")],
    "input": "How are you?"
})
```
