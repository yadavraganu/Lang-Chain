# Documents
A Document in LangChain is a core abstraction designed to represent a unit of text along with its associated metadata.
It acts as a standardized container for your data, making it easier to process, transform, and store content within your LLM applications. An individual Document often represents a smaller "chunk" of a larger source file.  

The Document object consists of the following attributes:
- `page_content`: A string representing the actual text content.
- `metadata`: A dictionary containing arbitrary metadata (e.g., source file name, page number, or author).
id: An optional string identifier for the document.

These documents are foundational for building RAG (Retrieval-Augmented Generation) pipelines, as they are the format used by document loaders, text splitters, embedding models, and vector stores. 
You can create Document objects easily in your Python code:
```python
from langchain_core.documents import Document

# Creating sample documents
documents = [
    Document(
        page_content="Dogs are loyal companions.",
        metadata={"source": "pets-guide.txt", "page": 1},
    ),
    Document(
        page_content="Cats enjoy their own space.",
        metadata={"source": "pets-guide.txt", "page": 2},
    ),
]
```

