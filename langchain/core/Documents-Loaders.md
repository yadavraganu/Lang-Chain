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
# Document Loaders
Document loaders in LangChain provide a standard interface for ingesting data from various external sources into LangChain's Document format.  

They act as the bridge between your raw data—whether stored in databases, cloud services, or local files—and your LLM-based application. By using a document loader, you ensure that data from disparate sources like Google Drive, Notion, Slack, or local PDFs can be processed and handled in a consistent, standardized way.  

There is no single library to install for all document loaders because they are split across different integration packages based on the data source.
LangChain uses a modular architecture where most loaders are maintained in specific integration packages (e.g., `langchain-community`, `langchain-google-community`, `langchain-aws`, etc.) to keep your project dependencies lightweight.
### How they work
All document loaders are designed to return one or more Document objects, which contain the raw text (page_content) and any associated metadata. This standardization is critical for the rest of the LangChain retrieval pipeline, such as text splitting, embedding, and vector storage.

### Key features
- Standardized output: Every loader, regardless of the source, produces the same Document object structure.
- Broad ecosystem: LangChain supports a massive range of sources, including cloud storage, databases, and collaboration platforms.
- Integration with retrieval: Once data is loaded as a Document, it is ready to be transformed (via text splitters) and indexed (via vector stores) for RAG applications.
- For a complete list of supported integrations and specific usage guides, you can refer to the official Document Loaders documentation.

### Loading from Local Directory 
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Define the path to your local directory
path = "./my_data_folder"

# Use DirectoryLoader to load files. 
# By default, it uses UnstructuredLoader, but you can specify a loader class.
# Here, we use TextLoader for all .txt files.
loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)

# Load the documents
docs = loader.load()

print(f"Loaded {len(docs)} documents.")
```
### Loading two specific documents
```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load files individually
loader1 = PyPDFLoader("./data/report.pdf")
loader2 = TextLoader("./data/notes.txt")

# Combine them into one list
documents = loader1.load() + loader2.load()

print(f"Total documents loaded: {len(documents)}")
```
### Loading from S3
```python
from langchain_community.document_loaders import S3DirectoryLoader, S3FileLoader

# 1. Load an entire directory (or prefix) from an S3 bucket
# This will load all files found under the specified prefix
dir_loader = S3DirectoryLoader(
    bucket="your-bucket-name", 
    prefix="path/to/directory/"
)
dir_docs = dir_loader.load()

# 2. Load a single file from an S3 bucket
# This will load only the specified file object
file_loader = S3FileLoader(
    bucket="your-bucket-name", 
    key="path/to/your/file.txt"
)
file_docs = file_loader.load()

# Print results
print(f"Loaded {len(dir_docs)} documents from directory.")
print(f"Loaded 1 document from file: {file_docs[0].metadata['source']}")
```
