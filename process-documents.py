# region Imports
import base64
import datetime
from openai import OpenAI
from typing import List, Optional
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
#from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
#from azure.core.credentials import AzureKeyCredential
from azure.identity import ClientSecretCredential
import os
import sys
import time
import hashlib
import math
from typing import List, Dict, Iterable, Optional, Tuple

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel


try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType, VectorSearch, VectorSearchAlgorithmConfiguration, VectorSearchProfile, 
    SemanticConfiguration, SemanticField, SemanticPrioritizedFields,SynonymMap,
    HnswAlgorithmConfiguration, AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters,SemanticSearch
)
from azure.search.documents.models import VectorizedQuery

try:
    import tiktoken
except ImportError:
    tiktoken = None


from structured_data_injestion import (
    AzureSQLDatabase,
    Incident,
    IncidentHeader,
    EmployeeImpact
)
# endregion


# region Environment Variables Management
# Clear Environment Variables Functions
def clear_specific_env_vars():
    """Clear specific environment variables"""
    vars_to_clear = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_ENDPOINT_V1",
        "GPT_MODEL_NAME",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_API_VERSION_RESPONSE",
        "SEARCH_ENDPOINT",
        "AOAI_ENDPOINT",
        "AOAI_EMBEDDING_MODEL",
        "AZURE_OPENAI_EMBEDDING_MODEL",
        "AOAI_EMBEDDING_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AOAI_GPT_MODEL",
        "AOAI_GPT_DEPLOYMENT",
        "INDEX_NAME",
        "KNOWLEDGE_SOURCE_NAME",
        "KNOWLEDGE_AGENT_NAME",
        "SEARCH_API_VERSION"
    ]
    
    cleared = []
    for var in vars_to_clear:
        if var in os.environ:
            del os.environ[var]
            cleared.append(var)
            print(f" Cleared: {var}")
        else:
            print(f" Not found: {var}")
    
    return cleared

def clear_azure_env_vars():
    """Clear all environment variables that start with AZURE or AOAI"""
    azure_vars = [key for key in os.environ.keys() if key.startswith(('AZURE', 'AOAI', 'SEARCH'))]
    
    for var in azure_vars:
        del os.environ[var]
        print(f" Cleared: {var}")
    
    return azure_vars

def show_current_env_vars():
    """Show current Azure/AI related environment variables"""
    relevant_vars = [key for key in os.environ.keys() 
                    if any(keyword in key.upper() for keyword in 
                          ['AZURE', 'OPENAI', 'SEARCH', 'AOAI', 'GPT', 'EMBEDDING'])]
    
    print("Current relevant environment variables:")
    if not relevant_vars:
        print("  No Azure/AI related environment variables found.")
        return
        
    for var in sorted(relevant_vars):
        # Hide sensitive values
        value = os.environ[var]
        if any(sensitive in var.upper() for sensitive in ['KEY', 'SECRET', 'TOKEN']):
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"  {var}: {masked_value}")
        else:
            print(f"  {var}: {value}")

def reset_environment():
    """Reset environment to clean state and reload from .env if it exists"""
    # Clear Azure-related vars
    cleared = clear_azure_env_vars()
    
    # Try to reload from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        print(" Reloaded environment variables from .env file")
    except ImportError:
        print("  dotenv not available - install with: pip install python-dotenv")
    except Exception as e:
        print(f"  Could not reload .env file: {e}")
    
    return cleared

#clear
clear_azure_env_vars()
# Show current state
show_current_env_vars()

os.environ.setdefault("AZURE_OPENAI_API_KEY", "")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT_V1", "https://.....openai.azure.com/openai/v1/")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://.....openai.azure.com/")
os.environ.setdefault("GPT_MODEL_NAME", "gpt-4o-mini")
os.environ.setdefault("AZURE_OPENAI_API_VERSION_RESPONSE", "2025-03-01-preview")
os.environ.setdefault("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
os.environ.setdefault("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

os.environ.setdefault("SEARCH_API_VERSION","2025-08-01-preview")
os.environ.setdefault("SEARCH_ENDPOINT","https://.....search.windows.net")
os.environ.setdefault("INDEX_NAME","Incidents_index")

show_current_env_vars()

# endregion


#region Define The Clients for OpenAI and Search

#azdemoapp the App registration to interact with Search
tenant_id = "...."
client_id = "...."
client_secret = "..." 

credential = ClientSecretCredential(tenant_id, client_id, client_secret)
#token_provider = get_bearer_token_provider(credential, "https://search.azure.com/.default")
token_provider = credential.get_token("https://cognitiveservices.azure.com/.default")


client_V1 = OpenAI(  
  base_url = os.getenv("AZURE_OPENAI_ENDPOINT_V1"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY") 
)

client = AzureOpenAI(  
  azure_endpoint= os.getenv("AZURE_OPENAI_ENDPOINT"),
  api_key=os.getenv("AZURE_OPENAI_API_KEY") ,
  api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)


search_client = SearchClient(endpoint=os.getenv("SEARCH_ENDPOINT"), index_name=os.getenv("INDEX_NAME"), credential=credential) 

index_client = SearchIndexClient(endpoint=os.getenv("SEARCH_ENDPOINT"), credential=credential)


#endregion

#region pydantic schema for documents
# Using a Pydantic model (and simple response format)
class IncidentHeader(BaseModel):
    documentId: str
    pageCount: str
    IncidentDate: str
    Category: str
    NumberOfImpactedEmployees: int
    EmployeeNames: str
    Location: str
    IncidentType: str
    Injuries: str
    Summary: str
 

class EmployeeImpact(BaseModel):
    EmployeeName: str
    EmployeeID: str
    InjuryDescription: str
    ActionTaken: str


class Incident(BaseModel):
    header: IncidentHeader
    impact_details: List[EmployeeImpact]


#endregion

#region Users Defined Functions
def get_env(name: str = "DEV", required: bool = True, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if required and not val:
        print(f"ERROR: Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return val

def embedding_model_dims(model: str) -> int:
# Update this mapping as Azure OpenAI adds models
# text-embedding-3-small: 1536, text-embedding-3-large: 3072, ada-002: 1536
    if model == "text-embedding-3-small":
        return 1536
    if model == "text-embedding-3-large":
        return 3072
    if model == "text-embedding-ada-002":
        return 1536
    # Default assumption
    return 1536

#create an Azure AI Search Index with vector search and semantic search
def create_new_index(index_client: SearchIndexClient, index_name: str, aoai_endpoint: str, aoai_embedding_deployment: str, aoai_embedding_model: str):
    """Create or update the Azure Cognitive Search index with vector search and semantic search configurations."""
    # Determine embedding dimensions
    embedding_dim=embedding_model_dims(aoai_embedding_model)
 
# Define the schema for the index using Python SDK objects
    
    fields = [
        SimpleField(name="DocumentId", type=SearchFieldDataType.String, key=True, filterable=True, facetable=True),
        SimpleField(name="PageCount", type=SearchFieldDataType.String, filterable=True, facetable=False),
        SimpleField(name="IncidentDate", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="Category", type=SearchFieldDataType.String, filterable=True, facetable=False),
        SimpleField(name="NumberOfImpactedEmployees", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="EmployeeNames", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="Location", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="IncidentType", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="Injuries", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="Summary", type=SearchFieldDataType.String, filterable=True, facetable=True, searchable=True),
        SearchField(  
            name="embedding",  
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),  
            searchable=True,  
            filterable=False,  
            sortable=False,  
            facetable=False,  
            vector_search_dimensions=embedding_dim,  
            vector_search_profile_name="vs_insident"
        ),  
    ]
    

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=VectorSearch(
            profiles=[VectorSearchProfile(name="vs_incident", algorithm_configuration_name="alg", vectorizer_name="azure_openai_text_3_large")],
            algorithms=[HnswAlgorithmConfiguration(name="alg")],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="azure_openai_text_3_large",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=aoai_endpoint,
                        deployment_name=aoai_embedding_deployment,
                        model_name=aoai_embedding_model
                    )
                )
            ]
        ),
        semantic_search=SemanticSearch(
            default_configuration_name="semantic_config",
            configurations=[
                SemanticConfiguration(
                    name="semantic_config",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[
                            SemanticField(field_name="Summary"), SemanticField(field_name="Injuries")
                        ],keywords_fields=[
                            SemanticField(field_name="Category"), SemanticField(field_name="Location"),
                            SemanticField(field_name="IncidentType")
                        ]
                    )
                )
            ]
        )
    )


#be careful with deleting the index in production scenarios
    index_client.delete_index(index_name)
    index_client.create_or_update_index(index)
    print(f"Index '{index_name}' created or updated successfully.")
    return

def load_text_from_file(path: Path, client: OpenAI, genAImodel: str) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")  
    if suffix == ".pdf":  
        with open(path, "rb") as f: # assumes PDF is in the same directory as the executing script
            data = f.read()

        base64_string = base64.b64encode(data).decode("utf-8")

        response = client.responses.create(
            model= genAImodel, # model deployment name
            input=[
                {"role": "system", "content": "Extract the information and present in markdown format"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "file-name.pdf",
                            "file_data": f"data:application/pdf;base64,{base64_string}",
                        },
                        {
                            "type": "input_text",
                            "text": "extract this data into markdown format",
                        },
                    ],
                },
            ],stream=False
        )

        result01 = response.output_text

        return result01  

# Unsupported file type  
    return ""  


## Extract Schema from the Markdown text for the Index
def load_schema_from_text(input_text: str, client: OpenAI, model: str) -> Incident:

    with client.responses.stream(
        model=model,
        input=[
            {"role": "system", "content": "Extract entities from the input text. You need to extract all the fields. If a field is missing in the input text, try to find it in the context, return it as empty string."},
            {
                "role": "user",
                "content": input_text,
            },
        ],
        text_format=Incident,
    ) as stream:
        for inv in stream:
            if inv.type == "response.error":
                print(inv.error, end="")
            elif inv.type == "response.completed":
                print("Completed")

    final_response = stream.get_final_response()
    
    # Extract the parsed Incident from the response
    if hasattr(final_response, 'output_parsed') and final_response.output_parsed:
        return final_response.output_parsed
    else:
        # If parsing failed
        print("Warning: Failed to parse response")
        return None

def get_encoding_for_model(model: str):
    if not tiktoken:
        return None
    # Most modern OpenAI embeddings/chat models use cl100k_base
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None
    
def count_tokens(text: str, enc) -> int:
    if enc is None:
    # Fallback heuristic if tiktoken isn't available
    # Approximate 4 chars per token
        return math.ceil(len(text) / 4)
    return len(enc.encode(text))


def chunk_text(text: str, model: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
    text = text.strip()
    if not text:
        return []

    enc = get_encoding_for_model(model)  
    if enc is None:  
        # Fallback: approximate by characters with overlap  
        approx_chars = max_tokens * 4  
        overlap_chars = overlap_tokens * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + approx_chars)
            chunk = text[start:end]
            chunks.append(chunk)
            start = max(0, end - overlap_chars)
            if end == len(text):  
                break  
            start = max(0, end - overlap_chars)
        return chunks

    # Token-aware chunking  
    tokens = enc.encode(text)  
    chunks = []  
    i = 0  
    while i < len(tokens):  
        j = min(i + max_tokens, len(tokens))  
        chunk_tokens = tokens[i:j]  
        chunk_text_str = enc.decode(chunk_tokens)  
        chunks.append(chunk_text_str)  
        if j == len(tokens):  
            break  
        i = max(0, j - overlap_tokens)  
    return chunks

def embed_texts(client: AzureOpenAI,model: str,texts: List[str],batch_size: int = 64,retry_max: int = 5,retry_backoff_sec: float = 2.0) -> List[List[float]]:
    print(f"Start embeddings process.")
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        print(f"Processing batch {start//batch_size + 1}: {len(batch)} texts")

        # Retry loop for this batch
        for attempt in range(retry_max):  
            try:  
                resp = client.embeddings.create(model=model, input=batch)
                print(f"Creating embeddings for batch {start//batch_size + 1}")  
                for datum in resp.data:  
                    embeddings.append(datum.embedding)  
                break  
            except Exception as e:  
                print(f"Attempt {attempt + 1} failed for batch {start//batch_size + 1}: {e}")
                if attempt == retry_max - 1:  
                    raise  
                time.sleep(retry_backoff_sec * (2 ** attempt))  
    
    print(f"Completed embeddings process. Generated {len(embeddings)} embeddings for {len(texts)} input texts.")
    return embeddings

def hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def build_documents_for_upload(folder: Path, client: OpenAI, model: str,max_tokens: int = 512,overlap_tokens: int = 50,include_extensions: Tuple[str, ...] = (".txt", ".md", ".pdf", ".docx"),) -> List[Dict]:
    docs: List[Dict] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in include_extensions:
            continue

        try:  
            text = load_text_from_file(path, client, model)  
        except Exception as e:  
            print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
            continue  

        if not text.strip():  
            continue  

        chunks = chunk_text(text, model=model, max_tokens=max_tokens, overlap_tokens=overlap_tokens)  

        base_id = hash_id(f"{path.resolve()}:{len(text)}")  
        title = path.stem  
        for idx, chunk in enumerate(chunks):  
            doc = {  
                "id": f"{base_id}-{idx}",  
                "filepath": str(path),  
                "title": title,  
                "chunk": idx,  
                "content": chunk,  
                # "embedding": [] will be filled later  
            }  
            docs.append(doc)  
    return docs


def upload_in_batches(search_client: SearchClient, docs: List[Dict], batch_size: int = 1000):
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        result = search_client.upload_documents(batch)
        failed = [r for r in result if not r.succeeded]
        if failed:
            raise RuntimeError(f"Upload failed for {len(failed)} documents in batch starting at {i}")
        print(f"Uploaded {len(batch)} docs ({i + len(batch)}/{len(docs)})")

def build_documents_for_upload_fixed(folder: Path, client: OpenAI, model: str, genAImodel: str, max_tokens: int = 512, overlap_tokens: int = 50, include_extensions: Tuple[str, ...] = (".txt", ".md", ".pdf", ".docx"), sql_db: AzureSQLDatabase=None) -> List[Dict]:
    docs: List[Dict] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in include_extensions:
            continue

        try:  
            text = load_text_from_file(path, client, genAImodel)
            tmp_case = load_schema_from_text(text, client, genAImodel) 

            # Insert the extracted schema fields into sql database
            
            incident = Incident(
                header=IncidentHeader(
                    documentId=tmp_case.DocumentId,
                    pageCount=tmp_case.PageCount,
                    IncidentDate=tmp_case.IncidentDate,
                    Category=tmp_case.Category,
                    NumberOfImpactedEmployees=tmp_case.NumberOfImpactedEmployees,
                    EmployeeNames=tmp_case.EmployeeNames,
                    Location=tmp_case.Location,
                    IncidentType=tmp_case.IncidentType,
                    Injuries=tmp_case.Injuries,
                    Summary=tmp_case.Summary,
                ),
                impact_details=[
                    EmployeeImpact(
                        EmployeeName=impact.EmployeeName,
                        EmployeeID=impact.EmployeeID,
                        InjuryDescription=impact.InjuryDescription,
                        ActionTaken=impact.ActionTaken,
                    )
                    for impact in tmp_case.impact_details
                ],
            )
            sql_db.insert_incident(incident)

        except Exception as e:  
            print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
            continue  

        if not text.strip():  
            continue  

        chunks = chunk_text(text, model=model, max_tokens=max_tokens, overlap_tokens=overlap_tokens)  

        for idx, chunk in enumerate(chunks):  
            doc = {  
                "DocumentId": tmp_case.DocumentId+"-"+str(idx),
                "PageCount": tmp_case.PageCount,
                "IncidentDate": tmp_case.IncidentDate,
                "Category": tmp_case.Category,
                "NumberOfImpactedEmployees": tmp_case.NumberOfImpactedEmployees,
                "EmployeeNames": tmp_case.EmployeeNames,
                "Location": tmp_case.Location,
                "IncidentType": tmp_case.IncidentType,
                "Injuries": tmp_case.Injuries,
                "Summary": tmp_case.Summary,
                "content": chunk,
            # "embedding": [] will be filled later
            }  
            docs.append(doc)  
    return docs  

def ingest_folder_to_search(
    folder_path: str,
    client: OpenAI,
    aoaiclient:AzureOpenAI,
    search_client: SearchClient,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    embedding_batch_size: int = 64
):
    # Env/config
    aoai_client = aoaiclient
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dim = embedding_model_dims(embedding_model)

      # Using the same key for now
    index_name = os.getenv("INDEX_NAME", "hearing-index")

    folder = Path(folder_path)  
    if not folder.exists():  
        print(f"ERROR: Folder not found: {folder}", file=sys.stderr)  
        sys.exit(1)  

    sql_db = AzureSQLDatabase(
                server=os.getenv("SQL_SERVER"),
                database=os.getenv("SQL_DATABASE"),
                use_azure_auth=False,
                username=os.getenv("SQL_USERNAME"),
                password=os.getenv("SQL_PASSWORD"),
            )
    sql_db.connect()
    # Build docs and chunk  
    print("Scanning and chunking documents...")  
    docs = build_documents_for_upload_fixed(  
        folder=folder,  
        client=client,  # Use the OpenAI client defined earlier
        model=embedding_model,
        genAImodel=os.getenv("GPT_MODEL_NAME", "gpt-4o-mini"),
        max_tokens=max_tokens,  
        overlap_tokens=overlap_tokens,  
        sql_db=sql_db
    )  
    if not docs:  
        print("No documents to ingest.")  
        return  

    # Embed content  
    print(f"Generating embeddings for {len(docs)} chunks with model {embedding_model}...")  
    contents = [d["content"] for d in docs]  
    vectors = embed_texts(aoai_client, embedding_model, contents, batch_size=embedding_batch_size)  
    if len(vectors) != len(docs):  
        raise RuntimeError("Mismatch between embeddings and documents count")  
    
    print(f"Generating embeddings completed.")
    for d, v in zip(docs, vectors):  
        d["embedding"] = v  

    # Upload to search  
     
    upload_in_batches(search_client, docs, batch_size=1000)  

    print("Ingestion complete.")  

#endregion


# region Main Code
if __name__ == "__main__":
    
    
    create_new_index(index_client=index_client,
                     index_name=os.getenv("INDEX_NAME", "hearing-index"),
                     aoai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                     aoai_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                     aoai_embedding_model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"))

    # Ingest documents from a folder to Azure Search Index
    # Set the folder path containing documents to process

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder_path  = os.path.join(script_dir, "data", "input")

    print(f"Looking for documents in: {data_folder_path}")

    # Check if folder exists
    if not os.path.exists(data_folder_path):
        print(f"ERROR: Folder not found: {data_folder_path}")
        print("Please ensure the folder exists and contains PDF, TXT, MD files to process.")
    else:
        # List files that will be processed
        supported_extensions = (".txt", ".md", ".pdf")
        files_to_process = []
        
        for root, dirs, files in os.walk(data_folder_path):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    files_to_process.append(os.path.join(root, file))
        
        print(f"Found {len(files_to_process)} documents to process:")
        for file in files_to_process[:5]:  # Show first 5 files
            print(f"  - {file}")
        if len(files_to_process) > 5:
            print(f"  ... and {len(files_to_process) - 5} more files")
        
        if files_to_process:
            print("\nStarting ingestion process...")
            try:
                # Call the ingest function
                ingest_folder_to_search(
                    folder_path=data_folder_path,
                    client=client_V1,
                    aoaiclient=client,
                    search_client=search_client,
                    max_tokens=512,
                    overlap_tokens=50,
                    embedding_batch_size=16  # Smaller batch size to avoid rate limits
                )
                print(" Document ingestion completed successfully!")
            except Exception as e:
                print(f" Error during ingestion: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No supported document files found in the folder.")
            print(f"Supported file types: {', '.join(supported_extensions)}")