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
from typing import List, Dict, Iterable, Optional, Tuple

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from zmq import TYPE


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
# endregion

# Load environment variables early
load_dotenv()


# endregion




credential = ClientSecretCredential(
    tenant_id = os.getenv("TENANT_ID"), client_id = os.getenv("CLIENT_ID"), client_secret = os.getenv("CLIENT_SECRET"))

token_provider = credential.get_token("https://cognitiveservices.azure.com/.default")


client_V1 = OpenAI (
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


try:  
    query = "What is in these documents?"  
    print(f"Running quick vector search for: {query}")  
    q_vec = embed_texts(client=client, model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"), texts=[query])[0]  
    results = search_client.search(  
        search_text=None,  
        vector_queries=[VectorizedQuery(vector=q_vec, k_nearest_neighbors=5, fields="embedding")],  
        select=["documentId", "filePath", "defendants", "summary"]  
    )  
    for rank, r in enumerate(results):  
        print(f"{rank+1}. {r['documentId']} ({r['filePath']} - Defendants: {r['defendants']})\n   Summary: {r['summary']}\n")  

except Exception as e:  
    print(f"Vector search test skipped/failed: {e}")

# Search by specific year

"""Search for cases from a specific date"""
try:
    searchdate = "2018-10-12"
    print(f"\n{'='*60}")
    print(f"🔍 Searching for incidents from date: {searchdate}")
    print(f"{'='*60}")
    
    results = search_client.search(
        search_text="*",  # Match all documents
        filter=f"IncidentDate eq {searchdate}",  # Filter by date
        select=["DocumentId", "filePath", "IncidentType", "IncidentDate", "Category", "NumberOfImpactedEmployees", "EmployeeNames", "Location", "Injuries", "Summary"],
        top=10,
        #order_by=["documentId desc"]  # Sort by document ID (newest first)
    )
    
    count = 0
    for result in results:
        count += 1
        print(f"\n{count}. Document: {result['DocumentId']}")
        print(f"    Incident Date: {result['IncidentDate']}")
        print(f"    Location: {result['Location']}")
        print(f"    Employee Names: {result['EmployeeNames']}")
        print(f"    Injuries: {result['Injuries']}")
        print(f"    Summary: {result['Summary'][:200]}{'...' if len(result['Summary']) > 200 else ''}")
        print(f"    File: {result['filePath']}")
        
    if count == 0:
        print(f" No Incidents found for date {searchdate}")
    else:
        print(f"\n Found {count} incident(s) from date {searchdate}")
        
except Exception as e:
    print(f" Error searching by date: {e}")



