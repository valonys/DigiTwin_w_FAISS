import os
import pandas as pd
from typing import List, Dict, Any, Optional

from langchain.schema import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings


class VectorStore:
    """
    A class to handle text processing and vector indexing using FAISS.
    This class provides methods to process documents, create embeddings,
    and perform similarity searches.
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_openai: bool = False,
                 openai_api_key: Optional[str] = None):
        """
        Initialize the VectorStore with specified parameters.
        
        Args:
            embedding_model: The HuggingFace model to use for embeddings
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            use_openai: Whether to use OpenAI embeddings instead of HuggingFace
            openai_api_key: OpenAI API key (required if use_openai is True)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_openai = use_openai
        
        # Initialize the text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize embeddings model
        if use_openai:
            if not openai_api_key and not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key must be provided when use_openai is True")
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        self.vectorstore = None
    
    def process_excel(self, file_path: str) -> pd.DataFrame:
        """
        Process an Excel file, specifically looking for the 'Global Notifications' worksheet
        and filtering to retain only specific columns.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Filtered DataFrame with only the required columns
        """
        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Excel file not found at {file_path}")
            
            # Read the Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Check if 'Global Notifications' worksheet exists
            if 'Global Notifications' in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name='Global Notifications')
                
                # Check if required columns exist
                required_columns = ["Notificn type", "Description", "Main WorkCtr", "FPSO"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
                
                # Filter to retain only specific columns
                df = df[["Notificn type", "Description", "Main WorkCtr", "FPSO"]]
                
                # Remove rows with no description
                df = df.dropna(subset=["Description"])
                
                return df
            else:
                raise ValueError("'Global Notifications' worksheet not found in the Excel file")
                
        except Exception as e:
            raise Exception(f"Error processing Excel file: {str(e)}")
    
    def process_excel_to_documents(self, file_path: str) -> List[LCDocument]:
        """
        Process an Excel file and convert it to a list of LangChain Document objects.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            List of LangChain Document objects
        """
        df = self.process_excel(file_path)
        
        documents = []
        for idx, row in df.iterrows():
            # Create a formatted text representation of the row
            content = f"Notification Type: {row['Notificn type']}\n" \
                     f"Description: {row['Description']}\n" \
                     f"Main Work Center: {row['Main WorkCtr']}\n" \
                     f"FPSO: {row['FPSO']}"
            
            # Create metadata
            metadata = {
                "source": file_path,
                "row_index": idx,
                "notification_type": row['Notificn type'],
                "main_workctr": row['Main WorkCtr'],
                "fpso": row['FPSO']
            }
            
            documents.append(LCDocument(page_content=content, metadata=metadata))
        
        return documents
    
    def process_documents(self, documents: List[LCDocument]) -> None:
        """
        Process a list of documents and create a FAISS vectorstore.
        
        Args:
            documents: List of LangChain Document objects
        """
        chunks = []
        for i, doc in enumerate(documents):
            # Split the document into chunks
            doc_chunks = self.splitter.split_text(doc.page_content)
            
            # Create LangChain Document objects for each chunk
            for chunk in doc_chunks:
                # Preserve the original document's metadata and add source identifier
                metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
                metadata['source'] = f"doc_{i}"
                
                chunks.append(LCDocument(page_content=chunk, metadata=metadata))
        
        # Create FAISS index from the chunks
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 5) -> List[LCDocument]:
        """
        Perform a similarity search on the vectorstore.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore available. Process documents first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def save_local(self, path: str) -> None:
        """
        Save the FAISS index to a local directory.
        
        Args:
            path: Directory path to save the index
        """
        if not self.vectorstore:
            raise ValueError("No vectorstore available to save.")
        
        self.vectorstore.save_local(path)
    
    @classmethod
    def load_local(cls, path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", use_openai: bool = False) -> 'VectorStore':
        """
        Load a FAISS index from a local directory.
        
        Args:
            path: Directory path where the index is stored
            embedding_model: The HuggingFace model to use for embeddings
            use_openai: Whether to use OpenAI embeddings
            
        Returns:
            VectorStore instance with loaded index
        """
        instance = cls(embedding_model=embedding_model, use_openai=use_openai)
        
        if use_openai:
            embeddings = OpenAIEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        instance.vectorstore = FAISS.load_local(path, embeddings)
        return instance