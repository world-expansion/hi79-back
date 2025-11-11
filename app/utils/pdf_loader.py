from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import os
import glob

class PDFProcessor:
    def __init__(self, data_dir: str = "data", chunk_size: int = 500, chunk_overlap: int = 100):
        """
        data_dir: PDF 파일들이 있는 디렉토리 경로
        chunk_size: 각 청크의 크기 (기본값: 500자)
        chunk_overlap: 청크 간 겹치는 부분 (기본값: 100자)
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self) -> tuple[List, List[str]]:
        """
        data_dir 내 모든 PDF 파일을 로드하고 청크로 분할

        Returns:
            tuple: (분할된 문서 리스트, 로드된 PDF 파일명 리스트)
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # data 폴더 내 모든 PDF 파일 찾기
        pdf_files = glob.glob(os.path.join(self.data_dir, "*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.data_dir}")

        all_documents = []
        loaded_files = []

        # 각 PDF 파일 로드
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                # PostgreSQL 호환성: NULL 문자 제거
                for doc in documents:
                    doc.page_content = doc.page_content.replace('\x00', '')

                all_documents.extend(documents)
                loaded_files.append(os.path.basename(pdf_path))
                print(f"✓ Loaded: {os.path.basename(pdf_path)} ({len(documents)} pages)")
            except Exception as e:
                print(f"✗ Failed to load {os.path.basename(pdf_path)}: {str(e)}")

        if not all_documents:
            raise ValueError("No documents were successfully loaded")

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        splits = text_splitter.split_documents(all_documents)
        return splits, loaded_files
