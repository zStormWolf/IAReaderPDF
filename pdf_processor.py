import PyPDF2
import pdfplumber
import io
from typing import Dict, List, Any
import re
from PIL import Image
import pandas as pd

class PDFProcessor:
    """Procesador avanzado de archivos PDF con múltiples métodos de extracción"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_content(self, file_path: str) -> Dict[str, Any]:
        """
        Extrae todo el contenido de un PDF usando múltiples métodos
        """
        content = {
            'text': '',
            'tables': [],
            'images': [],
            'metadata': {},
            'total_pages': 0,
            'pages_content': []
        }
        
        try:
            # Método 1: PyPDF2 para texto básico y metadata
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content['total_pages'] = len(pdf_reader.pages)
                content['metadata'] = self._extract_metadata(pdf_reader)
                
                # Extraer texto página por página
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    content['pages_content'].append({
                        'page_number': page_num + 1,
                        'text': page_text,
                        'word_count': len(page_text.split())
                    })
                    content['text'] += page_text + '\n'
            
            # Método 2: pdfplumber para tablas y layout avanzado
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extraer tablas
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            if table and len(table) > 1:  # Verificar que la tabla tenga contenido
                                df = pd.DataFrame(table[1:], columns=table[0])
                                content['tables'].append({
                                    'page': page_num + 1,
                                    'table_index': table_idx,
                                    'data': df.to_dict('records'),
                                    'columns': list(df.columns),
                                    'rows': len(df)
                                })
                    
                    # Extraer imágenes (información básica)
                    if hasattr(page, 'images'):
                        for img_idx, img in enumerate(page.images):
                            content['images'].append({
                                'page': page_num + 1,
                                'image_index': img_idx,
                                'bbox': img.get('bbox', []),
                                'width': img.get('width', 0),
                                'height': img.get('height', 0)
                            })
            
            # Limpiar y procesar texto
            content['text'] = self._clean_text(content['text'])
            content['word_count'] = len(content['text'].split())
            content['char_count'] = len(content['text'])
            
        except Exception as e:
            raise Exception(f"Error procesando PDF: {str(e)}")
        
        return content
    
    def _extract_metadata(self, pdf_reader) -> Dict[str, Any]:
        """Extrae metadata del PDF"""
        metadata = {}
        try:
            if pdf_reader.metadata:
                metadata = {
                    'title': pdf_reader.metadata.get('/Title', ''),
                    'author': pdf_reader.metadata.get('/Author', ''),
                    'subject': pdf_reader.metadata.get('/Subject', ''),
                    'creator': pdf_reader.metadata.get('/Creator', ''),
                    'producer': pdf_reader.metadata.get('/Producer', ''),
                    'creation_date': pdf_reader.metadata.get('/CreationDate', ''),
                    'modification_date': pdf_reader.metadata.get('/ModDate', '')
                }
        except:
            pass
        return metadata
    
    def _clean_text(self, text: str) -> str:
        """Limpia y normaliza el texto extraído"""
        if not text:
            return ""
        
        # Eliminar caracteres de control y espacios excesivos
        text = re.sub(r'\x00-\x1f\x7f-\x9f', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Eliminar líneas muy cortas que probablemente sean ruido
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 3:  # Mantener líneas con más de 3 caracteres
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_text_by_page(self, file_path: str) -> List[Dict[str, Any]]:
        """Extrae texto página por página con información detallada"""
        pages = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    cleaned_text = self._clean_text(text)
                    
                    pages.append({
                        'page_number': page_num + 1,
                        'text': cleaned_text,
                        'word_count': len(cleaned_text.split()),
                        'char_count': len(cleaned_text),
                        'line_count': len(cleaned_text.split('\n'))
                    })
        
        except Exception as e:
            raise Exception(f"Error extrayendo texto por página: {str(e)}")
        
        return pages
    
    def search_text_in_pdf(self, file_path: str, search_term: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Busca un término específico en el PDF y devuelve las coincidencias con contexto"""
        results = []
        
        try:
            pages = self.extract_text_by_page(file_path)
            
            for page_data in pages:
                text = page_data['text']
                if not case_sensitive:
                    search_text = text.lower()
                    term = search_term.lower()
                else:
                    search_text = text
                    term = search_term
                
                # Encontrar todas las posiciones del término
                start = 0
                while True:
                    pos = search_text.find(term, start)
                    if pos == -1:
                        break
                    
                    # Extraer contexto (50 caracteres antes y después)
                    context_start = max(0, pos - 50)
                    context_end = min(len(text), pos + len(term) + 50)
                    context = text[context_start:context_end]
                    
                    results.append({
                        'page': page_data['page_number'],
                        'position': pos,
                        'context': context,
                        'match': text[pos:pos + len(term)]
                    })
                    
                    start = pos + 1
        
        except Exception as e:
            raise Exception(f"Error buscando en PDF: {str(e)}")
        
        return results
    
    def get_document_structure(self, file_path: str) -> Dict[str, Any]:
        """Analiza la estructura del documento (títulos, secciones, etc.)"""
        structure = {
            'headings': [],
            'sections': [],
            'outline': []
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Intentar extraer outline/bookmarks
                if pdf_reader.outline:
                    structure['outline'] = self._parse_outline(pdf_reader.outline)
                
                # Analizar texto para encontrar posibles títulos
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    headings = self._detect_headings(text, page_num + 1)
                    structure['headings'].extend(headings)
        
        except Exception as e:
            print(f"Advertencia: No se pudo analizar la estructura: {str(e)}")
        
        return structure
    
    def _parse_outline(self, outline, level=0) -> List[Dict[str, Any]]:
        """Parsea el outline/bookmarks del PDF"""
        parsed_outline = []
        
        for item in outline:
            if isinstance(item, list):
                parsed_outline.extend(self._parse_outline(item, level + 1))
            else:
                try:
                    parsed_outline.append({
                        'title': item.title,
                        'level': level,
                        'page': item.page.idnum if hasattr(item, 'page') else None
                    })
                except:
                    continue
        
        return parsed_outline
    
    def _detect_headings(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Detecta posibles títulos en el texto basándose en patrones"""
        headings = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Patrones para detectar títulos
            if (len(line) > 5 and len(line) < 100 and
                (line.isupper() or  # Todo en mayúsculas
                 re.match(r'^\d+\.?\s+[A-Z]', line) or  # Numerado
                 re.match(r'^[A-Z][^.!?]*$', line))):  # Empieza con mayúscula, sin puntuación final
                
                headings.append({
                    'text': line,
                    'page': page_num,
                    'line': line_num + 1,
                    'type': 'detected_heading'
                })
        
        return headings
