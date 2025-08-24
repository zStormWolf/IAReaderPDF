import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Tuple
import pickle
import os
import re

class SemanticSearch:
    """Sistema de búsqueda semántica para documentos PDF"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Inicializar índice FAISS
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product para similitud coseno
        
        # Almacenar metadatos de documentos
        self.documents = {}  # doc_id -> document_data
        self.chunks = []     # Lista de chunks con metadata
        self.doc_embeddings = {}  # doc_id -> embeddings
        
    def add_document(self, doc_id: str, text: str, chunk_size: int = 500, overlap: int = 50):
        """Añade un documento al índice de búsqueda"""
        try:
            # Dividir texto en chunks
            chunks = self._split_text_into_chunks(text, chunk_size, overlap)
            
            # Generar embeddings para cada chunk
            chunk_embeddings = []
            chunk_metadata = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 10:  # Solo procesar chunks con contenido significativo
                    embedding = self.model.encode(chunk, convert_to_tensor=False)
                    chunk_embeddings.append(embedding)
                    
                    chunk_metadata.append({
                        'doc_id': doc_id,
                        'chunk_id': f"{doc_id}_chunk_{i}",
                        'text': chunk,
                        'chunk_index': i,
                        'word_count': len(chunk.split())
                    })
            
            if chunk_embeddings:
                # Normalizar embeddings para similitud coseno
                embeddings_array = np.array(chunk_embeddings).astype('float32')
                faiss.normalize_L2(embeddings_array)
                
                # Añadir al índice FAISS
                start_idx = len(self.chunks)
                self.index.add(embeddings_array)
                
                # Actualizar metadatos
                self.chunks.extend(chunk_metadata)
                self.documents[doc_id] = {
                    'text': text,
                    'chunk_count': len(chunk_embeddings),
                    'start_index': start_idx,
                    'end_index': start_idx + len(chunk_embeddings)
                }
                self.doc_embeddings[doc_id] = embeddings_array
                
                return True
            
        except Exception as e:
            print(f"Error añadiendo documento {doc_id}: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """Busca en todos los documentos"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Generar embedding de la consulta
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Buscar en el índice
            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= min_score:
                    chunk_data = self.chunks[idx].copy()
                    chunk_data['similarity_score'] = float(score)
                    chunk_data['relevance'] = self._calculate_relevance(query, chunk_data['text'])
                    results.append(chunk_data)
            
            # Ordenar por score y relevancia
            results.sort(key=lambda x: (x['similarity_score'], x['relevance']), reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Error en búsqueda: {str(e)}")
            return []
    
    def search_in_document(self, query: str, doc_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Busca específicamente en un documento"""
        try:
            if doc_id not in self.documents:
                return []
            
            doc_data = self.documents[doc_id]
            start_idx = doc_data['start_index']
            end_idx = doc_data['end_index']
            
            # Generar embedding de la consulta
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            query_embedding = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Buscar solo en los chunks de este documento
            doc_embeddings = self.doc_embeddings[doc_id]
            scores = np.dot(doc_embeddings, query_embedding.T).flatten()
            
            # Obtener top_k resultados
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for local_idx in top_indices:
                global_idx = start_idx + local_idx
                score = scores[local_idx]
                
                if score >= 0.3:  # Umbral mínimo
                    chunk_data = self.chunks[global_idx].copy()
                    chunk_data['similarity_score'] = float(score)
                    chunk_data['relevance'] = self._calculate_relevance(query, chunk_data['text'])
                    results.append(chunk_data)
            
            return results
            
        except Exception as e:
            print(f"Error buscando en documento {doc_id}: {str(e)}")
            return []
    
    def get_similar_chunks(self, doc_id: str, chunk_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Encuentra chunks similares a un texto dado"""
        try:
            # Generar embedding del chunk
            chunk_embedding = self.model.encode(chunk_text, convert_to_tensor=False)
            chunk_embedding = np.array([chunk_embedding]).astype('float32')
            faiss.normalize_L2(chunk_embedding)
            
            # Buscar chunks similares
            scores, indices = self.index.search(chunk_embedding, top_k + 5)  # Buscar más para filtrar
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= 0.4:
                    chunk_data = self.chunks[idx]
                    # Excluir chunks del mismo documento si se especifica
                    if chunk_data['doc_id'] != doc_id:
                        chunk_data_copy = chunk_data.copy()
                        chunk_data_copy['similarity_score'] = float(score)
                        results.append(chunk_data_copy)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Error encontrando chunks similares: {str(e)}")
            return []
    
    def remove_document(self, doc_id: str):
        """Elimina un documento del índice"""
        try:
            if doc_id in self.documents:
                # Nota: FAISS no permite eliminar elementos fácilmente
                # En una implementación completa, se reconstruiría el índice
                del self.documents[doc_id]
                if doc_id in self.doc_embeddings:
                    del self.doc_embeddings[doc_id]
                
                # Filtrar chunks
                self.chunks = [chunk for chunk in self.chunks if chunk['doc_id'] != doc_id]
                
                # Reconstruir índice (simplificado)
                self._rebuild_index()
                
        except Exception as e:
            print(f"Error eliminando documento {doc_id}: {str(e)}")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Divide el texto en chunks con solapamiento"""
        if not text or len(text.strip()) == 0:
            return []
        
        # Dividir por párrafos primero
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si el párrafo es muy largo, dividirlo por oraciones
            if len(paragraph) > chunk_size:
                sentences = self._split_into_sentences(paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Mantener overlap
                            words = current_chunk.split()
                            if len(words) > overlap:
                                current_chunk = ' '.join(words[-overlap:]) + ' ' + sentence
                            else:
                                current_chunk = sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk += ' ' + sentence if current_chunk else sentence
            else:
                if len(current_chunk) + len(paragraph) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Mantener overlap
                        words = current_chunk.split()
                        if len(words) > overlap:
                            current_chunk = ' '.join(words[-overlap:]) + ' ' + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk += '\n\n' + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones"""
        # Patrón simple para dividir oraciones
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calcula relevancia adicional basada en coincidencias de palabras clave"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calcular intersección
        common_words = query_words.intersection(text_words)
        relevance = len(common_words) / len(query_words)
        
        return relevance
    
    def _rebuild_index(self):
        """Reconstruye el índice FAISS (simplificado)"""
        try:
            # Crear nuevo índice
            self.index = faiss.IndexFlatIP(self.dimension)
            
            if self.chunks:
                # Regenerar embeddings para chunks restantes
                all_embeddings = []
                for chunk in self.chunks:
                    embedding = self.model.encode(chunk['text'], convert_to_tensor=False)
                    all_embeddings.append(embedding)
                
                if all_embeddings:
                    embeddings_array = np.array(all_embeddings).astype('float32')
                    faiss.normalize_L2(embeddings_array)
                    self.index.add(embeddings_array)
                    
                    # Actualizar índices de documentos
                    current_idx = 0
                    for doc_id in self.documents:
                        doc_chunks = [c for c in self.chunks if c['doc_id'] == doc_id]
                        if doc_chunks:
                            self.documents[doc_id]['start_index'] = current_idx
                            self.documents[doc_id]['end_index'] = current_idx + len(doc_chunks)
                            current_idx += len(doc_chunks)
                            
                            # Actualizar embeddings del documento
                            doc_embeddings = embeddings_array[
                                self.documents[doc_id]['start_index']:
                                self.documents[doc_id]['end_index']
                            ]
                            self.doc_embeddings[doc_id] = doc_embeddings
            
        except Exception as e:
            print(f"Error reconstruyendo índice: {str(e)}")
    
    def save_index(self, filepath: str):
        """Guarda el índice y metadatos"""
        try:
            data = {
                'documents': self.documents,
                'chunks': self.chunks,
                'model_name': self.model_name
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Guardar índice FAISS por separado
            faiss.write_index(self.index, filepath + '.faiss')
            
        except Exception as e:
            print(f"Error guardando índice: {str(e)}")
    
    def load_index(self, filepath: str):
        """Carga el índice y metadatos"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.chunks = data['chunks']
            
            # Cargar índice FAISS
            self.index = faiss.read_index(filepath + '.faiss')
            
            # Reconstruir embeddings por documento
            self._rebuild_doc_embeddings()
            
        except Exception as e:
            print(f"Error cargando índice: {str(e)}")
    
    def _rebuild_doc_embeddings(self):
        """Reconstruye los embeddings por documento después de cargar"""
        try:
            for doc_id, doc_data in self.documents.items():
                start_idx = doc_data['start_index']
                end_idx = doc_data['end_index']
                
                # Regenerar embeddings para este documento
                doc_chunks = self.chunks[start_idx:end_idx]
                doc_texts = [chunk['text'] for chunk in doc_chunks]
                
                if doc_texts:
                    embeddings = self.model.encode(doc_texts, convert_to_tensor=False)
                    embeddings_array = np.array(embeddings).astype('float32')
                    faiss.normalize_L2(embeddings_array)
                    self.doc_embeddings[doc_id] = embeddings_array
                    
        except Exception as e:
            print(f"Error reconstruyendo embeddings de documentos: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice"""
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal,
            'model_name': self.model_name,
            'embedding_dimension': self.dimension
        }
