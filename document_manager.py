import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import pandas as pd

class DocumentManager:
    """Gestor avanzado de documentos con organización y metadatos"""
    
    def __init__(self, storage_path: str = "documents"):
        self.storage_path = storage_path
        self.metadata_file = os.path.join(storage_path, "metadata.json")
        self.ensure_storage_directory()
        self.metadata = self.load_metadata()
    
    def ensure_storage_directory(self):
        """Crea el directorio de almacenamiento si no existe"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
    
    def add_document(self, doc_id: str, file_path: str, content: Dict[str, Any], 
                    tags: List[str] = None, category: str = "general") -> bool:
        """Añade un documento al sistema de gestión"""
        try:
            # Calcular hash del archivo para detectar duplicados
            file_hash = self._calculate_file_hash(file_path)
            
            # Verificar si ya existe un documento con el mismo hash
            existing_doc = self._find_document_by_hash(file_hash)
            if existing_doc:
                return False, f"Documento duplicado encontrado: {existing_doc['name']}"
            
            # Copiar archivo al almacenamiento
            filename = os.path.basename(file_path)
            stored_path = os.path.join(self.storage_path, f"{doc_id}_{filename}")
            shutil.copy2(file_path, stored_path)
            
            # Crear metadatos del documento
            doc_metadata = {
                'doc_id': doc_id,
                'name': filename,
                'original_path': file_path,
                'stored_path': stored_path,
                'file_hash': file_hash,
                'upload_date': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'file_size': os.path.getsize(file_path),
                'content_stats': {
                    'total_pages': content.get('total_pages', 0),
                    'word_count': content.get('word_count', 0),
                    'char_count': content.get('char_count', 0),
                    'table_count': len(content.get('tables', [])),
                    'image_count': len(content.get('images', []))
                },
                'tags': tags or [],
                'category': category,
                'metadata': content.get('metadata', {}),
                'access_count': 0,
                'favorite': False,
                'notes': ""
            }
            
            # Guardar metadatos
            self.metadata[doc_id] = doc_metadata
            self.save_metadata()
            
            return True, "Documento añadido correctamente"
            
        except Exception as e:
            return False, f"Error añadiendo documento: {str(e)}"
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un documento"""
        if doc_id in self.metadata:
            # Actualizar último acceso
            self.metadata[doc_id]['last_accessed'] = datetime.now().isoformat()
            self.metadata[doc_id]['access_count'] += 1
            self.save_metadata()
            return self.metadata[doc_id]
        return None
    
    def list_documents(self, category: str = None, tags: List[str] = None, 
                      sort_by: str = "upload_date", ascending: bool = False) -> List[Dict[str, Any]]:
        """Lista documentos con filtros opcionales"""
        documents = list(self.metadata.values())
        
        # Aplicar filtros
        if category:
            documents = [doc for doc in documents if doc['category'] == category]
        
        if tags:
            documents = [doc for doc in documents 
                        if any(tag in doc['tags'] for tag in tags)]
        
        # Ordenar
        reverse = not ascending
        if sort_by == "upload_date":
            documents.sort(key=lambda x: x['upload_date'], reverse=reverse)
        elif sort_by == "name":
            documents.sort(key=lambda x: x['name'].lower(), reverse=reverse)
        elif sort_by == "size":
            documents.sort(key=lambda x: x['file_size'], reverse=reverse)
        elif sort_by == "access_count":
            documents.sort(key=lambda x: x['access_count'], reverse=reverse)
        elif sort_by == "last_accessed":
            documents.sort(key=lambda x: x['last_accessed'], reverse=reverse)
        
        return documents
    
    def search_documents(self, query: str, search_in: List[str] = None) -> List[Dict[str, Any]]:
        """Busca documentos por nombre, tags o metadatos"""
        if search_in is None:
            search_in = ['name', 'tags', 'notes', 'category']
        
        query_lower = query.lower()
        results = []
        
        for doc in self.metadata.values():
            match_score = 0
            
            # Buscar en nombre
            if 'name' in search_in and query_lower in doc['name'].lower():
                match_score += 3
            
            # Buscar en tags
            if 'tags' in search_in:
                for tag in doc['tags']:
                    if query_lower in tag.lower():
                        match_score += 2
            
            # Buscar en notas
            if 'notes' in search_in and query_lower in doc['notes'].lower():
                match_score += 1
            
            # Buscar en categoría
            if 'category' in search_in and query_lower in doc['category'].lower():
                match_score += 1
            
            if match_score > 0:
                doc_copy = doc.copy()
                doc_copy['match_score'] = match_score
                results.append(doc_copy)
        
        # Ordenar por relevancia
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results
    
    def update_document_metadata(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Actualiza metadatos de un documento"""
        try:
            if doc_id not in self.metadata:
                return False
            
            # Campos permitidos para actualización
            allowed_fields = ['tags', 'category', 'notes', 'favorite']
            
            for field, value in updates.items():
                if field in allowed_fields:
                    self.metadata[doc_id][field] = value
            
            self.metadata[doc_id]['last_modified'] = datetime.now().isoformat()
            self.save_metadata()
            return True
            
        except Exception as e:
            print(f"Error actualizando metadatos: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Elimina un documento del sistema"""
        try:
            if doc_id not in self.metadata:
                return False
            
            doc_info = self.metadata[doc_id]
            
            # Eliminar archivo físico
            if os.path.exists(doc_info['stored_path']):
                os.remove(doc_info['stored_path'])
            
            # Eliminar metadatos
            del self.metadata[doc_id]
            self.save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error eliminando documento: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de documentos"""
        if not self.metadata:
            return {
                'total_documents': 0,
                'total_size': 0,
                'categories': {},
                'tags': {},
                'avg_pages': 0,
                'most_accessed': None
            }
        
        docs = list(self.metadata.values())
        
        # Estadísticas básicas
        total_docs = len(docs)
        total_size = sum(doc['file_size'] for doc in docs)
        total_pages = sum(doc['content_stats']['total_pages'] for doc in docs)
        
        # Categorías
        categories = {}
        for doc in docs:
            cat = doc['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        # Tags
        tags = {}
        for doc in docs:
            for tag in doc['tags']:
                tags[tag] = tags.get(tag, 0) + 1
        
        # Documento más accedido
        most_accessed = max(docs, key=lambda x: x['access_count']) if docs else None
        
        return {
            'total_documents': total_docs,
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'categories': categories,
            'tags': tags,
            'avg_pages': round(total_pages / total_docs, 1) if total_docs > 0 else 0,
            'most_accessed': most_accessed['name'] if most_accessed else None,
            'upload_dates': [doc['upload_date'] for doc in docs]
        }
    
    def export_metadata(self, format: str = "json") -> str:
        """Exporta metadatos en diferentes formatos"""
        try:
            if format.lower() == "json":
                return json.dumps(self.metadata, indent=2, ensure_ascii=False)
            
            elif format.lower() == "csv":
                # Convertir a DataFrame para CSV
                rows = []
                for doc_id, doc_data in self.metadata.items():
                    row = {
                        'doc_id': doc_id,
                        'name': doc_data['name'],
                        'category': doc_data['category'],
                        'tags': ', '.join(doc_data['tags']),
                        'upload_date': doc_data['upload_date'],
                        'file_size': doc_data['file_size'],
                        'pages': doc_data['content_stats']['total_pages'],
                        'word_count': doc_data['content_stats']['word_count'],
                        'access_count': doc_data['access_count'],
                        'favorite': doc_data['favorite']
                    }
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                return df.to_csv(index=False)
            
            else:
                return "Formato no soportado"
                
        except Exception as e:
            return f"Error exportando: {str(e)}"
    
    def get_duplicates(self) -> List[List[Dict[str, Any]]]:
        """Encuentra documentos duplicados basándose en el hash"""
        hash_groups = {}
        
        for doc in self.metadata.values():
            file_hash = doc['file_hash']
            if file_hash not in hash_groups:
                hash_groups[file_hash] = []
            hash_groups[file_hash].append(doc)
        
        # Retornar solo grupos con más de un documento
        duplicates = [group for group in hash_groups.values() if len(group) > 1]
        return duplicates
    
    def cleanup_orphaned_files(self) -> List[str]:
        """Limpia archivos huérfanos en el directorio de almacenamiento"""
        orphaned = []
        
        try:
            # Obtener todos los archivos en el directorio
            stored_files = set()
            for filename in os.listdir(self.storage_path):
                if filename != "metadata.json":
                    stored_files.add(os.path.join(self.storage_path, filename))
            
            # Obtener archivos referenciados en metadatos
            referenced_files = set()
            for doc in self.metadata.values():
                referenced_files.add(doc['stored_path'])
            
            # Encontrar archivos huérfanos
            orphaned_files = stored_files - referenced_files
            
            # Eliminar archivos huérfanos
            for file_path in orphaned_files:
                try:
                    os.remove(file_path)
                    orphaned.append(file_path)
                except Exception as e:
                    print(f"Error eliminando archivo huérfano {file_path}: {str(e)}")
            
        except Exception as e:
            print(f"Error en limpieza: {str(e)}")
        
        return orphaned
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcula hash SHA-256 de un archivo"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculando hash: {str(e)}")
            return ""
    
    def _find_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Encuentra un documento por su hash"""
        for doc in self.metadata.values():
            if doc['file_hash'] == file_hash:
                return doc
        return None
    
    def load_metadata(self) -> Dict[str, Any]:
        """Carga metadatos desde archivo"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error cargando metadatos: {str(e)}")
        
        return {}
    
    def save_metadata(self):
        """Guarda metadatos en archivo"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando metadatos: {str(e)}")
    
    def backup_metadata(self, backup_path: str = None) -> str:
        """Crea una copia de seguridad de los metadatos"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(self.storage_path, f"metadata_backup_{timestamp}.json")
            
            shutil.copy2(self.metadata_file, backup_path)
            return backup_path
            
        except Exception as e:
            print(f"Error creando backup: {str(e)}")
            return ""
