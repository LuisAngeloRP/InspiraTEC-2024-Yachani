# utils/document_manager.py
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
from pathlib import Path
import shutil

class DocumentManager:
    def __init__(self):
        # Definir estructura base de directorios
        self.BASE_DIR = "data"
        self.PROCESSED_DIR = os.path.join(self.BASE_DIR, "processed_docs")
        self.METADATA_FILE = os.path.join(self.BASE_DIR, "metadata.json")
        self.CATEGORIES_FILE = os.path.join(self.BASE_DIR, "categories.json")
        
        # Crear estructura de directorios
        self._ensure_directory_structure()
        
        # Inicializar o cargar datos
        self.metadata = self._load_metadata()
        self.categories = self._load_categories()

    def _ensure_directory_structure(self):
        """Crear estructura de directorios necesaria."""
        os.makedirs(self.BASE_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)

    def _load_metadata(self) -> Dict:
        """Cargar o crear archivo de metadatos."""
        try:
            if os.path.exists(self.METADATA_FILE):
                with open(self.METADATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {self.METADATA_FILE}, creating new file")
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
        
        # Si hay error o no existe, crear nuevo
        self._save_metadata({})
        return {}

    def _load_categories(self) -> Dict:
        """Cargar o crear estructura de categorías."""
        default_categories = {
            "categories": {
                "Matemáticas": ["Álgebra", "Cálculo", "Geometría", "Estadística"],
                "Ciencias": ["Física", "Química", "Biología", "Astronomía"],
                "Programación": ["Python", "JavaScript", "Java", "Web Development"],
                "Idiomas": ["Inglés", "Español", "Francés", "Alemán"],
                "Historia": ["Historia Mundial", "Historia del Arte", "Arqueología"],
                "Literatura": ["Narrativa", "Poesía", "Teatro", "Ensayo"]
            },
            "category_counts": {}
        }
        
        try:
            if os.path.exists(self.CATEGORIES_FILE):
                with open(self.CATEGORIES_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {self.CATEGORIES_FILE}, creating new file")
        except Exception as e:
            print(f"Error loading categories: {str(e)}")
        
        # Si hay error o no existe, crear nuevo
        self._save_categories(default_categories)
        return default_categories

    def _save_metadata(self, metadata: Dict) -> None:
        """Guardar metadatos de forma segura."""
        try:
            with open(self.METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")

    def _save_categories(self, categories: Dict) -> None:
        """Guardar categorías de forma segura."""
        try:
            with open(self.CATEGORIES_FILE, 'w', encoding='utf-8') as f:
                json.dump(categories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving categories: {str(e)}")

    def get_document_types(self) -> List[str]:
        """Obtener tipos de documentos disponibles."""
        return [
            "Libro de Texto",
            "Guía de Estudio",
            "Manual Técnico",
            "Paper Académico",
            "Presentación",
            "Material de Curso",
            "Documento de Investigación",
            "Apuntes",
            "Tutorial"
        ]

    def get_difficulty_levels(self) -> List[str]:
        """Obtener niveles de dificultad disponibles."""
        return [
            "Principiante",
            "Intermedio",
            "Avanzado",
            "Experto"
        ]

    def get_total_documents(self) -> int:
        """Obtener número total de documentos."""
        return len(self.metadata)

    def get_categories(self) -> Dict:
        """Obtener estructura de categorías."""
        return self.categories["categories"]

    def get_popular_categories(self) -> Dict:
        """Obtener categorías más populares."""
        counts = self.categories.get('category_counts', {})
        return dict(sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:8])

    def get_documents_by_category(self, category: str) -> List[Dict]:
        """Obtener documentos de una categoría específica."""
        return [
            doc for doc in self.metadata.values()
            if doc.get('category') == category
        ]

    def get_document(self, doc_hash: str) -> Optional[Dict]:
        """Obtener metadata de un documento específico."""
        return self.metadata.get(doc_hash)

    def get_new_documents_count(self, date: datetime) -> int:
        """Obtener cantidad de documentos nuevos para una fecha."""
        count = 0
        for doc in self.metadata.values():
            try:
                doc_date = datetime.fromisoformat(doc.get('processed_date', ''))
                if doc_date.date() == date.date():
                    count += 1
            except (ValueError, TypeError):
                continue
        return count

    def search_documents(self, query: str = None, filters: Dict = None) -> List[Dict]:
        """Buscar documentos con filtros."""
        results = self.metadata.values()
        
        if filters:
            for key, value in filters.items():
                if value and value != "Todas" and value != "Todos":
                    if key == "year_range":
                        results = [
                            doc for doc in results
                            if value[0] <= int(doc.get('year', 0)) <= value[1]
                        ]
                    else:
                        results = [
                            doc for doc in results
                            if doc.get(key) == value
                        ]
        
        if query:
            query = query.lower()
            results = [
                doc for doc in results
                if any(
                    query in str(value).lower()
                    for value in [
                        doc.get('title', ''),
                        doc.get('description', ''),
                        doc.get('author', ''),
                        *doc.get('tags', [])
                    ]
                )
            ]
        
        return list(results)

    def add_document(self, metadata: dict, vectorstore_path: str, original_path: str) -> str:
        """Agregar un nuevo documento."""
        try:
            # Generar hash único
            doc_hash = hashlib.sha256(
                f"{metadata['title']}_{metadata['author']}_{metadata['year']}".encode()
            ).hexdigest()
            
            # Agregar información adicional
            full_metadata = {
                **metadata,
                "hash": doc_hash,
                "vectorstore_path": vectorstore_path,
                "original_path": original_path,
                "processed_date": datetime.now().isoformat()
            }
            
            # Actualizar metadata
            self.metadata[doc_hash] = full_metadata
            self._save_metadata(self.metadata)
            
            # Actualizar conteo de categorías
            category = metadata['category']
            self.categories['category_counts'][category] = \
                self.categories['category_counts'].get(category, 0) + 1
            self._save_categories(self.categories)
            
            return doc_hash
            
        except Exception as e:
            raise Exception(f"Error adding document: {str(e)}")