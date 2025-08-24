import re
import nltk
import spacy
from collections import Counter
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# Descargar recursos necesarios de NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

class AdvancedAnalyzer:
    """Analizador avanzado de documentos con NLP y extracción de insights"""
    
    def __init__(self):
        self.setup_nlp()
        self.spanish_stopwords = self.get_spanish_stopwords()
    
    def setup_nlp(self):
        """Configura los modelos de NLP, descargándolos si es necesario."""
        model_es = "es_core_news_sm"
        model_en = "en_core_web_sm"
        
        try:
            self.nlp = spacy.load(model_es)
        except OSError:
            print(f"Modelo '{model_es}' no encontrado. Intentando descargar...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_es])
                self.nlp = spacy.load(model_es)
            except Exception as e:
                print(f"Error al descargar '{model_es}': {e}. Intentando con modelo en inglés.")
                try:
                    self.nlp = spacy.load(model_en)
                except OSError:
                    print(f"Modelo '{model_en}' no encontrado. Intentando descargar...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_en])
                        self.nlp = spacy.load(model_en)
                    except Exception as e2:
                        self.nlp = None
                        print(f"Error al descargar '{model_en}': {e2}.")
                        print("Advertencia: No se encontraron modelos de spaCy. Algunas funciones estarán limitadas.")
    
    def get_spanish_stopwords(self) -> set:
        """Obtiene stopwords en español"""
        try:
            from nltk.corpus import stopwords
            spanish_stops = set(stopwords.words('spanish'))
        except:
            spanish_stops = set()
        
        # Añadir stopwords adicionales
        additional_stops = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son',
            'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'han', 'ha', 'este', 'esta',
            'o', 'ser', 'si', 'ya', 'todo', 'esta', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta',
            'hay', 'donde', 'quien', 'desde', 'todos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros'
        }
        
        return spanish_stops.union(additional_stops)
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Genera un resumen automático del texto"""
        if not text or len(text.strip()) < 100:
            return "Texto demasiado corto para generar resumen."
        
        try:
            # Dividir en oraciones
            sentences = self._split_into_sentences(text)
            
            if len(sentences) <= max_sentences:
                return ' '.join(sentences)
            
            # Calcular puntuación de oraciones usando TF-IDF
            vectorizer = TfidfVectorizer(
                stop_words=list(self.spanish_stopwords),
                max_features=100,
                ngram_range=(1, 2)
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            except:
                # Fallback: usar longitud de oraciones
                sentence_scores = np.array([len(s.split()) for s in sentences])
            
            # Seleccionar las mejores oraciones
            top_indices = sentence_scores.argsort()[-max_sentences:][::-1]
            top_indices.sort()  # Mantener orden original
            
            summary_sentences = [sentences[i] for i in top_indices]
            return ' '.join(summary_sentences)
            
        except Exception as e:
            return f"Error generando resumen: {str(e)}"
    
    def extract_entities(self, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """Extrae entidades nombradas del texto y devuelve un posible error."""
        entities = []
        error_message = None
        
        try:
            if not self.nlp:
                error_message = "El modelo de NLP (spaCy) no está cargado. Se utilizará un análisis básico."
                entities = self._extract_entities_regex(text)
                return sorted(entities, key=lambda x: x['count'], reverse=True), error_message

            if not text or len(text.strip()) < 20:
                return [], "El texto es demasiado corto para extraer entidades."

            # Usar spaCy si está disponible
            doc = self.nlp(text[:1000000])  # Limitar texto para evitar problemas de memoria
            
            entity_counts = Counter()
            entity_examples = {}
            
            for ent in doc.ents:
                entity_type = ent.label_
                entity_text = ent.text.strip()
                
                if len(entity_text) > 1:  # Filtrar entidades muy cortas
                    entity_counts[entity_type] += 1
                    
                    if entity_type not in entity_examples:
                        entity_examples[entity_type] = []
                    
                    if entity_text not in entity_examples[entity_type]:
                        entity_examples[entity_type].append(entity_text)
            
            # Convertir a formato de salida
            for entity_type, count in entity_counts.items():
                entities.append({
                    'type': self._translate_entity_type(entity_type),
                    'count': count,
                    'examples': entity_examples[entity_type][:5]  # Máximo 5 ejemplos
                })
            
        except Exception as e:
            error_message = f"Error al extraer entidades: {str(e)}"
            print(error_message)
        
        return sorted(entities, key=lambda x: x['count'], reverse=True), error_message
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> Tuple[List[Tuple[str, float]], str]:
        """Extrae palabras clave usando TF-IDF y devuelve un posible error."""
        error_message = None
        try:
            # Limpiar y preparar texto
            cleaned_text = self._clean_text_for_analysis(text)
            
            if len(cleaned_text.split()) < 10:
                return [], "El texto es demasiado corto para extraer palabras clave significativas."
            
            # Usar TF-IDF para extraer palabras clave
            vectorizer = TfidfVectorizer(
                stop_words=list(self.spanish_stopwords),
                max_features=max_keywords * 2,
                ngram_range=(1, 3),  # Incluir n-gramas
                min_df=1,
                max_df=1.0  # Permitir términos que aparecen en todos los documentos (al procesar uno solo)
            )
            
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Combinar palabras con puntuaciones
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:max_keywords], None
            
        except Exception as e:
            error_message = f"Error al extraer palabras clave: {str(e)}"
            print(error_message)
            return [], error_message
    
    def get_document_stats(self, text: str) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas del documento"""
        if not text:
            return {}
        
        try:
            # Estadísticas básicas
            words = text.split()
            sentences = self._split_into_sentences(text)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Análisis de complejidad
            avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
            avg_chars_per_word = sum(len(word) for word in words) / len(words) if words else 0
            
            # Análisis de legibilidad (aproximado)
            readability_score = self._calculate_readability(text)
            
            # Distribución de longitud de palabras
            word_lengths = [len(word) for word in words if word.isalpha()]
            
            # Palabras más frecuentes (excluyendo stopwords)
            filtered_words = [word.lower() for word in words 
                            if word.isalpha() and word.lower() not in self.spanish_stopwords]
            most_common_words = Counter(filtered_words).most_common(10)
            
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'paragraph_count': len(paragraphs),
                'character_count': len(text),
                'avg_words_per_sentence': round(avg_words_per_sentence, 2),
                'avg_chars_per_word': round(avg_chars_per_word, 2),
                'readability_score': readability_score,
                'avg_word_length': round(np.mean(word_lengths), 2) if word_lengths else 0,
                'most_common_words': most_common_words,
                'unique_words': len(set(filtered_words)),
                'lexical_diversity': len(set(filtered_words)) / len(filtered_words) if filtered_words else 0
            }
            
        except Exception as e:
            print(f"Error calculando estadísticas: {str(e)}")
            return {}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Análisis básico de sentimiento"""
        try:
            # Palabras positivas y negativas básicas en español
            positive_words = {
                'bueno', 'excelente', 'fantástico', 'genial', 'perfecto', 'increíble', 'maravilloso',
                'positivo', 'beneficio', 'éxito', 'logro', 'ventaja', 'oportunidad', 'solución',
                'eficiente', 'efectivo', 'útil', 'valioso', 'importante', 'significativo'
            }
            
            negative_words = {
                'malo', 'terrible', 'horrible', 'pésimo', 'negativo', 'problema', 'error',
                'fallo', 'defecto', 'desventaja', 'riesgo', 'amenaza', 'dificultad', 'obstáculo',
                'ineficiente', 'inútil', 'peligroso', 'preocupante', 'crítico', 'grave'
            }
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            
            if total_sentiment_words == 0:
                sentiment = 'neutral'
                confidence = 0.5
            else:
                sentiment_score = (positive_count - negative_count) / total_sentiment_words
                
                if sentiment_score > 0.1:
                    sentiment = 'positivo'
                elif sentiment_score < -0.1:
                    sentiment = 'negativo'
                else:
                    sentiment = 'neutral'
                
                confidence = min(abs(sentiment_score) + 0.5, 1.0)
            
            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 2),
                'positive_words': positive_count,
                'negative_words': negative_count,
                'sentiment_score': round((positive_count - negative_count) / len(words), 4) if words else 0
            }
            
        except Exception as e:
            return {'sentiment': 'unknown', 'error': str(e)}
    
    def find_topics(self, text: str, num_topics: int = 3) -> List[Dict[str, Any]]:
        """Encuentra temas principales usando clustering de palabras clave"""
        try:
            # Extraer palabras clave
            keywords = self.extract_keywords(text, max_keywords=50)
            
            if len(keywords) < num_topics:
                return []
            
            # Preparar datos para clustering
            keyword_texts = [kw[0] for kw in keywords]
            keyword_scores = [kw[1] for kw in keywords]
            
            # Vectorizar palabras clave
            vectorizer = TfidfVectorizer(
                stop_words=list(self.spanish_stopwords),
                ngram_range=(1, 2)
            )
            
            # Crear un texto combinado para el análisis
            combined_text = ' '.join(keyword_texts)
            
            # Clustering simple basado en co-ocurrencia
            topics = []
            keywords_per_topic = len(keywords) // num_topics
            
            for i in range(num_topics):
                start_idx = i * keywords_per_topic
                end_idx = start_idx + keywords_per_topic
                
                if i == num_topics - 1:  # Último tema incluye palabras restantes
                    end_idx = len(keywords)
                
                topic_keywords = keywords[start_idx:end_idx]
                topic_score = sum(score for _, score in topic_keywords)
                
                topics.append({
                    'topic_id': i + 1,
                    'keywords': [kw[0] for kw in topic_keywords],
                    'scores': [kw[1] for kw in topic_keywords],
                    'total_score': round(topic_score, 4),
                    'representative_terms': [kw[0] for kw in topic_keywords[:3]]
                })
            
            return topics
            
        except Exception as e:
            print(f"Error encontrando temas: {str(e)}")
            return []
    
    def create_word_cloud_data(self, text: str, max_words: int = 100) -> List[Dict[str, Any]]:
        """Prepara datos para crear una nube de palabras"""
        try:
            keywords = self.extract_keywords(text, max_keywords=max_words)
            
            # Normalizar puntuaciones para el tamaño de las palabras
            if keywords:
                max_score = max(score for _, score in keywords)
                min_score = min(score for _, score in keywords)
                score_range = max_score - min_score if max_score != min_score else 1
                
                word_cloud_data = []
                for word, score in keywords:
                    normalized_size = 10 + (score - min_score) / score_range * 40  # Tamaño entre 10 y 50
                    word_cloud_data.append({
                        'text': word,
                        'size': round(normalized_size, 1),
                        'frequency': score
                    })
                
                return word_cloud_data
            
        except Exception as e:
            print(f"Error creando datos de nube de palabras: {str(e)}")
        
        return []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divide texto en oraciones"""
        try:
            if self.nlp:
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            else:
                # Fallback usando regex
                sentences = re.split(r'[.!?]+\s+', text)
                return [s.strip() for s in sentences if len(s.strip()) > 10]
        except:
            # Fallback simple
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _clean_text_for_analysis(self, text: str) -> str:
        """Limpia texto para análisis"""
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def _translate_entity_type(self, entity_type: str) -> str:
        """Traduce tipos de entidades de spaCy al español"""
        translations = {
            'PERSON': 'Personas',
            'ORG': 'Organizaciones',
            'GPE': 'Lugares',
            'LOC': 'Ubicaciones',
            'DATE': 'Fechas',
            'TIME': 'Tiempo',
            'MONEY': 'Dinero',
            'PERCENT': 'Porcentajes',
            'MISC': 'Misceláneos',
            'PER': 'Personas',
            'LOCATION': 'Ubicaciones',
            'ORGANIZATION': 'Organizaciones'
        }
        return translations.get(entity_type, entity_type)
    
    def _extract_entities_regex(self, text: str) -> List[Dict[str, Any]]:
        """Extrae entidades usando patrones regex como fallback"""
        entities = []
        
        try:
            # Patrones para fechas
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
            dates = re.findall(date_pattern, text)
            if dates:
                entities.append({
                    'type': 'Fechas',
                    'count': len(dates),
                    'examples': list(set(dates))[:5]
                })
            
            # Patrones para emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                entities.append({
                    'type': 'Emails',
                    'count': len(emails),
                    'examples': list(set(emails))[:5]
                })
            
            # Patrones para números de teléfono
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = re.findall(phone_pattern, text)
            if phones:
                entities.append({
                    'type': 'Teléfonos',
                    'count': len(phones),
                    'examples': list(set(phones))[:5]
                })
            
            # Patrones para URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, text)
            if urls:
                entities.append({
                    'type': 'URLs',
                    'count': len(urls),
                    'examples': list(set(urls))[:5]
                })
            
        except Exception as e:
            print(f"Error en extracción regex: {str(e)}")
        
        return entities
    
    def _calculate_readability(self, text: str) -> float:
        """Calcula un índice de legibilidad aproximado"""
        try:
            sentences = self._split_into_sentences(text)
            words = text.split()
            
            if not sentences or not words:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Fórmula simplificada (menor es más fácil de leer)
            readability = (avg_sentence_length * 0.5) + (avg_word_length * 2)
            
            # Normalizar a escala 0-100 (100 = más fácil de leer)
            normalized_score = max(0, min(100, 100 - readability))
            
            return round(normalized_score, 2)
            
        except:
            return 50.0  # Valor neutral por defecto
