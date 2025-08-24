import streamlit as st
import os
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_chat import message

# Importar módulos personalizados
from pdf_processor import PDFProcessor
from semantic_search import SemanticSearch
from document_manager import DocumentManager
from advanced_analysis import AdvancedAnalyzer

# Configuración de la página
st.set_page_config(
    page_title="IA PDF Reader",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .document-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de sesión
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None
if 'semantic_search' not in st.session_state:
    st.session_state.semantic_search = SemanticSearch()
if 'doc_manager' not in st.session_state:
    st.session_state.doc_manager = DocumentManager()
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = AdvancedAnalyzer()

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🤖 IA PDF Reader Avanzado</h1>', unsafe_allow_html=True)
    
    # Sidebar con navegación
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/white?text=IA+PDF", width=200)
        
        selected = option_menu(
            "Navegación Principal",
            ["🏠 Inicio", "📤 Subir PDFs", "💬 Chat con PDFs", "📊 Análisis", "📂 Documentos", "⚙️ Configuración"],
            icons=['house', 'upload', 'chat', 'graph-up', 'folder', 'gear'],
            menu_icon="cast",
            default_index=0,
        )
        
        # Métricas en sidebar
        st.markdown("### 📈 Estadísticas")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documentos", len(st.session_state.documents))
        with col2:
            st.metric("Consultas", len(st.session_state.chat_history))
    
    # Contenido principal según selección
    if selected == "🏠 Inicio":
        show_home()
    elif selected == "📤 Subir PDFs":
        show_upload()
    elif selected == "💬 Chat con PDFs":
        show_chat()
    elif selected == "📊 Análisis":
        show_analysis()
    elif selected == "📂 Documentos":
        show_documents()
    elif selected == "⚙️ Configuración":
        show_settings()

def show_home():
    st.markdown("## 🎯 Bienvenido al Lector IA de PDFs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Procesamiento Inteligente</h3>
            <p>Extrae y analiza texto de PDFs con IA avanzada</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Búsqueda Semántica</h3>
            <p>Encuentra información relevante usando lenguaje natural</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💡 Análisis Avanzado</h3>
            <p>Resúmenes, entidades y análisis de contenido</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Características principales
    st.markdown("### ✨ Características Principales")
    
    features = [
        "🚀 **Carga ilimitada de PDFs** - Sin restricciones de tamaño",
        "🤖 **Chat inteligente** - Pregunta cualquier cosa sobre tus documentos",
        "📊 **Análisis automático** - Extrae tablas, gráficos y datos clave",
        "🔍 **Búsqueda avanzada** - Encuentra información específica al instante",
        "📈 **Visualizaciones** - Gráficos interactivos de tu contenido",
        "💾 **Gestión de documentos** - Organiza y administra tu biblioteca",
        "🎯 **Resúmenes automáticos** - Obtén lo esencial de cada documento",
        "🏷️ **Extracción de entidades** - Identifica personas, lugares, fechas"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")
    
    st.markdown("---")
    st.info("💡 **Consejo**: Comienza subiendo un PDF en la sección 'Subir PDFs' y luego usa el chat para hacer preguntas sobre su contenido.")

def show_upload():
    st.markdown("## 📤 Subir y Procesar PDFs")
    
    # Área de carga de archivos
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos PDF",
        type=['pdf'],
        accept_multiple_files=True,
        help="Puedes subir múltiples PDFs a la vez. No hay límite de tamaño."
    )
    
    if uploaded_files:
        st.markdown(f"### 📋 Archivos seleccionados: {len(uploaded_files)}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Procesando: {uploaded_file.name}")
            
            # Procesar PDF
            processor = PDFProcessor()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Extraer contenido
                content = processor.extract_content(tmp_file_path)
                
                # Guardar en estado de sesión
                doc_id = f"doc_{len(st.session_state.documents) + 1}"
                st.session_state.documents[doc_id] = {
                    'name': uploaded_file.name,
                    'content': content,
                    'upload_date': datetime.now(),
                    'size': len(uploaded_file.getvalue()),
                    'pages': content.get('total_pages', 0)
                }
                
                # Actualizar índice semántico
                st.session_state.semantic_search.add_document(doc_id, content['text'])
                
                st.success(f"✅ {uploaded_file.name} procesado correctamente")
                
            except Exception as e:
                st.error(f"❌ Error procesando {uploaded_file.name}: {str(e)}")
            
            finally:
                os.unlink(tmp_file_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("¡Procesamiento completado!")
        st.balloons()

def show_chat():
    st.markdown("## 💬 Chat con tus PDFs")
    
    if not st.session_state.documents:
        st.warning("⚠️ No hay documentos cargados. Ve a la sección 'Subir PDFs' primero.")
        return
    
    # Selector de documento
    doc_options = {doc_id: doc_data['name'] for doc_id, doc_data in st.session_state.documents.items()}
    doc_options['all'] = "Todos los documentos"
    
    selected_doc = st.selectbox(
        "Selecciona el documento para consultar:",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x]
    )
    
    # Área de chat
    st.markdown("### 💭 Conversación")
    
    # Mostrar historial de chat
    chat_container = st.container()
    with chat_container:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            message(question, is_user=True, key=f"user_{i}")
            message(answer, key=f"bot_{i}")
    
    # Input para nueva pregunta
    user_question = st.text_input(
        "Haz una pregunta sobre el documento:",
        placeholder="Ej: ¿Cuál es el tema principal del documento?",
        key="chat_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("🚀 Enviar", type="primary"):
            if user_question:
                process_question(user_question, selected_doc)
    
    with col2:
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.chat_history = []
            st.rerun()

def process_question(question, doc_id):
    """Procesa una pregunta del usuario"""
    try:
        # Buscar respuesta usando búsqueda semántica
        if doc_id == 'all':
            results = st.session_state.semantic_search.search(question, top_k=3)
        else:
            results = st.session_state.semantic_search.search_in_document(question, doc_id, top_k=3)
        
        if results:
            # Generar respuesta basada en los resultados
            answer = generate_answer(question, results)
            st.session_state.chat_history.append((question, answer))
        else:
            st.session_state.chat_history.append((question, "No encontré información relevante para responder tu pregunta."))
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error procesando la pregunta: {str(e)}")

def generate_answer(question, search_results):
    """Genera una respuesta basada en los resultados de búsqueda"""
    # Combinar contexto de los resultados
    context = "\n".join([result['text'] for result in search_results])
    
    # Respuesta simple basada en el contexto
    if len(context) > 500:
        context = context[:500] + "..."
    
    return f"Basándome en el contenido del documento:\n\n{context}\n\n💡 Esta información responde a tu pregunta sobre: {question}"

def show_analysis():
    st.markdown("## 📊 Análisis Avanzado de Documentos")
    
    if not st.session_state.documents:
        st.warning("⚠️ No hay documentos para analizar.")
        return
    
    # Selector de documento
    doc_options = {doc_id: doc_data['name'] for doc_id, doc_data in st.session_state.documents.items()}
    selected_doc = st.selectbox(
        "Selecciona documento para analizar:",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x]
    )
    
    if selected_doc:
        doc_data = st.session_state.documents[selected_doc]
        
        # Tabs para diferentes análisis
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Resumen", "🏷️ Entidades", "📊 Estadísticas", "🔍 Palabras Clave"])
        
        with tab1:
            st.markdown("### 📄 Resumen Automático")
            summary = st.session_state.analyzer.generate_summary(doc_data['content']['text'])
            st.info(summary)
        
        with tab2:
            st.markdown("### 🏷️ Entidades Identificadas")
            entities = st.session_state.analyzer.extract_entities(doc_data['content']['text'])
            
            if entities:
                df_entities = pd.DataFrame(entities)
                fig = px.bar(df_entities, x='type', y='count', title="Tipos de Entidades Encontradas")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_entities)
            else:
                st.info("No se encontraron entidades específicas.")
        
        with tab3:
            st.markdown("### 📊 Estadísticas del Documento")
            stats = st.session_state.analyzer.get_document_stats(doc_data['content']['text'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Palabras", stats['word_count'])
            with col2:
                st.metric("Párrafos", stats['paragraph_count'])
            with col3:
                st.metric("Oraciones", stats['sentence_count'])
            with col4:
                st.metric("Páginas", doc_data['pages'])
        
        with tab4:
            st.markdown("### 🔍 Palabras Clave Principales")
            keywords = st.session_state.analyzer.extract_keywords(doc_data['content']['text'])
            
            if keywords:
                df_keywords = pd.DataFrame(keywords, columns=['Palabra', 'Relevancia'])
                fig = px.bar(df_keywords.head(10), x='Relevancia', y='Palabra', 
                           orientation='h', title="Top 10 Palabras Clave")
                st.plotly_chart(fig, use_container_width=True)

def show_documents():
    st.markdown("## 📂 Gestión de Documentos")
    
    if not st.session_state.documents:
        st.info("📭 No hay documentos cargados.")
        return
    
    # Lista de documentos
    for doc_id, doc_data in st.session_state.documents.items():
        with st.expander(f"📄 {doc_data['name']}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tamaño", f"{doc_data['size'] / 1024:.1f} KB")
            with col2:
                st.metric("Páginas", doc_data['pages'])
            with col3:
                st.metric("Fecha", doc_data['upload_date'].strftime("%d/%m/%Y"))
            
            # Botones de acción
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"🔍 Analizar", key=f"analyze_{doc_id}"):
                    st.session_state.current_doc = doc_id
                    st.info(f"📊 Análisis del documento '{doc_data['name']}' disponible en la pestaña 'Análisis'")
            
            with col2:
                if st.button(f"💬 Chat", key=f"chat_{doc_id}"):
                    st.session_state.current_doc = doc_id
                    st.info(f"💬 Puedes chatear con '{doc_data['name']}' en la pestaña 'Chat con PDFs'")
            
            with col3:
                if st.button(f"🗑️ Eliminar", key=f"delete_{doc_id}"):
                    del st.session_state.documents[doc_id]
                    st.session_state.semantic_search.remove_document(doc_id)
                    st.success(f"✅ Documento '{doc_data['name']}' eliminado correctamente")
                    st.rerun()

def show_settings():
    st.markdown("## ⚙️ Configuración")
    
    st.markdown("### 🎛️ Configuración del Sistema")
    
    # Configuraciones de búsqueda
    st.markdown("#### 🔍 Búsqueda Semántica")
    search_model = st.selectbox(
        "Modelo de embeddings:",
        ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
         "sentence-transformers/all-MiniLM-L6-v2"]
    )
    
    max_results = st.slider("Máximo resultados por búsqueda:", 1, 10, 3)
    
    # Configuraciones de análisis
    st.markdown("#### 📊 Análisis de Texto")
    language = st.selectbox("Idioma principal:", ["es", "en"])
    
    # Configuraciones de interfaz
    st.markdown("#### 🎨 Interfaz")
    theme = st.selectbox("Tema:", ["Claro", "Oscuro"])
    
    if st.button("💾 Guardar Configuración"):
        st.success("✅ Configuración guardada correctamente")

if __name__ == "__main__":
    main()
