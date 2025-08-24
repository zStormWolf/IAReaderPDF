# 🤖 IA PDF Reader Avanzado

## Descripción
Aplicación web avanzada construida con Streamlit que permite subir documentos PDF de cualquier tamaño y hacer preguntas inteligentes sobre su contenido usando IA y procesamiento de lenguaje natural.

## ✨ Características Principales

### 🚀 Procesamiento Avanzado de PDFs
- **Carga ilimitada**: Sin restricciones de tamaño de archivo
- **Extracción múltiple**: Texto, tablas, imágenes y metadatos
- **Procesamiento inteligente**: Limpieza automática y estructuración del contenido

### 🤖 Chat Inteligente
- **Búsqueda semántica**: Encuentra información relevante usando lenguaje natural
- **Respuestas contextuales**: Basadas en el contenido específico de tus documentos
- **Múltiples documentos**: Consulta uno o todos los documentos a la vez

### 📊 Análisis Avanzado
- **Resúmenes automáticos**: Extrae lo más importante de cada documento
- **Extracción de entidades**: Identifica personas, lugares, fechas, organizaciones
- **Palabras clave**: Encuentra los términos más relevantes
- **Estadísticas detalladas**: Análisis completo del contenido
- **Análisis de sentimiento**: Detecta el tono del documento
- **Detección de temas**: Identifica los temas principales

### 📂 Gestión de Documentos
- **Organización inteligente**: Categorías, tags y metadatos
- **Búsqueda avanzada**: Encuentra documentos por contenido o metadatos
- **Detección de duplicados**: Evita almacenar archivos repetidos
- **Estadísticas completas**: Métricas de uso y contenido

### 🎨 Interfaz Moderna
- **Diseño responsive**: Funciona en cualquier dispositivo
- **Navegación intuitiva**: Menú lateral con todas las funciones
- **Visualizaciones interactivas**: Gráficos y métricas en tiempo real
- **Chat en tiempo real**: Interfaz de conversación fluida

## 🛠️ Tecnologías Utilizadas

- **Frontend**: Streamlit, Plotly, Streamlit-Chat
- **Procesamiento PDF**: PyPDF2, pdfplumber
- **IA y NLP**: Sentence Transformers, spaCy, NLTK
- **Búsqueda**: FAISS (Facebook AI Similarity Search)
- **Análisis**: Scikit-learn, Pandas, NumPy
- **Visualización**: Plotly Express, Matplotlib

## 📋 Requisitos del Sistema

- Python 3.8 o superior
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio libre en disco
- Conexión a internet (para descargar modelos la primera vez)

## 🚀 Instalación y Uso

Sigue estos pasos para ejecutar la aplicación en tu máquina local:

### 1. Clona el Repositorio
```bash
git clone https://github.com/tu-usuario/ia-pdf-reader.git
cd ia-pdf-reader
```

### 2. Crea un Entorno Virtual (Recomendado)
```bash
python -m venv venv
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
```

### 3. Instala las Dependencias
```bash
pip install -r requirements.txt
```
**Nota**: La primera vez que ejecutes la aplicación, se descargarán automáticamente los modelos de lenguaje necesarios. Esto puede tardar unos minutos.

## 🎯 Uso de la Aplicación

### 1. Iniciar la aplicación
```bash
streamlit run app.py
```

### 2. Acceder desde el navegador
La aplicación se abrirá automáticamente en `http://localhost:8501`

### 3. Subir documentos
1. Ve a la sección "📤 Subir PDFs"
2. Selecciona uno o más archivos PDF
3. Espera a que se procesen automáticamente

### 4. Hacer preguntas
1. Ve a la sección "💬 Chat con PDFs"
2. Selecciona el documento o "Todos los documentos"
3. Escribe tu pregunta en lenguaje natural
4. Obtén respuestas basadas en el contenido

### 5. Analizar contenido
1. Ve a la sección "📊 Análisis"
2. Selecciona un documento
3. Explora resúmenes, entidades, palabras clave y estadísticas

## 💡 Ejemplos de Preguntas

### Preguntas Generales
- "¿De qué trata este documento?"
- "¿Cuáles son los puntos principales?"
- "¿Hay alguna conclusión importante?"

### Preguntas Específicas
- "¿Qué dice sobre [tema específico]?"
- "¿Cuándo ocurrió [evento]?"
- "¿Quién es [persona mencionada]?"
- "¿Cuáles son los datos financieros?"

### Preguntas de Análisis
- "¿Cuáles son las ventajas y desventajas mencionadas?"
- "¿Hay algún riesgo identificado?"
- "¿Qué recomendaciones se hacen?"

## 📁 Estructura del Proyecto

```
IAPDF/
├── app.py                 # Aplicación principal de Streamlit
├── pdf_processor.py       # Procesamiento de archivos PDF
├── semantic_search.py     # Sistema de búsqueda semántica
├── document_manager.py    # Gestión de documentos
├── advanced_analysis.py   # Análisis avanzado con NLP
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
└── documents/            # Carpeta de almacenamiento (se crea automáticamente)
    ├── metadata.json     # Metadatos de documentos
    └── [archivos PDF]    # Documentos almacenados
```

## ⚙️ Configuración Avanzada

### Modelos de Embeddings
Puedes cambiar el modelo de embeddings en la configuración:
- `paraphrase-multilingual-MiniLM-L12-v2` (por defecto, multiidioma)
- `all-MiniLM-L6-v2` (inglés, más rápido)

### Parámetros de Búsqueda
- **top_k**: Número de resultados por búsqueda (1-10)
- **chunk_size**: Tamaño de fragmentos de texto (300-1000)
- **overlap**: Solapamiento entre fragmentos (50-200)

### Análisis de Texto
- **max_keywords**: Máximo palabras clave a extraer (10-50)
- **summary_sentences**: Oraciones en resúmenes (2-5)

## 🔧 Solución de Problemas

### Error: Modelo spaCy no encontrado
La aplicación intentará descargar los modelos de spaCy automáticamente. Si esto falla (por ejemplo, por problemas de permisos o de red), puedes instalarlos manualmente en tu terminal:
```bash
# Modelo en español (principal)
python -m spacy download es_core_news_sm

# Modelo en inglés (secundario)
python -m spacy download en_core_web_sm
```

### Error: Memoria insuficiente
- Reduce el tamaño de chunk_size en la configuración
- Procesa documentos más pequeños
- Aumenta la RAM del sistema

### Error: FAISS no funciona
```bash
pip install faiss-cpu --force-reinstall
```

### Documentos no se procesan
- Verifica que el PDF no esté protegido con contraseña
- Asegúrate de que el archivo no esté corrupto
- Revisa los logs en la consola

## 📈 Rendimiento

### Tiempos de Procesamiento (aproximados)
- **Documento pequeño** (1-10 páginas): 5-15 segundos
- **Documento mediano** (10-50 páginas): 30-60 segundos
- **Documento grande** (50+ páginas): 1-5 minutos

### Optimización
- Los modelos se cargan una sola vez al inicio
- Los embeddings se almacenan en caché
- La búsqueda es casi instantánea después del procesamiento

## 🤝 Contribuciones

Este proyecto está abierto a mejoras. Algunas ideas:

### Funcionalidades Futuras
- [ ] Soporte para más formatos (Word, Excel, PowerPoint)
- [ ] Integración con APIs de IA (OpenAI, Claude)
- [ ] Exportación de análisis a PDF/Word
- [ ] Comparación entre documentos
- [ ] Anotaciones y comentarios
- [ ] Colaboración en tiempo real
- [ ] API REST para integración
- [ ] Versión móvil nativa

### Mejoras Técnicas
- [ ] Base de datos vectorial más robusta
- [ ] Caché distribuido
- [ ] Procesamiento en paralelo
- [ ] Compresión de embeddings
- [ ] Monitoreo y métricas

## 📄 Licencia

Este proyecto es de código abierto. Puedes usarlo, modificarlo y distribuirlo libremente.

## 📞 Soporte

Si encuentras algún problema o tienes sugerencias:

1. **Revisa la documentación** en este README
2. **Verifica los logs** en la consola de Streamlit
3. **Prueba con documentos más pequeños** para aislar el problema
4. **Reinicia la aplicación** si hay problemas de memoria

## 🎉 ¡Disfruta usando IA PDF Reader!

Esta aplicación te permitirá:
- ✅ Procesar documentos PDF de cualquier tamaño
- ✅ Hacer preguntas inteligentes sobre el contenido
- ✅ Obtener resúmenes y análisis automáticos
- ✅ Organizar y gestionar tu biblioteca de documentos
- ✅ Extraer insights valiosos de tus archivos

**¡Comienza subiendo tu primer PDF y descubre todo lo que puede hacer por ti!**
