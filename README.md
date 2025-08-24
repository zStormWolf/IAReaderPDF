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
- **Generación de respuestas**: Ollama + Mixtral 8x7B (configurable)

## 📋 Requisitos del Sistema

- Python 3.8 o superior
- RAM: 8GB mínimo (16–32GB recomendado para Mixtral 8x7B)
- Disco: 5GB mínimo (≈26GB adicionales si usas Mixtral 8x7B)
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
**Nota**: La generación de respuestas usa Ollama. Debes instalar Ollama y descargar el modelo que vayas a utilizar.

### 4. Instala Ollama y el Modelo
- Instala Ollama: https://ollama.ai
- Inicia el servidor (si no está ya corriendo):
```bash
ollama serve
```
- Descarga el modelo recomendado (Mixtral 8x7B):
```bash
ollama pull mixtral:8x7b
```
> Alternativas más ligeras: `mistral:latest`, `llama3:8b`

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

### 6. Configurar la Generación (Ollama)
En la pestaña "⚙️ Configuración" → "🧠 Generación (Ollama)" puedes ajustar:
- **Modelo**: por ejemplo `mixtral:8x7b` (requiere `ollama pull mixtral:8x7b`), o `mistral:latest`, `llama3:8b`.
- **Temperature**: 0.0–1.0 (recomendado 0.2–0.3 para precisión).
- **Timeout (segundos)**: aumenta si el modelo tarda en responder (p. ej., 300–600).
- **num_predict**: tokens de salida generados; reducirlo baja el uso de RAM/CPU (p. ej., 100–300).

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

### Generación (Ollama)
- **Modelo**: `mixtral:8x7b` (recomendado) u otros compatibles con Ollama.
- **Temperature**: controla creatividad vs precisión.
- **Timeout**: límite de espera de la petición a Ollama.
- **num_predict**: número de tokens de salida. Valores bajos reducen uso de memoria y latencia.

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
- Reduce `num_predict` en la Configuración (Generación).
- Usa un modelo más ligero (`mistral:latest`, `llama3:8b`).
- Sube gradualmente el `timeout` para evitar reintentos costosos.
- Reduce `chunk_size` si procesas documentos muy grandes.
- Cierra procesos que compitan por RAM.

### Error: 404 al generar con Ollama (modelo no encontrado)
- Verifica el nombre del modelo en la Configuración (exacto a `ollama list`).
- Descárgalo con `ollama pull <modelo>`.
- Si persiste, reinicia el servidor de Ollama:
  ```bash
  # Windows PowerShell
  Get-Process *ollama* | Stop-Process -Force
  ollama serve
  ```

### Error: ⏱️ Timeout al generar
- Aumenta el `Timeout (segundos)` en Configuración.
- Reduce `num_predict` (menos tokens a generar).
- Precalienta el modelo una vez:
  ```bash
  ollama run mixtral:8x7b "ok"
  ```

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
