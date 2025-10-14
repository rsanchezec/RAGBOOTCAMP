# 🚀 RAG Bootcamp

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Un curso completo sobre Retrieval-Augmented Generation (RAG) usando LangChain**

[Características](#-características-principales) •
[Instalación](#-instalación) •
[Vector Stores](#-bases-de-datos-vectoriales-vector-stores) •
[Búsqueda Híbrida](#-estrategias-de-búsqueda-híbrida) •
[Query Enhancement](#-mejora-de-consultas-query-enhancement) •
[RAG Multimodal](#️-rag-multimodal) •
[LangGraph](#-fundamentos-de-langgraph) •
[Agentes](#-arquitectura-de-agentes) •
[Debugging](#-debugging-con-langgraph-studio) •
[RAG Agéntico](#-rag-agéntico) •
[RAG Autónomo](#-rag-autónomo) •
[Multi-Agente](#-sistemas-rag-multi-agente) •
[RAG Correctivo](#-rag-correctivo-corrective-rag) •
[RAG Adaptativo](#-rag-adaptativo-adaptive-rag) •
[RAG Memory](#-rag-con-memoria-persistente) •
[Cache RAG](#️-cache-augmented-generation-cag) •
[Estructura](#-estructura-del-proyecto) •
[Uso](#-guía-de-uso)

</div>

---

## 📋 Descripción

RAG Bootcamp es un proyecto educativo diseñado para enseñar los fundamentos de la construcción de sistemas RAG (Retrieval-Augmented Generation) utilizando LangChain. El repositorio está estructurado como una ruta de aprendizaje progresiva con Jupyter notebooks que cubren desde la ingesta de datos hasta la implementación de bases de datos vectoriales.

Este proyecto es ideal para:
- 🎓 Estudiantes que quieren aprender sobre RAG y embeddings
- 💻 Desarrolladores que construyen aplicaciones de IA generativa
- 🔬 Investigadores explorando técnicas de recuperación de información
- 🏢 Profesionales implementando soluciones empresariales con LLMs

## ✨ Características Principales

- 📚 **Ingesta de Datos Completa**: Manejo de múltiples formatos (PDF, DOCX, CSV, JSON, bases de datos)
- 🧩 **Estrategias de Chunking**: Diferentes técnicas de división de texto optimizadas
- 🎯 **Advanced Chunking**: Técnicas avanzadas de semantic chunking para mejor recuperación
- 🔢 **Múltiples Embeddings**: Soporte para OpenAI, HuggingFace y sentence-transformers
- 💾 **5 Bases de Datos Vectoriales**: ChromaDB, FAISS, InMemory, AstraDB y Pinecone
- 🔍 **Búsqueda Híbrida Avanzada**: Dense+Sparse, Reranking y MMR para recuperación óptima
- 🎯 **Mejora de Consultas**: Query Expansion, Query Decomposition y HyDE para optimizar búsquedas
- 🖼️ **RAG Multimodal**: Procesamiento de PDFs con imágenes usando CLIP y GPT-4 Vision
- 🎯 **Filtrado de Metadatos**: Búsquedas precisas con filtros personalizados
- 🔄 **LangGraph Basics**: Construcción de grafos de estado con LangGraph para chatbots y agentes
- 🤖 **Arquitectura de Agentes**: Implementación de agentes ReAct con herramientas y memoria
- 🐛 **Debugging con LangGraph Studio**: Configuración y depuración de agentes con LangGraph Studio
- 🎯 **RAG Agéntico**: Sistema RAG con capacidades de razonamiento, evaluación y auto-corrección
- 🧠 **RAG Autónomo**: Chain-of-Thought, Auto-reflexión, Descomposición de Consultas, Recuperación Iterativa y Síntesis Multi-Fuente
- 🤖 **Sistemas RAG Multi-Agente**: Arquitecturas colaborativas, con supervisor y jerárquicas con equipos especializados
- 🔧 **RAG Correctivo (CRAG)**: Evaluación automática de relevancia, reescritura de consultas y búsqueda web adaptativa
- 🎯 **RAG Adaptativo (Adaptive RAG)**: Enrutamiento inteligente, detección de alucinaciones y auto-corrección con ciclos de retroalimentación
- 💾 **RAG con Memoria Persistente**: Implementación de memoria conversacional con LangGraph y MemorySaver para mantener contexto entre interacciones
- ⚡ **Cache-Augmented Generation (CAG)**: Sistema de caché semántico avanzado con FAISS para reutilizar respuestas y optimizar costos
- 📊 **Ejemplos Prácticos**: Notebooks interactivos con casos de uso reales
- 🌍 **Documentación en Español**: Código y comentarios completamente traducidos

## 🛠️ Instalación

### Requisitos Previos

- Python 3.12
- pip (gestor de paquetes de Python)
- Git

### Paso 1: Clonar el Repositorio

```bash
git clone <url-del-repositorio>
cd RAGBootcamp
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv .venv

# Activar entorno virtual
# En Windows:
.venv\Scripts\activate

# En macOS/Linux:
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Configurar Variables de Entorno

Copia el archivo `.env.example` y renómbralo a `.env`, luego completa tus claves API:

```bash
# Copiar el template
cp .env.example .env

# Editar el archivo .env con tus claves API
# En Windows puedes usar: notepad .env
# En macOS/Linux: nano .env
```

Tu archivo `.env` debe contener:

```env
OPENAI_API_KEY=tu_clave_openai_aqui
GROQ_API_KEY=tu_clave_groq_aqui
LANGSMITH_API_KEY=tu_clave_langsmith_aqui  # Opcional
```

**Obtener las claves API:**
- 🔑 **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- 🔑 **Groq**: [https://console.groq.com/keys](https://console.groq.com/keys)
- 🔑 **LangSmith** (opcional): [https://smith.langchain.com/settings](https://smith.langchain.com/settings)

### Paso 5: Iniciar Jupyter

```bash
jupyter notebook
```

## 💾 Bases de Datos Vectoriales (Vector Stores)

Los vector stores son componentes fundamentales en sistemas RAG que permiten almacenar y buscar embeddings de manera eficiente. Este proyecto incluye implementaciones de **5 bases de datos vectoriales diferentes**, cada una con características y casos de uso específicos.

### 📊 Comparativa de Vector Stores

| Vector Store | Tipo | Persistencia | Escalabilidad | Latencia | Costo | Uso Ideal |
|--------------|------|--------------|---------------|----------|-------|-----------|
| **InMemoryVectorStore** | Local | ❌ No | Baja | Ultra-baja | Gratis | Prototipos rápidos, demos |
| **FAISS** | Local | ⚠️ Manual | Media | Muy baja | Gratis | Aplicaciones locales, alta velocidad |
| **ChromaDB** | Local/Híbrido | ✅ Sí | Media | Baja | Gratis | Desarrollo, proyectos pequeños |
| **AstraDB** | Cloud | ✅ Sí | Muy alta | Baja | 💰 Pago | Producción, aplicaciones distribuidas |
| **Pinecone** | Cloud | ✅ Sí | Muy alta | Muy baja | 💰 Pago | Producción a gran escala |

### 🔍 Detalles por Vector Store

#### 1. **InMemoryVectorStore**

**Características:**
- Almacenamiento completamente en memoria RAM
- Utiliza diccionarios de Python y NumPy para cálculos
- Métrica de similitud: coseno

**Ventajas:**
- ⚡ Configuración instantánea (sin instalación adicional)
- 🚀 Velocidad ultra-rápida para conjuntos pequeños
- 💻 No requiere infraestructura externa

**Desventajas:**
- ❌ Los datos se pierden al cerrar la aplicación
- ❌ Limitado por la memoria RAM disponible
- ❌ No escalable para producción

**Cuándo usar:**
```python
✅ Prototipos y pruebas rápidas
✅ Demos y presentaciones
✅ Datasets pequeños (<1000 documentos)
✅ Aplicaciones de un solo uso
❌ Producción
❌ Datos que necesitan persistir
❌ Aplicaciones multiusuario
```

**Ejemplo de uso:**
```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = InMemoryVectorStore(embedding=OpenAIEmbeddings())
vector_store.add_documents(documents)
results = vector_store.similarity_search("consulta", k=5)
```

---

#### 2. **FAISS (Facebook AI Similarity Search)**

**Características:**
- Biblioteca de Facebook para búsqueda de similitud eficiente
- Múltiples algoritmos de indexación (Flat, IVF, HNSW)
- Optimizado para CPU y GPU

**Ventajas:**
- ⚡ Extremadamente rápido para millones de vectores
- 🔧 Altamente configurable y optimizable
- 💾 Puede guardarse y cargarse desde disco
- 🆓 Completamente gratuito y open-source

**Desventajas:**
- ⚠️ Requiere código manual para persistencia
- ❌ No incluye gestión de metadatos por defecto
- 🔧 Curva de aprendizaje para optimización avanzada

**Cuándo usar:**
```python
✅ Aplicaciones locales con alto rendimiento
✅ Datasets medianos a grandes (10K-10M vectores)
✅ Cuando necesitas control total sobre la indexación
✅ Aplicaciones que requieren baja latencia
❌ Cuando necesitas sincronización multi-dispositivo
❌ Si prefieres una solución managed
❌ Aplicaciones web sin servidor local
```

**Ejemplo de uso:**
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vector_store = FAISS.from_documents(documents, OpenAIEmbeddings())

# Guardar índice
vector_store.save_local("faiss_index")

# Cargar índice
vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings())
```

---

#### 3. **ChromaDB**

**Características:**
- Base de datos vectorial open-source diseñada para IA
- Persistencia automática en SQLite
- Soporte nativo para metadatos y filtros

**Ventajas:**
- 📦 Fácil de configurar (instalación con pip)
- 💾 Persistencia automática sin configuración
- 🏷️ Excelente manejo de metadatos y filtros
- 🔄 Soporte para actualizaciones y eliminaciones
- 🆓 Completamente gratuito

**Desventajas:**
- ⚠️ Rendimiento limitado para datasets muy grandes (>1M)
- ❌ No optimizado para entornos distribuidos
- ⚠️ Puede tener problemas de concurrencia

**Cuándo usar:**
```python
✅ Desarrollo y prototipado con persistencia
✅ Proyectos personales y pequeños
✅ Aplicaciones que requieren filtrado complejo
✅ Cuando necesitas actualizar/eliminar documentos
✅ MVPs y proyectos de tamaño mediano
❌ Producción a gran escala (>1M documentos)
❌ Aplicaciones distribuidas geográficamente
❌ Sistemas con alta concurrencia
```

**Ejemplo de uso:**
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vector_store = Chroma.from_documents(
    documents,
    OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Búsqueda con filtros
results = vector_store.similarity_search(
    "consulta",
    k=5,
    filter={"source": "news"}
)
```

---

#### 4. **AstraDB (DataStax)**

**Características:**
- Base de datos vectorial serverless en la nube
- Basada en Apache Cassandra
- Multi-región y multi-nube (AWS, GCP, Azure)

**Ventajas:**
- ☁️ Totalmente administrada (managed service)
- 🌍 Distribución global y baja latencia
- 📈 Escalabilidad automática
- 🔒 Seguridad y backup integrados
- 🔄 Alta disponibilidad (99.99% SLA)
- 🆓 Free tier generoso para empezar

**Desventajas:**
- 💰 Costo incrementa con el uso
- 🌐 Requiere conexión a internet
- ⚠️ Latencia de red vs. soluciones locales

**Cuándo usar:**
```python
✅ Aplicaciones en producción
✅ Sistemas distribuidos y multi-región
✅ Necesitas alta disponibilidad
✅ Equipos sin experiencia en DevOps
✅ Aplicaciones serverless (Lambda, Cloud Functions)
✅ Escalamiento automático requerido
❌ Presupuesto muy limitado
❌ Requisitos de soberanía de datos estrictos
❌ Aplicaciones totalmente offline
```

**Ejemplo de uso:**
```python
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = AstraDBVectorStore(
    embedding=OpenAIEmbeddings(),
    api_endpoint="https://...",
    token="AstraCS:...",
    collection_name="mi_coleccion"
)

# Búsqueda con score threshold
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.7}
)
```

---

#### 5. **Pinecone**

**Características:**
- Base de datos vectorial cloud-native líder del mercado
- Optimizada específicamente para búsqueda vectorial
- API simple y potente

**Ventajas:**
- 🚀 Rendimiento excepcional a cualquier escala
- ⚡ Latencia ultra-baja (p95 < 100ms)
- 🔧 Fácil de integrar y usar
- 📊 Dashboards y métricas avanzadas
- 🌐 Infraestructura global
- 🎯 Especializada en ML/AI workloads

**Desventajas:**
- 💰 Más costoso que alternativas
- 🌐 Requiere conexión a internet siempre
- ⚠️ Vendor lock-in

**Cuándo usar:**
```python
✅ Producción a gran escala (>10M vectores)
✅ Aplicaciones que requieren latencia mínima
✅ Sistemas de recomendación en tiempo real
✅ Búsqueda semántica de alto rendimiento
✅ Startups con inversión que priorizan velocidad
✅ Empresas que necesitan soporte enterprise
❌ Proyectos con presupuesto limitado
❌ Aplicaciones que necesitan funcionar offline
❌ Requisitos de hosting on-premise
```

**Ejemplo de uso:**
```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")
index = pc.Index("mi-indice")

vector_store = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings()
)

# Búsqueda MMR (diversidad)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)
```

---

### 🎯 Guía de Selección Rápida

#### Por Caso de Uso:

**🧪 Experimentación y Aprendizaje**
→ `InMemoryVectorStore` o `ChromaDB`

**💻 Aplicación Local de Alto Rendimiento**
→ `FAISS`

**🏗️ Desarrollo de MVP**
→ `ChromaDB`

**🚀 Startup/Producción (presupuesto medio)**
→ `AstraDB`

**🏢 Empresa/Producción (alto rendimiento)**
→ `Pinecone`

#### Por Tamaño de Dataset:

- **< 1,000 documentos**: InMemoryVectorStore
- **1K - 100K documentos**: ChromaDB o FAISS
- **100K - 1M documentos**: FAISS o ChromaDB
- **1M - 10M documentos**: AstraDB o Pinecone
- **> 10M documentos**: Pinecone

#### Por Presupuesto:

- **$0/mes**: InMemoryVectorStore, FAISS, ChromaDB
- **$0-50/mes**: AstraDB (free tier + paid)
- **$70+/mes**: Pinecone (standard tier)

## 🔍 Estrategias de Búsqueda Híbrida

Las estrategias de búsqueda híbrida son técnicas avanzadas que mejoran significativamente la calidad de la recuperación de documentos en sistemas RAG. El módulo 004 cubre tres estrategias fundamentales:

### 📊 Comparativa de Estrategias

| Estrategia | Tipo | Complejidad | Latencia | Mejora en Precisión | Mejora en Diversidad | Uso Ideal |
|------------|------|-------------|----------|---------------------|---------------------|-----------|
| **Dense + Sparse** | Híbrida | Media | Media | ⭐⭐⭐⭐ | ⭐⭐⭐ | Búsquedas que requieren tanto coincidencia exacta como semántica |
| **Reranking** | Post-procesamiento | Alta | Alta | ⭐⭐⭐⭐⭐ | ⭐⭐ | Cuando la precisión es crítica y hay presupuesto de latencia |
| **MMR** | Diversificación | Baja | Baja | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Evitar redundancia y cubrir múltiples aspectos de una consulta |

### 🔹 1. Búsqueda Híbrida (Dense + Sparse)

**¿Qué es?**
Combina dos enfoques complementarios de recuperación:
- **Recuperación Densa (FAISS)**: Usa embeddings vectoriales para capturar similitud semántica
- **Recuperación Dispersa (BM25)**: Usa coincidencia de palabras clave para términos exactos

**Ventajas:**
- ✅ Captura tanto significado semántico como coincidencias exactas
- ✅ Mejor rendimiento en consultas con términos específicos
- ✅ Reduce falsos negativos de cada método individual

**Cuándo usar:**
```python
✅ Búsquedas técnicas con terminología específica
✅ Consultas que mezclan conceptos y nombres propios
✅ Cuando necesitas balance entre precisión y cobertura
❌ Consultas muy simples donde un método bastaría
```

**Ejemplo:**
```python
from langchain.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]  # 70% semántico, 30% palabras clave
)
```

---

### 🔹 2. Reranking (Reordenamiento)

**¿Qué es?**
Proceso de dos etapas donde primero se recuperan documentos rápidamente y luego se reordenan con un modelo más preciso:
1. **Primera etapa**: Recuperador rápido obtiene top-k documentos (ej: k=20)
2. **Segunda etapa**: LLM o cross-encoder re-puntúa y reordena por relevancia real

**Ventajas:**
- ✅ Mejora significativa en precisión sin perder velocidad inicial
- ✅ Los documentos más relevantes aparecen primero
- ✅ Reduce documentos poco relevantes en el contexto del LLM
- ✅ Mejora calidad de respuestas finales

**Cuándo usar:**
```python
✅ Aplicaciones donde la precisión es crítica
✅ Cuando tienes presupuesto de latencia (añade ~500ms-2s)
✅ Datasets grandes con mucho ruido
✅ Consultas complejas con múltiples intenciones
❌ Aplicaciones de latencia ultra-baja
❌ Datasets pequeños y bien curados
```

**Ejemplo:**
```python
# 1. Recuperar top-8 documentos
retrieved_docs = retriever.invoke(query)

# 2. Reordenar con LLM
reranking_prompt = """Clasifica estos documentos por relevancia..."""
reranked_docs = llm_rerank(retrieved_docs, query)
```

---

### 🔹 3. MMR (Maximal Marginal Relevance)

**¿Qué es?**
Técnica que balancea dos objetivos al seleccionar documentos:
- **Relevancia**: Qué tan relacionado está con la consulta
- **Diversidad**: Qué tan diferente es de documentos ya seleccionados

**Ventajas:**
- ✅ Evita información repetitiva y redundante
- ✅ Cubre múltiples aspectos de una consulta
- ✅ Perspectiva más amplia del tema
- ✅ Mejora respuestas para preguntas multifacéticas

**Cuándo usar:**
```python
✅ Consultas amplias con múltiples sub-temas
✅ Cuando quieres cobertura completa de un tema
✅ Evitar documentos muy similares entre sí
✅ Exploratory search (búsqueda exploratoria)
❌ Consultas muy específicas de una sola respuesta
❌ Cuando necesitas documentos altamente enfocados
```

**Ejemplo:**
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20  # Primero obtiene 20, luego MMR selecciona 5 diversos
    }
)
```

---

### 🎯 Guía de Selección Rápida

**Por Caso de Uso:**

**📚 Búsqueda de documentación técnica**
→ `Dense + Sparse` (captura términos técnicos exactos + contexto)

**🎯 Sistema de Q&A de alta precisión**
→ `Reranking` (máxima precisión en respuestas)

**🔬 Investigación exploratoria**
→ `MMR` (máxima diversidad y cobertura)

**🏢 Chatbot empresarial**
→ `Dense + Sparse + Reranking` (combinación de todas)

**Por Prioridad:**

- **Prioridad: Precisión** → Reranking
- **Prioridad: Cobertura** → MMR
- **Prioridad: Balance** → Dense + Sparse
- **Prioridad: Latencia** → MMR (más rápido)

### 🔄 Combinando Estrategias

Puedes combinar múltiples estrategias para mejores resultados:

```python
# Pipeline óptimo: Híbrido → Reranking → MMR
# 1. Búsqueda híbrida (Dense + Sparse)
hybrid_results = hybrid_retriever.invoke(query, k=20)

# 2. Reranking con LLM
reranked_results = rerank(hybrid_results)

# 3. MMR para diversidad final
final_results = apply_mmr(reranked_results, k=5)
```

Este enfoque multicapa es ideal para aplicaciones de producción donde la calidad es crítica.

---

## 🎯 Mejora de Consultas (Query Enhancement)

Las técnicas de mejora de consultas transforman y optimizan las consultas del usuario antes de la recuperación de documentos, mejorando significativamente la calidad de los resultados en sistemas RAG. El módulo 005 cubre tres estrategias fundamentales para optimizar consultas.

### 📊 Comparativa de Técnicas de Query Enhancement

| Técnica | Tipo | Complejidad | Latencia | Costo | Mejora en Precisión | Casos de Uso |
|---------|------|-------------|----------|-------|---------------------|--------------|
| **Query Expansion** | Enriquecimiento | Baja | Media (~500ms-1s) | Medio | ⭐⭐⭐⭐ | Consultas cortas, vagas o técnicas |
| **Query Decomposition** | División | Alta | Alta (1s-3s) | Alto | ⭐⭐⭐⭐⭐ | Consultas complejas multi-aspecto |
| **HyDE** | Transformación | Media | Alta (~500ms-2s) | Alto | ⭐⭐⭐⭐⭐ | Vocabulary mismatch, consultas cortas |

### 🔹 1. Query Expansion (Expansión de Consultas)

**¿Qué es?**
Técnica que enriquece la consulta original agregando sinónimos, términos técnicos, variaciones ortográficas y contexto adicional usando un LLM.

**¿Cómo funciona?**
```
Consulta: "Langchain memory"
    ↓ (LLM expande)
Consulta expandida: "Langchain memory, ConversationBufferMemory, ConversationSummaryMemory,
                     memory management, session context, state management, context storage..."
    ↓ (recuperación)
Documentos relevantes
```

**Ventajas:**
- ✅ Captura variaciones semánticas y sinónimos
- ✅ Mejora recall (recuperación) de documentos relevantes
- ✅ Enriquece consultas vagas con terminología específica
- ✅ Reduce falsos negativos
- ✅ Especialmente útil cuando el usuario no conoce términos exactos

**Desventajas:**
- ⚠️ Agrega latencia (~500ms-1s por llamada al LLM)
- ⚠️ Puede sobre-expandir consultas ya muy específicas
- ⚠️ Costo de tokens por cada consulta
- ⚠️ Calidad depende del LLM utilizado

**Cuándo usar:**
```python
✅ Consultas cortas o vagas ("memoria en IA")
✅ Búsquedas técnicas que requieren terminología específica
✅ Cuando el usuario usa lenguaje coloquial
✅ Bases de conocimiento con vocabulario diverso
❌ Consultas ya muy específicas y técnicas
❌ Aplicaciones de latencia crítica
```

**Ejemplo de implementación:**
```python
query_expansion_prompt = PromptTemplate.from_template("""
Expande la siguiente consulta agregando sinónimos y términos técnicos:
Consulta: "{query}"
""")

expansion_chain = query_expansion_prompt | llm | StrOutputParser()
expanded_query = expansion_chain.invoke({"query": "Langchain memory"})
```

---

### 🔹 2. Query Decomposition (Descomposición de Consultas)

**¿Qué es?**
Técnica que divide una consulta compleja de múltiples partes en sub-preguntas más simples y atómicas que se procesan individualmente.

**¿Cómo funciona?**
```
Consulta compleja: "¿Cómo usa LangChain memoria y agentes vs CrewAI?"
    ↓ (LLM descompone)
Sub-pregunta 1: "¿Qué mecanismos de memoria ofrece LangChain?"
Sub-pregunta 2: "¿Cómo funcionan los agentes en LangChain?"
Sub-pregunta 3: "¿Qué mecanismos de memoria ofrece CrewAI?"
Sub-pregunta 4: "¿Cómo funcionan los agentes en CrewAI?"
    ↓ (recuperación individual)
Respuestas combinadas
```

**Ventajas:**
- ✅ Precisión excepcional en consultas complejas
- ✅ Permite razonamiento multi-hop (paso a paso)
- ✅ Cada sub-pregunta obtiene contexto específico
- ✅ No se pierden aspectos de la pregunta original
- ✅ Recuperación más enfocada por aspecto
- ✅ Respuestas bien estructuradas

**Desventajas:**
- ⚠️ Alta latencia (múltiples llamadas al LLM)
- ⚠️ Mayor costo (1 descomposición + N respuestas)
- ⚠️ Complejidad en el parseo de sub-preguntas
- ⚠️ Puede ser overkill para consultas simples

**Cuándo usar:**
```python
✅ Preguntas comparativas ("A vs B")
✅ Consultas multi-aspecto (memoria + agentes + herramientas)
✅ Análisis complejos con múltiples entidades
✅ Investigación exploratoria de temas amplios
✅ Cuando se necesita razonamiento estructurado
❌ Consultas simples de un solo aspecto
❌ Presupuesto de latencia muy limitado
```

**Ejemplo de implementación:**
```python
decomposition_prompt = PromptTemplate.from_template("""
Descompón esta pregunta en 2-4 sub-preguntas simples:
Pregunta: "{question}"
""")

# Descomponer y procesar cada sub-pregunta
sub_questions = decomposition_chain.invoke({"question": query})
for subq in sub_questions:
    docs = retriever.invoke(subq)
    answer = qa_chain.invoke({"input": subq, "context": docs})
```

---

### 🔹 3. HyDE (Hypothetical Document Embeddings)

**¿Qué es?**
Técnica que genera una respuesta hipotética a la consulta usando un LLM, y luego busca documentos similares a esa respuesta hipotética en lugar de a la consulta original.

**¿Cómo funciona?**
```
Consulta: "¿Cuándo despidieron a Steve Jobs?"
    ↓ (LLM genera respuesta hipotética)
Respuesta hipotética: "Steve Jobs fue despedido de Apple en septiembre de 1985
                       debido a conflictos con la junta directiva..."
    ↓ (embedding de la respuesta)
    ↓ (búsqueda vectorial)
Documentos similares a la respuesta hipotética
```

**Ventajas:**
- ✅ Resuelve vocabulary mismatch (pregunta vs respuesta)
- ✅ Excelente para consultas cortas
- ✅ Alinea formato pregunta-respuesta
- ✅ Captura el estilo de respuestas esperadas
- ✅ Mejora significativa en recall

**Desventajas:**
- ⚠️ Alta latencia (LLM + búsqueda)
- ⚠️ Costo adicional por generación
- ⚠️ Calidad depende del documento hipotético generado
- ⚠️ Puede ser innecesario si documentos están bien alineados

**Cuándo usar:**
```python
✅ Consultas en lenguaje natural vs documentos técnicos
✅ Preguntas cortas que necesitan expansión semántica
✅ Vocabulario diferente entre usuario y documentos
✅ FAQs donde buscas respuestas, no preguntas
✅ Dominios especializados (medicina, leyes)
❌ Búsqueda de palabras clave exactas
❌ Latencia crítica
❌ Documentos y consultas ya alineados
```

**Ejemplo de implementación:**
```python
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder

# Opción 1: Con prompt predeterminado
hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search"  # o "sci_fact", "fiqa", "trec_news"
)

# Opción 2: Con prompt personalizado
custom_prompt = PromptTemplate.from_template(
    "Genera una respuesta hipotética para: {query}"
)
hyde_embedder = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    custom_prompt=custom_prompt
)

vectorstore = Chroma.from_documents(docs, hyde_embedder)
```

---

### 🎯 Guía de Selección Rápida

**Por Problema:**

**Consulta vaga o corta**
→ `Query Expansion` (agrega contexto y sinónimos)

**Consulta compleja multi-aspecto**
→ `Query Decomposition` (divide en sub-preguntas)

**Vocabulary mismatch**
→ `HyDE` (genera respuesta hipotética)

**Por Prioridad:**

- **Prioridad: Máxima precisión** → Query Decomposition
- **Prioridad: Resolver vocabulary gap** → HyDE
- **Prioridad: Balance costo/beneficio** → Query Expansion
- **Prioridad: Latencia mínima** → Query Expansion (más rápida)

**Por Tipo de Consulta:**

- **"memoria IA"** (vaga) → Query Expansion
- **"¿Cómo se compara X con Y en A, B y C?"** (compleja) → Query Decomposition
- **"¿Cuándo pasó X?"** (corta, respuesta específica) → HyDE

### 🔄 Comparación con Otras Estrategias

| Técnica | Latencia | Costo | Precisión | Recall | Uso Ideal |
|---------|----------|-------|-----------|--------|-----------|
| **Búsqueda Simple** | ⚡ Muy baja | 💰 Bajo | ⭐⭐⭐ | ⭐⭐⭐ | Consultas bien formadas |
| **Query Expansion** | ⚡ Media | 💰 Medio | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Consultas vagas |
| **Query Decomposition** | 🐌 Alta | 💰💰 Alto | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Consultas complejas |
| **HyDE** | 🐌 Alta | 💰💰 Alto | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Vocabulary mismatch |
| **Dense+Sparse** | ⚡ Media | 💰 Bajo | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Balance semántico/exacto |
| **Reranking** | 🐌 Alta | 💰💰 Alto | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Precisión crítica |

### 🔄 Combinando Técnicas

Puedes combinar Query Enhancement con otras estrategias para resultados óptimos:

```python
# Pipeline óptimo: Decomposition → Expansion → Dense+Sparse → Reranking
# 1. Descomponer consulta compleja
sub_questions = decomposition_chain.invoke({"question": complex_query})

# 2. Expandir cada sub-pregunta
for subq in sub_questions:
    expanded_subq = expansion_chain.invoke({"query": subq})

    # 3. Búsqueda híbrida (Dense + Sparse)
    results = hybrid_retriever.invoke(expanded_subq, k=20)

    # 4. Reranking para precisión final
    final_docs = rerank(results, subq, k=5)
```

Este enfoque multicapa es ideal para aplicaciones enterprise donde la calidad es crítica y hay presupuesto de latencia.

---

## 🖼️ RAG Multimodal

El RAG Multimodal extiende las capacidades de RAG tradicional para procesar y comprender no solo texto, sino también imágenes, gráficos, diagramas y otros elementos visuales presentes en documentos. Esta técnica es fundamental para trabajar con documentos del mundo real que combinan múltiples modalidades de información.

### 🎯 ¿Qué es RAG Multimodal?

**RAG Tradicional vs RAG Multimodal:**

| Aspecto | RAG Tradicional | RAG Multimodal |
|---------|----------------|----------------|
| **Entrada** | Solo texto | Texto + Imágenes + Gráficos |
| **Embeddings** | Un modelo de texto | CLIP (unificado texto-imagen) |
| **LLM** | GPT-3.5/4 estándar | GPT-4 Vision / LLaVA |
| **Casos de uso** | Documentos textuales | PDFs con gráficos, presentaciones, informes |
| **Complejidad** | Media | Alta |

**¿Por qué es importante?**

Los documentos del mundo real raramente son solo texto:
- 📊 **Informes financieros** con gráficos de barras y tendencias
- 📄 **Documentos técnicos** con diagramas y arquitecturas
- 🏥 **Registros médicos** con imágenes de rayos X y escaneos
- 📑 **Presentaciones** con infografías y visualizaciones
- 📚 **Libros educativos** con ilustraciones y esquemas

### 🔧 Componentes Clave

#### 1. **CLIP (Contrastive Language-Image Pre-training)**

**¿Qué es CLIP?**
- Modelo de OpenAI entrenado con 400 millones de pares (imagen, texto)
- Genera embeddings en el **mismo espacio vectorial** para texto e imágenes
- Permite buscar imágenes con texto y viceversa

**Arquitectura:**
```
Texto: "gráfico de ventas"  →  [Vision Transformer]  →  Vector 512D
                                                           ↓ (mismo espacio)
Imagen: [gráfico.png]       →  [Vision Transformer]  →  Vector 512D
```

**Ventajas de CLIP:**
- ✅ Búsqueda semántica unificada (texto puede encontrar imágenes)
- ✅ No requiere etiquetado manual de imágenes
- ✅ Funciona con múltiples idiomas
- ✅ Generaliza bien a conceptos nuevos

**Alternativas a CLIP:**
- **BLIP-2**: Modelo más reciente con mejor comprensión
- **LLaVA**: Combina CLIP con LLaMA para generación
- **ALIGN**: Modelo de Google similar a CLIP

#### 2. **GPT-4 Vision (GPT-4V)**

**Capacidades:**
- 👁️ Interpreta imágenes, gráficos, tablas, diagramas
- 📊 Extrae datos de visualizaciones
- 🎨 Describe contenido visual en detalle
- 🔗 Relaciona información visual con contexto textual

**Formato de entrada:**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "¿Qué muestra este gráfico?"},
    {"type": "image_url", "image_url": "data:image/png;base64,..."}
  ]
}
```

### 🔄 Flujo del Pipeline Multimodal

```
                    PDF CON TEXTO E IMÁGENES
                             ↓
              ┌──────────────┴──────────────┐
              ↓                             ↓
         TEXTO                          IMÁGENES
              ↓                             ↓
    [Text Splitter]                  [Extracción]
              ↓                             ↓
      Chunks de Texto              Imágenes PIL
              ↓                             ↓
    [CLIP Text Encoder]          [CLIP Image Encoder]
              ↓                             ↓
       Vectores 512D                 Vectores 512D
              └──────────────┬──────────────┘
                             ↓
                    FAISS VECTOR STORE
                    (Embeddings Unificados)
                             ↓
                    ┌────────┴────────┐
                    ↓                 ↓
              CONSULTA           RECUPERACIÓN
            (texto usuario)    (texto + imágenes)
                    ↓                 ↓
              [CLIP Encoder]    Docs Relevantes
                    ↓                 ↓
                    └────────┬────────┘
                             ↓
                      GPT-4 VISION
                   (procesa texto + imágenes)
                             ↓
                        RESPUESTA
```

### 📋 Implementación Paso a Paso

#### **Paso 1: Extraer Texto e Imágenes del PDF**

```python
import fitz  # PyMuPDF
from PIL import Image
import io

# Abrir PDF
doc = fitz.open("documento.pdf")

for page in doc:
    # Extraer texto
    text = page.get_text()

    # Extraer imágenes
    for img in page.get_images():
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        pil_image = Image.open(io.BytesIO(image_bytes))
```

#### **Paso 2: Generar Embeddings con CLIP**

```python
from transformers import CLIPProcessor, CLIPModel
import torch

# Inicializar CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image):
    """Generar embedding de imagen"""
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

def embed_text(text):
    """Generar embedding de texto"""
    inputs = clip_processor(text=text, return_tensors="pt", max_length=77)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)
```

#### **Paso 3: Almacenar en Vector Store Unificado**

```python
from langchain_community.vectorstores import FAISS

# Crear vector store con embeddings de texto e imágenes
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings)],
    embedding=None,  # Ya tenemos embeddings precalculados
    metadatas=[doc.metadata for doc in all_docs]
)
```

#### **Paso 4: Recuperación Multimodal**

```python
def retrieve_multimodal(query, k=5):
    """Recuperar texto e imágenes relevantes"""
    # Convertir consulta a embedding CLIP
    query_embedding = embed_text(query)

    # Buscar documentos similares (texto e imágenes)
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    return results
```

#### **Paso 5: Crear Mensaje para GPT-4 Vision**

```python
import base64
from langchain.schema.messages import HumanMessage

def create_multimodal_message(query, retrieved_docs):
    """Crear mensaje con texto e imágenes para GPT-4V"""
    content = [{"type": "text", "text": f"Pregunta: {query}\n\nContexto:\n"}]

    # Agregar texto recuperado
    for doc in retrieved_docs:
        if doc.metadata["type"] == "text":
            content.append({"type": "text", "text": doc.page_content})

        # Agregar imágenes en base64
        elif doc.metadata["type"] == "image":
            image_base64 = image_data_store[doc.metadata["image_id"]]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            })

    return HumanMessage(content=content)
```

#### **Paso 6: Generar Respuesta con GPT-4 Vision**

```python
from langchain.chat_models import init_chat_model

# Inicializar GPT-4 Vision
llm = init_chat_model("openai:gpt-4-vision-preview")

def multimodal_rag_pipeline(query):
    # 1. Recuperar documentos relevantes
    docs = retrieve_multimodal(query, k=5)

    # 2. Crear mensaje multimodal
    message = create_multimodal_message(query, docs)

    # 3. Generar respuesta
    response = llm.invoke([message])
    return response.content
```

### 🎯 Casos de Uso

#### **1. Análisis de Informes Financieros**
```python
query = "¿Cuáles fueron las tendencias de ingresos en Q3 según el gráfico?"
# Recupera: texto sobre Q3 + gráfico de barras
# GPT-4V interpreta el gráfico y combina con texto
```

#### **2. Documentación Técnica**
```python
query = "Explica la arquitectura del sistema mostrada en el diagrama"
# Recupera: descripción textual + diagrama de arquitectura
# GPT-4V analiza el diagrama y lo relaciona con el texto
```

#### **3. Informes Médicos**
```python
query = "¿Qué anomalías se observan en la radiografía?"
# Recupera: notas médicas + imagen de rayos X
# GPT-4V examina la imagen y proporciona análisis
```

### ✅ Ventajas del RAG Multimodal

| Ventaja | Descripción |
|---------|-------------|
| **🎯 Comprensión Completa** | Procesa toda la información, no solo texto |
| **📊 Análisis Visual** | Interpreta gráficos, tablas, diagramas |
| **🔍 Búsqueda Unificada** | Una consulta encuentra texto e imágenes relevantes |
| **💡 Contexto Rico** | El LLM ve exactamente lo que ve el usuario |
| **🎨 Versatilidad** | Funciona con cualquier tipo de documento visual |

### ⚠️ Desafíos y Consideraciones

| Desafío | Solución |
|---------|----------|
| **💰 Costo Alto** | GPT-4 Vision es más costoso que GPT-4 estándar |
| **🐌 Latencia** | Procesamiento de imágenes añade tiempo (~2-5s extra) |
| **📦 Tamaño de Contexto** | Imágenes consumen muchos tokens (cada imagen ≈ 85-170 tokens) |
| **🎨 Calidad de Imagen** | Imágenes de baja resolución o borrosas limitan la comprensión |
| **💾 Almacenamiento** | Guardar imágenes en base64 ocupa mucho espacio |

### 🔄 Comparación con Alternativas

| Enfoque | Ventajas | Desventajas |
|---------|----------|-------------|
| **RAG Multimodal (CLIP + GPT-4V)** | ✅ Comprensión visual completa<br>✅ Búsqueda semántica unificada | ❌ Costoso<br>❌ Alta latencia |
| **OCR + RAG Tradicional** | ✅ Más económico<br>✅ Más rápido | ❌ Pierde información visual<br>❌ No interpreta gráficos |
| **Image Captioning + RAG** | ✅ Balance costo/beneficio | ❌ Captions pueden ser inexactos<br>❌ Pierde detalles |
| **Table Extraction + RAG** | ✅ Bueno para tablas | ❌ No funciona con gráficos<br>❌ Limitado a estructuras |

### 📊 Cuándo Usar RAG Multimodal

**✅ Usa RAG Multimodal cuando:**
- Tus documentos contienen información visual crítica (gráficos, diagramas)
- Necesitas respuestas que requieren interpretar imágenes
- El presupuesto y latencia no son limitantes
- La precisión visual es más importante que la velocidad

**❌ NO uses RAG Multimodal cuando:**
- Tus documentos son mayormente texto
- El presupuesto es muy limitado
- La latencia debe ser ultra-baja (<1s)
- Las imágenes son decorativas, no informativas

### 🚀 Mejoras y Optimizaciones

**1. Caché de Embeddings**
```python
# Guardar embeddings CLIP para evitar recalcular
vector_store.save_local("embeddings_cache")
```

**2. Compresión de Imágenes**
```python
# Reducir tamaño de imágenes para ahorrar tokens
from PIL import Image
image = image.resize((800, 600), Image.LANCZOS)
```

**3. Filtrado Inteligente**
```python
# Solo enviar imágenes realmente relevantes a GPT-4V
if doc.metadata["type"] == "image" and similarity_score > 0.8:
    # Incluir imagen
```

**4. Batch Processing**
```python
# Procesar múltiples páginas en paralelo
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    embeddings = list(executor.map(embed_image, images))
```

### 🛠️ Herramientas y Librerías

| Herramienta | Propósito | Alternativa |
|-------------|-----------|-------------|
| **CLIP** | Embeddings unificados | BLIP-2, ALIGN |
| **GPT-4 Vision** | LLM multimodal | LLaVA, Gemini Pro Vision |
| **PyMuPDF (fitz)** | Extracción de PDF | pdfplumber, PyPDF2 |
| **PIL/Pillow** | Procesamiento de imágenes | OpenCV |
| **FAISS** | Vector store | ChromaDB, Pinecone |

---

## 🔄 Fundamentos de LangGraph

LangGraph es un framework de LangChain diseñado para construir aplicaciones con estado usando grafos. Es especialmente útil para crear agentes, chatbots complejos y flujos de trabajo que requieren gestión avanzada de estado y enrutamiento condicional.

### 🎯 ¿Qué es LangGraph?

**LangGraph** permite construir aplicaciones basadas en grafos donde:
- **Nodos** representan funciones que procesan el estado
- **Aristas** definen el flujo entre nodos
- **Estado** se comparte y actualiza a través del grafo
- **Enrutamiento condicional** permite decisiones dinámicas basadas en el estado

### 📊 Conceptos Clave

| Concepto | Descripción | Ejemplo |
|----------|-------------|---------|
| **State Schema** | Define la estructura de datos del grafo | `TypedDict`, `DataClass`, `Pydantic` |
| **Nodos** | Funciones que procesan y actualizan el estado | `chatbot(state)`, `tool_executor(state)` |
| **Aristas** | Conexiones entre nodos (fijas o condicionales) | `START → chatbot → END` |
| **Reducers** | Funciones que combinan estados (ej: `add_messages`) | Agregar mensajes sin sobrescribir |
| **Herramientas** | Funciones externas que el LLM puede llamar | Búsqueda web, cálculos, APIs |

### 🔹 Esquemas de Estado

LangGraph soporta 3 formas de definir el estado del grafo:

#### 1. **TypedDict** (Solo Type Hints)
```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**Características:**
- ✅ Simple y rápido
- ✅ Integración nativa con Python
- ❌ **NO valida en tiempo de ejecución**
- ❌ Los errores de tipo solo se detectan durante la ejecución

**Cuándo usar:**
- Prototipos rápidos
- Cuando confías en los datos de entrada
- Testing y desarrollo

#### 2. **DataClass** (Estructura de Datos)
```python
from dataclasses import dataclass

@dataclass
class State:
    name: str
    game: Literal["cricket", "badminton"]
```

**Características:**
- ✅ Sintaxis concisa y limpia
- ✅ Acceso a atributos con notación de punto (`state.name`)
- ✅ Métodos autogenerados (`__init__`, `__repr__`)
- ⚠️ Validación básica solo de tipos

**Cuándo usar:**
- Cuando necesitas clases de datos estructuradas
- Mejor legibilidad del código
- Proyectos de tamaño mediano

#### 3. **Pydantic** (Validación Robusta) ⭐ Recomendado
```python
from pydantic import BaseModel

class State(BaseModel):
    name: str
    age: int
```

**Características:**
- ✅ **Validación completa en tiempo de ejecución**
- ✅ Mensajes de error claros y descriptivos
- ✅ Conversión automática de tipos cuando es posible
- ✅ Validadores personalizados
- ✅ Integración perfecta con FastAPI y otros frameworks

**Cuándo usar:**
- Aplicaciones de producción ✅
- Cuando necesitas validación robusta
- APIs y servicios web
- Cuando recibes datos de fuentes externas

**Ejemplo de validación:**
```python
# TypedDict: ❌ No valida, error en tiempo de ejecución posterior
graph.invoke({"name": 123})  # Acepta pero falla después

# Pydantic: ✅ Valida inmediatamente
graph.invoke({"name": 123})  # ValidationError: Input should be a valid string
```

### 🛠️ Componentes de un Grafo LangGraph

#### **1. Nodos (Nodes)**
Funciones que procesan el estado:

```python
def chatbot(state: State):
    """Nodo que procesa mensajes con un LLM"""
    return {"messages": [llm.invoke(state["messages"])]}
```

#### **2. Aristas (Edges)**
Conexiones entre nodos:

```python
# Arista fija
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Arista condicional
builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # Función que decide la ruta
)
```

#### **3. Reductores (Reducers)**
Controlan cómo se actualizan los valores del estado:

```python
from langgraph.graph.message import add_messages

# Sin reducer: sobrescribe
messages: list  # Nuevo valor reemplaza el anterior

# Con reducer: agrega
messages: Annotated[list, add_messages]  # Nuevo valor se agrega al anterior
```

### 🔧 Herramientas en LangGraph

LangGraph permite que los LLMs llamen a herramientas externas:

#### **Herramientas Disponibles:**

| Herramienta | Descripción | Uso Ideal |
|-------------|-------------|-----------|
| **Arxiv** | Búsqueda de papers científicos | Investigación académica |
| **Wikipedia** | Información enciclopédica | Conocimiento general |
| **Tavily** | Búsqueda web optimizada para LLMs | Noticias y contenido actual |
| **Custom Functions** | Funciones Python personalizadas | Cálculos, transformaciones |

#### **Flujo con Herramientas:**

```
Usuario → LLM (con herramientas vinculadas)
           ↓
      ¿Necesita herramienta?
           ↓
        SÍ ┌─┴─┐ NO
           ↓     ↓
    Ejecutar → END
    herramienta
           ↓
      Respuesta
```

#### **Ejemplo de Implementación:**

```python
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Definir herramientas
tools = [arxiv_tool, wiki_tool, tavily_tool]

# 2. Vincular herramientas al LLM
llm_with_tools = llm.bind_tools(tools)

# 3. Crear nodos
def tool_calling_llm(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 4. Construir grafo
builder.add_node("llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

# 5. Enrutamiento condicional
builder.add_conditional_edges("llm", tools_condition)
# tools_condition enruta a "tools" si hay llamada a herramienta, sino a END
```

### 📋 Notebooks del Módulo

#### **2-chatbot.ipynb** - Chatbot Simple
- Construcción de un chatbot básico con LangGraph
- Uso de mensajes como estado
- Reducer `add_messages` para mantener historial
- Streaming de respuestas

#### **3-DataclassStateSchema.ipynb** - Esquemas de Estado
- Comparación entre TypedDict y DataClass
- Nodos y aristas condicionales
- Enrutamiento aleatorio con `decide_play`
- Diferencias en validación y acceso a datos

#### **4-pydantic.ipynb** - Validación con Pydantic
- Uso de Pydantic para validación robusta
- Ventajas sobre TypedDict
- Detección temprana de errores
- Mejores prácticas para producción

#### **5-ChainsLangGraph.ipynb** - Cadenas y Herramientas
- Integración de herramientas con LLMs
- Uso de `bind_tools()` y `ToolNode`
- Enrutamiento condicional con `tools_condition`
- Manejo de mensajes multimodales

#### **6-chatbotswithmultipletools.ipynb** - Chatbot Avanzado
- Chatbot con múltiples herramientas (Arxiv, Wikipedia, Tavily)
- Selección inteligente de herramientas por el LLM
- Pipeline completo: consulta → decisión → herramienta → respuesta
- Casos de uso: investigación, enciclopedia, noticias

### ✅ Ventajas de LangGraph

| Ventaja | Descripción |
|---------|-------------|
| **🎯 Control Total** | Defines exactamente el flujo de tu aplicación |
| **🔄 Estado Compartido** | El estado se propaga automáticamente entre nodos |
| **🛤️ Enrutamiento Flexible** | Decisiones dinámicas basadas en el estado actual |
| **🛠️ Herramientas Integradas** | Conecta fácilmente APIs, bases de datos, funciones |
| **📊 Visualización** | Genera diagramas del grafo automáticamente |
| **🧩 Modular** | Cada nodo es independiente y reutilizable |

### 🎯 Casos de Uso

**🤖 Chatbots Avanzados**
- Mantener contexto de conversación
- Llamar herramientas cuando sea necesario
- Enrutamiento basado en intención del usuario

**🔍 Agentes de Investigación**
- Búsqueda en múltiples fuentes (Arxiv, Wikipedia, Web)
- Agregación de información
- Razonamiento multi-paso

**🏢 Flujos de Trabajo Empresariales**
- Aprobaciones multi-nivel
- Procesamiento condicional de documentos
- Integración con sistemas legacy

**🧪 Pipelines RAG Complejos**
- Query Enhancement → Retrieval → Reranking → Generation
- Decisiones adaptativas basadas en la calidad de resultados

### 🔄 Comparación con Alternativas

| Framework | Estado | Enrutamiento | Curva Aprendizaje | Uso Ideal |
|-----------|--------|--------------|-------------------|-----------|
| **LangGraph** | ✅ Explícito | ✅ Condicional | Media | Agentes complejos, flujos personalizados |
| **LangChain LCEL** | ⚠️ Implícito | ❌ Lineal | Baja | Cadenas simples, pipelines lineales |
| **CrewAI** | ✅ Automático | ✅ Automático | Baja | Multi-agentes colaborativos |
| **AutoGen** | ✅ Conversacional | ✅ Automático | Alta | Agentes autónomos, investigación |

### 🚀 Mejores Prácticas

1. **Usa Pydantic para Producción**
   ```python
   # ✅ Bueno
   class State(BaseModel):
       messages: list

   # ❌ Evitar en producción
   class State(TypedDict):
       messages: list
   ```

2. **Siempre Usa Reducers para Listas**
   ```python
   # ✅ Bueno - los mensajes se agregan
   messages: Annotated[list, add_messages]

   # ❌ Malo - los mensajes se sobrescriben
   messages: list
   ```

3. **Nombra Nodos Descriptivamente**
   ```python
   # ✅ Bueno
   builder.add_node("validate_input", validate_fn)
   builder.add_node("call_llm", llm_fn)

   # ❌ Malo
   builder.add_node("node1", validate_fn)
   builder.add_node("node2", llm_fn)
   ```

4. **Usa tools_condition para Herramientas**
   ```python
   # ✅ Bueno - enrutamiento automático
   builder.add_conditional_edges("llm", tools_condition)

   # ❌ Malo - lógica manual propensa a errores
   builder.add_conditional_edges("llm", custom_routing)
   ```

---

## 🤖 Arquitectura de Agentes

Los agentes son sistemas de IA que pueden tomar decisiones, usar herramientas y ejecutar tareas de manera autónoma basándose en las entradas del usuario. El módulo 008 se enfoca en la arquitectura de agentes ReAct, una de las arquitecturas más efectivas para construir agentes inteligentes.

### 🎯 ¿Qué es ReAct?

**ReAct (Reason + Act)** es un paradigma de arquitectura de agentes que combina razonamiento y acción de manera iterativa. El agente alterna entre:

1. **Razonar (Reason)**: El LLM piensa sobre qué hacer a continuación
2. **Actuar (Act)**: El agente ejecuta una herramienta específica
3. **Observar (Observe)**: El agente recibe los resultados de la herramienta
4. **Repetir**: Vuelve a razonar basándose en la nueva información

### 📊 Ciclo ReAct

```
Usuario: "¿Cuáles son las últimas noticias de IA?"
    ↓
[RAZONAR] LLM decide: "Necesito buscar en internet"
    ↓
[ACTUAR] Ejecuta herramienta: tavily.search("noticias IA")
    ↓
[OBSERVAR] Recibe: [lista de artículos]
    ↓
[RAZONAR] LLM decide: "Tengo suficiente información"
    ↓
[RESPONDER] Genera respuesta estructurada al usuario
```

### 🛠️ Componentes Clave

#### 1. **Herramientas (Tools)**

Las herramientas son funciones que el agente puede invocar:

| Herramienta | Tipo | Descripción | Caso de Uso |
|-------------|------|-------------|-------------|
| **Arxiv** | Búsqueda académica | Consulta papers científicos | Investigación, referencias académicas |
| **Wikipedia** | Enciclopedia | Información general y conceptos | Definiciones, contexto histórico |
| **Tavily** | Búsqueda web | Noticias y contenido actualizado | Información reciente, tendencias |
| **Custom Functions** | Matemáticas/Lógica | add(), multiply(), divide() | Cálculos, transformaciones |

**Ejemplo de implementación:**
```python
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults

# Configurar herramientas
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
tavily = TavilySearchResults()

# Funciones personalizadas
def add(a: int, b: int) -> int:
    """Suma dos números"""
    return a + b

# Lista de herramientas disponibles
tools = [arxiv, wiki, tavily, add, multiply, divide]
```

#### 2. **LLM con Herramientas Vinculadas**

El modelo de lenguaje debe estar "consciente" de las herramientas disponibles:

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

# Vincular herramientas al LLM
llm_with_tools = llm.bind_tools(tools)
```

Cuando se vinculan herramientas, el LLM:
- ✅ Recibe las descripciones y parámetros de cada herramienta
- ✅ Puede decidir cuándo y cómo usar cada herramienta
- ✅ Genera llamadas a herramientas en formato estructurado

#### 3. **Grafo ReAct con LangGraph**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Definir el nodo del LLM
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Construir grafo
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

# Flujo del grafo
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition  # Enruta a "tools" o END según si hay llamadas
)
builder.add_edge("tools", "tool_calling_llm")  # Loop de vuelta al LLM

graph = builder.compile()
```

**Flujo del grafo:**
```
START → tool_calling_llm ──┐
            ↑               │
            │               ↓
         tools ← [tools_condition] → END
```

### 💾 Memoria en Agentes ReAct

Los agentes pueden mantener contexto entre múltiples interacciones usando **checkpointers**:

#### **Sin Memoria:**
```python
# Cada invocación es independiente
graph.invoke({"messages": "¿Cuánto es 5 + 8?"})  # → 13
graph.invoke({"messages": "Divide eso por 5"})   # ❌ No sabe qué es "eso"
```

#### **Con Memoria (MemorySaver):**
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph_memory = builder.compile(checkpointer=memory)

# Configurar thread_id para sesión
config = {"configurable": {"thread_id": "1"}}

# Primera interacción
graph_memory.invoke({"messages": "¿Cuánto es 5 + 8?"}, config)  # → 13

# Segunda interacción - mantiene contexto
graph_memory.invoke({"messages": "Divide eso por 5"}, config)   # ✅ 13 / 5 = 2.6
```

**Ventajas de la memoria:**
- ✅ Conversaciones naturales con referencias contextuales
- ✅ El agente "recuerda" resultados previos
- ✅ Seguimiento de tareas multi-paso
- ✅ Personalización basada en interacciones anteriores

### 🔄 Streaming de Respuestas

El módulo también cubre técnicas avanzadas de streaming para mejorar la experiencia del usuario:

#### **1. Stream Mode: "updates"**
Solo muestra los cambios incrementales (nuevos mensajes):

```python
for chunk in graph.stream({"messages": "Hola"}, config, stream_mode="updates"):
    print(chunk)  # Solo el mensaje nuevo del bot
```

#### **2. Stream Mode: "values"**
Muestra el estado completo en cada paso:

```python
for chunk in graph.stream({"messages": "Hola"}, config, stream_mode="values"):
    print(chunk)  # Todo el historial + mensaje nuevo
```

#### **3. Async Stream Events (Token por Token)**
Transmite cada token a medida que se genera:

```python
async for event in graph.astream_events({"messages": "Hola"}, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

**Comparación:**

| Modo | Latencia Percibida | Ancho de Banda | Caso de Uso |
|------|-------------------|----------------|-------------|
| **updates** | Media | Bajo | APIs, procesamiento por lotes |
| **values** | Media | Alto | Debugging, auditoría completa |
| **astream_events** | Baja | Muy alto | UIs tipo ChatGPT, experiencia fluida |

### 📋 Notebooks del Módulo

#### **1-ReActAgents.ipynb**
- Implementación completa de agentes ReAct
- Integración de múltiples herramientas (Arxiv, Wikipedia, Tavily)
- Funciones personalizadas como herramientas
- Construcción del grafo con enrutamiento condicional
- Agentes sin memoria vs. con memoria (MemorySaver)
- Ejemplos de consultas complejas multi-herramienta

**Temas cubiertos:**
- Configuración de herramientas externas
- Vinculación de herramientas al LLM con `bind_tools()`
- Estado del grafo con `TypedDict` y `add_messages`
- Uso de `tools_condition` para enrutamiento automático
- Implementación de memoria con checkpointers
- Gestión de threads de conversación

#### **2-streaming.ipynb**
- Chatbot simple con LangGraph
- Técnicas de streaming síncronas (.stream())
- Streaming asíncrono token por token (.astream_events())
- Comparación entre stream_mode="updates" vs "values"
- Casos de uso para cada tipo de streaming

**Temas cubiertos:**
- Construcción de chatbot básico con un solo nodo
- Configuración de memoria con MemorySaver
- Métodos `.stream()` y `.astream_events()`
- Diferencias entre streaming de estado vs tokens
- Aplicaciones prácticas de cada modo

### ✅ Ventajas de los Agentes ReAct

| Ventaja | Descripción |
|---------|-------------|
| **🎯 Autonomía** | El agente decide qué herramientas usar y cuándo |
| **🔄 Iterativo** | Puede ejecutar múltiples pasos para tareas complejas |
| **🛠️ Extensible** | Agregar nuevas herramientas es trivial |
| **💭 Transparente** | El razonamiento del agente es visible (tool_calls) |
| **🧩 Composable** | Combina múltiples herramientas de manera inteligente |

### 🎯 Casos de Uso

**🔬 Investigación Académica**
```python
query = "Explica el paper 'Attention is All You Need' y busca papers relacionados"
# Agente usa: Arxiv → Wikipedia → Genera resumen
```

**📰 Análisis de Noticias**
```python
query = "Últimas noticias de IA, resume las 5 más importantes"
# Agente usa: Tavily → Procesa resultados → Genera resumen
```

**🧮 Asistente Matemático**
```python
query = "Calcula (12 + 8) * 3 y luego divide el resultado entre 5"
# Agente usa: add() → multiply() → divide() → Responde
```

**🤝 Consultas Multi-Fuente**
```python
query = "¿Qué es machine learning según Wikipedia y hay papers recientes sobre el tema?"
# Agente usa: Wikipedia → Arxiv → Combina información
```

### ⚠️ Consideraciones y Limitaciones

| Aspecto | Consideración |
|---------|---------------|
| **💰 Costo** | Cada llamada a herramienta = 1 llamada extra al LLM |
| **🐌 Latencia** | Agentes multi-paso pueden ser lentos (5-10s) |
| **🔁 Loops Infinitos** | Necesitas límites en el número de iteraciones |
| **🎲 No Determinismo** | El LLM puede elegir herramientas diferentes en cada ejecución |
| **🛡️ Seguridad** | Herramientas externas requieren validación y rate limiting |

### 🚀 Mejores Prácticas

1. **Limita Iteraciones del Agente**
   ```python
   # Agregar límite de recursión
   graph = builder.compile(checkpointer=memory, recursion_limit=10)
   ```

2. **Descripciones Claras de Herramientas**
   ```python
   def add(a: int, b: int) -> int:
       """Suma dos números enteros.  # Descripción clara

       Args:
           a: Primer número
           b: Segundo número

       Returns:
           La suma de a + b
       """
       return a + b
   ```

3. **Usa Memoria para Conversaciones**
   ```python
   # Siempre usa checkpointer para agentes conversacionales
   graph = builder.compile(checkpointer=MemorySaver())
   ```

4. **Monitorea con LangSmith**
   ```python
   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_PROJECT"] = "mi-agente"
   ```

---

## 🐛 Debugging con LangGraph Studio

LangGraph Studio es una herramienta de desarrollo y debugging que permite visualizar, inspeccionar y depurar grafos de LangGraph en tiempo real. El módulo 009 introduce las configuraciones necesarias para trabajar con LangGraph Studio.

### 🎯 ¿Qué es LangGraph Studio?

**LangGraph Studio** es una aplicación de escritorio que proporciona:
- 🔍 Visualización interactiva de grafos
- 🐛 Debugging paso a paso de ejecuciones
- 📊 Inspección de estado en cada nodo
- ⚡ Ejecución local de agentes
- 🔄 Recarga en caliente (hot reload) de cambios

### 📋 Configuración con langgraph.json

El archivo `langgraph.json` define la configuración del proyecto para LangGraph Studio:

```json
{
    "dependencies": ["."],
    "graphs": {
      "openai_agent": "./openai_agent.py:agent"
    },
    "env": "../.env"
}
```

**Campos:**

| Campo | Descripción | Ejemplo |
|-------|-------------|---------|
| `dependencies` | Directorios con código fuente | `["."]` - directorio actual |
| `graphs` | Mapeo nombre → ruta del grafo | `"openai_agent": "./openai_agent.py:agent"` |
| `env` | Ruta al archivo .env | `"../.env"` |

### 🛠️ Archivo openai_agent.py

Este archivo implementa dos grafos para debugging:

#### **1. Grafo Básico (make_default_graph)**
```python
def make_default_graph():
    """Grafo simple: consulta → LLM → respuesta"""
    graph_workflow = StateGraph(State)

    def call_model(state):
        return {"messages": [model.invoke(state['messages'])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    return graph_workflow.compile()
```

**Flujo:**
```
START → agent → END
```

#### **2. Grafo con Herramientas (make_alternative_graph)**
```python
def make_alternative_graph():
    """Grafo con herramienta 'add' y enrutamiento condicional"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    def should_continue(state: State):
        """Decide si ejecutar herramienta o terminar"""
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)
    graph_workflow.add_edge("tools", "agent")

    return graph_workflow.compile()
```

**Flujo:**
```
START → agent ──┐
         ↑      │
         │      ↓
      tools ← [should_continue] → END
```

### 🚀 Uso de LangGraph Studio

#### **1. Instalación**
```bash
# Descargar desde: https://studio.langchain.com/
# Disponible para macOS, Windows y Linux
```

#### **2. Abrir Proyecto**
```bash
# Navegar a la carpeta con langgraph.json
cd 009_Debugging

# LangGraph Studio detectará automáticamente el proyecto
```

#### **3. Seleccionar Grafo**
En la interfaz de LangGraph Studio:
1. Seleccionar "openai_agent" del dropdown
2. Ver visualización del grafo
3. Ejecutar con inputs personalizados

#### **4. Debugging Paso a Paso**
```python
# LangGraph Studio permite:
- Ver el estado después de cada nodo
- Inspeccionar mensajes y herramientas invocadas
- Identificar dónde falla el grafo
- Modificar el código y ver cambios en tiempo real
```

### 📊 Comparación: Desarrollo vs Producción

| Aspecto | LangGraph Studio | Producción (Python) |
|---------|------------------|---------------------|
| **Visualización** | ✅ Gráfica interactiva | ❌ Solo código |
| **Debugging** | ✅ Paso a paso | ⚠️ Logs/print |
| **Hot Reload** | ✅ Automático | ❌ Reinicio manual |
| **Inspección de Estado** | ✅ UI visual | ⚠️ Breakpoints |
| **Velocidad de Desarrollo** | 🚀 Muy rápida | 🐌 Media |
| **Deployment** | ❌ Solo desarrollo | ✅ Código Python |

### ✅ Ventajas de LangGraph Studio

| Ventaja | Descripción |
|---------|-------------|
| **👁️ Visualización** | Ve tu grafo en tiempo real mientras se ejecuta |
| **🐛 Debugging Visual** | Identifica problemas rápidamente sin print() |
| **⚡ Iteración Rápida** | Cambios en el código se reflejan inmediatamente |
| **📊 Inspección de Estado** | Ve exactamente qué datos pasan entre nodos |
| **🧪 Testing Interactivo** | Prueba diferentes inputs sin escribir tests |

### 🎯 Casos de Uso

**🔍 Desarrollar Nuevo Agente**
- Visualizar flujo antes de escribir código complejo
- Verificar que el enrutamiento condicional funcione correctamente

**🐛 Depurar Agente Existente**
- Identificar por qué el agente elige herramientas incorrectas
- Ver el estado exacto cuando ocurre un error

**🧪 Experimentar con Prompts**
- Probar diferentes prompts y ver su efecto inmediatamente
- Comparar comportamiento entre modelos (GPT-4 vs Llama)

**📚 Aprender LangGraph**
- Entender cómo fluyen los datos en grafos complejos
- Ver ejemplos funcionando en tiempo real

### 🛠️ Estructura del Módulo

```
009_Debugging/
├── langgraph.json         # Configuración de LangGraph Studio
└── openai_agent.py        # Implementación de grafos para debugging
    ├── make_default_graph()      # Grafo simple
    └── make_alternative_graph()  # Grafo con herramientas
```

### 🚀 Mejores Prácticas

1. **Usa langgraph.json para Todos tus Proyectos**
   ```json
   {
       "dependencies": ["."],
       "graphs": {
         "mi_agente": "./agent.py:graph",
         "mi_chatbot": "./chatbot.py:chatbot_graph"
       },
       "env": ".env"
   }
   ```

2. **Implementa Múltiples Variantes de Grafos**
   ```python
   # Útil para A/B testing y experimentación
   def make_basic_agent(): ...
   def make_agent_with_memory(): ...
   def make_agent_with_tools(): ...
   ```

3. **Usa Nombres Descriptivos**
   ```python
   # ✅ Bueno
   graph_workflow.add_node("validate_user_input", validate_fn)

   # ❌ Malo
   graph_workflow.add_node("node1", validate_fn)
   ```

4. **Documenta Funciones de Decisión**
   ```python
   def should_continue(state: State):
       """Decide si continuar con herramientas o terminar.

       Returns:
           "tools" si hay tool_calls pendientes
           END si la respuesta está lista
       """
       ...
   ```

---

## 🎯 RAG Agéntico

El RAG Agéntico representa la evolución de los sistemas RAG tradicionales, donde en lugar de un flujo lineal simple (recuperar → generar), el sistema implementa capacidades de razonamiento, evaluación y auto-corrección. El módulo 010 introduce estos conceptos avanzados usando LangGraph.

### 🤖 ¿Qué es RAG Agéntico?

**RAG Agéntico (Agentic RAG)** es un sistema de Recuperación Aumentada Generativa donde un agente inteligente:

1. **Razona**: Analiza la pregunta y decide qué herramientas usar
2. **Recupera**: Busca información en múltiples fuentes de conocimiento
3. **Evalúa**: Determina si los documentos recuperados son relevantes
4. **Reformula**: Mejora la consulta si los documentos no son adecuados
5. **Genera**: Crea una respuesta fundamentada en evidencia

### 🔄 Flujo de RAG Agéntico

```
Usuario: "¿Qué es LangGraph?"
    ↓
[AGENTE] Analiza la pregunta y decide usar herramienta de recuperación
    ↓
[RECUPERAR] Busca en vectorstore de LangGraph
    ↓
[EVALUAR] ¿Los documentos son relevantes?
    ├─ SÍ → [GENERAR] Crea respuesta basada en contexto
    └─ NO → [REFORMULAR] Mejora la pregunta → Vuelve a [AGENTE]
    ↓
[RESPUESTA] Entrega respuesta final al usuario
```

### 🛠️ Componentes del Sistema

#### 1. **Múltiples Vectorstores**
```python
# Vectorstore para documentación de LangGraph
vectorstore_langgraph = FAISS.from_documents(docs_langgraph, embeddings)

# Vectorstore para documentación de LangChain
vectorstore_langchain = FAISS.from_documents(docs_langchain, embeddings)

# El agente decide cuál usar según la consulta
```

#### 2. **Nodos del Grafo**

- **Nodo Agent**: Razona y decide qué herramienta usar
  ```python
  def agent(state):
      model = ChatGroq(model="qwen-qwq-32b")
      model = model.bind_tools(tools)
      response = model.invoke(state["messages"])
      return {"messages": [response]}
  ```

- **Nodo Retrieve**: Ejecuta herramientas de recuperación
  ```python
  retrieve = ToolNode([retriever_tool_langgraph, retriever_tool_langchain])
  ```

- **Nodo Grade**: Evalúa relevancia de documentos
  ```python
  def grade_documents(state) -> Literal["generate", "rewrite"]:
      # Usa un LLM para evaluar si los docs son relevantes
      scored_result = chain.invoke({"question": question, "context": docs})
      return "generate" if scored_result.binary_score == "yes" else "rewrite"
  ```

- **Nodo Rewrite**: Reformula consultas no exitosas
  ```python
  def rewrite(state):
      # Mejora la pregunta basándose en la intención semántica
      msg = HumanMessage(content=f"Formula una pregunta mejorada: {question}")
      response = model.invoke(msg)
      return {"messages": [response]}
  ```

- **Nodo Generate**: Crea respuesta final
  ```python
  def generate(state):
      prompt = hub.pull("rlm/rag-prompt")
      response = rag_chain.invoke({"context": docs, "question": question})
      return {"messages": [response]}
  ```

#### 3. **Aristas Condicionales**

```python
# Desde Agent: ¿Usar herramientas o terminar?
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "retrieve", END: END}
)

# Desde Retrieve: ¿Documentos relevantes?
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,  # Retorna "generate" o "rewrite"
)
```

### 📚 Notebooks del Módulo

#### 1. **1-agenticrag.ipynb**: RAG Agéntico Básico
- Construcción de grafo simple con StateGraph
- Nodos de recuperación y generación
- Flujo lineal: recuperar → generar

#### 2. **2-ReAct.ipynb**: Framework ReAct
- Implementación del patrón Reasoning + Acting
- Agente que decide qué herramientas usar
- Múltiples herramientas (RAG, Wikipedia, ArXiv)
- Herramientas personalizadas desde archivos de texto

#### 3. **3-AgenticRAG.ipynb**: Sistema Completo
- Múltiples vectorstores (LangGraph y LangChain)
- Evaluación de relevancia con LLM
- Reformulación automática de consultas
- Ciclos en el grafo para auto-corrección
- Decisiones inteligentes con aristas condicionales

### 🎯 Diferencias: RAG Tradicional vs RAG Agéntico

| Característica | RAG Tradicional | RAG Agéntico |
|----------------|-----------------|---------------|
| **Flujo** | Lineal (recuperar → generar) | Cíclico con decisiones |
| **Herramientas** | Una fuente de datos | Múltiples fuentes |
| **Evaluación** | No evalúa relevancia | Evalúa y decide |
| **Auto-corrección** | No | Sí (reformula consultas) |
| **Complejidad** | Baja | Alta |
| **Precisión** | Moderada | Alta |

### 🚀 Mejores Prácticas

1. **Usa Múltiples Fuentes de Conocimiento**
   ```python
   tools = [
       retriever_tool_docs,      # Documentación
       retriever_tool_research,  # Papers de investigación
       wiki_tool,                # Conocimiento general
   ]
   ```

2. **Implementa Evaluación de Relevancia**
   ```python
   class Grade(BaseModel):
       binary_score: str = Field(description="'yes' or 'no'")

   llm_grader = llm.with_structured_output(Grade)
   ```

3. **Reformula Consultas Fallidas**
   ```python
   if score == "no":
       # Mejora la consulta y reintenta
       improved_query = rewrite_query(original_query)
       return "rewrite"
   ```

4. **Limita Ciclos de Reformulación**
   ```python
   class State(TypedDict):
       messages: list
       retry_count: int  # Evita bucles infinitos

   def should_retry(state):
       return "rewrite" if state["retry_count"] < 3 else END
   ```

---

## 🧠 RAG Autónomo

El RAG Autónomo lleva los sistemas RAG un paso más allá al implementar **Chain-of-Thought (CoT)**, una técnica que descompone preguntas complejas en pasos de razonamiento intermedios. Este enfoque permite al sistema "pensar" antes de recuperar información, similar a cómo los humanos abordan problemas complejos.

### 🤔 ¿Qué es Chain-of-Thought (CoT) en RAG?

**Chain-of-Thought (CoT) RAG** es un sistema que descompone preguntas complejas en sub-problemas más manejables, recupera información relevante para cada sub-problema, y sintetiza una respuesta coherente considerando todo el razonamiento.

**Diferencia clave**: En lugar de una sola recuperación, CoT RAG realiza recuperación multi-paso guiada por razonamiento.

### 🔄 Flujo de CoT RAG

```
Usuario: "¿Cuáles son los experimentos adicionales en la evaluación de Transformers?"
    ↓
[PLANNER] Descompone en sub-pasos:
    1. "Identificar áreas clave de evaluación de Transformers"
    2. "Determinar categorías de experimentos adicionales"
    3. "Refinar y especificar experimentos por categoría"
    ↓
[RETRIEVER] Para cada sub-paso:
    - Busca documentos relevantes específicos
    - Acumula todos los documentos encontrados
    ↓
[RESPONDER] Sintetiza respuesta:
    - Combina contexto de todos los sub-pasos
    - Genera respuesta razonada y coherente
```

### 🎯 Arquitectura del Sistema

#### **Estado del Grafo**
```python
class RAGCoTState(BaseModel):
    question: str                    # Pregunta original compleja
    sub_steps: List[str] = []       # Sub-pasos de razonamiento
    retrieved_docs: List[Document]  # Documentos de todos los sub-pasos
    answer: str = ""                # Respuesta final sintetizada
```

#### **Nodo 1: Planner (Planificador)**
Descompone la pregunta compleja en 2-3 pasos de razonamiento:

```python
def plan_steps(state: RAGCoTState) -> RAGCoTState:
    prompt = f"Divide la pregunta en 2-3 pasos: {state.question}"
    result = llm.invoke(prompt).content
    sub_steps = [line.strip("- ") for line in result.split("\n") if line.strip()]
    return state.model_copy(update={"sub_steps": sub_steps})
```

**Ejemplo de salida**:
```
Pregunta: "¿Cómo optimizar Transformers para producción?"
Sub-pasos:
1. Identificar cuellos de botella de rendimiento en Transformers
2. Explorar técnicas de optimización (quantización, pruning, destilación)
3. Evaluar trade-offs entre velocidad y precisión
```

#### **Nodo 2: Retriever (Recuperador Multi-paso)**
Recupera documentos relevantes para cada sub-paso:

```python
def retrieve_per_step(state: RAGCoTState) -> RAGCoTState:
    all_docs = []
    for sub_step in state.sub_steps:
        docs = retriever.invoke(sub_step)  # Recuperación enfocada
        all_docs.extend(docs)
    return state.model_copy(update={"retrieved_docs": all_docs})
```

**Ventaja**: Cada sub-paso recupera documentos específicos, evitando ruido.

#### **Nodo 3: Responder (Sintetizador)**
Genera respuesta final considerando todo el razonamiento:

```python
def generate_answer(state: RAGCoTState) -> RAGCoTState:
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = f"""
Pregunta: {state.question}
Información Relevante: {context}
Sintetiza una respuesta bien razonada.
"""
    result = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"answer": result})
```

### 📊 CoT RAG vs RAG Tradicional vs RAG Agéntico

| Característica | RAG Tradicional | RAG Agéntico | CoT RAG |
|----------------|-----------------|---------------|---------|
| **Descomposición** | No | No | ✅ Sí (2-3 sub-pasos) |
| **Recuperación** | Una sola vez | Múltiple (con evaluación) | Multi-paso (guiada) |
| **Razonamiento** | No explícito | Evaluación de relevancia | ✅ Explícito paso a paso |
| **Ciclos** | No | Sí (reescritura) | No (lineal) |
| **Transparencia** | Baja | Media | ✅ Alta (muestra sub-pasos) |
| **Complejidad** | Baja | Alta | Media |
| **Mejor para** | Preguntas simples | Múltiples fuentes | ✅ Preguntas complejas |

### 🎯 Casos de Uso Ideales para CoT RAG

1. **Preguntas que requieren múltiples perspectivas**
   - ❌ RAG Tradicional: "¿Qué es un Transformer?" (simple)
   - ✅ CoT RAG: "¿Cómo han evolucionado los Transformers y qué optimizaciones existen para producción?"

2. **Análisis comparativo**
   - ✅ "Compare técnicas de optimización de Transformers: cuantización, pruning y destilación"

3. **Preguntas con múltiples sub-componentes**
   - ✅ "¿Cuáles son las mejores prácticas para implementar RAG: desde chunking hasta evaluación?"

4. **Investigación profunda**
   - ✅ "¿Qué experimentos se han realizado en evaluación de Transformers y cuáles son sus resultados?"

### 🚀 Mejores Prácticas

1. **Limita los Sub-pasos**
   ```python
   # ✅ Bueno - 2-3 pasos manejables
   prompt = "Divide en 2-3 pasos de razonamiento"

   # ❌ Malo - demasiados pasos
   prompt = "Divide en 10 pasos detallados"
   ```

2. **Valida la Calidad de Descomposición**
   ```python
   def plan_steps(state):
       result = llm.invoke(prompt).content
       sub_steps = parse_steps(result)

       # Validar que hay entre 2 y 4 pasos
       if len(sub_steps) < 2 or len(sub_steps) > 4:
           # Re-intentar o usar pregunta original
           pass
   ```

3. **Evita Redundancia en Recuperación**
   ```python
   # ✅ Bueno - elimina documentos duplicados
   all_docs = []
   seen_ids = set()
   for sub_step in sub_steps:
       docs = retriever.invoke(sub_step)
       for doc in docs:
           if doc.id not in seen_ids:
               all_docs.append(doc)
               seen_ids.add(doc.id)
   ```

4. **Muestra el Razonamiento al Usuario**
   ```python
   print("🪜 Pasos de Razonamiento:")
   for i, step in enumerate(final["sub_steps"], 1):
       print(f"{i}. {step}")
   print("\n✅ Respuesta Final:", final["answer"])
   ```

### 💡 Cuándo Usar CoT RAG

**✅ Usa CoT RAG cuando**:
- La pregunta es genuinamente compleja y multi-facética
- Necesitas transparencia en el razonamiento
- Quieres mejorar la calidad de respuestas para preguntas difíciles
- El usuario valora ver los pasos de pensamiento

**❌ No uses CoT RAG cuando**:
- La pregunta es simple y directa
- La latencia es crítica (CoT añade overhead)
- No necesitas explicabilidad
- El costo de llamadas LLM es una restricción

### 🔗 Flujo del Grafo

```
START → planner → retriever → responder → END
         ↓           ↓            ↓
    sub_steps   retrieved_docs  answer
```

**Características**:
- **Lineal**: No hay ciclos (a diferencia de RAG Agéntico)
- **Determinista**: Siempre ejecuta los 3 nodos en orden
- **Explicable**: Cada paso es visible y auditable

### 🔁 Auto-Reflexión en RAG

**Auto-reflexión** es una técnica donde el LLM evalúa su propia respuesta para determinar si es completa, precisa y satisfactoria. Si la respuesta no cumple con los estándares, el sistema puede refinar la consulta y recuperar información adicional.

**Combina**: Recuperación Iterativa + Auto-crítica

#### Flujo de Auto-Reflexión

```
Usuario: "¿Cuáles son las variantes de transformers en despliegues de producción?"
    ↓
[RETRIEVE] Recupera documentos
    ↓
[GENERATE] Genera respuesta inicial
    ↓
[REFLECT] Evalúa la respuesta:
    - ¿Es factualmente suficiente?
    - ¿Responde completamente la pregunta?
    ↓
SI: Respuesta aprobada → FIN
NO: Refina y vuelve a recuperar (máx 2 intentos)
```

#### Estado del Sistema

```python
class RAGReflectionState(BaseModel):
    question: str                    # Pregunta original
    retrieved_docs: List[Document]   # Documentos recuperados
    answer: str = ""                 # Respuesta generada
    reflection: str = ""             # Evaluación de la respuesta
    revised: bool = False            # ¿Necesita revisión?
    attempts: int = 0                # Contador de intentos
```

#### Nodos Clave

**1. Retrieve**: Recupera documentos del vector store
**2. Generate**: Genera respuesta basándose en el contexto
**3. Reflect**: Evalúa la calidad de la respuesta (LLM como juez)
**4. Finalize**: Marca el final del proceso

**Flujo del Grafo**:
```
START → retrieve → generate → reflect → [done o retrieve (si necesita revisión)]
                                ↓
                              END (si verificado o attempts >= 2)
```

**Ventajas**:
- ✅ Mejora la calidad de respuestas automáticamente
- ✅ Detecta respuestas incompletas o imprecisas
- ✅ Proceso de mejora iterativo y controlado
- ✅ Límite de intentos evita ciclos infinitos

### 🎯 Planificación y Descomposición de Consultas

**Query Planning and Decomposition** es una técnica que divide consultas complejas en sub-preguntas más simples, permitiendo recuperación más precisa y completa de información.

**Es como**: Ingeniería inversa de una pregunta en pasos manejables antes de responderla.

#### ¿Por qué es necesario?

En consultas complejas como:
> "Explica cómo funcionan los bucles de agentes y cuáles son los desafíos en la generación de video por difusión"

Esta pregunta tiene **dos componentes independientes**:
1. Bucles de agentes
2. Desafíos en generación de video por difusión

**Problema**: Una sola búsqueda vectorial puede no capturar ambos aspectos adecuadamente.

**Solución**: Descomponer en sub-preguntas y buscar cada una individualmente.

#### Arquitectura del Sistema

```python
class RAGState(BaseModel):
    question: str                    # Pregunta compleja original
    sub_questions: List[str] = []    # 2-3 sub-preguntas generadas
    retrieved_docs: List[Document]   # Todos los docs recuperados
    answer: str = ""                 # Respuesta final consolidada
```

#### Flujo de Trabajo

```
[PLANNER] Descompone la pregunta:
    "Pregunta compleja" → ["Sub-pregunta 1", "Sub-pregunta 2", "Sub-pregunta 3"]
    ↓
[RETRIEVER] Para cada sub-pregunta:
    - Busca documentos específicos
    - Acumula todos los documentos
    ↓
[RESPONDER] Sintetiza respuesta final:
    - Combina contexto de todas las sub-preguntas
    - Genera respuesta coherente y completa
```

#### Nodos del Grafo

**1. Planner**: Divide pregunta en 2-3 sub-preguntas usando LLM
**2. Retriever**: Recupera documentos para cada sub-pregunta
**3. Responder**: Sintetiza respuesta final consolidada

**Flujo Secuencial**:
```
START → planner → retriever → responder → END
```

**Ventajas**:
- ✅ Mejor cobertura para preguntas multifacéticas
- ✅ Recuperación más precisa y completa
- ✅ Reduce el ruido en documentos recuperados
- ✅ Razonamiento paso a paso más claro

**Casos de Uso Ideales**:
- Preguntas con múltiples temas
- Consultas que requieren información de diferentes dominios
- Análisis comparativos o multi-dimensionales

### 🔄 Recuperación Iterativa

**Iterative Retrieval** combina recuperación iterativa con auto-reflexión en un ciclo de retroalimentación continua. El sistema no se conforma con la primera recuperación; evalúa, refina y vuelve a buscar hasta obtener información suficiente.

**Diferencia clave**: Similar a Auto-reflexión, pero con **refinamiento de consulta** cuando la respuesta es insuficiente.

#### ¿Cómo Funciona?

```
Usuario: "bucles de agentes y sistemas basados en transformers?"
    ↓
[RETRIEVE] Recupera con pregunta original
    ↓
[ANSWER] Genera respuesta
    ↓
[REFLECT] Evalúa calidad
    ↓
¿Verificada? NO → [REFINE] Mejora la consulta
    ↓                       ↓
   SÍ                  Vuelve a RETRIEVE
    ↓
  END
```

#### Estado del Sistema

```python
class IterativeRAGState(BaseModel):
    question: str                    # Pregunta original
    refined_question: str = ""       # Versión refinada de la consulta
    retrieved_docs: List[Document]   # Documentos recuperados
    answer: str = ""                 # Respuesta generada
    verified: bool = False           # ¿Respuesta verificada?
    attempts: int = 0                # Contador de iteraciones
```

#### Nodos del Grafo

**1. Retrieve**: Usa pregunta refinada (si existe) o la original
**2. Answer**: Genera respuesta e incrementa contador
**3. Reflect**: Evalúa si la respuesta es suficiente
**4. Refine**: Genera versión mejorada de la consulta

**Ciclo Iterativo**:
```
START → retrieve → answer → reflect → refine → retrieve (ciclo)
                               ↓
                              END (si verificado o attempts >= 2)
```

**El "Ciclo Mágico"**:
- Si `verified=True` O `attempts>=2` → END
- Si `verified=False` Y `attempts<2` → refine → retrieve (reintentar)

**Ventajas**:
- ✅ Mejora automática de consultas vagas o mal formuladas
- ✅ Recuperación adaptativa basada en resultados previos
- ✅ Similar a cómo un investigador humano refina búsquedas
- ✅ Maximiza calidad de respuesta dentro de límites de iteraciones

**Casos de Uso**:
- Consultas inicialmente vagas o imprecisas
- Cuando la primera recuperación no proporciona contexto suficiente
- Temas que requieren refinamiento progresivo de búsqueda

### 🎨 Síntesis de Respuestas desde Múltiples Fuentes

**Answer Synthesis from Multiple Sources** es el proceso donde un agente de IA recopila información de diferentes herramientas de recuperación o bases de conocimiento, y las fusiona en una única respuesta coherente y contextualmente rica.

**Capacidad fundamental** en RAG Agéntico: El sistema no solo recupera, sino que **planifica, recupera de múltiples fuentes, y sintetiza**.

#### ¿Por qué es Necesario?

La mayoría de consultas del mundo real son:
- **Multifacéticas**: Requieren múltiples tipos de información
- **Ambiguas**: Necesitan refinamiento contextual
- **Abiertas**: No se mapean a un solo documento

**Limitación de RAG tradicional**: Una sola base de datos vectorial es insuficiente.

**Solución**: Agente que recupera de múltiples fuentes y sintetiza.

#### Fuentes de Información

Este sistema integra **4 fuentes diferentes**:

1. **📄 Documentos Internos** (Vector Store local)
   - Archivos de texto propios de la organización
   - Documentación interna y privada

2. **🎥 YouTube** (Transcripciones)
   - Contenido multimedia/educativo
   - Explicaciones conceptuales en video

3. **🌐 Wikipedia** (API pública)
   - Conocimiento enciclopédico general
   - Definiciones y contexto amplio

4. **📚 ArXiv** (Papers académicos)
   - Investigación científica actualizada
   - Papers y publicaciones académicas

#### Arquitectura del Sistema

```python
class MultiSourceRAGState(BaseModel):
    question: str                    # Pregunta del usuario
    text_docs: List[Document] = []   # Docs internos
    yt_docs: List[Document] = []     # Transcripciones YouTube
    wiki_context: str = ""           # Contenido Wikipedia
    arxiv_context: str = ""          # Papers ArXiv
    final_answer: str = ""           # Respuesta sintetizada
```

#### Flujo de Trabajo

```
Usuario: "¿Qué son los agentes transformers y cómo están evolucionando?"
    ↓
[RETRIEVE TEXT] Documentos internos sobre transformers
    ↓
[RETRIEVE YOUTUBE] Videos explicativos sobre agentes
    ↓
[RETRIEVE WIKIPEDIA] Artículos sobre transformers y agentes
    ↓
[RETRIEVE ARXIV] Papers recientes sobre transformer agents
    ↓
[SYNTHESIZE] Combina toda la información:
    - Organiza contexto por fuente
    - Genera respuesta que integra todas las perspectivas
    - Proporciona visión completa y multidimensional
```

#### Nodos del Grafo

**Nodos de Recuperación** (4):
1. `retrieve_text`: Documentación interna
2. `retrieve_yt`: Contenido multimedia
3. `retrieve_wikipedia`: Conocimiento enciclopédico
4. `retrieve_arxiv`: Investigación científica

**Nodo de Síntesis** (1):
5. `synthesize`: Fusiona toda la información en respuesta coherente

**Flujo Secuencial**:
```
START → retrieve_text → retrieve_yt → retrieve_wiki → retrieve_arxiv → synthesize → END
```

#### Proceso de Síntesis

El nodo de síntesis realiza:

1. **Organización del Contexto**:
```python
context = """
[Documentos Internos]
<contenido de text_docs>

[Transcripción de YouTube]
<contenido de yt_docs>

[Wikipedia]
<wiki_context>

[ArXiv]
<arxiv_context>
"""
```

2. **Prompt de Síntesis**:
```python
prompt = f"""Has recuperado contexto de múltiples fuentes.
Sintetiza una respuesta completa y coherente.

Pregunta: {question}
Contexto: {context}
"""
```

3. **Generación Unificada**: El LLM analiza toda la información y genera una respuesta que:
   - Combina perspectivas de todas las fuentes
   - Identifica patrones comunes
   - Resuelve contradicciones
   - Proporciona respuesta rica y completa

#### Ventajas

- ✅ **Cobertura Completa**: Información de múltiples dominios
- ✅ **Perspectivas Diversas**: Documentación interna + conocimiento público + investigación
- ✅ **Actualización**: Combina conocimiento histórico (Wikipedia) con investigación reciente (ArXiv)
- ✅ **Flexibilidad**: Fácil agregar/quitar fuentes según necesidad
- ✅ **Robustez**: Si una fuente falla, otras compensan

#### Casos de Uso Ideales

1. **Investigación Profunda**:
   - "¿Qué son los agentes transformers y cómo están evolucionando en la investigación reciente?"

2. **Análisis Multidimensional**:
   - Combinar documentación interna + papers académicos + explicaciones públicas

3. **Verificación Cruzada**:
   - Contrastar información de múltiples fuentes para mayor confiabilidad

4. **Síntesis de Conocimiento**:
   - Generar respuestas que integran múltiples perspectivas y tipos de información

#### Mejores Prácticas

1. **Orden de Fuentes**: Prioriza fuentes más específicas primero (docs internos) antes que generales (Wikipedia)

2. **Manejo de Errores**: Implementa fallbacks si alguna fuente falla
```python
try:
    wiki_context = wikipedia_search(query)
except Exception:
    wiki_context = "Wikipedia no disponible"
```

3. **Limitación de Contenido**: Para evitar context window overflow, limita documentos por fuente
```python
arxiv_results = arxiv_loader.load()[:2]  # Solo primeros 2 papers
```

4. **Identificación de Fuentes**: Marca claramente cada fuente en el contexto para trazabilidad

5. **Deduplicación**: Elimina información redundante entre fuentes

#### Cuándo Usar Multi-Source RAG

**✅ Usa Multi-Source cuando**:
- Necesitas cobertura completa de un tema
- La pregunta requiere múltiples tipos de información
- Quieres contrastar información de diferentes fuentes
- Necesitas combinar conocimiento interno y externo

**❌ No uses Multi-Source cuando**:
- La pregunta es simple y una fuente es suficiente
- La latencia es crítica (múltiples fuentes añaden tiempo)
- Costos de API son restricción (más llamadas = más costo)
- Solo tienes una fuente de información confiable

---

## 🤖 Sistemas RAG Multi-Agente

Los **Sistemas RAG Multi-Agente** representan la evolución más avanzada de RAG, donde el pipeline se divide en múltiples agentes especializados que colaboran para resolver tareas complejas. Cada agente tiene un rol específico y herramientas dedicadas, permitiendo una división inteligente del trabajo.

### 🎯 ¿Qué son los Sistemas Multi-Agente?

Un Sistema RAG Multi-Agente divide el pipeline RAG en múltiples agentes especializados — cada uno responsable de un rol específico — y les permite **colaborar** en una sola consulta o tarea.

**Diferencia clave con RAG tradicional**:
- **RAG Tradicional**: Un solo agente hace todo (recuperación + generación)
- **RAG Multi-Agente**: Múltiples agentes especializados colaboran, cada uno con su expertise

### 📊 Tres Arquitecturas Multi-Agente

Este módulo cubre tres arquitecturas progresivamente más complejas:

#### 1️⃣ **Sistema Multi-Agente Colaborativo Básico**

**Arquitectura**: Dos agentes que se pasan trabajo entre sí.

```
Usuario: "Escribe un blog sobre transformers"
    ↓
[RESEARCHER] Busca información:
    - Usa búsqueda web (Tavily)
    - Consulta documentos internos (FAISS)
    - Recopila datos relevantes
    ↓
[BLOG GENERATOR] Escribe contenido:
    - Recibe investigación del researcher
    - Genera blog detallado y estructurado
    - Añade "FINAL ANSWER" al terminar
    ↓
RESULTADO: Blog completo sobre transformers
```

**Agentes**:
- **Researcher**: Especializado en búsqueda y recuperación de información
  - Herramientas: `internal_tool_1` (FAISS), `tavily_tool` (búsqueda web)
- **Blog Generator**: Especializado en escritura de contenido
  - Herramientas: Ninguna (solo procesa y escribe)

**Patrón de Terminación**: Cualquier agente puede añadir "FINAL ANSWER" para indicar que el trabajo está completo.

**Flujo**:
```
START → researcher → blog_generator → END (si "FINAL ANSWER")
                          ↓
                    researcher (si necesita más info)
```

#### 2️⃣ **Supervisor Multi-Agente con RAG**

**Arquitectura**: Un supervisor central coordina agentes especializados.

```
Usuario: "Lista transformers del retriever y calcula 5 + 10"
    ↓
[SUPERVISOR] Analiza y delega:
    - Identifica dos tareas diferentes
    - Decide qué agente usar para cada una
    ↓
[RESEARCH AGENT] Tarea 1:
    - Busca en documentos internos
    - Lista variantes de transformers
    - Reporta al supervisor
    ↓
[MATH AGENT] Tarea 2:
    - Usa herramientas matemáticas
    - Calcula 5 + 10 = 15
    - Reporta al supervisor
    ↓
[SUPERVISOR] Consolida:
    - Combina resultados de ambos agentes
    - Genera respuesta final unificada
```

**Componentes**:
- **Supervisor**: Agente de coordinación que:
  - Analiza la consulta del usuario
  - Decide qué agente usar para cada tarea
  - Consolida respuestas
  - **Regla importante**: Delega a un agente a la vez (no en paralelo)

- **Research Agent**:
  - Herramientas: `web_search` (Tavily), `internal_tool_1` (docs internos)
  - Restricción: **SOLO investigación**, NO matemáticas

- **Math Agent**:
  - Herramientas: `add()`, `multiply()`, `divide()`
  - Restricción: **SOLO matemáticas**, NO investigación

**Librería**: Usa `langgraph_supervisor.create_supervisor()` pre-construido.

**Ventajas**:
- ✅ Especialización clara de agentes
- ✅ Delegación inteligente de tareas
- ✅ Fácil agregar nuevos agentes especializados
- ✅ Supervisor maneja la coordinación automáticamente

#### 3️⃣ **Equipos Jerárquicos de Agentes con RAG**

**Arquitectura**: Jerarquía de 3 niveles con equipos completos y supervisores anidados.

```
[SUPERVISOR DE EQUIPOS] (Nivel Superior)
    ↓
    ├─→ [EQUIPO DE INVESTIGACIÓN] (Nivel Medio)
    │       [Supervisor de Investigación]
    │           ↓
    │           ├─→ [Search Agent]: Búsqueda general (Tavily + docs)
    │           └─→ [Web Scraper Agent]: Scraping de URLs específicas
    │
    └─→ [EQUIPO DE ESCRITURA] (Nivel Medio)
            [Supervisor de Escritura]
                ↓
                ├─→ [Note Taker]: Crea esquemas/outlines
                ├─→ [Doc Writer]: Escribe documentos completos
                └─→ [Chart Generator]: Crea visualizaciones con Python
```

**Flujo de Trabajo Completo**:

```
Usuario: "Escribe sobre transformers en producción"
    ↓
1. [SUPERVISOR DE EQUIPOS] Analiza y delega a EQUIPO DE INVESTIGACIÓN
    ↓
2. [SUPERVISOR DE INVESTIGACIÓN] Coordina búsqueda:
   2.1. [SEARCH AGENT] busca información general
   2.2. [WEB SCRAPER AGENT] obtiene contenido detallado de páginas específicas
   2.3. Reportan resultados al supervisor de investigación
   2.4. Supervisor reporta al supervisor de equipos
    ↓
3. [SUPERVISOR DE EQUIPOS] Delega a EQUIPO DE ESCRITURA
    ↓
4. [SUPERVISOR DE ESCRITURA] Coordina creación:
   4.1. [NOTE TAKER] crea outline del documento
   4.2. [DOC WRITER] escribe documento completo basándose en outline
   4.3. [CHART GENERATOR] (opcional) crea gráficos si es necesario
   4.4. Reportan al supervisor de escritura
   4.5. Supervisor reporta al supervisor de equipos
    ↓
5. [SUPERVISOR DE EQUIPOS] Confirma finalización
```

**Herramientas Avanzadas**:

**Equipo de Investigación**:
- `tavily_tool`: Búsqueda web optimizada para IA (máx 5 resultados)
- `internal_tool_1`: Recuperación vectorial con FAISS de docs internos
- `scrape_webpages`: Scraping web con BeautifulSoup para contenido detallado

**Equipo de Escritura**:
- `create_outline`: Crea esquemas numerados de documentos
- `write_document`: Escribe documentos completos
- `edit_document`: Inserta texto en líneas específicas
- `read_document`: Lee documentos (completos o rangos de líneas)
- `python_repl_tool`: Ejecuta código Python para crear visualizaciones

**Directorio de Trabajo**: Usa `TemporaryDirectory` para gestión automática de archivos.

### 🎯 Comparación de Arquitecturas

| Característica | Colaborativo Básico | Supervisor | Jerárquico |
|----------------|---------------------|------------|------------|
| **Niveles** | 1 nivel | 2 niveles | 3 niveles |
| **Agentes** | 2 agentes | 2+ agentes | 6+ agentes |
| **Coordinación** | Auto-coordinación | Supervisor central | Supervisores anidados |
| **Complejidad** | Baja | Media | Alta |
| **Escalabilidad** | Limitada | Buena | Excelente |
| **Especialización** | Básica | Alta | Muy alta |
| **Mejor para** | Tareas simples | Tareas múltiples | Proyectos complejos |

### 💡 Casos de Uso por Arquitectura

#### **Colaborativo Básico**
✅ Usa cuando:
- Solo necesitas 2-3 agentes
- La tarea tiene flujo lineal simple
- No hay muchas decisiones de enrutamiento

**Ejemplo**: Investigar + escribir blog

#### **Supervisor**
✅ Usa cuando:
- Necesitas múltiples agentes especializados
- Las tareas son claramente separables
- Quieres delegación inteligente automática

**Ejemplo**: Combinar investigación + cálculos + análisis

#### **Jerárquico**
✅ Usa cuando:
- El proyecto es muy complejo
- Necesitas equipos completos trabajando juntos
- Cada equipo tiene múltiples subagentes
- Quieres máxima escalabilidad

**Ejemplo**: Investigar en múltiples fuentes + crear documentos complejos + generar visualizaciones

### 🔧 Componentes Clave en Multi-Agente

#### **1. Estado (State)**
```python
class State(MessagesState):
    next: str  # Tracking del siguiente nodo
```
- Mantiene historial de mensajes entre agentes
- Puede extenderse con campos adicionales

#### **2. Nodos (Nodes)**
Cada nodo es una función que:
- Recibe el estado actual
- Ejecuta un agente específico
- Retorna `Command` con actualización y navegación

#### **3. Comando (Command)**
```python
return Command(
    update={"messages": [...]},  # Actualiza estado
    goto="next_node"              # Navega al siguiente nodo
)
```

#### **4. Supervisor Pattern**
```python
def make_supervisor_node(llm, members):
    # Crea supervisor que decide: "¿A quién delego?"
    # Usa LLM con structured output para decisiones
```

#### **5. Herramientas (Tools)**
- Funciones Python decoradas con `@tool`
- Descripciones claras para que el LLM sepa cuándo usarlas
- Type hints con `Annotated` para documentación

### 🚀 Mejores Prácticas

#### **1. Diseño de Agentes**
```python
# ✅ Bueno - Agente con rol y restricciones claras
research_agent = create_react_agent(
    llm,
    tools=[web_search, internal_docs],
    prompt=(
        "Eres un agente de investigación.\n"
        "SOLO asiste con investigación.\n"
        "NO hagas matemáticas.\n"
        "Reporta al supervisor cuando termines."
    )
)

# ❌ Malo - Agente con rol ambiguo
general_agent = create_react_agent(
    llm,
    tools=[everything],
    prompt="Haz lo que sea necesario"
)
```

#### **2. Delegación Secuencial**
```python
# ✅ Bueno - Un agente a la vez
prompt = "Asigna trabajo a un agente a la vez, no en paralelo."

# ❌ Malo - Llamadas paralelas sin coordinación
# Puede causar conflictos y duplicación de trabajo
```

#### **3. Mensajes de Reporte**
```python
# ✅ Bueno - Identifica claramente el origen
HumanMessage(
    content=result["messages"][-1].content,
    name="research_agent"  # Identifica quién responde
)
```

#### **4. Manejo de Errores en Herramientas**
```python
@tool
def python_repl_tool(code: str):
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Fallo al ejecutar. Error: {repr(e)}"
    return f"Ejecutado exitosamente:\n{result}"
```

### ⚠️ Consideraciones Importantes

#### **Costos**
- Sistemas multi-agente hacen **múltiples llamadas al LLM**
- Supervisor también usa el LLM para decisiones
- Usa modelos económicos como GPT-4o-mini

#### **Latencia**
- Cada agente añade tiempo de procesamiento
- Supervisores añaden overhead de decisión
- Considera si la complejidad justifica el tiempo

#### **Complejidad**
- Más agentes = más difícil de depurar
- Usa LangGraph Studio para visualización
- Implementa logging detallado

#### **Coordinación**
- Define claramente roles y restricciones
- Evita solapamiento de responsabilidades
- Documenta el flujo esperado

### 🎯 Cuándo Usar Multi-Agente RAG

**✅ Usa Multi-Agente cuando**:
- La tarea requiere múltiples especialidades (investigación + escritura + análisis)
- Necesitas dividir trabajo complejo en subtareas manejables
- Quieres agentes reutilizables con roles claros
- La calidad justifica el costo adicional
- Necesitas escalabilidad y mantenibilidad

**❌ No uses Multi-Agente cuando**:
- La tarea es simple y un solo agente basta
- Los costos de API son prohibitivos
- La latencia es crítica
- El overhead de coordinación no vale la pena
- No hay beneficio claro de la especialización

### 📚 Recursos y Referencias

- **LangGraph**: Framework para construir grafos de agentes
- **langgraph_supervisor**: Librería pre-construida para supervisores
- **create_react_agent**: Constructor de agentes ReAct
- **Command Pattern**: Para navegación y actualización de estado
- **Structured Output**: Para decisiones supervisores con LLM

---

## 🔧 RAG Correctivo (Corrective RAG)

**Corrective RAG (CRAG)** es una técnica avanzada que mejora la calidad y confiabilidad de sistemas RAG mediante la **evaluación automática de documentos recuperados** y la **corrección adaptativa** del flujo de recuperación. A diferencia del RAG tradicional que asume que todos los documentos recuperados son relevantes, CRAG evalúa cada documento y toma decisiones inteligentes sobre cómo proceder.

### 🎯 ¿Qué es Corrective RAG?

CRAG introduce un **ciclo de retroalimentación inteligente** en el pipeline RAG:

```
Usuario: "¿Qué es la memoria de agentes?"
    ↓
1. [RECUPERAR] Busca en vectorstore local (FAISS)
    ↓
2. [EVALUAR] LLM califica cada documento: ¿Es relevante?
    ↓
    ├─→ [SI RELEVANTE] → Genera respuesta directamente
    │
    └─→ [NO RELEVANTE] → Reescribe consulta → Busca en web → Genera respuesta
```

**Diferencia clave con RAG tradicional**:
- **RAG Tradicional**: Recupera → Genera (asume que los documentos son relevantes)
- **Corrective RAG**: Recupera → **Evalúa** → Corrige si es necesario → Genera

### 🧩 Componentes de CRAG

#### 1️⃣ **Evaluador de Relevancia (Retrieval Grader)**

Un LLM especializado que califica documentos con puntuación binaria (yes/no).

```python
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documentos son relevantes a la pregunta, 'yes' o 'no'"
    )

# LLM con salida estructurada usando Pydantic
structured_llm_grader = llm.with_structured_output(GradeDocuments)
```

**¿Cómo funciona?**
- Recibe: pregunta del usuario + contenido del documento
- Analiza: similitud semántica y presencia de keywords
- Retorna: `{"binary_score": "yes"}` o `{"binary_score": "no"}`

**Ventajas de salida estructurada**:
- ✅ Respuestas determinísticas y parseables
- ✅ No necesita parsear texto libre
- ✅ Validación automática con Pydantic
- ✅ Integración directa en lógica condicional

#### 2️⃣ **Reescritor de Consultas (Query Rewriter)**

Optimiza consultas cuando los documentos locales no son suficientes.

```python
system = """Eres un reescritor de preguntas que convierte una pregunta
de entrada en una mejor versión optimizada para búsqueda web."""

question_rewriter = re_write_prompt | llm | StrOutputParser()
```

**Transformaciones típicas**:
- Vaga: "memoria de agentes" → Precisa: "¿Cuál es el rol de la memoria en agentes de IA?"
- Ambigua: "transformers" → Contextual: "arquitectura transformer en deep learning"
- Técnica: "RAG chunking" → Buscable: "mejores prácticas para dividir documentos en RAG"

**Por qué es importante**:
- Los vectorstores locales pueden no tener información actualizada
- Las preguntas mal formuladas obtienen resultados pobres
- La web tiene información más amplia que requiere consultas optimizadas

#### 3️⃣ **Búsqueda Web Adaptativa (Tavily Integration)**

Cuando los documentos locales fallan, CRAG busca en la web.

```python
web_search_tool = TavilySearchResults(k=3)  # Top 3 resultados

# Integra resultados web con documentos locales
docs = web_search_tool.invoke({"query": better_question})
web_results = "\n".join([d["content"] for d in docs])
documents.append(Document(page_content=web_results))
```

**Tavily vs Google/Bing**:
- ✅ Optimizado para agentes de IA (respuestas estructuradas)
- ✅ Sin límites de rate estrictos
- ✅ Resultados limpios sin ads
- ✅ API simple con respuestas JSON

#### 4️⃣ **Flujo de Decisión con LangGraph**

El cerebro del sistema que toma decisiones basándose en la evaluación.

```python
def decide_to_generate(state):
    """Decide: ¿Generar directamente o buscar en web?"""
    web_search = state["web_search"]

    if web_search == "Yes":
        # Documentos no relevantes → transformar consulta
        return "transform_query"
    else:
        # Documentos relevantes → generar respuesta
        return "generate"

# Arista condicional en el grafo
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)
```

### 📊 Arquitectura del Flujo CRAG

```
START
  ↓
[retrieve]
  Recupera documentos del vectorstore local (FAISS)
  Retorna: {"documents": [...], "question": "..."}
  ↓
[grade_documents]
  Por cada documento:
    - Evalúa relevancia con LLM
    - Si relevante: agrega a filtered_docs
    - Si no relevante: marca web_search = "Yes"
  Retorna: {"documents": filtered_docs, "web_search": "Yes/No"}
  ↓
[decide_to_generate] ← DECISIÓN
  ↓
  ├─→ [SI web_search == "No"] → [generate] → END
  │     Documentos relevantes encontrados
  │     Genera respuesta directamente
  │
  └─→ [SI web_search == "Yes"] → [transform_query]
        Documentos no relevantes
        ↓
      [transform_query]
        Reescribe la pregunta para búsqueda web
        Retorna: {"question": "mejor_pregunta"}
        ↓
      [web_search_node]
        Busca en web usando Tavily
        Agrega resultados a documents
        Retorna: {"documents": [...incluye web]}
        ↓
      [generate] → END
        Genera respuesta con documentos web
```

### 🎯 Estado del Grafo (GraphState)

```python
class GraphState(TypedDict):
    question: str       # Pregunta original o reescrita
    generation: str     # Respuesta generada final
    web_search: str     # "Yes" o "No" - necesita búsqueda web
    documents: List[str]  # Documentos locales + web (si aplica)
```

**Flujo de datos**:
1. **Entrada**: `{"question": "memoria de agentes"}`
2. **Después de retrieve**: `{"question": "...", "documents": [...]}`
3. **Después de grade**: `{"...", "web_search": "Yes", "documents": filtered}`
4. **Después de transform**: `{"question": "mejor pregunta", ...}`
5. **Después de generate**: `{"...", "generation": "respuesta final"}`

### 💡 Casos de Uso de CRAG

#### **✅ Usa CRAG cuando**:
- Tu vectorstore tiene información limitada o desactualizada
- Las preguntas de usuarios son impredecibles o mal formuladas
- Necesitas alta confiabilidad (evitar alucinaciones)
- Quieres combinar conocimiento local + web automáticamente
- La calidad de respuestas es crítica (mejor que velocidad)

**Ejemplos**:
- **Soporte técnico**: Base de conocimiento interna + Stack Overflow
- **Investigación**: Papers locales + búsqueda académica web
- **E-commerce**: Catálogo interno + reviews web
- **Legal/Compliance**: Documentos corporativos + regulaciones públicas

#### **❌ No uses CRAG cuando**:
- Tu vectorstore es completo y siempre tiene respuestas
- La latencia es crítica (CRAG añade evaluación + posible búsqueda web)
- Los costos de API son prohibitivos (evaluación = llamada LLM extra)
- Las consultas siempre son relevantes a tus documentos
- No tienes acceso a búsqueda web (Tavily API)

### 🔧 Implementación Paso a Paso

#### **Paso 1: Construir Vectorstore**
```python
# Cargar documentos web
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Crear vectorstore FAISS
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
```

#### **Paso 2: Crear Evaluador**
```python
# Modelo Pydantic para salida estructurada
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' o 'no'")

# LLM con structured output
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt de evaluación
system = """Eres un evaluador que analiza relevancia de documentos.
Si el documento contiene keywords o significado semántico relacionado
con la pregunta, califícalo como relevante. Da 'yes' o 'no'."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Documento: {document}\nPregunta: {question}")
])

retrieval_grader = grade_prompt | structured_llm_grader
```

#### **Paso 3: Definir Nodos del Grafo**
```python
def retrieve(state):
    """Recuperar documentos del vectorstore"""
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}

def grade_documents(state):
    """Evaluar relevancia de cada documento"""
    filtered_docs = []
    web_search = "No"

    for doc in state["documents"]:
        score = retrieval_grader.invoke({
            "question": state["question"],
            "document": doc.page_content
        })
        if score.binary_score == "yes":
            filtered_docs.append(doc)
        else:
            web_search = "Yes"

    return {
        "documents": filtered_docs,
        "question": state["question"],
        "web_search": web_search
    }

def transform_query(state):
    """Reescribir pregunta para búsqueda web"""
    better_question = question_rewriter.invoke({
        "question": state["question"]
    })
    return {"documents": state["documents"], "question": better_question}

def web_search(state):
    """Buscar en web usando Tavily"""
    docs = web_search_tool.invoke({"query": state["question"]})
    web_results = "\n".join([d["content"] for d in docs])
    state["documents"].append(Document(page_content=web_results))
    return {"documents": state["documents"], "question": state["question"]}

def generate(state):
    """Generar respuesta final"""
    generation = rag_chain.invoke({
        "context": state["documents"],
        "question": state["question"]
    })
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": generation
    }
```

#### **Paso 4: Construir el Grafo**
```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(GraphState)

# Agregar nodos
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

# Agregar aristas
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Arista condicional: decide según relevancia
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)

workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compilar
app = workflow.compile()
```

#### **Paso 5: Ejecutar el Flujo**
```python
# Ejecutar con una pregunta
result = app.invoke({
    "question": "What are the types of agent memory?"
})

print(result["generation"])
# Output: "Los tipos de memoria de agentes son: memoria a corto plazo
#          (in-context learning) y memoria a largo plazo (almacenamiento
#          externo con recuperación basada en relevancia, recencia e
#          importancia)..."
```

### 🚀 Optimizaciones y Mejores Prácticas

#### **1. Threshold de Relevancia**
```python
# Ajustar sensibilidad del evaluador
if len(filtered_docs) >= 2:  # Al menos 2 docs relevantes
    web_search = "No"  # No buscar en web
else:
    web_search = "Yes"  # Buscar aunque haya 1 doc relevante
```

#### **2. Cache de Evaluaciones**
```python
# Evitar evaluar el mismo documento múltiples veces
evaluation_cache = {}

def grade_documents_cached(state):
    for doc in state["documents"]:
        doc_hash = hash(doc.page_content)
        if doc_hash not in evaluation_cache:
            evaluation_cache[doc_hash] = retrieval_grader.invoke(...)
        score = evaluation_cache[doc_hash]
```

#### **3. Fallback Strategy**
```python
# Si web search también falla, usar respuesta genérica
def generate(state):
    if not state["documents"]:
        return {
            "generation": "No encontré información relevante. "
                          "¿Puedes reformular tu pregunta?"
        }
    # Generar normalmente...
```

#### **4. Logging y Observabilidad**
```python
def grade_documents(state):
    print(f"---EVALUANDO {len(state['documents'])} DOCUMENTOS---")

    for i, doc in enumerate(state["documents"]):
        score = retrieval_grader.invoke(...)
        if score.binary_score == "yes":
            print(f"  ✓ Doc {i+1}: RELEVANTE")
            filtered_docs.append(doc)
        else:
            print(f"  ✗ Doc {i+1}: NO RELEVANTE")
            web_search = "Yes"

    print(f"---RESULTADO: {len(filtered_docs)} relevantes, "
          f"búsqueda web = {web_search}---")
```

### ⚠️ Consideraciones Importantes

#### **Costos**
- Cada documento evaluado = 1 llamada LLM adicional
- 4 documentos recuperados = 4 evaluaciones = costo significativo
- Usa modelos económicos (gpt-3.5-turbo) para evaluación
- Considera evaluar solo top-k documentos (e.g., top 2)

#### **Latencia**
- Evaluación añade ~1-2 segundos por documento
- Búsqueda web añade ~2-3 segundos adicionales
- Total: puede ser 5-10 segundos vs 2 segundos de RAG tradicional
- Considera evaluación paralela si tienes muchos documentos

#### **Precisión del Evaluador**
- El evaluador puede cometer errores (falsos positivos/negativos)
- Usa temperature=0 para consistencia
- Considera usar un modelo más potente (GPT-4) para evaluación crítica
- Evalúa el evaluador periódicamente con ground truth

#### **Dependencia de Búsqueda Web**
- Requiere Tavily API key (servicio de pago)
- Tiene límites de rate (considera caching)
- Búsquedas web pueden retornar información desactualizada o incorrecta
- Considera implementar filtrado adicional de resultados web

### 📊 Métricas de Éxito

**Comparación CRAG vs RAG Tradicional**:

| Métrica | RAG Tradicional | CRAG |
|---------|----------------|------|
| **Precisión** | 70-80% | 85-95% |
| **Tasa de alucinación** | 15-25% | 5-10% |
| **Latencia promedio** | 2s | 6s |
| **Costo por consulta** | $0.002 | $0.008 |
| **Cobertura** | Solo docs locales | Local + Web |
| **Adaptabilidad** | Baja | Alta |

### 🎯 Cuándo Usar CRAG vs Alternativas

**CRAG** es ideal para:
- Sistemas donde la precisión es más importante que la velocidad
- Dominios donde el conocimiento local es limitado
- Aplicaciones que requieren información actualizada
- Casos donde las alucinaciones son inaceptables

**Alternativas**:
- **RAG Tradicional**: Cuando velocidad > precisión
- **RAG Agéntico**: Cuando necesitas razonamiento complejo
- **Self-RAG**: Cuando necesitas auto-reflexión iterativa
- **Adaptive RAG**: Cuando necesitas enrutamiento multi-fuente

### 📚 Recursos y Referencias

- **Paper Original**: "Corrective Retrieval Augmented Generation" (CRAG)
- **LangGraph**: Framework para grafos de estado con nodos condicionales
- **Tavily API**: Motor de búsqueda optimizado para IA
- **Structured Output**: `with_structured_output()` en LangChain
- **Pydantic**: Validación de datos y esquemas

---

## 🎯 RAG Adaptativo (Adaptive RAG)

**Adaptive RAG** es el patrón más completo y robusto de RAG, que combina **enrutamiento inteligente**, **evaluación multi-nivel** y **auto-corrección automática** en un solo flujo adaptativo. Este es el enfoque ideal para aplicaciones de producción donde la calidad y confiabilidad son críticas.

### 🎯 ¿Qué es Adaptive RAG?

Adaptive RAG toma decisiones inteligentes en **múltiples puntos** del flujo, adaptándose dinámicamente según la calidad de los documentos y respuestas:

```
INICIO → Router Inteligente
    ↓
    ¿Vectorstore o Web?
    ↓
┌───┴───┐
│  WEB  │ → Generar → Validar → ✓ FIN
└───────┘
    │
┌───┴────┐
│ VECTOR │ → Evaluar Docs → ¿Relevantes?
└────────┘        ↓
              SÍ ↓   NO↓
                 ↓    Reescribir → Reintentar
           Generar
                 ↓
           Validar → ¿Alucinaciones?
                 ↓
              NO ↓   SÍ↓
                 ↓    Regenerar
           ¿Contesta pregunta?
                 ↓
              SÍ ↓   NO↓
                 ↓    Reescribir → Reintentar
                 ✓
               FIN
```

**Diferencias clave**:
- **RAG Tradicional**: Recupera → Genera (sin validación)
- **CRAG**: Recupera → Evalúa → Corrige si es necesario → Genera
- **Adaptive RAG**: **Enruta** → Recupera → **Evalúa** → Genera → **Valida Alucinaciones** → **Valida Respuesta** → **Auto-Corrige** hasta éxito

### 🧩 Componentes de Adaptive RAG

#### 1️⃣ **Router (Enrutador Inteligente)**

El primer componente que decide la fuente de datos ANTES de recuperar.

```python
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(...)

# Router LLM con salida estructurada
question_router = route_prompt | llm.with_structured_output(RouteQuery)
```

**¿Cómo funciona?**
- Analiza la pregunta del usuario
- Compara con los temas del vectorstore (agentes, prompts, ataques)
- Decide: "vectorstore" si es un tema local, "web_search" para todo lo demás

**Ventajas**:
- ✅ Evita recuperación innecesaria si claramente necesita web
- ✅ Ahorra costos de embedding cuando la respuesta está en web
- ✅ Reduce latencia al evitar búsquedas vectoriales innecesarias

#### 2️⃣ **Retrieval Grader (Evaluador de Relevancia)**

Igual que en CRAG: califica cada documento recuperado.

```python
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' o 'no'")

retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)
```

**Criterio**: La prueba NO debe ser estricta. El objetivo es filtrar solo recuperaciones **erróneas**, no mediocres.

#### 3️⃣ **Hallucination Grader (Detector de Alucinaciones)**

**Novedad en Adaptive RAG**: Verifica que la respuesta esté fundamentada en los documentos.

```python
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="'yes' = grounded, 'no' = hallucination")

hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)
```

**¿Qué detecta?**
- Información NO presente en los documentos
- Hechos inventados por el LLM
- Extrapolaciones no fundamentadas

**Ejemplo**:
- Documentos: "Los agentes tienen memoria a corto plazo"
- Respuesta con alucinación: "Los agentes tienen memoria a corto, largo y ultra-largo plazo"
- Detector: `binary_score='no'` → Regenerar

#### 4️⃣ **Answer Grader (Validador de Respuestas)**

**Novedad en Adaptive RAG**: Verifica que la respuesta REALMENTE conteste la pregunta.

```python
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="'yes' = answers question, 'no' = doesn't")

answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)
```

**¿Qué valida?**
- La respuesta ABORDA la pregunta del usuario
- No es solo información relacionada, sino la RESPUESTA ESPECÍFICA

**Ejemplo**:
- Pregunta: "¿Cuántos tipos de memoria tienen los agentes?"
- Respuesta inútil: "Los agentes usan memoria para aprender"
- Validador: `binary_score='no'` → Reescribir pregunta y reintentar

#### 5️⃣ **Question Rewriter (Reescritor de Preguntas)**

Optimiza la pregunta cuando falla la recuperación o respuesta.

```python
question_rewriter = re_write_prompt | llm | StrOutputParser()
```

**Diferencia vs CRAG**:
- CRAG: Optimiza para **búsqueda web**
- Adaptive RAG: Optimiza para **recuperación en vectorstore**

### 📊 Arquitectura del Flujo Adaptive RAG

El flujo más complejo con 3 decisiones condicionales:

```
START
  ↓
[route_question] ← DECISIÓN 1: ¿Vectorstore o Web?
  ↓
  ├─→ "web_search"
  │     ↓
  │   [web_search] → [generate]
  │                      ↓
  │                [grade_generation_v_documents_and_question] ← DECISIÓN 3
  │                      ↓
  │                      ├─→ "useful" → END ✓
  │                      ├─→ "not useful" → [transform_query] → [retrieve] (ciclo)
  │                      └─→ "not supported" → [generate] (ciclo)
  │
  └─→ "vectorstore"
        ↓
      [retrieve]
        ↓
      [grade_documents] ← DECISIÓN 2: ¿Documentos relevantes?
        ↓
        ├─→ "generate"
        │     ↓
        │   [generate]
        │     ↓
        │   [grade_generation_v_documents_and_question] ← DECISIÓN 3
        │     ↓
        │     ├─→ "useful" → END ✓
        │     ├─→ "not useful" → [transform_query] → [retrieve] (ciclo)
        │     └─→ "not supported" → [generate] (ciclo)
        │
        └─→ "transform_query" → [retrieve] (ciclo)

CICLOS DE AUTO-CORRECCIÓN:
1. transform_query → retrieve → grade_documents (si docs no relevantes)
2. generate → grade_generation (si alucinaciones)
3. transform_query → retrieve → generate (si respuesta no útil)
```

### 🎯 Tres Decisiones Condicionales

**DECISIÓN 1: route_question()**
```python
def route_question(state):
    source = question_router.invoke({"question": state["question"]})
    if source.datasource == "web_search":
        return "web_search"  # Ir directo a búsqueda web
    else:
        return "vectorstore"  # Ir a recuperación vectorial
```

**DECISIÓN 2: decide_to_generate()**
```python
def decide_to_generate(state):
    if not state["documents"]:  # No hay documentos relevantes
        return "transform_query"  # Reescribir y reintentar
    else:
        return "generate"  # Generar respuesta
```

**DECISIÓN 3: grade_generation_v_documents_and_question()**
```python
def grade_generation_v_documents_and_question(state):
    # Paso 1: ¿Hay alucinaciones?
    if hallucination_grader(...).binary_score == "yes":
        # Paso 2: ¿Contesta la pregunta?
        if answer_grader(...).binary_score == "yes":
            return "useful"  # ✓ Todo bien, terminar
        else:
            return "not useful"  # Reescribir pregunta y reintentar
    else:
        return "not supported"  # Regenerar (hay alucinaciones)
```

### 💡 Casos de Uso de Adaptive RAG

#### **✅ Usa Adaptive RAG cuando**:
- Necesitas el **máximo nivel de calidad** y confiabilidad
- Las alucinaciones son **inaceptables** (legal, médico, financiero)
- El conocimiento está **distribuido** entre local + web
- Los usuarios hacen **preguntas impredecibles** de múltiples dominios
- Tienes presupuesto para **múltiples llamadas LLM** (evaluadores)
- La latencia es **secundaria** a la calidad

**Ejemplos**:
- **Asistentes legales**: Combina precedentes locales + leyes públicas
- **Soporte médico**: Base de conocimiento interna + investigación web
- **Consultoría financiera**: Datos corporativos + mercados en tiempo real
- **Investigación académica**: Papers locales + búsqueda web actualizada

#### **❌ No uses Adaptive RAG cuando**:
- Necesitas **baja latencia** (cada evaluador añade ~1-2s)
- Presupuesto de API es **limitado** (3-4 evaluaciones por consulta)
- El conocimiento es **completo** en una sola fuente
- RAG tradicional o CRAG ya dan **resultados aceptables**

### 🚀 Comparación: RAG Tradicional vs CRAG vs Adaptive RAG

| Característica | RAG Tradicional | CRAG | Adaptive RAG |
|----------------|----------------|------|--------------|
| **Enrutamiento** | ❌ No | ❌ No | ✅ Sí (Router) |
| **Evaluación de docs** | ❌ No | ✅ Sí | ✅ Sí |
| **Búsqueda web** | ❌ No | ✅ Condicional | ✅ Enrutada |
| **Detección alucinaciones** | ❌ No | ❌ No | ✅ Sí |
| **Validación respuesta** | ❌ No | ❌ No | ✅ Sí |
| **Ciclos auto-corrección** | ❌ No | ✅ 1 ciclo | ✅ 3 ciclos |
| **Complejidad** | Baja | Media | Alta |
| **Llamadas LLM promedio** | 1 | 2-4 | 4-8 |
| **Latencia** | ~2s | ~6s | ~10-15s |
| **Costo por consulta** | $0.002 | $0.008 | $0.015 |
| **Precisión** | 70-80% | 85-95% | 95-98% |
| **Tasa alucinación** | 15-25% | 5-10% | 2-5% |
| **Mejor para** | Demos, MVPs | Producción media | Producción crítica |

### ⚠️ Consideraciones Importantes

#### **Costos**
- **3-4 evaluadores por consulta**: router + grader + hallucination + answer
- Si hay ciclos: puede llegar a 8-10 llamadas LLM
- Usa `gpt-4o-mini` en vez de `gpt-4` para evaluadores (10x más barato)

#### **Latencia**
- Cada evaluador: ~1-2 segundos
- Flujo completo exitoso: ~10-15 segundos
- Con ciclos de corrección: puede llegar a 20-30 segundos
- Considera evaluación paralela si es posible

#### **Ciclos Infinitos**
- Implementa un **límite de iteraciones** (max 3 reintentos)
- Tracking de estados visitados para evitar loops
- Fallback a respuesta genérica si se agotan intentos

#### **Calidad de Evaluadores**
- Los evaluadores pueden cometer errores (falsos positivos/negativos)
- Usa `temperature=0` para consistencia
- Considera GPT-4 para evaluación crítica (más caro pero más preciso)
- Evalúa los evaluadores periódicamente con ground truth

### 🎯 Cuándo Usar Qué

**RAG Tradicional** → Demos, prototipos, latencia crítica
**CRAG** → Producción con presupuesto medio, necesitas corrección básica
**Adaptive RAG** → Producción crítica, calidad máxima, presupuesto holgado

### 📚 Recursos y Referencias

- **LangGraph**: Framework para grafos complejos con múltiples decisiones
- **Structured Output**: `with_structured_output()` con Pydantic para evaluadores
- **Tavily API**: Motor de búsqueda para agentes de IA
- **Literal Types**: Para enrutamiento con opciones limitadas
- **Conditional Edges**: `add_conditional_edges()` para flujos adaptativos

---

## 💾 RAG con Memoria Persistente

**RAG con Memoria Persistente** permite que tu sistema RAG **mantenga el contexto completo** de conversaciones entre múltiples interacciones con el usuario. A diferencia del RAG tradicional (sin memoria), este patrón usa **LangGraph con MemorySaver** para recordar preguntas previas, respuestas anteriores y mantener el contexto conversacional.

### 🧠 ¿Qué es la Memoria Persistente?

La memoria persistente en RAG significa que el sistema puede:
- 🔄 **Recordar conversaciones completas** entre sesiones
- 💬 **Entender referencias contextuales** ("¿y qué más?", "explica eso mejor")
- 🎯 **Mantener múltiples hilos** independientes con thread IDs únicos
- 📝 **Reutilizar contexto previo** sin repetir preguntas

```
SIN MEMORIA:
Usuario: ¿Qué es LangGraph?
Bot: LangGraph es un framework para construir grafos de estado...

Usuario: ¿Qué ventajas tiene?
Bot: ❌ No sé de qué hablas (no recuerda el contexto)

CON MEMORIA:
Usuario: ¿Qué es LangGraph?
Bot: LangGraph es un framework para construir grafos de estado...

Usuario: ¿Qué ventajas tiene?
Bot: ✅ Las ventajas de LangGraph incluyen... (recuerda que hablamos de LangGraph)
```

### 🎯 Componentes Clave

#### 1️⃣ **MemorySaver y Checkpoints**

El **MemorySaver** guarda el estado completo del grafo en cada paso.

```python
from langgraph.checkpoint.memory import MemorySaver

# Crear gestor de memoria
memory = MemorySaver()

# Compilar grafo con memoria
graph = graph_builder.compile(checkpointer=memory)
```

**¿Qué se guarda?**
- 📨 Todos los mensajes (usuario + asistente + herramientas)
- 🔧 Estado del grafo (nodo actual, variables)
- 🕐 Historial de ejecución completo

#### 2️⃣ **Thread IDs para Sesiones**

Cada conversación se identifica con un **thread_id** único.

```python
# Conversación del usuario A
config_user_a = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": [...]}, config_user_a)

# Conversación del usuario B (independiente)
config_user_b = {"configurable": {"thread_id": "user-456"}}
graph.invoke({"messages": [...]}, config_user_b)
```

**Ventajas**:
- ✅ Múltiples usuarios simultáneos sin mezclar contextos
- ✅ Retomar conversaciones en cualquier momento
- ✅ Aislar sesiones (web, móvil, etc.)

#### 3️⃣ **MessagesState**

`MessagesState` mantiene automáticamente el historial de mensajes.

```python
from langgraph.graph import MessagesState

class RAGState(MessagesState):
    # MessagesState ya incluye:
    # - messages: List[BaseMessage]
    # Añadimos campos adicionales:
    context_docs: List[Document]
    answer: Optional[str]
```

**Tipos de mensajes**:
- `HumanMessage`: Mensajes del usuario
- `AIMessage`: Respuestas del asistente
- `ToolMessage`: Resultados de herramientas
- `SystemMessage`: Instrucciones del sistema

### 📊 Arquitectura del Flujo con Memoria

```
INICIO
  ↓
[query_or_respond] → Decide si necesita recuperar info
  ↓
  ¿Necesita herramienta?
  ↓
  SÍ → [tools] → Ejecuta retrieve → [generate]
  NO → Responde directo
  ↓
[generate] → Genera respuesta con contexto + historial
  ↓
END (guarda checkpoint con MemorySaver)

SEGUNDA INTERACCIÓN:
INICIO (carga checkpoint previo)
  ↓
[query_or_respond] → Analiza con HISTORIAL completo
  ↓
...continúa...
```

### 🔄 Flujo de Trabajo Típico

**Primera Pregunta** (sin contexto):
```python
# Usuario pregunta sobre Task Decomposition
state = {
    "messages": [HumanMessage("¿Qué es Task Decomposition?")]
}
result = graph.invoke(state, config)

# Sistema:
# 1. Invoca retrieve para buscar docs
# 2. Genera respuesta con contexto recuperado
# 3. Guarda: [HumanMessage, AIMessage, ToolMessage]
```

**Pregunta de Seguimiento** (con contexto):
```python
# Usuario hace pregunta vaga que requiere contexto
state = {
    "messages": [HumanMessage("¿Puedes darme ejemplos de eso?")]
}
result = graph.invoke(state, config)

# Sistema:
# 1. Carga historial: [mensaje anterior sobre Task Decomposition]
# 2. Entiende que "eso" = Task Decomposition
# 3. Recupera ejemplos específicos
# 4. Genera respuesta contextualizada
```

### 💡 Casos de Uso

#### ✅ **Usa Memoria Persistente cuando**:
- 🎯 **Conversaciones multi-turno**: Chatbots, asistentes, soporte técnico
- 🔄 **Referencia contextual**: Usuarios usan "eso", "aquello", "lo anterior"
- 📋 **Tareas secuenciales**: Cada paso depende del anterior
- 👥 **Múltiples usuarios**: Necesitas aislar sesiones independientes
- 🕐 **Sesiones largas**: Conversaciones que duran minutos u horas

**Ejemplos**:
- 💬 **Chatbot de Soporte**: Recuerda el problema reportado y el historial de troubleshooting
- 🎓 **Tutor Educativo**: Mantiene progreso del estudiante y conceptos ya explicados
- 📝 **Asistente de Escritura**: Recuerda el contexto del documento que se está editando
- 🛒 **E-commerce**: Mantiene preferencias, búsquedas previas, artículos consultados

#### ❌ **No uses Memoria cuando**:
- ⚡ **Consultas independientes**: Cada pregunta no requiere contexto previo
- 🔒 **Privacidad estricta**: No debes guardar historial de conversación
- 💾 **Límites de almacenamiento**: Memoria crece con el tiempo (puede requerir limpieza)
- 🚀 **Latencia crítica**: Cargar historial añade overhead

### 🎨 Implementación con ReAct

El patrón **ReAct** (Reasoning + Acting) se puede combinar con memoria:

```python
from langgraph.prebuilt import create_react_agent

# Agente ReAct con memoria persistente
agent = create_react_agent(
    llm=llm,
    tools=[retrieve],  # Herramientas disponibles
    checkpointer=memory  # Habilita memoria
)

# Conversación continua
agent.invoke({"messages": [...]}, config)
```

**Ventajas del ReAct con Memoria**:
- ✅ El agente **recuerda decisiones previas**
- ✅ **No repite recuperaciones** innecesarias
- ✅ **Referencia resultados anteriores** sin volver a buscar

### 📈 Gestión de Memoria

#### **Recuperar Historial**
```python
# Obtener estado guardado
state = graph.get_state(config)
chat_history = state.values["messages"]

for message in chat_history:
    print(f"{message.type}: {message.content}")
```

#### **Limpiar Memoria (opcional)**
```python
# Si necesitas reiniciar conversación
# Simplemente usa un nuevo thread_id
config_new = {"configurable": {"thread_id": "new-conversation"}}
```

#### **Límites de Contexto**
- ⚠️ **Crecimiento ilimitado**: El historial puede crecer indefinidamente
- 🔧 **Solución**: Implementa **ventana deslizante** (últimos N mensajes)
- 💡 **Alternativa**: **Resumir historial antiguo** periódicamente

### 🚀 Comparación: Sin Memoria vs Con Memoria

| Característica | Sin Memoria | Con Memoria |
|----------------|-------------|-------------|
| **Contexto conversacional** | ❌ No | ✅ Sí |
| **Referencias ("eso", "aquello")** | ❌ No entiende | ✅ Entiende |
| **Múltiples turnos** | ❌ Cada pregunta aislada | ✅ Conversación fluida |
| **Sesiones independientes** | ❌ No aplica | ✅ Thread IDs |
| **Overhead de memoria** | ✅ Ninguno | ⚠️ Crece con el tiempo |
| **Latencia** | ✅ Baja | ⚠️ +100-300ms (carga historial) |
| **Complejidad** | ✅ Simple | ⚠️ Media |
| **Mejor para** | APIs stateless | Chatbots, asistentes |

---

## ⚡ Cache-Augmented Generation (CAG)

**Cache-Augmented Generation (CAG)** es una técnica de optimización que **reutiliza respuestas previas** cuando detecta preguntas similares, reduciendo drásticamente costos de API y latencia. A diferencia del RAG tradicional que siempre recupera y genera, CAG **usa caché semántico** para evitar llamadas innecesarias al LLM.

### 🎯 ¿Qué es CAG?

CAG **precomputa y almacena respuestas** para reutilizarlas cuando aparecen preguntas semánticamente similares:

```
SIN CACHÉ (RAG Tradicional):
Usuario: ¿Qué es LangGraph?
Sistema: Retrieve → Generate (17 segundos, $0.005)

Usuario: ¿Qué es Langgraph?  (casi idéntica)
Sistema: Retrieve → Generate (17 segundos, $0.005)  ← INEFICIENTE

CON CACHÉ (CAG):
Usuario: ¿Qué es LangGraph?
Sistema: Retrieve → Generate → Cache (17 segundos, $0.005)

Usuario: ¿Qué es Langgraph?  (detecta similitud)
Sistema: Cache Hit! (0.01 segundos, $0.000)  ← 1700x MÁS RÁPIDO
```

### 🔑 Tipos de Caché

#### 1️⃣ **Caché Simple (Diccionario)**

Coincidencia **exacta** de strings:

```python
cache = {}

def cache_model(query):
    if query in cache:
        return cache[query]  # ✅ Cache hit
    else:
        response = llm.invoke(query)  # ❌ Cache miss
        cache[query] = response
        return response
```

**Problema**: Solo funciona con strings **idénticos**:
- ✅ "¿Qué es LangGraph?" → ✅ "¿Qué es LangGraph?"
- ❌ "¿Qué es LangGraph?" → ❌ "¿Qué es Langgraph?" (falla por mayúscula)

#### 2️⃣ **Caché Semántico (FAISS)**

Usa **embeddings vectoriales** para detectar similitud semántica:

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Crear índice FAISS para caché
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
cache_index = faiss.IndexFlatL2(384)  # 384 dimensiones

qa_cache = FAISS(
    embedding_function=embeddings,
    index=cache_index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)
```

**Ventajas**:
- ✅ Detecta preguntas **parafraseadas**: "¿Qué es X?" ≈ "Explícame X" ≈ "Define X"
- ✅ Tolerante a **errores tipográficos**: "LangGraph" ≈ "Langgraph"
- ✅ Funciona en **múltiples idiomas** (si el modelo lo soporta)

### 📊 Arquitectura del Flujo CAG

```
INICIO
  ↓
[normalize_query] → Convierte a minúsculas, limpia espacios
  ↓
[semantic_cache_lookup] → Busca en FAISS con umbral de similitud
  ↓
  ¿Cache hit?
  ↓
  SÍ → [respond_from_cache] → Retorna respuesta guardada → END
  NO → [retrieve] → Recupera docs del vector store
          ↓
        [generate] → Genera respuesta con LLM
          ↓
        [cache_write] → Guarda en caché con metadatos
          ↓
        END
```

### 🧩 Componentes de CAG

#### 1️⃣ **Normalización de Consultas**

```python
def normalize_query(state):
    q = state["question"].strip().lower()
    state["normalized_question"] = q
    return state
```

**¿Por qué normalizar?**
- 🔄 Reduce duplicados: "LangGraph" = "langgraph"
- 📏 Mejora coincidencias: "  Qué es X  " = "Qué es X"

#### 2️⃣ **Búsqueda Semántica en Caché**

```python
def semantic_cache_lookup(state):
    q = state["normalized_question"]

    # Buscar preguntas similares en caché
    hits = qa_cache.similarity_search_with_score(q, k=3)

    if hits:
        doc, distance = hits[0]

        # Si la distancia L2 es menor al umbral → cache hit
        if distance <= CACHE_DISTANCE_THRESHOLD:
            state["answer"] = doc.metadata["answer"]
            state["cache_hit"] = True

    return state
```

**Parámetros clave**:
- `CACHE_DISTANCE_THRESHOLD`: Umbral de similitud (ej: 0.45)
  - Menor = **más estricto** (solo preguntas muy similares)
  - Mayor = **más permisivo** (preguntas menos similares)
- Distancia L2: **0.0** = idéntico, **>1.0** = muy diferente

#### 3️⃣ **Escritura en Caché**

```python
def cache_write(state):
    q = state["normalized_question"]
    a = state["answer"]

    # Guardar pregunta + respuesta + timestamp
    qa_cache.add_texts(
        texts=[q],
        metadatas=[{
            "answer": a,
            "ts": time.time()  # Para TTL
        }]
    )
    return state
```

**Metadatos guardados**:
- `answer`: La respuesta generada
- `ts`: Timestamp para Time-To-Live (opcional)

### ⏰ Time-To-Live (TTL)

Opcional: Las entradas caducadas se ignoran.

```python
CACHE_TTL_SEC = 3600  # 1 hora

# En semantic_cache_lookup:
ts = doc.metadata.get("ts")
if time.time() - ts > CACHE_TTL_SEC:
    # Cache expirado, ignorar
    return state
```

**Casos de uso**:
- 📰 **Noticias**: Caché de 5 minutos (información cambia rápido)
- 📚 **Documentación técnica**: Caché de días/semanas (información estable)
- 🔄 **Sin TTL**: Para información que nunca cambia

### 💡 Casos de Uso de CAG

#### ✅ **Usa CAG cuando**:
- 🔁 **Preguntas repetitivas**: FAQ, soporte técnico, educación
- 💰 **Optimización de costos**: Reduce llamadas a GPT-4 (caro)
- ⚡ **Baja latencia**: Respuestas instantáneas para usuarios
- 📈 **Alto volumen**: Miles de usuarios con preguntas similares
- 🎓 **Onboarding**: Nuevos usuarios hacen las mismas preguntas básicas

**Ejemplos**:
- 🎓 **Chatbot Educativo**: "¿Qué es X?" preguntado por cientos de estudiantes
- 🛠️ **Soporte Técnico**: "¿Cómo reseteo mi contraseña?" (top FAQ)
- 📄 **Documentación**: "¿Cómo instalo X?" (pregunta muy común)
- 🏢 **Onboarding Corporativo**: Políticas, beneficios (preguntas repetidas)

#### ❌ **No uses CAG cuando**:
- 🔄 **Información en tiempo real**: Precios de bolsa, clima actual
- 🎨 **Respuestas personalizadas**: Cada usuario necesita respuesta única
- 🗣️ **Conversaciones únicas**: Pocas o ninguna pregunta repetida
- 🚫 **Privacidad sensible**: No debes guardar preguntas de usuarios

### 🚀 Métricas de Optimización

#### **Ahorro de Tiempo**
```
Sin Caché:     17s por consulta
Con Caché:     0.01s por consulta
Reducción:     1700x más rápido
```

#### **Ahorro de Costos (GPT-4o-mini)**
```
Sin Caché:     $0.005 por consulta
Con Caché:     $0.000 (solo embedding: ~$0.00001)
Reducción:     500x más barato
```

#### **Tasa de Acierto de Caché**
```
Cache Hit Rate = (Cache Hits / Total Queries) × 100%

Ejemplo con 1000 consultas:
- 400 cache hits
- 600 cache misses
Cache Hit Rate = 40%

Ahorro: 400 × ($0.005 - $0.00001) = $1.996
```

### ⚖️ Umbral de Distancia

El umbral controla qué tan similares deben ser las preguntas:

| Umbral | Comportamiento | Ejemplo |
|--------|---------------|---------|
| **0.0** | Solo idénticos | "¿Qué es X?" = "¿Qué es X?" |
| **0.3** | Muy estricto | "¿Qué es LangGraph?" ≈ "¿Qué es Langgraph?" |
| **0.45** | **Recomendado** | "¿Qué es LangGraph?" ≈ "Explica LangGraph" |
| **0.7** | Permisivo | "¿Qué es LangGraph?" ≈ "¿Cómo funciona LangGraph?" |
| **>1.0** | Demasiado permisivo | Preguntas diferentes se tratan como iguales ❌ |

**Recomendación**: Empieza con `0.45` y ajusta según tus métricas:
- ⬇️ Si muchos **falsos positivos** (respuestas incorrectas) → Reduce umbral
- ⬆️ Si pocos **cache hits** (bajo aprovechamiento) → Aumenta umbral

### 🛠️ Limpieza y Mantenimiento

#### **Límite de Tamaño del Caché**
```python
MAX_CACHE_SIZE = 10000

if qa_cache.index.ntotal > MAX_CACHE_SIZE:
    # Eliminar entradas más antiguas
    # Estrategia: FIFO, LRU, o por timestamp
```

#### **Evaluación de Calidad**
```python
# Monitorear métricas
metrics = {
    "cache_hits": 0,
    "cache_misses": 0,
    "false_positives": 0  # Cache hit con respuesta incorrecta
}

# Logging
if state["cache_hit"]:
    logger.info(f"Cache hit: {query} → {answer}")
```

### 📈 Comparación: Sin Caché vs CAG

| Característica | Sin Caché | CAG |
|----------------|-----------|-----|
| **Latencia promedio** | ~15s | ~2s (85% hits) |
| **Costo por 1000 queries** | $5.00 | $1.50 (70% savings) |
| **Escalabilidad** | ⚠️ Crece linealmente | ✅ Mejor con más tráfico |
| **Complejidad** | ✅ Simple | ⚠️ Media |
| **Almacenamiento** | ✅ Ninguno | ⚠️ Crece con el tiempo |
| **Mejor para** | Prototipado | Producción con tráfico alto |

---

## 📁 Estructura del Proyecto

```
RAGBootcamp/
│
├── 000_DataIngestParsing/          # Módulo 1: Ingesta de Datos
│   ├── 1-dataingestion.ipynb       # Carga de archivos de texto
│   ├── 2-dataparsingpdf.ipynb      # Parseo de PDFs (PyPDF, PyMuPDF)
│   ├── 3-dataparsingdoc.ipynb      # Parseo de documentos Word
│   ├── 4-csvexcelparsing.ipynb     # Datos estructurados (CSV/Excel)
│   ├── 5-jsonparsing.ipynb         # Manejo de archivos JSON
│   ├── 6-databaseparsing.ipynb     # Conexión a bases de datos
│   └── data/                        # Datasets de ejemplo
│       ├── text_files/
│       ├── pdf/
│       ├── word_files/
│       ├── structured_files/
│       ├── json_files/
│       └── databases/
│
├── 001_VectorEmbeddingAndDatabases/ # Módulo 2: Embeddings
│   ├── 1-embedding.ipynb            # Conceptos de embeddings
│   └── 2-openaiembeddings.ipynb     # Embeddings de OpenAI
│
├── 002_VectorStores/                # Módulo 3: Bases de Datos Vectoriales
│   ├── 1-chromadb.ipynb             # ChromaDB
│   ├── 2-faiss.ipynb                # FAISS
│   ├── 3-Othervectorstores.ipynb    # InMemoryVectorStore
│   ├── 4-Datastaxdb.ipynb           # AstraDB
│   ├── 5-PineconeVectorDB.ipynb     # Pinecone
│   ├── chroma_db/                   # Almacenamiento ChromaDB
│   └── faiss_index/                 # Índices FAISS guardados
│
├── 003_AdvancedChuking/             # Módulo 4: Técnicas Avanzadas de Chunking
│   └── 1-semantichunking.ipynb      # Semantic Chunking
│
├── 004_HybridSearchStrategies/      # Módulo 5: Estrategias de Búsqueda Híbrida
│   ├── 1-densesparse.ipynb          # Búsqueda Híbrida (Dense + Sparse)
│   ├── 2-reranking.ipynb            # Reranking con LLM
│   └── 3-mmr.ipynb                  # MMR (Maximal Marginal Relevance)
│
├── 005_QueryEnhancement/            # Módulo 6: Mejora de Consultas
│   ├── 1-queryexpansion.ipynb       # Query Expansion (Expansión de Consultas)
│   ├── 2-querydecomposition.ipynb   # Query Decomposition (Descomposición)
│   └── 3-HyDE.ipynb                 # HyDE (Hypothetical Document Embeddings)
│
├── 006_MultimodalRag/               # Módulo 7: RAG Multimodal
│   └── 1-multimodalopenai.ipynb     # RAG con CLIP + GPT-4 Vision
│
├── 007_LanggraphBasics/             # Módulo 8: Fundamentos de LangGraph
│   ├── 2-chatbot.ipynb              # Chatbot simple con LangGraph
│   ├── 3-DataclassStateSchema.ipynb # Esquemas de estado (TypedDict vs DataClass)
│   ├── 4-pydantic.ipynb             # Validación de datos con Pydantic
│   ├── 5-ChainsLangGraph.ipynb      # Cadenas y herramientas en LangGraph
│   └── 6-chatbotswithmultipletools.ipynb # Chatbot con múltiples herramientas
│
├── 008_AgentsArchitecture/          # Módulo 9: Arquitectura de Agentes
│   ├── 1-ReActAgents.ipynb          # Agentes ReAct con herramientas y memoria
│   └── 2-streaming.ipynb            # Streaming de respuestas en tiempo real
│
├── 009_Debugging/                   # Módulo 10: Debugging y LangGraph Studio
│   ├── langgraph.json               # Configuración de LangGraph Studio
│   └── openai_agent.py              # Agente con herramientas para debugging
│
├── 010_AgenticRag/                  # Módulo 11: RAG Agéntico
│   ├── 1-agenticrag.ipynb           # Introducción a RAG Agéntico con LangGraph
│   ├── 2-ReAct.ipynb                # Framework ReAct: Reasoning + Acting
│   ├── 3-AgenticRAG.ipynb           # Sistema RAG Agéntico completo con evaluación
│   ├── internal_docs.txt            # Documentos internos de ejemplo
│   └── research_notes.txt           # Notas de investigación de ejemplo
│
├── 011_AutonomousRag/               # Módulo 12: RAG Autónomo
│   ├── 1-COTRag.ipynb               # Chain-of-Thought RAG (Razonamiento paso a paso)
│   ├── 2-Selfreflection.ipynb       # Auto-reflexión: El LLM evalúa su propia respuesta
│   ├── 3-QueryPlanningdecomposition.ipynb  # Descomposición de consultas complejas
│   ├── 4-Iterativeretrieval.ipynb   # Recuperación iterativa con refinamiento de consultas
│   ├── 5-answersynthesis.ipynb      # Síntesis de respuestas desde múltiples fuentes
│   ├── internal_docs.txt            # Documentos internos de ejemplo
│   └── research_notes.txt           # Notas de investigación de ejemplo
│
├── 012_MultiAgentsRags/             # Módulo 13: Sistemas RAG Multi-Agente
│   └── 1-multiagent.ipynb           # Sistema multi-agente: Colaborativo, Supervisor y Jerárquico
│
├── 013_CorrectiveRag/               # Módulo 14: RAG Correctivo (Corrective RAG - CRAG)
│   └── 1-CorrectiveRAG.ipynb        # Sistema CRAG con evaluación de relevancia y búsqueda web
│
├── 014_AdaptiveRag/                 # Módulo 15: RAG Adaptativo (Adaptive RAG)
│   └── 1-AdaptiveRAG.ipynb          # Sistema completo con enrutamiento, evaluadores y auto-corrección
│
├── 015_RagMemory/                   # Módulo 16: RAG con Memoria Persistente
│   └── 1-ragmemory.ipynb            # Sistema RAG con memoria conversacional usando LangGraph y MemorySaver
│
├── 016_CacheRagLangGraph/           # Módulo 17: Cache-Augmented Generation (CAG)
│   └── 1-cache_augment_generation.ipynb  # Sistema de caché semántico con FAISS para optimizar respuestas
│
├── .env                             # Variables de entorno (API keys)
├── .gitignore                       # Archivos ignorados por Git
├── .python-version                  # Versión de Python (3.12)
├── requirements.txt                 # Dependencias del proyecto
├── CLAUDE.md                        # Guía para Claude Code
└── README.md                        # Este archivo
```

## 🎓 Guía de Uso

### Orden de Aprendizaje Recomendado

1. **Módulo 000: Ingesta de Datos** (4-6 horas)
   - Comienza con `1-dataingestion.ipynb`
   - Aprende diferentes técnicas de carga de documentos
   - Explora estrategias de text splitting

2. **Módulo 001: Embeddings** (2-3 horas)
   - Comprende qué son los embeddings
   - Compara diferentes modelos de embeddings
   - Aprende similitud del coseno

3. **Módulo 002: Vector Stores** (4-5 horas)
   - Experimenta con cada base de datos vectorial
   - Compara rendimiento y características
   - Implementa búsquedas avanzadas con filtros

4. **Módulo 003: Advanced Chunking** (2-3 horas)
   - Aprende técnicas avanzadas de chunking semántico
   - Optimiza la división de documentos para mejor recuperación
   - Implementa semantic chunking con embeddings

5. **Módulo 004: Estrategias de Búsqueda Híbrida** (3-4 horas)
   - Combina recuperación densa y dispersa (Dense + Sparse)
   - Implementa reranking con LLM para mayor precisión
   - Usa MMR para diversidad en resultados
   - Aprende cuándo aplicar cada estrategia

6. **Módulo 005: Mejora de Consultas** (3-4 horas)
   - Implementa Query Expansion para enriquecer consultas vagas
   - Descompone consultas complejas con Query Decomposition
   - Usa HyDE para resolver vocabulary mismatch
   - Aprende a combinar técnicas de mejora de consultas
   - Optimiza el balance entre precisión, recall y latencia

7. **Módulo 006: RAG Multimodal** (4-5 horas)
   - Comprende cómo procesar documentos con texto e imágenes
   - Implementa embeddings unificados con CLIP
   - Integra GPT-4 Vision para análisis multimodal
   - Construye pipelines que recuperan y procesan imágenes y texto
   - Aprende optimizaciones para reducir costos y latencia

8. **Módulo 007: LangGraph Basics** (4-6 horas)
   - Comprende los conceptos fundamentales de grafos de estado
   - Aprende las diferencias entre TypedDict, DataClass y Pydantic
   - Construye chatbots simples y avanzados con LangGraph
   - Integra múltiples herramientas (Arxiv, Wikipedia, Tavily)
   - Implementa enrutamiento condicional y gestión de estado
   - Domina el uso de reductores (reducers) y mensajes
   - Crea agentes que toman decisiones inteligentes

9. **Módulo 008: Arquitectura de Agentes** (4-5 horas)
   - Comprende la arquitectura ReAct (Reason + Act)
   - Implementa agentes que razonan y actúan iterativamente
   - Integra herramientas de búsqueda (Arxiv, Wikipedia, Tavily)
   - Crea funciones personalizadas como herramientas
   - Implementa memoria conversacional con MemorySaver
   - Aprende técnicas de streaming de respuestas
   - Domina stream_mode="updates" vs "values"
   - Implementa streaming token por token con astream_events()

10. **Módulo 009: Debugging con LangGraph Studio** (2-3 horas)
   - Configura LangGraph Studio para desarrollo local
   - Visualiza grafos de agentes en tiempo real
   - Depura agentes paso a paso con inspección de estado
   - Experimenta con diferentes configuraciones de grafos
   - Aprende el flujo de desarrollo con hot reload
   - Implementa grafos básicos y con herramientas
   - Domina el archivo langgraph.json para configuración

11. **Módulo 010: RAG Agéntico** (5-6 horas)
   - Comprende qué es RAG Agéntico y cómo difiere del RAG tradicional
   - Construye un sistema RAG básico con LangGraph y StateGraph
   - Implementa el framework ReAct (Reasoning + Acting)
   - Crea agentes que razonan, recuperan, evalúan y reformulan consultas
   - Implementa múltiples fuentes de conocimiento (vectorstores separados)
   - Aprende a evaluar relevancia de documentos con LLM
   - Implementa nodos condicionales para flujo inteligente
   - Domina la reformulación automática de consultas (query rewriting)
   - Crea herramientas personalizadas de recuperación con metadatos
   - Construye grafos complejos con ciclos y toma de decisiones

12. **Módulo 011: RAG Autónomo** (6-8 horas)
   - **Chain-of-Thought (CoT)**: Descompone preguntas complejas en sub-pasos razonados
   - **Auto-Reflexión**: El LLM evalúa su propia respuesta y mejora iterativamente
   - **Planificación de Consultas**: Divide consultas en sub-preguntas para búsqueda precisa
   - **Recuperación Iterativa**: Ciclo de retroalimentación con refinamiento de consultas
   - **Síntesis Multi-Fuente**: Combina información de 4 fuentes (docs, YouTube, Wikipedia, ArXiv)
   - Implementa sistemas que razonan antes de recuperar información
   - Domina ciclos de retroalimentación con límites de iteraciones
   - Construye flujos transparentes y explicables paso a paso
   - Aprende cuándo usar cada patrón según el caso de uso
   - Integra múltiples fuentes de conocimiento en respuestas coherentes

13. **Módulo 012: Sistemas RAG Multi-Agente** (8-10 horas)
   - **Arquitectura Colaborativa**: Dos agentes (researcher + blog writer) colaboran
   - **Patrón Supervisor**: Supervisor central coordina agentes especializados (research + math)
   - **Jerarquía de Equipos**: 3 niveles de supervisión con equipos completos
   - Construye equipos de investigación (search + web scraper)
   - Implementa equipos de escritura (note taker + doc writer + chart generator)
   - Usa `langgraph_supervisor` para coordinación automática
   - Crea herramientas personalizadas con `@tool` decorator
   - Implementa scraping web con BeautifulSoup
   - Ejecuta código Python dinámico con REPL para visualizaciones
   - Domina el patrón Command para navegación y actualización de estado
   - Aprende cuándo usar cada arquitectura según complejidad del proyecto
   - Gestiona múltiples agentes especializados con roles claros

14. **Módulo 013: RAG Correctivo (Corrective RAG - CRAG)** (4-5 horas)
   - **Evaluación Automática de Relevancia**: Usa un LLM para calificar documentos recuperados
   - **Reescritura Inteligente de Consultas**: Optimiza preguntas cuando los documentos no son relevantes
   - **Búsqueda Web Adaptativa**: Integra Tavily para buscar información externa cuando es necesario
   - **Flujo de Decisión con LangGraph**: Implementa lógica condicional basada en relevancia
   - Construye un evaluador binario (yes/no) con salida estructurada usando Pydantic
   - Aprende a usar `with_structured_output()` para respuestas determinísticas
   - Implementa nodos de decisión con `add_conditional_edges()`
   - Crea flujos adaptativos que se auto-corrigen cuando fallan
   - Integra múltiples fuentes: vectorstore local + búsqueda web
   - Reduce alucinaciones filtrando documentos irrelevantes
   - Domina el patrón CRAG para sistemas RAG más robustos y confiables
   - Aprende cuándo usar CRAG vs RAG tradicional según el caso de uso

15. **Módulo 014: RAG Adaptativo (Adaptive RAG)** (5-6 horas)
   - **Enrutamiento Inteligente**: Router LLM decide automáticamente entre vectorstore local y búsqueda web
   - **Evaluación de Relevancia**: Califica cada documento recuperado del vectorstore
   - **Detección de Alucinaciones**: Verifica que la respuesta esté fundamentada en los documentos
   - **Validación de Respuestas**: Confirma que la respuesta realmente conteste la pregunta
   - **Auto-Corrección con Ciclos**: Reescribe consultas y reintenta hasta obtener una respuesta de calidad
   - Implementa un router con `Literal["vectorstore", "web_search"]` y Pydantic
   - Crea tres evaluadores independientes (relevancia, alucinaciones, respuestas)
   - Construye flujos con múltiples decisiones condicionales anidadas
   - Implementa ciclos de retroalimentación (transform_query → retrieve → generate)
   - Aprende a usar `add_conditional_edges()` con 3 opciones de salida
   - Maneja el flujo complejo: enrutamiento → recuperación → evaluación → generación → validación
   - Domina el patrón más completo y robusto de RAG para producción
   - Aprende cuándo usar Adaptive RAG vs CRAG vs RAG tradicional

16. **Módulo 015: RAG con Memoria Persistente** (4-5 horas)
   - **Memoria Conversacional con LangGraph**: Mantiene el historial completo de interacciones entre el usuario y el agente
   - **MemorySaver y Checkpoints**: Implementa persistencia de estado usando checkpointers
   - **Thread IDs para Sesiones**: Gestiona múltiples conversaciones independientes con identificadores únicos
   - **Arquitectura de Grafo Personalizada**: Construye flujos con query_or_respond, retrieve, generate y cache_write
   - **Agente ReAct con Memoria**: Implementa el patrón ReAct usando `create_react_agent` con memoria persistente
   - **Contexto entre Mensajes**: El sistema recuerda preguntas previas y puede responder seguimientos
   - Implementa herramientas personalizadas de recuperación con el decorador `@tool`
   - Construye nodos de generación que inyectan contexto recuperado en prompts
   - Aprende a usar `MessagesState` para mantener el historial de mensajes
   - Domina el flujo: normalizar → buscar caché → recuperar → generar → escribir caché
   - Implementa respuestas que hacen referencia a conversaciones previas
   - Aprende cuándo usar memoria persistente vs conversaciones stateless

17. **Módulo 016: Cache-Augmented Generation (CAG)** (5-6 horas)
   - **Caché Simple con Diccionarios**: Implementa caché básico basado en coincidencia exacta de strings
   - **Caché Semántico con FAISS**: Usa embeddings vectoriales para detectar preguntas similares, no solo idénticas
   - **Similitud L2 y Umbrales**: Configura distancia L2 y thresholds para determinar aciertos de caché
   - **Optimización de Costos**: Reduce llamadas a API reutilizando respuestas previas (0.00s vs 12-17s)
   - **TTL (Time To Live)**: Implementa expiración automática de entradas de caché
   - **Arquitectura Avanzada con LangGraph**: Construye flujo con normalización, búsqueda semántica y escritura de caché
   - Implementa funciones de nodo: `normalize_query`, `semantic_cache_lookup`, `retrieve`, `generate`, `cache_write`
   - Usa HuggingFace embeddings (sentence-transformers) para vectorización
   - Crea índices FAISS con `IndexFlatL2` para búsqueda eficiente
   - Aprende edges condicionales: si cache_hit → responder, si no → RAG completo
   - Integra metadatos (respuesta, timestamp) en documentos cacheados
   - Domina el patrón CAG para sistemas RAG optimizados en producción
   - Comprende cuándo usar caché exacto vs caché semántico según el caso de uso

### Ejecutar un Notebook

```bash
# Opción 1: Jupyter Notebook
jupyter notebook 000_DataIngestParsing/1-dataingestion.ipynb

# Opción 2: Jupyter Lab (interfaz moderna)
jupyter lab

# Opción 3: VS Code con extensión Jupyter
code .
```

### Ejemplos de Código Rápido

#### Ejemplo 1: Cargar y Dividir Documento

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cargar documento
loader = TextLoader("documento.txt")
documents = loader.load()

# Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

#### Ejemplo 2: Crear Vector Store y Buscar

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Crear vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./mi_db"
)

# Búsqueda semántica
results = vector_store.similarity_search(
    "¿Cuál es la capital de Francia?",
    k=3
)

for doc in results:
    print(doc.page_content)
```

#### Ejemplo 3: Retriever con Filtros

```python
# Convertir a retriever
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7,
        "filter": {"source": "wikipedia"}
    }
)

# Usar en cadena RAG
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=retriever
)

respuesta = qa_chain.invoke("¿Qué dice el documento sobre...?")
```

## 🛠️ Tecnologías Utilizadas

### Core Framework
- **LangChain 0.3**: Framework principal para RAG
- **LangChain Community**: Extensiones y utilidades
- **LangChain OpenAI**: Integración con modelos OpenAI

### Embeddings
- **OpenAI Embeddings**: text-embedding-3-small, text-embedding-ada-002
- **HuggingFace Transformers**: Modelos open-source
- **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2

### Vector Databases
- **ChromaDB**: Base de datos vectorial local
- **FAISS**: Biblioteca de Facebook para búsqueda de similitud
- **Pinecone**: Servicio cloud de bases de datos vectoriales
- **AstraDB**: Base de datos vectorial serverless de DataStax

### Document Processing
- **PyPDF / PyMuPDF**: Procesamiento de PDFs
- **python-docx / docx2txt**: Documentos de Word
- **Pandas / OpenPyXL**: Datos estructurados
- **Unstructured**: Parser avanzado de documentos

### Utilities
- **tiktoken**: Tokenización para OpenAI
- **python-dotenv**: Gestión de variables de entorno
- **Jupyter**: Notebooks interactivos

## 📚 Recursos Adicionales

### Documentación Oficial
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Pinecone Docs](https://docs.pinecone.io/)
- [AstraDB Docs](https://docs.datastax.com/en/astra-serverless/docs/)

### Tutoriales Relacionados
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Vector Database Comparison](https://www.datastax.com/guides/what-is-a-vector-database)

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Áreas de Contribución
- 📝 Mejorar documentación
- 🐛 Reportar y corregir bugs
- ✨ Agregar nuevos ejemplos
- 🌐 Traducciones a otros idiomas
- 🧪 Agregar tests
- 📊 Benchmarks de rendimiento

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👤 Autor

**RAG Bootcamp Team**

## 🙏 Agradecimientos

- A la comunidad de LangChain por el framework excepcional
- A OpenAI por los modelos de embeddings
- A todos los contribuidores de las bibliotecas open-source utilizadas
- A la comunidad de desarrolladores de IA que comparten conocimiento

---

<div align="center">

**¿Te gustó este proyecto? Dale una ⭐ en GitHub!**

[⬆ Volver arriba](#-rag-bootcamp)

</div>
