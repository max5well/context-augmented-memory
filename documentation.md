# Context Augmented Memory (CAM) Documentation

## 📖 Project Overview

Context Augmented Memory (CAM) is an intelligent memory system that enhances Large Language Model (LLM) interactions by maintaining contextual continuity across conversations. It combines vector-based memory storage with intelligent context retrieval and multi-provider LLM routing capabilities.

## 🏗️ System Architecture

The CAM system operates on a dual-mode architecture:
- **CLI Mode**: Interactive command-line interface for direct user interaction
- **API Mode**: FastAPI-based proxy server with OpenAI-compatible endpoints

## 📁 Core Files and Components

### **Entry Points**

#### `main.py` - CLI Application Entry Point
**Purpose**: Standalone CLI runner for Context-Augmented Memory (CAM)
**Why we need it**: Provides interactive command-line interface for testing and direct interaction with the memory system
**Data flow**: User input → Intent classification → Context decision → LLM processing → Memory storage
**What it does**:
- Classifies user prompts (fact/query/meta)
- Retrieves relevant context when needed
- Sends augmented prompts to OpenAI GPT-4o-mini
- Stores responses with metadata in ChromaDB
- Maintains short-term sliding context window for factual continuity

#### `proxy_api/app.py` - FastAPI Application Entry Point
**Purpose**: Web API entry point for the memory proxy system
**Why we need it**: Enables REST API access to CAM functionality for external applications
**Data flow**: HTTP requests → Router → Service layer → Response
**What it does**: Initializes FastAPI application and includes router endpoints

### **Configuration Files**

#### `config.json` - System Configuration
**Purpose**: Central configuration for all CAM system parameters
**Why we need it**: Allows runtime customization without code changes
**Data flow**: Loaded at startup by `config_manager.py`
**What it does**:
- Configures retrieval distance thresholds (max_distance: 0.8)
- Sets usefulness filter parameters (min word count, blacklist phrases)
- Defines context decision thresholds (continuity base/std factors)

#### `requirements.txt` - Python Dependencies
**Purpose**: Lists all required Python packages and versions
**Why we need it**: Ensures reproducible environment setup
**Data flow**: Used by pip for dependency installation
**Key dependencies**: ChromaDB, OpenAI, Anthropic, FastAPI, scikit-learn

### **Core Modules (`/modules/`)**

#### `embedding.py` - Embedding Generation & Similarity
**Purpose**: Handles OpenAI embeddings and similarity calculations
**Why we need it**: Converts text to vectors for semantic search and comparison
**Data flow**: Text input → OpenAI API → Embedding vector → Similarity calculation
**What it does**:
- Generates text embeddings using OpenAI's text-embedding-3-large model
- Computes cosine similarity between embedding vectors
- Handles API errors gracefully with fallback empty lists

#### `memory.py` - Vector Database Integration
**Purpose**: ChromaDB integration for persistent vector storage and retrieval
**Why we need it**: Provides long-term memory storage with fast semantic search
**Data flow**: Documents + embeddings → ChromaDB → Vector search → Retrieved context
**What it does**:
- Manages ChromaDB collection lifecycle
- Stores documents with metadata and embeddings
- Performs similarity-based document retrieval
- Handles connection errors and initialization

#### `retrieval.py` - Advanced Context Retrieval
**Purpose**: Intelligent context retrieval with pronoun-aware re-ranking
**Why we need it**: Ensures retrieved context is relevant and resolves pronouns correctly
**Data flow**: Query intent → Retrieval mode selection → Search → Re-ranking → Context assembly
**What it does**:
- Implements dual-mode retrieval (contextual vs global)
- Uses adaptive distance thresholds for semantic filtering
- Performs pronoun-aware re-ranking using named entity recognition
- Boosts entries with named entities for better pronoun resolution

#### `intent_classifier.py` - Prompt Intent Classification
**Purpose**: Classifies user prompts as 'fact', 'query', or 'meta'
**Why we need it**: Determines appropriate processing pipeline for different prompt types
**Data flow**: User prompt → OpenAI API → Intent classification → Processing decision
**What it does**:
- Uses OpenAI API for accurate intent classification
- Enables different handling for facts (store), queries (retrieve), meta (skip)
- Provides confidence scores for classification decisions

#### `context_decider.py` - Semantic Continuity Detection
**Purpose**: Determines when context retrieval is needed using adaptive thresholds
**Why we need it**: Avoids unnecessary retrieval for unrelated prompts, maintaining conversation flow
**Data flow**: Current prompt → Embedding similarity → Adaptive threshold → Retrieval decision
**What it does**:
- Calculates semantic similarity between current and previous prompts
- Uses dynamic thresholds based on conversation semantic variance
- Provides continuity decisions with confidence scores

#### `auto_tagger.py` - Automatic Content Tagging
**Purpose**: Automatically tags conversations with predefined categories
**Why we need it**: Enables better organization and filtering of stored memories
**Data flow**: User prompt → Category matching → Tag assignment → Metadata storage
**What it does**:
- Matches prompts against predefined tag categories
- Supports multiple tags per conversation
- Enhances metadata for better searchability

#### `usefulness_filter.py` - Content Quality Filtering
**Purpose**: Filters out trivial or low-value prompts before storage
**Why we need it**: Prevents memory pollution with insignificant interactions
**Data flow**: User prompt → Quality checks → Filter decision → Storage skip/allow
**What it does**:
- Applies minimum word/character count filters
- Uses blacklist of common low-value phrases
- Ensures only meaningful content is stored

#### `config_manager.py` - Dynamic Configuration
**Purpose**: Handles loading and validation of system configuration
**Why we need it**: Centralizes configuration management with validation
**Data flow**: config.json → Validation → Runtime config → Application access
**What it does**:
- Loads configuration from JSON file
- Validates configuration schema and values
- Provides typed configuration access throughout system

### **API Layer (`/proxy_api/`)**

#### `proxy_api/router.py` - API Routes
**Purpose**: Defines all API endpoints and request/response handling
**Why we need it**: Provides HTTP interface for external applications to use CAM
**Data flow**: HTTP requests → Route handlers → Service calls → HTTP responses
**What it does**:
- Implements OpenAI-compatible `/v1/chat/completions` endpoint
- Provides memory debugging endpoint at `/v1/memory/debug`
- Handles request validation and response formatting

#### `proxy_api/services/context_injector.py` - Context Injection Service
**Purpose**: Core service for context injection and memory management in API mode
**Why we need it**: Orchestrates the entire memory lifecycle for API requests
**Data flow**: API request → Intent classification → Context retrieval → LLM call → Memory storage
**What it does**:
- Coordinates all CAM modules for processing API requests
- Manages context injection and response augmentation
- Handles memory storage with proper metadata
- Provides error handling and fallback mechanisms

#### `proxy_api/utils/normalizer.py` - Response Normalization
**Purpose**: Normalizes responses from different LLM providers to standard format
**Why we need it**: Ensures consistent API responses regardless of underlying LLM provider
**Data flow**: LLM responses → Provider-specific parsing → Standardized format → API response
**What it does**:
- Handles different response formats from OpenAI, Anthropic, Mistral, Gemini
- Converts all responses to OpenAI-compatible format
- Maintains API contract consistency

#### `proxy_api/utils/fallback_llm.py` - LLM Fallback Handler
**Purpose**: Provides fallback LLM functionality when primary provider fails
**Why we need it**: Ensures system reliability and availability
**Data flow**: Primary LLM failure → Fallback selection → Secondary LLM call → Response
**What it does**:
- Detects LLM provider failures
- Automatically switches to backup providers
- Maintains service continuity during outages

### **LLM Provider Clients (`/proxy_api/clients/`)**

#### `clients/openai_client.py` - OpenAI API Client
**Purpose**: Handles communication with OpenAI API
**Why we need it**: Provides standardized interface to OpenAI models
**Data flow**: API requests → OpenAI client → OpenAI API → Response processing
**What it does**:
- Manages OpenAI API authentication and rate limiting
- Handles chat completions with proper error handling
- Implements retry logic for transient failures

#### `clients/anthropic_client.py` - Anthropic API Client
**Purpose**: Handles communication with Anthropic Claude API
**Why we need it**: Enables access to Anthropic models through unified interface
**Data flow**: API requests → Anthropic client → Anthropic API → Response processing
**What it does**:
- Converts between OpenAI and Anthropic API formats
- Manages Claude-specific parameters and constraints
- Provides consistent error handling

#### `clients/mistral_client.py` - Mistral API Client
**Purpose**: Handles communication with Mistral AI API
**Why we need it**: Provides access to Mistral models with cost-effective alternatives
**Data flow**: API requests → Mistral client → Mistral API → Response processing
**What it does**:
- Adapts requests to Mistral API specification
- Handles model-specific parameters and limits
- Maintains API compatibility

#### `clients/gemini_client.py` - Google Gemini API Client
**Purpose**: Handles communication with Google Gemini API
**Why we need it**: Adds Google's multimodal capabilities to the system
**Data flow**: API requests → Gemini client → Gemini API → Response processing
**What it does**:
- Manages Gemini-specific authentication and parameters
- Handles multimodal input capabilities
- Provides consistent response format

### **Supporting Files**

#### `start_cam.sh` - System Startup Script
**Purpose**: Automated startup script for CAM services
**Why we need it**: Simplifies deployment and ensures proper service initialization
**Data flow**: Script execution → Service startup → Health checks → Status reporting
**What it does**:
- Initializes ChromaDB server
- Starts FastAPI proxy server
- Performs health checks and logging
- Handles graceful shutdown procedures

## 🔗 Key Data Flows

### **CLI Mode Workflow**
```
User Input → Intent Classification → Context Decision →
[Retrieval if relevant] → Context Injection → LLM Processing →
Response Normalization → Memory Storage
```

### **API Mode Workflow**
```
HTTP Request → Router → Context Injector Service →
Intent Classification → Context Retrieval → LLM Client Selection →
LLM Call → Response Normalization → Memory Storage → HTTP Response
```

### **Memory Storage Workflow**
```
User Prompt + LLM Response → Metadata Generation →
Embedding Generation → ChromaDB Storage → Confirmation
```

## 🛠️ Configuration Management

The system uses a hierarchical configuration approach:
1. **Base configuration** in `config.json`
2. **Environment variables** for API keys and secrets
3. **Runtime parameters** for dynamic adjustments
4. **Module-level defaults** for fallback values

## 🔧 Key Features and Capabilities

### **Intelligent Context Management**
- Semantic similarity-based context retrieval
- Adaptive thresholds based on conversation patterns
- Pronoun-aware re-ranking for better context relevance
- Short-term sliding window for factual continuity

### **Multi-Provider LLM Support**
- Unified interface for OpenAI, Anthropic, Mistral, and Gemini
- Automatic fallback handling for high availability
- Response normalization for consistent API contract
- Provider-specific optimizations

### **Robust Memory System**
- ChromaDB-based vector storage with metadata
- Automatic tagging and categorization
- Quality filtering to prevent memory pollution
- Session and episode tracking for context continuity

### **Enterprise-Grade Features**
- Error handling and graceful degradation
- Rate limiting and API abuse protection
- Health checks and monitoring endpoints
- Configurable performance parameters

## 🚀 Usage Examples

### **CLI Usage**
```bash
python main.py
```
Interactive prompts with automatic context retrieval and memory storage.

### **API Usage**
```bash
./start_cam.sh  # Start the proxy server
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

Standard OpenAI-compatible API with automatic memory augmentation.

---

This documentation provides a comprehensive overview of the CAM system architecture, data flows, and the purpose of each component. The system is designed for scalability, reliability, and easy integration with existing LLM applications.