# 🎯 Intent Classifier API

API RESTful robusta para classificação de intenção usando **Few-Shot Prompting** com **Gemini 2.5 Flash**.

Construída com **Clean Architecture**, **SOLID principles** e **Python 3.11+**.

---

## 📋 Índice

- [Características](#-características)
- [Arquitetura](#-arquitetura)
- [Tecnologias](#-tecnologias)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Uso](#-uso)
- [Endpoints](#-endpoints)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Desenvolvimento](#-desenvolvimento)
- [Deploy](#-deploy)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Características

- ✅ **Clean Architecture** com separação clara de responsabilidades
- ✅ **SOLID Principles** aplicados em toda a codebase
- ✅ **Few-Shot Learning** com exemplos configuráveis
- ✅ **Gemini 2.5 Flash** integração assíncrona
- ✅ **FastAPI** com validação Pydantic V2
- ✅ **Logging Estruturado** (JSON) para observabilidade
- ✅ **Health Check** completo com verificação de dependências
- ✅ **Classificação em Lote** para processar múltiplos textos
- ✅ **Request Tracking** com IDs únicos
- ✅ **CORS** configurável
- ✅ **Documentação Automática** (OpenAPI/Swagger)
- ✅ **Type Hints** completos
- ✅ **Async/Await** para I/O não-bloqueante

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Endpoints  │  │ Middlewares  │  │ Dependencies│ │
│  └────────────┘  └──────────────┘  └─────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               Service Layer (Business Logic)         │
│  ┌─────────────────┐        ┌──────────────────┐   │
│  │ IntentService   │◄───────┤ PromptManager    │   │
│  └─────────────────┘        └──────────────────┘   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Domain Layer (Core)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Interfaces  │  │    Models    │  │Exceptions │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            Infrastructure Layer (Providers)          │
│  ┌──────────────────────────────────────────────┐  │
│  │       GeminiClient (LLM Provider)            │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Princípios Aplicados:**
- **Dependency Inversion**: Camadas superiores dependem de abstrações (interfaces)
- **Single Responsibility**: Cada classe tem uma única responsabilidade
- **Open/Closed**: Extensível sem modificar código existente
- **Interface Segregation**: Interfaces específicas para cada contexto
- **Liskov Substitution**: Implementações podem ser substituídas sem quebrar o sistema

---

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|-----------|------------|
| **Framework** | FastAPI 0.115.5 |
| **Validação** | Pydantic V2 |
| **LLM** | Google Gemini 2.5 Flash |
| **Server** | Uvicorn (ASGI) |
| **Logging** | python-json-logger |
| **Async HTTP** | httpx |
| **Python** | 3.11+ |

---

## 📦 Pré-requisitos

- **Python 3.11+** instalado
- **Google Gemini API Key** ([Obtenha aqui](https://makersuite.google.com/app/apikey))
- **Git** (opcional)

---

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/intent-classifier-api.git
cd intent-classifier-api
```

### 2. Crie um ambiente virtual

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuração

### 1. Configure as variáveis de ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Edite o arquivo .env e adicione sua API Key
# GEMINI_API_KEY=sua_chave_aqui
```

### 2. Variáveis obrigatórias

```bash
GEMINI_API_KEY=your_gemini_api_key_here  # ⚠️ OBRIGATÓRIO
```

### 3. Variáveis opcionais (com defaults)

```bash
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.3
GEMINI_MAX_TOKENS=512
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## 💻 Uso

### Iniciar o servidor

```bash
# Modo desenvolvimento (com auto-reload)
python src/main.py

# Ou usando uvicorn diretamente
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Acessar a documentação

Abra no navegador:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📡 Endpoints

### 1. Classificar Intenção (Único)

```bash
POST /api/v1/classify
```

**Request:**
```json
{
  "text": "Olá, bom dia!",
  "request_id": "req_001",
  "include_raw_response": false,
  "include_metadata": false
}
```

**Response:**
```json
{
  "intent": "greeting",
  "confidence": 0.95,
  "confidence_level": "high",
  "processing_time_ms": 234.56,
  "timestamp": "2026-01-22T18:36:00.000000",
  "request_id": "req_001"
}
```

### 2. Classificar em Lote

```bash
POST /api/v1/classify/batch
```

**Request:**
```json
{
  "texts": [
    "Bom dia!",
    "Preciso de ajuda",
    "Obrigado!"
  ],
  "request_id": "batch_001"
}
```

**Response:**
```json
{
  "results": [...],
  "total_processed": 3,
  "total_successful": 3,
  "total_failed": 0,
  "total_processing_time_ms": 456.78,
  "timestamp": "2026-01-22T18:36:00.000000",
  "request_id": "batch_001"
}
```

### 3. Informações do Modelo

```bash
GET /api/v1/classify/model/info
```

**Response:**
```json
{
  "model_name": "gemini-2.5-flash",
  "provider": "Google Gemini",
  "temperature": 0.3,
  "max_tokens": 512,
  "examples_count": 25,
  "supported_intents": [
    "greeting", "farewell", "question", "complaint",
    "compliment", "request", "information", "help",
    "cancellation", "confirmation", "unknown"
  ],
  "timestamp": "2026-01-22T18:36:00.000000"
}
```

### 4. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-22T18:36:00.000000",
  "version": "1.0.0",
  "checks": {
    "llm_provider": true,
    "examples_loaded": true,
    "configuration": true
  },
  "details": {
    "model": "gemini-2.5-flash",
    "examples_count": 25,
    "environment": "development"
  }
}
```

---

## 📂 Estrutura do Projeto

```
intent-classifier-api/
├── src/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   └── classifier.py    # Rotas REST
│   │   │   └── dependencies.py      # Injeção de dependências
│   │   └── middlewares/
│   │       └── logging_middleware.py # Logging automático
│   ├── core/
│   │   ├── config.py                # Configurações
│   │   ├── logger.py                # Logging estruturado
│   │   └── exceptions.py            # Exceções customizadas
│   ├── domain/
│   │   ├── models.py                # Modelos de domínio
│   │   └── interfaces.py            # Abstrações/Contratos
│   ├── services/
│   │   ├── intent_service.py        # Lógica de negócio
│   │   └── prompt_manager.py        # Gerenciamento de prompts
│   ├── providers/
│   │   └── gemini/
│   │       └── client.py            # Cliente Gemini
│   ├── schemas/
│   │   ├── request.py               # DTOs de entrada
│   │   └── response.py              # DTOs de saída
│   ├── data/
│   │   └── examples.json            # Exemplos few-shot
│   └── main.py                      # Entry point
├── .env.example                     # Variáveis de ambiente
├── requirements.txt                 # Dependências
└── README.md                        # Este arquivo
```

---

## 🔧 Desenvolvimento

### Adicionar novos exemplos

Edite `src/data/examples.json`:

```json
{
  "examples": [
    {
      "user_input": "Seu novo exemplo",
      "intent": "greeting",
      "confidence": 0.95,
      "metadata": {}
    }
  ]
}
```

### Adicionar nova intenção

1. Edite `src/domain/models.py`:
```python
class IntentType(str, Enum):
    # ... existing intents
    NEW_INTENT = "new_intent"
```

2. Adicione exemplos em `src/data/examples.json`

3. Atualize a system instruction em `src/services/prompt_manager.py`

---

## 🐳 Deploy

### Docker (em breve)

```bash
docker build -t intent-classifier-api .
docker run -p 8000:8000 --env-file .env intent-classifier-api
```

### Produção

```bash
# Instale dependências
pip install -r requirements.txt

# Configure variáveis de ambiente
export ENVIRONMENT=production
export DEBUG=false
export GEMINI_API_KEY=your_key

# Inicie com Gunicorn
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 🐛 Troubleshooting

### Erro: "GEMINI_API_KEY não pode estar vazia"

✅ **Solução**: Configure a variável `GEMINI_API_KEY` no arquivo `.env`

### Erro: "Arquivo de exemplos não encontrado"

✅ **Solução**: Verifique se `src/data/examples.json` existe

### Erro: "Module not found"

✅ **Solução**: Certifique-se de que todas as dependências estão instaladas:
```bash
pip install -r requirements.txt
```

### Performance lenta

✅ **Solução**: Ajuste `GEMINI_TEMPERATURE` para valores menores (ex: 0.1) ou reduza `MAX_EXAMPLES_IN_PROMPT`

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 👨‍💻 Autor

Desenvolvido seguindo **Clean Architecture** e **SOLID principles**.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Add: Nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

---

## 📞 Suporte

Para questões e suporte, abra uma [issue](https://github.com/seu-usuario/intent-classifier-api/issues).
```

***