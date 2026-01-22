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
- [Exemplos de Uso](#-exemplos-de-uso)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Deploy](#-deploy)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Características

- ✅ **Clean Architecture** com separação clara de responsabilidades
- ✅ **SOLID Principles** aplicados em toda a codebase
- ✅ **Few-Shot Learning** com exemplos configuráveis (25 exemplos em português)
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
- ✅ **Docker Ready** com docker-compose

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

| Categoria | Tecnologia | Versão |
|-----------|------------|--------|
| **Framework** | FastAPI | 0.115.5 |
| **Validação** | Pydantic V2 | 2.10.3 |
| **LLM** | Google Gemini 2.5 Flash | - |
| **Server** | Uvicorn (ASGI) | 0.32.1 |
| **Logging** | python-json-logger | 3.2.1 |
| **Async HTTP** | httpx | 0.28.1 |
| **Python** | 3.10+ | - |

---

## 📦 Pré-requisitos

- **Python 3.10+** instalado
- **Google Gemini API Key** ([Obtenha aqui](https://makersuite.google.com/app/apikey))
- **pip** atualizado

---

## 🚀 Instalação

### **1. Clone o repositório**

```bash
git clone https://github.com/seu-usuario/intent-classifier-api.git
cd intent-classifier-api
```

### **2. Crie um ambiente virtual**

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### **3. Instale as dependências**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuração

### **1. Configure as variáveis de ambiente**

```bash
# Copie o arquivo de exemplo
cp .env.example .env
```

### **2. Edite o arquivo `.env` e adicione sua API Key**

```bash
# OBRIGATÓRIO
GEMINI_API_KEY=sua_chave_api_do_gemini_aqui

# Opcional (já possui defaults)
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.3
ENVIRONMENT=development
DEBUG=true
```

> 🔑 **Importante**: Obtenha sua API Key gratuita em: https://makersuite.google.com/app/apikey

### **3. Variáveis Disponíveis**

| Variável | Descrição | Default | Obrigatória |
|----------|-----------|---------|-------------|
| `GEMINI_API_KEY` | Chave de API do Google Gemini | - | ✅ Sim |
| `GEMINI_MODEL` | Modelo Gemini a usar | `gemini-2.5-flash` | ❌ |
| `GEMINI_TEMPERATURE` | Temperatura (0.0-2.0) | `0.3` | ❌ |
| `GEMINI_MAX_TOKENS` | Máximo de tokens | `512` | ❌ |
| `ENVIRONMENT` | Ambiente (development/production) | `development` | ❌ |
| `DEBUG` | Modo debug | `false` | ❌ |
| `LOG_LEVEL` | Nível de log | `INFO` | ❌ |
| `PORT` | Porta do servidor | `8000` | ❌ |

---

## 💻 Uso

### **Iniciar o servidor**

```powershell
# Adicione o path ao PYTHONPATH
$env:PYTHONPATH = (Get-Location).Path

# Inicie com uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Ou crie um arquivo `run.py` na raiz:**

```python
import sys
from pathlib import Path

# Adiciona o diretório raiz ao PYTHONPATH
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

if __name__ == "__main__":
    import uvicorn
    from src.core.config import settings
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
```

E execute:
```powershell
python run.py
```

### **Acessar a documentação**

Após iniciar o servidor, acesse:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📡 Endpoints

### **1. Health Check**

```http
GET /health
```

**Resposta:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-22T19:44:00.000000",
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

### **2. Classificar Intenção (Único)**

```http
POST /api/v1/classify
```

**Request Body:**
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
  "processing_time_ms": 1109.02,
  "timestamp": "2026-01-22T19:44:00.000000",
  "request_id": "req_001"
}
```

---

### **3. Classificar em Lote**

```http
POST /api/v1/classify/batch
```

**Request Body:**
```json
{
  "texts": [
    "Bom dia!",
    "Como faço para rastrear?",
    "Obrigado!"
  ],
  "request_id": "batch_001"
}
```

**Response:**
```json
{
  "results": [
    {
      "intent": "greeting",
      "confidence": 0.95,
      "confidence_level": "high",
      "processing_time_ms": 1649.69
    },
    {
      "intent": "question",
      "confidence": 0.95,
      "confidence_level": "high",
      "processing_time_ms": 1470.58
    },
    {
      "intent": "compliment",
      "confidence": 0.95,
      "confidence_level": "high",
      "processing_time_ms": 1301.95
    }
  ],
  "total_processed": 3,
  "total_successful": 3,
  "total_failed": 0,
  "total_processing_time_ms": 4930.59,
  "timestamp": "2026-01-22T19:44:00.000000",
  "request_id": "batch_001"
}
```

---

### **4. Informações do Modelo**

```http
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
  "timestamp": "2026-01-22T19:44:00.000000"
}
```

---

## 🧪 Exemplos de Uso

### **Python (requests)**

```python
import requests

# Classificação única
response = requests.post(
    "http://localhost:8000/api/v1/classify",
    json={"text": "Olá, bom dia!"}
)
print(response.json())

# Classificação em lote
response = requests.post(
    "http://localhost:8000/api/v1/classify/batch",
    json={
        "texts": [
            "Bom dia!",
            "Preciso de ajuda",
            "Obrigado!"
        ]
    }
)
print(response.json())
```

### **cURL**

```bash
# Classificação única
curl -X POST "http://localhost:8000/api/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá, bom dia!"}'

# Health check
curl http://localhost:8000/health
```

### **PowerShell**

```powershell
# Health Check
Invoke-RestMethod -Uri http://localhost:8000/health

# Classificação
$body = '{"text": "Olá, bom dia!"}' 
Invoke-RestMethod -Uri http://localhost:8000/api/v1/classify -Method Post -Body $body -ContentType "application/json"
```

---

## 📂 Estrutura do Projeto

```
intent-classifier-api/
├── src/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   └── classifier.py      # Rotas REST
│   │   │   └── dependencies.py        # Injeção de dependências
│   │   └── middlewares/
│   │       └── logging_middleware.py  # Logging automático
│   ├── core/
│   │   ├── config.py                  # Configurações
│   │   ├── logger.py                  # Logging estruturado
│   │   └── exceptions.py              # Exceções customizadas
│   ├── domain/
│   │   ├── models.py                  # Modelos de domínio
│   │   └── interfaces.py              # Abstrações/Contratos
│   ├── services/
│   │   ├── intent_service.py          # Lógica de negócio
│   │   └── prompt_manager.py          # Gerenciamento de prompts
│   ├── providers/
│   │   └── gemini/
│   │       └── client.py              # Cliente Gemini
│   ├── schemas/
│   │   ├── request.py                 # DTOs de entrada
│   │   └── response.py                # DTOs de saída
│   ├── data/
│   │   └── examples.json              # 25 exemplos few-shot (pt-BR)
│   └── main.py                        # Entry point
├── .env.example                       # Template de variáveis de ambiente
├── .gitignore                         # Arquivos ignorados pelo Git
├── requirements.txt                   # Dependências Python
├── Dockerfile                         # Container Docker
├── docker-compose.yml                 # Orquestração Docker
├── pyproject.toml                     # Configuração do projeto
└── README.md                          # Este arquivo
```

---

## 🎯 Intenções Suportadas

A API classifica textos em 11 categorias:

| Intenção | Descrição | Exemplo |
|----------|-----------|---------|
| `greeting` | Saudações | "Olá, bom dia!" |
| `farewell` | Despedidas | "Até logo, obrigado!" |
| `question` | Perguntas | "Como faço para rastrear?" |
| `complaint` | Reclamações | "Produto com defeito!" |
| `compliment` | Elogios | "Muito obrigado!" |
| `request` | Solicitações | "Quero trocar o produto" |
| `information` | Informações | "Meu pedido é #12345" |
| `help` | Ajuda | "Preciso de ajuda" |
| `cancellation` | Cancelamentos | "Quero cancelar" |
| `confirmation` | Confirmações | "Sim, confirmo" |
| `unknown` | Não identificado | Textos ambíguos |

---

## 🐳 Deploy com Docker

### **Build e Run**

```bash
# Build da imagem
docker build -t intent-classifier-api .

# Run do container
docker run -p 8000:8000 --env-file .env intent-classifier-api
```

### **Docker Compose**

```bash
# Inicie todos os serviços
docker-compose up -d

# Veja os logs
docker-compose logs -f

# Pare os serviços
docker-compose down
```

---

## 🐛 Troubleshooting

### **Erro: "GEMINI_API_KEY não pode estar vazia"**

✅ **Solução**: Configure a variável `GEMINI_API_KEY` no arquivo `.env`

```bash
GEMINI_API_KEY=sua_chave_aqui
```

### **Erro: "Module not found: 'src'"**

✅ **Solução**: Configure o PYTHONPATH antes de executar

```powershell
# PowerShell
$env:PYTHONPATH = (Get-Location).Path
uvicorn src.main:app --reload

# Linux/macOS
export PYTHONPATH=$(pwd)
uvicorn src.main:app --reload
```

### **Erro: "Arquivo de exemplos não encontrado"**

✅ **Solução**: Verifique se `src/data/examples.json` existe

### **Performance lenta (>3s por requisição)****

✅ **Soluções**:
- Reduza `GEMINI_TEMPERATURE` para 0.1
- Reduza `MAX_EXAMPLES_IN_PROMPT` para 3
- Aumente `GEMINI_TIMEOUT` para 60

### **Erro: "Rate limit excedido"**

✅ **Solução**: Aguarde alguns segundos entre requisições ou use a API Key paga do Gemini

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 👨‍💻 Desenvolvido com

- ❤️ **Clean Architecture**
- 🎯 **SOLID Principles**
- 🚀 **FastAPI + Gemini 2.5 Flash**
- 🐍 **Python 3.10+**

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Add: Nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

---

## 📞 Suporte

Para dúvidas, problemas ou sugestões, abra uma [issue](https://github.com/seu-usuario/intent-classifier-api/issues).

---

**⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!**
```
