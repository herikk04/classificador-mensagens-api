# Intent Classifier API

API RESTful para classificação de intenção em mensagens de texto, usando Few-Shot Prompting com Gemini 2.5 Flash. Construída com Clean Architecture, princípios SOLID e Python 3.11+.

---

## Índice

- [Características](#características)
- [Arquitetura](#arquitetura)
- [Tecnologias](#tecnologias)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Endpoints](#endpoints)
- [Exemplos de uso](#exemplos-de-uso)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Deploy](#deploy-com-docker)
- [Troubleshooting](#troubleshooting)

---

## Características

- Clean Architecture com separação clara de responsabilidades
- Princípios SOLID aplicados em toda a codebase
- Few-Shot Learning com exemplos configuráveis (25 exemplos em português)
- Integração assíncrona com Gemini 2.5 Flash
- FastAPI com validação Pydantic V2
- Logging estruturado (JSON) para observabilidade
- Health check completo, com verificação de dependências
- Classificação em lote para processar múltiplos textos
- Request tracking com IDs únicos
- CORS configurável
- Documentação automática (OpenAPI/Swagger)
- Type hints completos
- Async/await para I/O não-bloqueante
- Pronto para Docker (com docker-compose)

---

## Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Endpoints  │  │ Middlewares  │  │ Dependencies│ │
│  └────────────┘  └──────────────┘  └─────────────┘ │
└──────────────────────┬──────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│               Service Layer (Business Logic)         │
│  ┌─────────────────┐        ┌──────────────────┐    │
│  │ IntentService   │◄───────┤ PromptManager    │    │
│  └─────────────────┘        └──────────────────┘    │
└──────────────────────┬───────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│                  Domain Layer (Core)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Interfaces  │  │    Models    │  │Exceptions │  │
│  └──────────────┘  └──────────────┘  └───────────┘  │
└──────────────────────┬───────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────┐
│            Infrastructure Layer (Providers)           │
│  ┌──────────────────────────────────────────────┐    │
│  │       GeminiClient (LLM Provider)            │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

**Princípios aplicados:**

- **Dependency Inversion** — camadas superiores dependem de abstrações (interfaces)
- **Single Responsibility** — cada classe tem uma única responsabilidade
- **Open/Closed** — extensível sem modificar código existente
- **Interface Segregation** — interfaces específicas para cada contexto
- **Liskov Substitution** — implementações podem ser substituídas sem quebrar o sistema

---

## Tecnologias

| Categoria      | Tecnologia              | Versão  |
| -------------- | ----------------------- | ------- |
| Framework      | FastAPI                 | 0.115.5 |
| Validação      | Pydantic V2              | 2.10.3  |
| LLM            | Google Gemini 2.5 Flash | -       |
| Server         | Uvicorn (ASGI)           | 0.32.1  |
| Logging        | python-json-logger       | 3.2.1   |
| Async HTTP     | httpx                    | 0.28.1  |
| Python         | 3.10+                    | -       |

---

## Pré-requisitos

- Python 3.10+
- Google Gemini API Key ([obtenha aqui](https://makersuite.google.com/app/apikey))
- pip atualizado

---

## Instalação

**1. Clone o repositório**

```bash
git clone https://github.com/herikk04/classificador-mensagens-api.git
cd classificador-mensagens-api
```

**2. Crie um ambiente virtual**

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

**3. Instale as dependências**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuração

**1. Configure as variáveis de ambiente**

```bash
cp .env.example .env
```

**2. Edite o `.env` e adicione sua API key**

```env
# Obrigatório
GEMINI_API_KEY=sua_chave_api_do_gemini_aqui

# Opcional (já possui defaults)
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.3
ENVIRONMENT=development
DEBUG=true
```

> Chave de API gratuita em: <https://makersuite.google.com/app/apikey>

**3. Variáveis disponíveis**

| Variável             | Descrição                         | Default            | Obrigatória |
| -------------------- | --------------------------------- | ------------------ | ----------- |
| `GEMINI_API_KEY`     | Chave de API do Google Gemini     | -                  | Sim         |
| `GEMINI_MODEL`       | Modelo Gemini a usar              | `gemini-2.5-flash` | Não         |
| `GEMINI_TEMPERATURE` | Temperatura (0.0-2.0)             | `0.3`              | Não         |
| `GEMINI_MAX_TOKENS`  | Máximo de tokens                  | `512`              | Não         |
| `ENVIRONMENT`        | Ambiente (development/production) | `development`      | Não         |
| `DEBUG`              | Modo debug                        | `false`             | Não         |
| `LOG_LEVEL`          | Nível de log                      | `INFO`              | Não         |
| `PORT`               | Porta do servidor                 | `8000`              | Não         |

---

## Uso

**Iniciar o servidor**

```bash
export PYTHONPATH=$(pwd)   # Linux/macOS
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Acessar a documentação**

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- Health Check: <http://localhost:8000/health>

---

## Endpoints

**Health Check**

```
GET /health
```

**Classificar intenção (único)**

```
POST /api/v1/classify
```

```json
{
  "text": "Olá, bom dia!",
  "request_id": "req_001"
}
```

**Classificar em lote**

```
POST /api/v1/classify/batch
```

```json
{
  "texts": ["Bom dia!", "Como faço para rastrear?", "Obrigado!"],
  "request_id": "batch_001"
}
```

**Informações do modelo**

```
GET /api/v1/classify/model/info
```

A API classifica textos em 11 categorias: `greeting`, `farewell`, `question`, `complaint`, `compliment`, `request`, `information`, `help`, `cancellation`, `confirmation`, `unknown`.

---

## Exemplos de uso

**Python**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/classify",
    json={"text": "Olá, bom dia!"}
)
print(response.json())
```

**cURL**

```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá, bom dia!"}'
```

---

## Estrutura do projeto

```
classificador-mensagens-api/
├── src/
│   ├── api/v1/endpoints/classifier.py
│   ├── api/v1/dependencies.py
│   ├── api/middlewares/logging_middleware.py
│   ├── core/config.py
│   ├── core/logger.py
│   ├── core/exceptions.py
│   ├── domain/models.py
│   ├── domain/interfaces.py
│   ├── services/intent_service.py
│   ├── services/prompt_manager.py
│   ├── providers/gemini/client.py
│   ├── schemas/request.py
│   ├── schemas/response.py
│   ├── data/examples.json
│   └── main.py
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Deploy com Docker

```bash
docker build -t classificador-mensagens-api .
docker run -p 8000:8000 --env-file .env classificador-mensagens-api

# ou, com docker-compose
docker-compose up -d
```

---

## Troubleshooting

**"GEMINI_API_KEY não pode estar vazia"** — configure a variável `GEMINI_API_KEY` no `.env`.

**"Module not found: 'src'"** — configure o `PYTHONPATH` antes de executar (`export PYTHONPATH=$(pwd)`).

**"Arquivo de exemplos não encontrado"** — verifique se `src/data/examples.json` existe.

**Performance lenta (>3s por requisição)** — reduza `GEMINI_TEMPERATURE` ou `MAX_EXAMPLES_IN_PROMPT`, ou aumente `GEMINI_TIMEOUT`.

**"Rate limit excedido"** — aguarde entre requisições ou use uma chave paga do Gemini.

---

## Licença

MIT.
