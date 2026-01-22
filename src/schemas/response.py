from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.domain.models import ConfidenceLevel, IntentType


class ClassifyIntentResponse(BaseModel):
    """
    Schema de resposta para classificação de intenção.
    Retornado pelo endpoint POST /api/v1/classify
    """
    intent: IntentType = Field(
        ...,
        description="Intenção identificada"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nível de confiança da classificação (0.0 a 1.0)"
    )
    
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Categoria do nível de confiança (high, medium, low)"
    )
    
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo de processamento em milissegundos"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp da classificação (UTC)"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="ID da requisição para rastreamento"
    )
    
    raw_response: Optional[str] = Field(
        default=None,
        description="Resposta bruta do LLM (se solicitado)"
    )
    
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Metadados adicionais (se solicitados)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "intent": "greeting",
                "confidence": 0.95,
                "confidence_level": "high",
                "processing_time_ms": 234.56,
                "timestamp": "2026-01-22T18:26:00.000000",
                "request_id": "req_20260122_001",
                "raw_response": None,
                "metadata": None
            }
        }
    }


class BatchClassifyIntentResponse(BaseModel):
    """
    Schema de resposta para classificação em lote.
    Retornado pelo endpoint POST /api/v1/classify/batch
    """
    results: list[ClassifyIntentResponse] = Field(
        ...,
        description="Lista de resultados de classificação"
    )
    
    total_processed: int = Field(
        ...,
        ge=0,
        description="Número total de textos processados"
    )
    
    total_successful: int = Field(
        ...,
        ge=0,
        description="Número de classificações bem-sucedidas"
    )
    
    total_failed: int = Field(
        ...,
        ge=0,
        description="Número de classificações que falharam"
    )
    
    total_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo total de processamento em milissegundos"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp do processamento em lote (UTC)"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="ID da requisição em lote para rastreamento"
    )
    
    errors: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Lista de erros ocorridos (se houver)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "results": [
                    {
                        "intent": "greeting",
                        "confidence": 0.95,
                        "confidence_level": "high",
                        "processing_time_ms": 120.5,
                        "timestamp": "2026-01-22T18:26:00.000000"
                    },
                    {
                        "intent": "help",
                        "confidence": 0.87,
                        "confidence_level": "high",
                        "processing_time_ms": 135.2,
                        "timestamp": "2026-01-22T18:26:00.100000"
                    }
                ],
                "total_processed": 2,
                "total_successful": 2,
                "total_failed": 0,
                "total_processing_time_ms": 255.7,
                "timestamp": "2026-01-22T18:26:00.200000",
                "request_id": "batch_req_001",
                "errors": None
            }
        }
    }


class HealthCheckResponse(BaseModel):
    """
    Schema de resposta para health check.
    Retornado pelo endpoint GET /api/v1/health
    """
    status: str = Field(
        ...,
        description="Status geral da aplicação (healthy, degraded, unhealthy)"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Timestamp da verificação (UTC)"
    )
    
    version: str = Field(
        ...,
        description="Versão da aplicação"
    )
    
    checks: dict[str, bool] = Field(
        ...,
        description="Status de cada componente verificado"
    )
    
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Detalhes adicionais sobre os componentes"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2026-01-22T18:26:00.000000",
                "version": "1.0.0",
                "checks": {
                    "llm_provider": True,
                    "examples_loaded": True,
                    "configuration": True
                },
                "details": {
                    "model": "gemini-2.5-flash",
                    "examples_count": 15,
                    "environment": "production"
                }
            }
        }
    }


class ErrorResponse(BaseModel):
    """
    Schema de resposta para erros.
    Usado em todos os endpoints quando ocorrem erros.
    """
    error_code: str = Field(
        ...,
        description="Código identificador do erro"
    )
    
    message: str = Field(
        ...,
        description="Mensagem descritiva do erro"
    )
    
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Detalhes adicionais sobre o erro"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp do erro (UTC)"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="ID da requisição para rastreamento"
    )
    
    path: Optional[str] = Field(
        default=None,
        description="Path da requisição que gerou o erro"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "message": "O texto não pode estar vazio",
                "details": {
                    "field": "text",
                    "constraint": "min_length"
                },
                "timestamp": "2026-01-22T18:26:00.000000",
                "request_id": "req_error_001",
                "path": "/api/v1/classify"
            }
        }
    }


class ModelInfoResponse(BaseModel):
    """
    Schema de resposta com informações do modelo.
    Retornado pelo endpoint GET /api/v1/model/info
    """
    model_name: str = Field(
        ...,
        description="Nome do modelo LLM sendo usado"
    )
    
    provider: str = Field(
        ...,
        description="Provider do modelo (ex: Google Gemini)"
    )
    
    temperature: float = Field(
        ...,
        description="Temperatura padrão configurada"
    )
    
    max_tokens: int = Field(
        ...,
        description="Máximo de tokens padrão configurado"
    )
    
    examples_count: int = Field(
        ...,
        ge=0,
        description="Número de exemplos few-shot carregados"
    )
    
    supported_intents: list[str] = Field(
        ...,
        description="Lista de intenções suportadas"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp da consulta (UTC)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_name": "gemini-2.5-flash",
                "provider": "Google Gemini",
                "temperature": 0.3,
                "max_tokens": 512,
                "examples_count": 15,
                "supported_intents": [
                    "greeting",
                    "farewell",
                    "question",
                    "complaint",
                    "help"
                ],
                "timestamp": "2026-01-22T18:26:00.000000"
            }
        }
    }


class MetricsResponse(BaseModel):
    """
    Schema de resposta para métricas da aplicação.
    Retornado pelo endpoint GET /api/v1/metrics (opcional)
    """
    total_requests: int = Field(
        ...,
        ge=0,
        description="Total de requisições processadas"
    )
    
    successful_requests: int = Field(
        ...,
        ge=0,
        description="Requisições bem-sucedidas"
    )
    
    failed_requests: int = Field(
        ...,
        ge=0,
        description="Requisições que falharam"
    )
    
    average_processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo médio de processamento em milissegundos"
    )
    
    intent_distribution: dict[str, int] = Field(
        ...,
        description="Distribuição de intenções classificadas"
    )
    
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Tempo de atividade em segundos"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp da consulta de métricas (UTC)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_requests": 1523,
                "successful_requests": 1498,
                "failed_requests": 25,
                "average_processing_time_ms": 187.34,
                "intent_distribution": {
                    "greeting": 342,
                    "help": 289,
                    "question": 456,
                    "complaint": 123
                },
                "uptime_seconds": 86400.0,
                "timestamp": "2026-01-22T18:26:00.000000"
            }
        }
    }
