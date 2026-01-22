from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class IntentType(str, Enum):
    """
    Enumeração dos tipos de intenção suportados.
    Adicione novas intenções conforme necessário.
    """

    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    REQUEST = "request"
    INFORMATION = "information"
    HELP = "help"
    CANCELLATION = "cancellation"
    CONFIRMATION = "confirmation"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "IntentType":
        """
        Converte string para IntentType de forma case-insensitive.

        Args:
            value: String representando a intenção

        Returns:
            IntentType: Tipo de intenção correspondente
        """
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


class ConfidenceLevel(str, Enum):
    """Nível de confiança da classificação."""

    HIGH = "high"  # >= 0.8
    MEDIUM = "medium"  # >= 0.5 e < 0.8
    LOW = "low"  # < 0.5


class FewShotExample(BaseModel):
    """
    Modelo para exemplos few-shot usados no prompt.
    Representa pares de input/output para aprendizado do LLM.
    """

    user_input: str = Field(
        ..., min_length=1, max_length=500, description="Texto de exemplo fornecido pelo usuário"
    )
    intent: IntentType = Field(..., description="Intenção classificada para este exemplo")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confiança da classificação (0.0 a 1.0)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadados adicionais do exemplo"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_input": "Olá, bom dia!",
                "intent": "greeting",
                "confidence": 0.95,
                "metadata": {"language": "pt-BR"},
            }
        }
    }

    @field_validator("user_input")
    @classmethod
    def validate_user_input(cls, v: str) -> str:
        """Valida e normaliza o input do usuário."""
        v = v.strip()
        if not v:
            raise ValueError("user_input não pode estar vazio")
        return v


class ClassificationResult(BaseModel):
    """
    Resultado da classificação de intenção.
    Modelo principal retornado pelo serviço de classificação.
    """

    intent: IntentType = Field(..., description="Intenção identificada")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Nível de confiança da classificação (0.0 a 1.0)"
    )
    confidence_level: ConfidenceLevel = Field(..., description="Categoria do nível de confiança")
    raw_response: str = Field(..., description="Resposta bruta do LLM")
    processing_time_ms: float = Field(
        ..., ge=0.0, description="Tempo de processamento em milissegundos"
    )
    model_used: str = Field(..., description="Nome do modelo LLM utilizado")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp da classificação (UTC)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadados adicionais da classificação"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "intent": "greeting",
                "confidence": 0.92,
                "confidence_level": "high",
                "raw_response": "greeting",
                "processing_time_ms": 234.56,
                "model_used": "gemini-2.5-flash",
                "timestamp": "2026-01-22T18:24:00.000000",
                "metadata": {},
            }
        }
    }

    @field_validator("confidence_level", mode="before")
    @classmethod
    def calculate_confidence_level(cls, v: Any, info: Any) -> ConfidenceLevel:
        """Calcula o nível de confiança baseado no score numérico."""
        # Se já foi fornecido, retorna
        if isinstance(v, ConfidenceLevel):
            return v

        # Obtém o valor de confidence dos dados
        confidence = info.data.get("confidence", 0.0)

        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class LLMRequest(BaseModel):
    """
    Modelo para requisições ao LLM.
    Encapsula parâmetros necessários para chamadas ao provider.
    """

    prompt: str = Field(..., min_length=1, description="Prompt a ser enviado ao LLM")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Temperatura da geração")
    max_tokens: int = Field(default=512, ge=1, le=8192, description="Máximo de tokens na resposta")
    model: str = Field(default="gemini-2.5-flash", description="Nome do modelo a ser usado")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadados da requisição")


class LLMResponse(BaseModel):
    """
    Modelo para respostas do LLM.
    Encapsula dados retornados pelo provider.
    """

    text: str = Field(..., description="Texto gerado pelo LLM")
    model: str = Field(..., description="Modelo que gerou a resposta")
    tokens_used: Optional[int] = Field(
        default=None, ge=0, description="Número de tokens utilizados"
    )
    finish_reason: Optional[str] = Field(default=None, description="Razão de término da geração")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadados adicionais da resposta"
    )


class HealthStatus(BaseModel):
    """
    Status de saúde da aplicação e suas dependências.
    """

    status: str = Field(..., description="Status geral (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp da verificação"
    )
    checks: dict[str, bool] = Field(
        default_factory=dict, description="Status de cada componente verificado"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Detalhes adicionais do health check"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2026-01-22T18:24:00.000000",
                "checks": {"llm_provider": True, "examples_loaded": True},
                "details": {"model": "gemini-2.5-flash", "examples_count": 15},
            }
        }
    }
