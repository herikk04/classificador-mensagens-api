from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ClassifyIntentRequest(BaseModel):
    """
    Schema para requisição de classificação de intenção.
    Usado no endpoint POST /api/v1/classify
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Texto do usuário a ser classificado",
        examples=["Olá, bom dia!", "Quero cancelar meu pedido"]
    )
    
    request_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="ID único para rastreamento da requisição (opcional)",
        examples=["req_abc123"]
    )
    
    include_raw_response: bool = Field(
        default=False,
        description="Incluir resposta bruta do LLM no resultado"
    )
    
    include_metadata: bool = Field(
        default=False,
        description="Incluir metadados adicionais no resultado"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Preciso de ajuda com meu pedido",
                "request_id": "req_20260122_001",
                "include_raw_response": False,
                "include_metadata": False
            }
        }
    }
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """
        Valida e normaliza o texto de entrada.
        Remove espaços extras e verifica conteúdo válido.
        """
        v = v.strip()
        
        if not v:
            raise ValueError("O texto não pode estar vazio ou conter apenas espaços")
        
        # Verifica se contém pelo menos um caractere alfanumérico
        if not any(c.isalnum() for c in v):
            raise ValueError("O texto deve conter pelo menos um caractere alfanumérico")
        
        return v
    
    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, v: Optional[str]) -> Optional[str]:
        """Valida e normaliza o request_id."""
        if v is None:
            return None
        
        v = v.strip()
        
        if not v:
            return None
        
        # Remove caracteres inválidos
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "request_id deve conter apenas letras, números, hífens e underscores"
            )
        
        return v


class BatchClassifyIntentRequest(BaseModel):
    """
    Schema para requisição de classificação em lote.
    Usado no endpoint POST /api/v1/classify/batch
    """
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Lista de textos a serem classificados",
        examples=[["Olá!", "Preciso de ajuda", "Até logo"]]
    )
    
    request_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="ID único para rastreamento da requisição (opcional)"
    )
    
    include_raw_response: bool = Field(
        default=False,
        description="Incluir resposta bruta do LLM em cada resultado"
    )
    
    include_metadata: bool = Field(
        default=False,
        description="Incluir metadados adicionais em cada resultado"
    )
    
    fail_on_first_error: bool = Field(
        default=False,
        description="Interromper processamento ao encontrar primeiro erro"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "Bom dia!",
                    "Como faço para cancelar?",
                    "Muito obrigado pela ajuda"
                ],
                "request_id": "batch_req_001",
                "include_raw_response": False,
                "include_metadata": False,
                "fail_on_first_error": False
            }
        }
    }
    
    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        """
        Valida cada texto na lista.
        Aplica as mesmas regras do ClassifyIntentRequest.
        """
        validated_texts = []
        
        for idx, text in enumerate(v):
            text = text.strip()
            
            if not text:
                raise ValueError(
                    f"Texto na posição {idx} está vazio ou contém apenas espaços"
                )
            
            if not any(c.isalnum() for c in text):
                raise ValueError(
                    f"Texto na posição {idx} deve conter pelo menos um caractere alfanumérico"
                )
            
            if len(text) > 1000:
                raise ValueError(
                    f"Texto na posição {idx} excede o limite de 1000 caracteres"
                )
            
            validated_texts.append(text)
        
        return validated_texts
    
    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, v: Optional[str]) -> Optional[str]:
        """Valida e normaliza o request_id."""
        if v is None:
            return None
        
        v = v.strip()
        
        if not v:
            return None
        
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "request_id deve conter apenas letras, números, hífens e underscores"
            )
        
        return v


class LLMConfigOverrideRequest(BaseModel):
    """
    Schema para override de configurações do LLM (opcional).
    Permite ajustes fine-tuned por requisição.
    """
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperatura para geração (override do padrão)"
    )
    
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=8192,
        description="Máximo de tokens (override do padrão)"
    )
    
    max_examples: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Número de exemplos few-shot (override do padrão)"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "temperature": 0.5,
                "max_tokens": 256,
                "max_examples": 3
            }
        }
    }


class ClassifyIntentWithConfigRequest(ClassifyIntentRequest):
    """
    Schema estendido que permite override de configurações do LLM.
    Combina requisição de classificação com ajustes de configuração.
    """
    config_override: Optional[LLMConfigOverrideRequest] = Field(
        default=None,
        description="Configurações customizadas para esta requisição"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Preciso de suporte técnico urgente",
                "request_id": "req_custom_001",
                "include_raw_response": True,
                "include_metadata": True,
                "config_override": {
                    "temperature": 0.1,
                    "max_tokens": 128,
                    "max_examples": 3
                }
            }
        }
    }
