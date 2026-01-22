"""
Configurações centralizadas da aplicação.
Gerencia variáveis de ambiente e settings usando Pydantic V2.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configurações da aplicação com validação via Pydantic V2.
    Todas as configs podem ser sobrescritas via variáveis de ambiente.
    """
    
    # API Configuration
    app_name: str = Field(default="Intent Classifier API", description="Nome da aplicação")
    app_version: str = Field(default="1.0.0", description="Versão da API")
    debug: bool = Field(default=False, description="Modo debug")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Ambiente de execução"
    )
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Host do servidor")
    port: int = Field(default=8000, ge=1024, le=65535, description="Porta do servidor")
    reload: bool = Field(default=False, description="Auto-reload (dev only)")
    
    # Gemini Configuration
    gemini_api_key: str = Field(..., description="Chave de API do Google Gemini")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Modelo do Gemini a ser utilizado"
    )
    gemini_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperatura para geração (0.0 = determinístico, 2.0 = criativo)"
    )
    gemini_max_tokens: int = Field(
        default=512,
        ge=1,
        le=8192,
        description="Máximo de tokens na resposta"
    )
    gemini_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout para chamadas ao Gemini (segundos)"
    )
    
    # Few-Shot Configuration
    examples_file_path: str = Field(
        default="src/data/examples.json",
        description="Caminho para o arquivo de exemplos few-shot"
    )
    max_examples_in_prompt: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número máximo de exemplos no prompt"
    )
    
    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Nível de log"
    )
    log_format: str = Field(
        default="json",
        description="Formato do log (json ou text)"
    )
    
    # CORS Configuration - CORRIGIDO para aceitar string separada por vírgulas
    cors_origins: str = Field(
        default="*",
        description="Origens permitidas para CORS (separadas por vírgula)"
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: str = Field(default="*")
    cors_allow_headers: str = Field(default="*")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Habilitar rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests por minuto")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Valida se a API key não está vazia."""
        if not v or v.strip() == "":
            raise ValueError("GEMINI_API_KEY não pode estar vazia")
        if len(v) < 20:
            raise ValueError("GEMINI_API_KEY parece inválida (muito curta)")
        return v.strip()
    
    @field_validator("gemini_model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Valida se o modelo é compatível."""
        allowed_models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        if v not in allowed_models:
            raise ValueError(
                f"Modelo '{v}' não suportado. Use um dos seguintes: {', '.join(allowed_models)}"
            )
        return v
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ajusta configurações baseadas no ambiente."""
        return v.lower()
    
    # Métodos auxiliares para retornar listas
    def get_cors_origins_list(self) -> list[str]:
        """Retorna lista de origens CORS."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    def get_cors_methods_list(self) -> list[str]:
        """Retorna lista de métodos CORS."""
        if self.cors_allow_methods == "*":
            return ["*"]
        return [method.strip() for method in self.cors_allow_methods.split(",")]
    
    def get_cors_headers_list(self) -> list[str]:
        """Retorna lista de headers CORS."""
        if self.cors_allow_headers == "*":
            return ["*"]
        return [header.strip() for header in self.cors_allow_headers.split(",")]


@lru_cache
def get_settings() -> Settings:
    """
    Retorna instância singleton das configurações.
    Usa cache para evitar recarregar .env múltiplas vezes.
    
    Returns:
        Settings: Instância das configurações da aplicação
    """
    return Settings()


# Instância global para facilitar imports
settings = get_settings()
