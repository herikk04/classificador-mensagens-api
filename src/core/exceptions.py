from typing import Any, Optional


class AppBaseException(Exception):
    """
    Exceção base da aplicação.
    Todas as exceções customizadas devem herdar desta classe.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a exceção para dict (útil para respostas HTTP)."""
        return {"error_code": self.error_code, "message": self.message, "details": self.details}


# === Domain Exceptions ===


class DomainException(AppBaseException):
    """Exceções relacionadas à lógica de domínio."""

    pass


class InvalidIntentException(DomainException):
    """Intenção inválida ou não reconhecida."""

    def __init__(self, intent: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Intenção inválida: '{intent}'",
            error_code="INVALID_INTENT",
            details=details or {"intent": intent},
        )


class ClassificationFailedException(DomainException):
    """Falha ao classificar a intenção do usuário."""

    def __init__(self, reason: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Falha na classificação: {reason}",
            error_code="CLASSIFICATION_FAILED",
            details=details,
        )


# === Service Exceptions ===


class ServiceException(AppBaseException):
    """Exceções relacionadas à camada de serviço."""

    pass


class PromptBuildException(ServiceException):
    """Erro ao construir o prompt few-shot."""

    def __init__(self, reason: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=f"Erro ao construir prompt: {reason}",
            error_code="PROMPT_BUILD_ERROR",
            details=details,
        )


class ExamplesLoadException(ServiceException):
    """Erro ao carregar exemplos few-shot do arquivo."""

    def __init__(self, file_path: str, reason: str) -> None:
        super().__init__(
            message=f"Erro ao carregar exemplos de '{file_path}': {reason}",
            error_code="EXAMPLES_LOAD_ERROR",
            details={"file_path": file_path, "reason": reason},
        )


# === Provider Exceptions ===


class ProviderException(AppBaseException):
    """Exceções relacionadas a provedores externos."""

    pass


class LLMProviderException(ProviderException):
    """Exceção base para erros de provedores LLM."""

    pass


class GeminiAPIException(LLMProviderException):
    """Erro ao comunicar com a API do Gemini."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if status_code:
            error_details["status_code"] = status_code

        super().__init__(
            message=f"Erro na API Gemini: {message}",
            error_code="GEMINI_API_ERROR",
            details=error_details,
        )


class GeminiTimeoutException(LLMProviderException):
    """Timeout ao chamar a API do Gemini."""

    def __init__(self, timeout: int) -> None:
        super().__init__(
            message=f"Timeout ao chamar Gemini API ({timeout}s)",
            error_code="GEMINI_TIMEOUT",
            details={"timeout_seconds": timeout},
        )


class GeminiRateLimitException(LLMProviderException):
    """Rate limit excedido na API do Gemini."""

    def __init__(self, retry_after: Optional[int] = None) -> None:
        details = {"retry_after_seconds": retry_after} if retry_after else {}
        super().__init__(
            message="Rate limit excedido na API do Gemini",
            error_code="GEMINI_RATE_LIMIT",
            details=details,
        )


class GeminiInvalidResponseException(LLMProviderException):
    """Resposta inválida ou malformada do Gemini."""

    def __init__(self, reason: str, response_text: Optional[str] = None) -> None:
        details = {"reason": reason}
        if response_text:
            details["response_preview"] = response_text[:200]

        super().__init__(
            message=f"Resposta inválida do Gemini: {reason}",
            error_code="GEMINI_INVALID_RESPONSE",
            details=details,
        )


# === Infrastructure Exceptions ===


class InfrastructureException(AppBaseException):
    """Exceções relacionadas à infraestrutura."""

    pass


class ConfigurationException(InfrastructureException):
    """Erro de configuração da aplicação."""

    def __init__(self, config_key: str, reason: str) -> None:
        super().__init__(
            message=f"Erro de configuração '{config_key}': {reason}",
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key, "reason": reason},
        )


class FileNotFoundException(InfrastructureException):
    """Arquivo não encontrado."""

    def __init__(self, file_path: str) -> None:
        super().__init__(
            message=f"Arquivo não encontrado: '{file_path}'",
            error_code="FILE_NOT_FOUND",
            details={"file_path": file_path},
        )


# === Validation Exceptions ===


class ValidationException(AppBaseException):
    """Exceções de validação de entrada."""

    def __init__(
        self, message: str, field: Optional[str] = None, details: Optional[dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        if field:
            error_details["field"] = field

        super().__init__(message=message, error_code="VALIDATION_ERROR", details=error_details)
