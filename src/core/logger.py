import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger

from src.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Formatter JSON customizado com campos adicionais.
    Adiciona contexto útil para rastreamento e debugging.
    """
    
    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any]
    ) -> None:
        """Adiciona campos customizados ao log JSON."""
        super().add_fields(log_record, record, message_dict)
        
        # Campos obrigatórios
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        
        # Adiciona informações da aplicação
        log_record["app_name"] = settings.app_name
        log_record["environment"] = settings.environment
        
        # Adiciona informações de contexto se disponíveis
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        
        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id
        
        # Informações de exceção
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Configura e retorna um logger com formatação apropriada.
    
    Args:
        name: Nome do logger (geralmente __name__ do módulo)
    
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evita duplicação de handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, settings.log_level))
    logger.propagate = False
    
    # Handler para stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, settings.log_level))
    
    # Seleciona formatter baseado na configuração
    if settings.log_format == "json":
        formatter = CustomJsonFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Obtém um logger configurado para o módulo.
    
    Args:
        name: Nome do logger (use __name__ do módulo)
    
    Returns:
        logging.Logger: Logger configurado
    """
    return setup_logger(name)


# Logger padrão da aplicação
app_logger = get_logger("intent_classifier_api")


class LoggerAdapter(logging.LoggerAdapter):
    """
    Adapter para adicionar contexto extra aos logs.
    Útil para rastreamento de requests e usuários.
    """
    
    def __init__(self, logger: logging.Logger, extra: dict[str, Any]) -> None:
        super().__init__(logger, extra)
    
    def process(
        self,
        msg: str,
        kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Adiciona campos extras ao log."""
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        
        kwargs["extra"].update(self.extra)
        return msg, kwargs


def get_request_logger(request_id: str, logger: logging.Logger = None) -> LoggerAdapter:
    """
    Cria um logger com request_id para rastreamento.
    
    Args:
        request_id: ID único da requisição
        logger: Logger base (opcional, usa app_logger por padrão)
    
    Returns:
        LoggerAdapter: Logger com contexto de request
    """
    base_logger = logger or app_logger
    return LoggerAdapter(base_logger, {"request_id": request_id})


# Funções de conveniência para logging estruturado

def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    request_id: str,
    **extra: Any
) -> None:
    """Loga requisição HTTP recebida."""
    logger.info(
        f"API Request: {method} {path}",
        extra={
            "event_type": "api_request",
            "request_id": request_id,
            "method": method,
            "path": path,
            **extra
        }
    )


def log_api_response(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    request_id: str,
    duration_ms: float,
    **extra: Any
) -> None:
    """Loga resposta HTTP enviada."""
    logger.info(
        f"API Response: {method} {path} - {status_code} ({duration_ms:.2f}ms)",
        extra={
            "event_type": "api_response",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **extra
        }
    )


def log_llm_call(
    logger: logging.Logger,
    model: str,
    prompt_length: int,
    request_id: str,
    **extra: Any
) -> None:
    """Loga chamada ao LLM."""
    logger.info(
        f"LLM Call: {model} (prompt: {prompt_length} chars)",
        extra={
            "event_type": "llm_call",
            "request_id": request_id,
            "model": model,
            "prompt_length": prompt_length,
            **extra
        }
    )


def log_llm_response(
    logger: logging.Logger,
    model: str,
    response_length: int,
    request_id: str,
    duration_ms: float,
    **extra: Any
) -> None:
    """Loga resposta do LLM."""
    logger.info(
        f"LLM Response: {model} ({response_length} chars, {duration_ms:.2f}ms)",
        extra={
            "event_type": "llm_response",
            "request_id": request_id,
            "model": model,
            "response_length": response_length,
            "duration_ms": duration_ms,
            **extra
        }
    )


def log_error(
    logger: logging.Logger,
    error: Exception,
    context: str,
    request_id: str = None,
    **extra: Any
) -> None:
    """Loga erro com contexto completo."""
    error_data = {
        "event_type": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        **extra
    }
    
    if request_id:
        error_data["request_id"] = request_id
    
    logger.error(
        f"Error in {context}: {type(error).__name__} - {str(error)}",
        exc_info=True,
        extra=error_data
    )
