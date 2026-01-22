"""
Middleware de logging para requisições HTTP.
Captura e loga todas as requisições/respostas da API.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para logging automático de requisições e respostas.
    Adiciona request_id, timing e informações de contexto.
    """
    
    def __init__(self, app: ASGIApp) -> None:
        """
        Inicializa o middleware.
        
        Args:
            app: Aplicação ASGI
        """
        super().__init__(app)
        logger.info("LoggingMiddleware inicializado")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa a requisição, adiciona logging e timing.
        
        Args:
            request: Requisição HTTP
            call_next: Próximo middleware/handler na cadeia
        
        Returns:
            Response: Resposta HTTP
        """
        # Gera request_id único
        request_id = self._get_or_generate_request_id(request)
        
        # Adiciona request_id ao state para acesso em endpoints
        request.state.request_id = request_id
        
        # Marca tempo de início
        start_time = time.time()
        
        # Loga requisição recebida
        self._log_request(request, request_id)
        
        # Variáveis para resposta
        response = None
        error = None
        
        try:
            # Processa a requisição
            response = await call_next(request)
            
            # Calcula tempo de processamento
            process_time_ms = (time.time() - start_time) * 1000
            
            # Adiciona headers customizados
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-MS"] = f"{process_time_ms:.2f}"
            
            # Loga resposta
            self._log_response(request, response, request_id, process_time_ms)
            
            return response
            
        except Exception as e:
            # Calcula tempo até o erro
            process_time_ms = (time.time() - start_time) * 1000
            
            # Loga erro
            self._log_error(request, e, request_id, process_time_ms)
            
            # Re-lança a exceção para ser tratada pelo exception handler
            raise
    
    def _get_or_generate_request_id(self, request: Request) -> str:
        """
        Obtém request_id do header ou gera um novo.
        
        Args:
            request: Requisição HTTP
        
        Returns:
            str: Request ID único
        """
        # Verifica se cliente enviou request_id no header
        request_id = request.headers.get("X-Request-ID")
        
        if request_id:
            return request_id
        
        # Gera novo UUID
        return str(uuid.uuid4())
    
    def _log_request(self, request: Request, request_id: str) -> None:
        """
        Loga informações da requisição recebida.
        
        Args:
            request: Requisição HTTP
            request_id: ID da requisição
        """
        # Extrai informações do cliente
        client_host = request.client.host if request.client else "unknown"
        client_port = request.client.port if request.client else 0
        
        # Extrai informações da requisição
        method = request.method
        url = str(request.url)
        path = request.url.path
        query_params = dict(request.query_params)
        headers = dict(request.headers)
        
        # Remove headers sensíveis
        sensitive_headers = ["authorization", "x-api-key", "cookie"]
        for header in sensitive_headers:
            if header in headers:
                headers[header] = "***REDACTED***"
        
        logger.info(
            f"HTTP Request: {method} {path}",
            extra={
                "event_type": "http_request",
                "request_id": request_id,
                "method": method,
                "path": path,
                "url": url,
                "client_host": client_host,
                "client_port": client_port,
                "query_params": query_params,
                "headers": headers,
                "user_agent": headers.get("user-agent", "unknown")
            }
        )
    
    def _log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        process_time_ms: float
    ) -> None:
        """
        Loga informações da resposta enviada.
        
        Args:
            request: Requisição HTTP
            response: Resposta HTTP
            request_id: ID da requisição
            process_time_ms: Tempo de processamento em ms
        """
        method = request.method
        path = request.url.path
        status_code = response.status_code
        
        # Determina nível de log baseado no status code
        if status_code >= 500:
            log_level = "error"
        elif status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        log_method = getattr(logger, log_level)
        
        log_method(
            f"HTTP Response: {method} {path} - {status_code} ({process_time_ms:.2f}ms)",
            extra={
                "event_type": "http_response",
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "process_time_ms": process_time_ms,
                "response_headers": dict(response.headers)
            }
        )
    
    def _log_error(
        self,
        request: Request,
        error: Exception,
        request_id: str,
        process_time_ms: float
    ) -> None:
        """
        Loga erros ocorridos durante processamento.
        
        Args:
            request: Requisição HTTP
            error: Exceção capturada
            request_id: ID da requisição
            process_time_ms: Tempo até o erro em ms
        """
        method = request.method
        path = request.url.path
        
        logger.error(
            f"HTTP Error: {method} {path} - {type(error).__name__}: {str(error)}",
            exc_info=True,
            extra={
                "event_type": "http_error",
                "request_id": request_id,
                "method": method,
                "path": path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "process_time_ms": process_time_ms
            }
        )


class CORSLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware adicional para logar requisições CORS preflight.
    Útil para debugging de problemas de CORS.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Loga requisições OPTIONS (CORS preflight).
        
        Args:
            request: Requisição HTTP
            call_next: Próximo middleware/handler
        
        Returns:
            Response: Resposta HTTP
        """
        if request.method == "OPTIONS":
            logger.debug(
                f"CORS Preflight: {request.url.path}",
                extra={
                    "event_type": "cors_preflight",
                    "path": request.url.path,
                    "origin": request.headers.get("origin", "unknown"),
                    "access_control_request_method": request.headers.get(
                        "access-control-request-method", "unknown"
                    ),
                    "access_control_request_headers": request.headers.get(
                        "access-control-request-headers", "unknown"
                    )
                }
            )
        
        response = await call_next(request)
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware para limitar o tamanho das requisições.
    Previne ataques de DoS com payloads grandes.
    """
    
    def __init__(self, app: ASGIApp, max_size_mb: float = 10.0) -> None:
        """
        Inicializa o middleware.
        
        Args:
            app: Aplicação ASGI
            max_size_mb: Tamanho máximo da requisição em MB
        """
        super().__init__(app)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        logger.info(f"RequestSizeLimitMiddleware: max_size={max_size_mb}MB")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Verifica tamanho da requisição antes de processar.
        
        Args:
            request: Requisição HTTP
            call_next: Próximo middleware/handler
        
        Returns:
            Response: Resposta HTTP
        """
        # Verifica Content-Length header
        content_length = request.headers.get("content-length")
        
        if content_length:
            content_length = int(content_length)
            
            if content_length > self.max_size_bytes:
                logger.warning(
                    f"Requisição rejeitada: tamanho excede limite "
                    f"({content_length} bytes > {self.max_size_bytes} bytes)",
                    extra={
                        "event_type": "request_size_exceeded",
                        "content_length": content_length,
                        "max_size": self.max_size_bytes,
                        "path": request.url.path
                    }
                )
                
                from fastapi import HTTPException, status
                from src.schemas.response import ErrorResponse
                
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "error_code": "REQUEST_TOO_LARGE",
                        "message": f"Requisição excede o limite de {self.max_size_bytes / (1024 * 1024):.1f}MB",
                        "details": {
                            "content_length": content_length,
                            "max_size": self.max_size_bytes
                        }
                    }
                )
        
        response = await call_next(request)
        return response
