from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.api.middlewares.logging_middleware import (
    CORSLoggingMiddleware,
    LoggingMiddleware,
    RequestSizeLimitMiddleware,
)
from src.api.v1.dependencies import cleanup_dependencies, startup_dependencies, validate_configuration
from src.api.v1.endpoints import classifier
from src.core.config import settings
from src.core.exceptions import AppBaseException
from src.core.logger import get_logger
from src.schemas.response import ErrorResponse, HealthCheckResponse

logger = get_logger(__name__)


# === Lifecycle Events ===

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Gerencia o ciclo de vida da aplicação.
    Executa tarefas no startup e shutdown.
    
    Args:
        app: Instância do FastAPI
    
    Yields:
        None
    """
    # === STARTUP ===
    logger.info(f"Iniciando {settings.app_name} v{settings.app_version}")
    logger.info(f"Ambiente: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        # Valida configurações críticas
        validate_configuration()
        
        # Inicializa dependências
        await startup_dependencies()
        
        logger.info("✅ Aplicação iniciada com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar aplicação: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # === SHUTDOWN ===
    logger.info("Encerrando aplicação...")
    
    try:
        # Limpa dependências
        await cleanup_dependencies()
        
        logger.info("✅ Aplicação encerrada com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro ao encerrar aplicação: {str(e)}", exc_info=True)


# === Aplicação FastAPI ===

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API RESTful para classificação de intenção usando Few-Shot Prompting com Gemini 2.5 Flash",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)


# === Middlewares ===

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Request Size Limit Middleware (10MB)
app.add_middleware(RequestSizeLimitMiddleware, max_size_mb=10.0)

# CORS Logging Middleware (debug)
if settings.debug:
    app.add_middleware(CORSLoggingMiddleware)

# Logging Middleware (sempre ativo)
app.add_middleware(LoggingMiddleware)


# === Exception Handlers ===

@app.exception_handler(AppBaseException)
async def app_exception_handler(request: Request, exc: AppBaseException) -> JSONResponse:
    """
    Handler para exceções customizadas da aplicação.
    
    Args:
        request: Requisição HTTP
        exc: Exceção capturada
    
    Returns:
        JSONResponse: Resposta de erro formatada
    """
    logger.error(
        f"AppException: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path
        }
    )
    
    # Determina status code baseado no tipo de erro
    status_code_map = {
        "VALIDATION_ERROR": status.HTTP_400_BAD_REQUEST,
        "CLASSIFICATION_FAILED": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "GEMINI_TIMEOUT": status.HTTP_504_GATEWAY_TIMEOUT,
        "GEMINI_RATE_LIMIT": status.HTTP_429_TOO_MANY_REQUESTS,
        "GEMINI_API_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "FILE_NOT_FOUND": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "CONFIGURATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    
    status_code = status_code_map.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handler para erros de validação do Pydantic.
    
    Args:
        request: Requisição HTTP
        exc: Erro de validação
    
    Returns:
        JSONResponse: Resposta de erro formatada
    """
    logger.warning(
        f"Validation Error: {str(exc)}",
        extra={
            "errors": exc.errors(),
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Erro de validação nos dados fornecidos",
            details={"errors": exc.errors()},
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path
        ).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handler global para exceções não tratadas.
    
    Args:
        request: Requisição HTTP
        exc: Exceção capturada
    
    Returns:
        JSONResponse: Resposta de erro formatada
    """
    logger.error(
        f"Unhandled Exception: {type(exc).__name__} - {str(exc)}",
        exc_info=True,
        extra={
            "error_type": type(exc).__name__,
            "path": request.url.path
        }
    )
    
    # Em produção, não expõe detalhes internos
    if settings.environment == "production":
        message = "Erro interno do servidor"
        details = None
    else:
        message = str(exc)
        details = {"error_type": type(exc).__name__}
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message=message,
            details=details,
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path
        ).model_dump()
    )


# === Rotas ===

# Health Check Endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Verifica saúde da aplicação",
    description="Endpoint para health check da aplicação e suas dependências"
)
async def health_check() -> HealthCheckResponse:
    """
    Verifica a saúde da aplicação.
    
    Returns:
        HealthCheckResponse: Status de saúde
    """
    from datetime import datetime
    from src.api.v1.dependencies import get_intent_service, get_prompt_manager
    
    try:
        # Verifica serviço de classificação
        intent_service = get_intent_service()
        is_healthy = await intent_service.health_check()
        
        # Verifica exemplos carregados
        prompt_manager = get_prompt_manager()
        examples_count = prompt_manager.get_examples_count()
        
        # Se exemplos não estão carregados, carrega
        if examples_count == 0:
            examples = await prompt_manager.load_examples()
            examples_count = len(examples)
        
        checks = {
            "llm_provider": is_healthy,
            "examples_loaded": examples_count > 0,
            "configuration": True
        }
        
        # Determina status geral
        if all(checks.values()):
            overall_status = "healthy"
        elif any(checks.values()):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=settings.app_version,
            checks=checks,
            details={
                "model": settings.gemini_model,
                "examples_count": examples_count,
                "environment": settings.environment
            }
        )
        
    except Exception as e:
        logger.error(f"Health check falhou: {str(e)}")
        
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version=settings.app_version,
            checks={
                "llm_provider": False,
                "examples_loaded": False,
                "configuration": False
            },
            details={"error": str(e)}
        )


# Root Endpoint
@app.get(
    "/",
    tags=["Root"],
    summary="Informações da API",
    description="Retorna informações básicas sobre a API"
)
async def root() -> dict:
    """
    Endpoint raiz com informações da API.
    
    Returns:
        dict: Informações básicas
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
        "docs_url": "/docs" if settings.debug else None,
        "health_check_url": "/health"
    }


# Inclui routers de endpoints
app.include_router(classifier.router, prefix="/api/v1")


# === Entry Point ===

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
