import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.v1.dependencies import IntentServiceDep, PromptManagerDep
from src.core.exceptions import (
    AppBaseException,
    ClassificationFailedException,
    ValidationException,
)
from src.core.logger import get_logger, log_api_request, log_api_response, log_error
from src.domain.models import IntentType
from src.schemas.request import (
    BatchClassifyIntentRequest,
    ClassifyIntentRequest,
    ClassifyIntentWithConfigRequest,
)
from src.schemas.response import (
    BatchClassifyIntentResponse,
    ClassifyIntentResponse,
    ErrorResponse,
    ModelInfoResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/classify", tags=["Classification"])


@router.post(
    "",
    response_model=ClassifyIntentResponse,
    status_code=status.HTTP_200_OK,
    summary="Classifica intenção de texto único",
    description="Classifica a intenção de um texto fornecido pelo usuário usando Few-Shot Learning com Gemini",
    responses={
        200: {"description": "Classificação realizada com sucesso"},
        400: {"model": ErrorResponse, "description": "Requisição inválida"},
        422: {"model": ErrorResponse, "description": "Erro de validação"},
        500: {"model": ErrorResponse, "description": "Erro interno do servidor"},
        503: {"model": ErrorResponse, "description": "Serviço indisponível"}
    }
)
async def classify_intent(
    request: ClassifyIntentRequest,
    intent_service: IntentServiceDep
) -> ClassifyIntentResponse:
    """
    Classifica a intenção de um texto único.
    
    Args:
        request: Dados da requisição com texto a classificar
        intent_service: Serviço de classificação (injetado)
    
    Returns:
        ClassifyIntentResponse: Resultado da classificação
    """
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    log_api_request(
        logger,
        method="POST",
        path="/api/v1/classify",
        request_id=request_id,
        text_length=len(request.text)
    )
    
    try:
        # Classifica a intenção
        result = await intent_service.classify(
            user_input=request.text,
            request_id=request_id
        )
        
        # Monta resposta
        response = ClassifyIntentResponse(
            intent=result.intent,
            confidence=result.confidence,
            confidence_level=result.confidence_level,
            processing_time_ms=result.processing_time_ms,
            timestamp=result.timestamp,
            request_id=request_id,
            raw_response=result.raw_response if request.include_raw_response else None,
            metadata=result.metadata if request.include_metadata else None
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        log_api_response(
            logger,
            method="POST",
            path="/api/v1/classify",
            status_code=200,
            request_id=request_id,
            duration_ms=duration_ms,
            intent=result.intent.value,
            confidence=result.confidence
        )
        
        return response
        
    except ValidationException as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(logger, e, "classify_intent", request_id)
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    
    except ClassificationFailedException as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(logger, e, "classify_intent", request_id)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.to_dict()
        )
    
    except AppBaseException as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(logger, e, "classify_intent", request_id)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.to_dict()
        )
    
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(logger, e, "classify_intent", request_id)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "Erro inesperado ao processar requisição",
                "details": {"error": str(e)}
            }
        )


@router.post(
    "/batch",
    response_model=BatchClassifyIntentResponse,
    status_code=status.HTTP_200_OK,
    summary="Classifica múltiplas intenções em lote",
    description="Classifica a intenção de múltiplos textos fornecidos em uma única requisição",
    responses={
        200: {"description": "Classificação em lote realizada"},
        400: {"model": ErrorResponse, "description": "Requisição inválida"},
        422: {"model": ErrorResponse, "description": "Erro de validação"},
        500: {"model": ErrorResponse, "description": "Erro interno do servidor"}
    }
)
async def classify_intent_batch(
    request: BatchClassifyIntentRequest,
    intent_service: IntentServiceDep
) -> BatchClassifyIntentResponse:
    """
    Classifica a intenção de múltiplos textos em lote.
    
    Args:
        request: Dados da requisição com lista de textos
        intent_service: Serviço de classificação (injetado)
    
    Returns:
        BatchClassifyIntentResponse: Resultados da classificação em lote
    """
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    log_api_request(
        logger,
        method="POST",
        path="/api/v1/classify/batch",
        request_id=request_id,
        batch_size=len(request.texts)
    )
    
    try:
        # Classifica em lote
        results = await intent_service.classify_batch(
            user_inputs=request.texts,
            request_id=request_id
        )
        
        # Conta sucessos e falhas
        successful = sum(1 for r in results if r.intent != IntentType.UNKNOWN or r.confidence > 0)
        failed = len(results) - successful
        
        # Coleta erros se houver
        errors = []
        for idx, result in enumerate(results):
            if result.intent == IntentType.UNKNOWN and result.confidence == 0.0:
                error_info = result.metadata.get("error", "Unknown error")
                errors.append({
                    "index": idx,
                    "text": request.texts[idx][:100],
                    "error": error_info
                })
        
        # Monta respostas individuais
        response_results = []
        for result in results:
            response_results.append(
                ClassifyIntentResponse(
                    intent=result.intent,
                    confidence=result.confidence,
                    confidence_level=result.confidence_level,
                    processing_time_ms=result.processing_time_ms,
                    timestamp=result.timestamp,
                    request_id=result.metadata.get("request_id"),
                    raw_response=result.raw_response if request.include_raw_response else None,
                    metadata=result.metadata if request.include_metadata else None
                )
            )
        
        total_processing_time_ms = (time.time() - start_time) * 1000
        
        # Monta resposta em lote
        response = BatchClassifyIntentResponse(
            results=response_results,
            total_processed=len(request.texts),
            total_successful=successful,
            total_failed=failed,
            total_processing_time_ms=total_processing_time_ms,
            timestamp=results[0].timestamp if results else None,
            request_id=request_id,
            errors=errors if errors else None
        )
        
        log_api_response(
            logger,
            method="POST",
            path="/api/v1/classify/batch",
            status_code=200,
            request_id=request_id,
            duration_ms=total_processing_time_ms,
            batch_size=len(request.texts),
            successful=successful,
            failed=failed
        )
        
        return response
        
    except ValidationException as e:
        log_error(logger, e, "classify_intent_batch", request_id)
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    
    except Exception as e:
        log_error(logger, e, "classify_intent_batch", request_id)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "Erro inesperado ao processar lote",
                "details": {"error": str(e)}
            }
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Informações do modelo",
    description="Retorna informações sobre o modelo LLM e configurações atuais",
    responses={
        200: {"description": "Informações recuperadas com sucesso"},
        500: {"model": ErrorResponse, "description": "Erro interno do servidor"}
    }
)
async def get_model_info(
    intent_service: IntentServiceDep,
    prompt_manager: PromptManagerDep
) -> ModelInfoResponse:
    """
    Retorna informações sobre o modelo e configurações.
    
    Args:
        intent_service: Serviço de classificação (injetado)
        prompt_manager: Gerenciador de prompts (injetado)
    
    Returns:
        ModelInfoResponse: Informações do modelo
    """
    request_id = str(uuid.uuid4())
    
    log_api_request(
        logger,
        method="GET",
        path="/api/v1/classify/model/info",
        request_id=request_id
    )
    
    try:
        # Obtém estatísticas
        stats = await intent_service.get_statistics()
        
        # Carrega exemplos para contar
        examples = await prompt_manager.load_examples()
        
        from src.core.config import settings
        from datetime import datetime
        
        response = ModelInfoResponse(
            model_name=stats["model"],
            provider="Google Gemini",
            temperature=settings.gemini_temperature,
            max_tokens=settings.gemini_max_tokens,
            examples_count=len(examples),
            supported_intents=stats["supported_intents"],
            timestamp=datetime.utcnow()
        )
        
        log_api_response(
            logger,
            method="GET",
            path="/api/v1/classify/model/info",
            status_code=200,
            request_id=request_id,
            duration_ms=0,
            model=stats["model"]
        )
        
        return response
        
    except Exception as e:
        log_error(logger, e, "get_model_info", request_id)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "Erro ao recuperar informações do modelo",
                "details": {"error": str(e)}
            }
        )
