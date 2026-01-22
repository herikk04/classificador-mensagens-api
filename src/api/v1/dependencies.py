from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.core.config import settings
from src.core.logger import get_logger
from src.domain.interfaces import IIntentClassifier, ILLMProvider, IPromptManager
from src.providers.gemini.client import GeminiClient
from src.services.intent_service import IntentService
from src.services.prompt_manager import PromptManager

logger = get_logger(__name__)


# === Provider Dependencies ===


@lru_cache
def get_llm_provider() -> ILLMProvider:
    """
    Factory para o provedor de LLM.
    Retorna instância singleton do GeminiClient.

    Returns:
        ILLMProvider: Cliente Gemini configurado
    """
    logger.info("Inicializando LLM Provider (GeminiClient)")

    return GeminiClient(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
        temperature=settings.gemini_temperature,
        max_tokens=settings.gemini_max_tokens,
        timeout=settings.gemini_timeout,
    )


@lru_cache
def get_prompt_manager() -> IPromptManager:
    """
    Factory para o gerenciador de prompts.
    Retorna instância singleton do PromptManager.

    Returns:
        IPromptManager: Gerenciador de prompts configurado
    """
    logger.info("Inicializando Prompt Manager")

    return PromptManager(
        examples_file_path=settings.examples_file_path, max_examples=settings.max_examples_in_prompt
    )


# === Service Dependencies ===


@lru_cache
def get_intent_service(
    llm_provider: Annotated[ILLMProvider, Depends(get_llm_provider)],
    prompt_manager: Annotated[IPromptManager, Depends(get_prompt_manager)],
) -> IIntentClassifier:
    """
    Factory para o serviço de classificação de intenção.
    Retorna instância singleton do IntentService.

    Args:
        llm_provider: Provider de LLM injetado
        prompt_manager: Gerenciador de prompts injetado

    Returns:
        IIntentClassifier: Serviço de classificação configurado
    """
    logger.info("Inicializando Intent Service")

    return IntentService(llm_provider=llm_provider, prompt_manager=prompt_manager)


# === Type Aliases for Dependency Injection ===

# Tipos anotados para injeção nas rotas
LLMProviderDep = Annotated[ILLMProvider, Depends(get_llm_provider)]
PromptManagerDep = Annotated[IPromptManager, Depends(get_prompt_manager)]
IntentServiceDep = Annotated[IIntentClassifier, Depends(get_intent_service)]


# === Utility Dependencies ===


def get_request_id_generator():
    """
    Gerador de IDs únicos para requisições.
    Pode ser substituído por implementação mais robusta (UUID, etc.).

    Yields:
        str: ID único da requisição
    """
    import uuid

    return str(uuid.uuid4())


async def verify_api_health(intent_service: IntentServiceDep) -> bool:
    """
    Verifica se a API está saudável antes de processar requisições.

    Args:
        intent_service: Serviço de classificação injetado

    Returns:
        bool: True se saudável

    Raises:
        HTTPException: Se algum componente crítico estiver indisponível
    """
    from fastapi import HTTPException, status

    is_healthy = await intent_service.health_check()

    if not is_healthy:
        logger.error("Health check falhou - API em estado degradado")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço temporariamente indisponível",
        )

    return True


# === Cleanup Functions ===


async def cleanup_dependencies() -> None:
    """
    Limpa recursos e caches ao desligar a aplicação.
    Deve ser chamado no evento shutdown do FastAPI.
    """
    logger.info("Limpando dependências...")

    # Limpa caches do lru_cache
    get_llm_provider.cache_clear()
    get_prompt_manager.cache_clear()
    get_intent_service.cache_clear()

    logger.info("Dependências limpas com sucesso")


async def startup_dependencies() -> None:
    """
    Inicializa dependências no startup da aplicação.
    Pré-carrega recursos para melhor performance.
    """
    logger.info("Inicializando dependências no startup...")

    try:
        # Pré-instancia os serviços para carregar recursos
        llm_provider = get_llm_provider()
        prompt_manager = get_prompt_manager()

        # Pré-carrega exemplos
        examples = await prompt_manager.load_examples()
        logger.info(f"Pré-carregados {len(examples)} exemplos few-shot")

        # Verifica saúde do LLM
        llm_healthy = await llm_provider.health_check()

        if llm_healthy:
            logger.info("LLM Provider está operacional")
        else:
            logger.warning("LLM Provider health check falhou no startup")

        logger.info("Dependências inicializadas com sucesso")

    except Exception as e:
        logger.error(f"Erro ao inicializar dependências: {str(e)}", exc_info=True)
        raise


# === Configuration Validators ===


def validate_configuration() -> None:
    """
    Valida configurações críticas no startup.

    Raises:
        ValueError: Se alguma configuração crítica estiver inválida
    """
    from pathlib import Path

    logger.info("Validando configurações...")

    # Valida API Key
    if not settings.gemini_api_key or len(settings.gemini_api_key) < 20:
        raise ValueError("GEMINI_API_KEY inválida ou não configurada")

    # Valida arquivo de exemplos
    examples_path = Path(settings.examples_file_path)
    if not examples_path.exists():
        logger.warning(f"Arquivo de exemplos não encontrado: {settings.examples_file_path}")

    # Valida modelo
    allowed_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    if settings.gemini_model not in allowed_models:
        raise ValueError(
            f"Modelo '{settings.gemini_model}' não suportado. "
            f"Use um dos seguintes: {', '.join(allowed_models)}"
        )

    logger.info("Configurações validadas com sucesso")
