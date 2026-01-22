import asyncio
from typing import Any, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import GenerateContentResponse

from src.core.config import settings
from src.core.exceptions import (
    GeminiAPIException,
    GeminiInvalidResponseException,
    GeminiRateLimitException,
    GeminiTimeoutException,
)
from src.core.logger import get_logger
from src.domain.interfaces import ILLMProvider

logger = get_logger(__name__)


class GeminiClient(ILLMProvider):
    """
    Cliente para comunicação com Google Gemini API.
    Implementa ILLMProvider com suporte assíncrono.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Inicializa o cliente Gemini.

        Args:
            api_key: API key do Google (usa settings se None)
            model_name: Nome do modelo (usa settings se None)
            temperature: Temperatura padrão (usa settings se None)
            max_tokens: Max tokens padrão (usa settings se None)
            timeout: Timeout em segundos (usa settings se None)
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.temperature = temperature or settings.gemini_temperature
        self.max_tokens = max_tokens or settings.gemini_max_tokens
        self.timeout = timeout or settings.gemini_timeout

        # Configura a API
        genai.configure(api_key=self.api_key)

        # Inicializa o modelo
        self._model = genai.GenerativeModel(
            model_name=self.model_name, generation_config=self._get_generation_config()
        )

        logger.info(
            f"GeminiClient inicializado: model={self.model_name}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}"
        )

    def _get_generation_config(
        self, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Constrói configuração de geração para o modelo.

        Args:
            temperature: Override de temperatura (usa padrão se None)
            max_tokens: Override de max_tokens (usa padrão se None)

        Returns:
            dict[str, Any]: Configuração de geração
        """
        return {
            "temperature": temperature if temperature is not None else self.temperature,
            "max_output_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "top_p": 0.95,
            "top_k": 40,
        }

    async def generate_completion(
        self, prompt: str, temperature: float = None, max_tokens: int = None, **kwargs: Any
    ) -> str:
        """
        Gera uma completion usando Gemini API.

        Args:
            prompt: Prompt para o modelo
            temperature: Override de temperatura
            max_tokens: Override de max_tokens
            **kwargs: Parâmetros adicionais

        Returns:
            str: Texto gerado pelo modelo

        Raises:
            GeminiAPIException: Erro na API
            GeminiTimeoutException: Timeout na requisição
            GeminiRateLimitException: Rate limit excedido
            GeminiInvalidResponseException: Resposta inválida
        """
        try:
            logger.debug(f"Gerando completion com Gemini: prompt_length={len(prompt)}")

            # Atualiza configuração se houver overrides
            if temperature is not None or max_tokens is not None:
                generation_config = self._get_generation_config(temperature, max_tokens)
                model = genai.GenerativeModel(
                    model_name=self.model_name, generation_config=generation_config
                )
            else:
                model = self._model

            # Executa a chamada com timeout
            response = await asyncio.wait_for(
                self._async_generate_content(model, prompt), timeout=self.timeout
            )

            # Valida e extrai o texto da resposta
            text = self._extract_text_from_response(response)

            logger.debug(f"Completion gerada com sucesso: response_length={len(text)}")
            return text

        except asyncio.TimeoutError as e:
            logger.error(f"Timeout ao chamar Gemini API: {self.timeout}s")
            raise GeminiTimeoutException(timeout=self.timeout) from e

        except google_exceptions.ResourceExhausted as e:
            logger.error(f"Rate limit excedido na Gemini API: {str(e)}")
            raise GeminiRateLimitException() from e

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Erro na Gemini API: {str(e)}")
            raise GeminiAPIException(message=str(e), status_code=getattr(e, "code", None)) from e

        except Exception as e:
            logger.error(f"Erro inesperado ao chamar Gemini: {str(e)}", exc_info=True)
            raise GeminiAPIException(message=f"Erro inesperado: {str(e)}") from e

    async def _async_generate_content(
        self, model: genai.GenerativeModel, prompt: str
    ) -> GenerateContentResponse:
        """
        Wrapper assíncrono para generate_content.
        A API do Gemini é síncrona, então executamos em thread separada.

        Args:
            model: Modelo Gemini
            prompt: Prompt para geração

        Returns:
            GenerateContentResponse: Resposta do modelo
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, model.generate_content, prompt)

    def _extract_text_from_response(self, response: GenerateContentResponse) -> str:
        """
        Extrai o texto da resposta do Gemini.

        Args:
            response: Resposta do modelo

        Returns:
            str: Texto extraído

        Raises:
            GeminiInvalidResponseException: Se a resposta for inválida
        """
        try:
            # Verifica se há conteúdo
            if not response.candidates:
                raise GeminiInvalidResponseException(
                    reason="Resposta sem candidates", response_text=str(response)
                )

            # Pega o primeiro candidate
            candidate = response.candidates[0]

            # Verifica finish_reason
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                finish_reason = str(candidate.finish_reason)
                if finish_reason not in ["STOP", "1", "FinishReason.STOP"]:
                    logger.warning(f"Finish reason inesperado: {finish_reason}")

            # Extrai o texto
            if not candidate.content or not candidate.content.parts:
                raise GeminiInvalidResponseException(
                    reason="Candidate sem conteúdo", response_text=str(response)
                )

            text = candidate.content.parts[0].text.strip()

            if not text:
                raise GeminiInvalidResponseException(
                    reason="Texto vazio na resposta", response_text=str(response)
                )

            return text

        except AttributeError as e:
            raise GeminiInvalidResponseException(
                reason=f"Estrutura de resposta inválida: {str(e)}", response_text=str(response)
            ) from e

    async def health_check(self) -> bool:
        """
        Verifica se o cliente está funcionando corretamente.

        Returns:
            bool: True se operacional
        """
        try:
            # Testa com um prompt simples
            test_prompt = "Responda apenas: OK"
            response = await asyncio.wait_for(
                self._async_generate_content(self._model, test_prompt), timeout=10
            )

            # Verifica se obteve resposta válida
            text = self._extract_text_from_response(response)

            logger.info("Health check do GeminiClient bem-sucedido")
            return True

        except Exception as e:
            logger.error(f"Health check do GeminiClient falhou: {str(e)}")
            return False

    def get_model_name(self) -> str:
        """
        Retorna o nome do modelo sendo usado.

        Returns:
            str: Nome do modelo
        """
        return self.model_name

    def update_config(
        self, temperature: Optional[float] = None, max_tokens: Optional[int] = None
    ) -> None:
        """
        Atualiza configurações do cliente.

        Args:
            temperature: Nova temperatura
            max_tokens: Novo max_tokens
        """
        if temperature is not None:
            self.temperature = temperature
            logger.info(f"Temperatura atualizada para: {temperature}")

        if max_tokens is not None:
            self.max_tokens = max_tokens
            logger.info(f"Max tokens atualizado para: {max_tokens}")

        # Recria o modelo com nova configuração
        self._model = genai.GenerativeModel(
            model_name=self.model_name, generation_config=self._get_generation_config()
        )
