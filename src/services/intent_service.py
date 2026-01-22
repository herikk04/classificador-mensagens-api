import re
import time
from typing import Optional

from src.core.exceptions import ClassificationFailedException, ValidationException
from src.core.logger import get_logger
from src.domain.interfaces import IIntentClassifier, ILLMProvider, IPromptManager
from src.domain.models import ClassificationResult, IntentType

logger = get_logger(__name__)


class IntentService(IIntentClassifier):
    """
    Serviço principal de classificação de intenção.
    Implementa IIntentClassifier usando Dependency Injection.
    """
    
    def __init__(
        self,
        llm_provider: ILLMProvider,
        prompt_manager: IPromptManager
    ) -> None:
        """
        Inicializa o serviço de classificação.
        
        Args:
            llm_provider: Provider de LLM (ex: GeminiClient)
            prompt_manager: Gerenciador de prompts few-shot
        """
        self.llm_provider = llm_provider
        self.prompt_manager = prompt_manager
        
        logger.info(
            f"IntentService inicializado: model={self.llm_provider.get_model_name()}"
        )
    
    async def classify(
        self,
        user_input: str,
        request_id: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classifica a intenção do usuário a partir do texto fornecido.
        
        Args:
            user_input: Texto fornecido pelo usuário
            request_id: ID da requisição para rastreamento (opcional)
        
        Returns:
            ClassificationResult: Resultado da classificação com intenção e confiança
        
        Raises:
            ClassificationFailedException: Se a classificação falhar
            ValidationException: Se o input for inválido
        """
        start_time = time.time()
        
        try:
            # Valida input
            self._validate_input(user_input)
            
            logger.info(
                f"Iniciando classificação: input='{user_input[:50]}...', request_id={request_id}"
            )
            
            # Constrói o prompt few-shot
            prompt = await self.prompt_manager.build_prompt(user_input)
            
            logger.debug(f"Prompt construído: {len(prompt)} caracteres")
            
            # Chama o LLM
            raw_response = await self.llm_provider.generate_completion(prompt)
            
            logger.debug(f"Resposta do LLM: '{raw_response}'")
            
            # Processa a resposta
            intent, confidence = self._parse_llm_response(raw_response)
            
            # Calcula tempo de processamento
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Cria resultado
            result = ClassificationResult(
                intent=intent,
                confidence=confidence,
                confidence_level="",  # Calculado automaticamente pelo validator
                raw_response=raw_response,
                processing_time_ms=processing_time_ms,
                model_used=self.llm_provider.get_model_name(),
                metadata={
                    "request_id": request_id,
                    "input_length": len(user_input),
                    "prompt_length": len(prompt)
                }
            )
            
            logger.info(
                f"Classificação concluída: intent={intent.value}, "
                f"confidence={confidence:.2f}, time={processing_time_ms:.2f}ms"
            )
            
            return result
            
        except ValidationException:
            raise
        
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Erro na classificação: {str(e)}, time={processing_time_ms:.2f}ms",
                exc_info=True
            )
            raise ClassificationFailedException(
                reason=str(e),
                details={
                    "request_id": request_id,
                    "input_preview": user_input[:100],
                    "processing_time_ms": processing_time_ms
                }
            ) from e
    
    async def classify_batch(
        self,
        user_inputs: list[str],
        request_id: Optional[str] = None
    ) -> list[ClassificationResult]:
        """
        Classifica múltiplas intenções em lote.
        
        Args:
            user_inputs: Lista de textos a serem classificados
            request_id: ID da requisição para rastreamento (opcional)
        
        Returns:
            list[ClassificationResult]: Lista de resultados de classificação
        
        Raises:
            ClassificationFailedException: Se alguma classificação falhar
            ValidationException: Se algum input for inválido
        """
        logger.info(
            f"Iniciando classificação em lote: count={len(user_inputs)}, "
            f"request_id={request_id}"
        )
        
        results = []
        errors = []
        
        for idx, user_input in enumerate(user_inputs):
            try:
                # Cria sub-request_id para rastreamento
                sub_request_id = f"{request_id}_item_{idx}" if request_id else None
                
                result = await self.classify(user_input, sub_request_id)
                results.append(result)
                
            except Exception as e:
                error_msg = f"Erro no item {idx}: {str(e)}"
                logger.error(error_msg)
                errors.append({
                    "index": idx,
                    "input": user_input[:100],
                    "error": str(e)
                })
                
                # Adiciona resultado com UNKNOWN se falhar
                results.append(
                    ClassificationResult(
                        intent=IntentType.UNKNOWN,
                        confidence=0.0,
                        confidence_level="",
                        raw_response="",
                        processing_time_ms=0.0,
                        model_used=self.llm_provider.get_model_name(),
                        metadata={
                            "request_id": sub_request_id,
                            "error": str(e)
                        }
                    )
                )
        
        logger.info(
            f"Classificação em lote concluída: total={len(user_inputs)}, "
            f"successful={len(user_inputs) - len(errors)}, errors={len(errors)}"
        )
        
        return results
    
    def _validate_input(self, user_input: str) -> None:
        """
        Valida o input do usuário.
        
        Args:
            user_input: Texto a validar
        
        Raises:
            ValidationException: Se o input for inválido
        """
        if not user_input or not user_input.strip():
            raise ValidationException(
                message="O texto não pode estar vazio",
                field="user_input"
            )
        
        if len(user_input) > 1000:
            raise ValidationException(
                message="O texto excede o limite de 1000 caracteres",
                field="user_input",
                details={"length": len(user_input), "max_length": 1000}
            )
        
        # Verifica se contém pelo menos um caractere alfanumérico
        if not any(c.isalnum() for c in user_input):
            raise ValidationException(
                message="O texto deve conter pelo menos um caractere alfanumérico",
                field="user_input"
            )
    
    def _parse_llm_response(self, raw_response: str) -> tuple[IntentType, float]:
        """
        Processa a resposta do LLM e extrai intenção e confiança.
        
        Args:
            raw_response: Resposta bruta do LLM
        
        Returns:
            tuple[IntentType, float]: Tupla (intenção, confiança)
        
        Raises:
            ClassificationFailedException: Se não conseguir parsear a resposta
        """
        try:
            # Limpa a resposta
            cleaned_response = raw_response.strip().lower()
            
            # Remove pontuação e espaços extras
            cleaned_response = re.sub(r'[^\w\s]', '', cleaned_response)
            cleaned_response = cleaned_response.strip()
            
            logger.debug(f"Resposta limpa: '{cleaned_response}'")
            
            # Tenta converter para IntentType
            intent = IntentType.from_string(cleaned_response)
            
            # Calcula confiança baseada na resposta
            confidence = self._calculate_confidence(cleaned_response, intent)
            
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Erro ao parsear resposta do LLM: {str(e)}")
            raise ClassificationFailedException(
                reason=f"Não foi possível parsear a resposta do LLM: {str(e)}",
                details={"raw_response": raw_response[:200]}
            ) from e
    
    def _calculate_confidence(self, cleaned_response: str, intent: IntentType) -> float:
        """
        Calcula a confiança da classificação.
        
        Args:
            cleaned_response: Resposta limpa do LLM
            intent: Intenção classificada
        
        Returns:
            float: Score de confiança (0.0 a 1.0)
        """
        # Se a resposta for exatamente o nome da intent, alta confiança
        if cleaned_response == intent.value:
            return 0.95
        
        # Se for unknown, baixa confiança
        if intent == IntentType.UNKNOWN:
            return 0.3
        
        # Se a resposta contém o nome da intent, média-alta confiança
        if intent.value in cleaned_response:
            return 0.85
        
        # Caso contrário, média confiança
        return 0.7
    
    async def get_statistics(self) -> dict[str, any]:
        """
        Retorna estatísticas do serviço.
        
        Returns:
            dict[str, any]: Estatísticas do serviço
        """
        examples_count = self.prompt_manager.get_examples_count()
        
        return {
            "model": self.llm_provider.get_model_name(),
            "examples_loaded": examples_count,
            "supported_intents": [intent.value for intent in IntentType]
        }
    
    async def health_check(self) -> bool:
        """
        Verifica a saúde do serviço e suas dependências.
        
        Returns:
            bool: True se todas as dependências estão operacionais
        """
        try:
            # Verifica LLM provider
            llm_healthy = await self.llm_provider.health_check()
            
            if not llm_healthy:
                logger.warning("LLM provider health check falhou")
                return False
            
            # Verifica se exemplos estão carregados
            examples = await self.prompt_manager.load_examples()
            
            if not examples:
                logger.warning("Nenhum exemplo foi carregado")
                return False
            
            logger.info("Health check bem-sucedido")
            return True
            
        except Exception as e:
            logger.error(f"Health check falhou: {str(e)}")
            return False
