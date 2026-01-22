import json
from pathlib import Path
from typing import Optional

from src.core.config import settings
from src.core.exceptions import ExamplesLoadException, PromptBuildException
from src.core.logger import get_logger
from src.domain.interfaces import IPromptManager
from src.domain.models import FewShotExample, IntentType

logger = get_logger(__name__)


class PromptManager(IPromptManager):
    """
    Implementação do gerenciador de prompts few-shot.
    Carrega exemplos de arquivo JSON e constrói prompts estruturados.
    """
    
    def __init__(
        self,
        examples_file_path: Optional[str] = None,
        max_examples: Optional[int] = None
    ) -> None:
        """
        Inicializa o gerenciador de prompts.
        
        Args:
            examples_file_path: Caminho para arquivo de exemplos (usa settings se None)
            max_examples: Número máximo de exemplos (usa settings se None)
        """
        self.examples_file_path = examples_file_path or settings.examples_file_path
        self.max_examples = max_examples or settings.max_examples_in_prompt
        self._examples_cache: Optional[list[FewShotExample]] = None
        
        logger.info(
            f"PromptManager inicializado: file={self.examples_file_path}, "
            f"max_examples={self.max_examples}"
        )
    
    async def load_examples(self) -> list[FewShotExample]:
        """
        Carrega exemplos few-shot do arquivo JSON.
        Utiliza cache para evitar recarregar múltiplas vezes.
        
        Returns:
            list[FewShotExample]: Lista de exemplos carregados
        
        Raises:
            ExamplesLoadException: Se houver erro ao carregar exemplos
        """
        # Retorna do cache se já carregado
        if self._examples_cache is not None:
            logger.debug(f"Retornando {len(self._examples_cache)} exemplos do cache")
            return self._examples_cache
        
        try:
            logger.info(f"Carregando exemplos de: {self.examples_file_path}")
            
            file_path = Path(self.examples_file_path)
            
            if not file_path.exists():
                raise ExamplesLoadException(
                    file_path=str(file_path),
                    reason="Arquivo não encontrado"
                )
            
            if not file_path.is_file():
                raise ExamplesLoadException(
                    file_path=str(file_path),
                    reason="Path não é um arquivo válido"
                )
            
            # Lê o arquivo JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Valida estrutura
            if not isinstance(data, dict) or "examples" not in data:
                raise ExamplesLoadException(
                    file_path=str(file_path),
                    reason="Estrutura JSON inválida. Esperado: {'examples': [...]}"
                )
            
            examples_data = data["examples"]
            
            if not isinstance(examples_data, list):
                raise ExamplesLoadException(
                    file_path=str(file_path),
                    reason="Campo 'examples' deve ser uma lista"
                )
            
            # Converte para objetos FewShotExample
            examples = []
            for idx, item in enumerate(examples_data):
                try:
                    example = FewShotExample(**item)
                    examples.append(example)
                except Exception as e:
                    logger.warning(
                        f"Exemplo {idx} inválido e será ignorado: {str(e)}"
                    )
            
            if not examples:
                raise ExamplesLoadException(
                    file_path=str(file_path),
                    reason="Nenhum exemplo válido encontrado"
                )
            
            # Armazena no cache
            self._examples_cache = examples
            
            logger.info(f"Carregados {len(examples)} exemplos com sucesso")
            return examples
            
        except json.JSONDecodeError as e:
            raise ExamplesLoadException(
                file_path=str(self.examples_file_path),
                reason=f"Erro ao decodificar JSON: {str(e)}"
            ) from e
        
        except ExamplesLoadException:
            raise
        
        except Exception as e:
            raise ExamplesLoadException(
                file_path=str(self.examples_file_path),
                reason=f"Erro inesperado: {str(e)}"
            ) from e
    
    async def build_prompt(
        self,
        user_input: str,
        examples: Optional[list[FewShotExample]] = None,
        max_examples: Optional[int] = None
    ) -> str:
        """
        Constrói o prompt few-shot para classificação de intenção.
        
        Args:
            user_input: Texto do usuário a ser classificado
            examples: Exemplos a usar (None = carrega automaticamente)
            max_examples: Número máximo de exemplos (None = usa padrão)
        
        Returns:
            str: Prompt formatado pronto para envio ao LLM
        
        Raises:
            PromptBuildException: Se houver erro ao construir o prompt
        """
        try:
            logger.debug(f"Construindo prompt para input: '{user_input[:50]}...'")
            
            # Carrega exemplos se não fornecidos
            if examples is None:
                examples = await self.load_examples()
            
            # Determina número de exemplos a usar
            num_examples = max_examples or self.max_examples
            selected_examples = examples[:num_examples]
            
            # Constrói o prompt
            prompt_parts = [
                self.get_system_instruction(),
                "",
                "## Exemplos de Classificação:",
                ""
            ]
            
            # Adiciona exemplos few-shot
            for idx, example in enumerate(selected_examples, 1):
                prompt_parts.append(f"Exemplo {idx}:")
                prompt_parts.append(f"Input: {example.user_input}")
                prompt_parts.append(f"Output: {example.intent.value}")
                prompt_parts.append("")
            
            # Adiciona o input do usuário
            prompt_parts.extend([
                "## Tarefa:",
                f"Input: {user_input}",
                "Output:"
            ])
            
            prompt = "\n".join(prompt_parts)
            
            logger.debug(
                f"Prompt construído: {len(prompt)} caracteres, "
                f"{len(selected_examples)} exemplos"
            )
            
            return prompt
            
        except Exception as e:
            raise PromptBuildException(
                reason=f"Erro ao construir prompt: {str(e)}",
                details={"user_input": user_input[:100]}
            ) from e
    
    def get_system_instruction(self) -> str:
        """
        Retorna a instrução do sistema para o LLM.
        Define o comportamento e formato de resposta esperado.
        
        Returns:
            str: Instrução base do sistema
        """
        intent_list = ", ".join([intent.value for intent in IntentType])
        
        return f"""Você é um classificador de intenções de texto em português brasileiro.

Sua tarefa é analisar o texto fornecido pelo usuário e classificar a intenção em UMA das seguintes categorias:
{intent_list}

## Regras:
1. Retorne APENAS o nome da categoria (exemplo: "greeting", "question", "help")
2. NÃO inclua explicações, pontuação extra ou texto adicional
3. A resposta deve ser uma única palavra em minúsculas
4. Se não tiver certeza, retorne "unknown"
5. Analise o contexto e o tom da mensagem para determinar a intenção correta

## Descrição das Intenções:
- greeting: Saudações, cumprimentos iniciais
- farewell: Despedidas, encerramentos
- question: Perguntas ou dúvidas
- complaint: Reclamações, insatisfações
- compliment: Elogios, agradecimentos positivos
- request: Solicitações, pedidos de ação
- information: Fornecimento de informações
- help: Pedidos de ajuda ou suporte
- cancellation: Solicitações de cancelamento
- confirmation: Confirmações ou validações
- unknown: Intenção não identificada"""
    
    def clear_cache(self) -> None:
        """Limpa o cache de exemplos carregados."""
        self._examples_cache = None
        logger.info("Cache de exemplos limpo")
    
    async def reload_examples(self) -> list[FewShotExample]:
        """
        Recarrega os exemplos do arquivo, ignorando o cache.
        
        Returns:
            list[FewShotExample]: Novos exemplos carregados
        """
        logger.info("Recarregando exemplos (forçando reload)")
        self.clear_cache()
        return await self.load_examples()
    
    def get_examples_count(self) -> int:
        """
        Retorna o número de exemplos carregados no cache.
        
        Returns:
            int: Número de exemplos (0 se não carregado)
        """
        if self._examples_cache is None:
            return 0
        return len(self._examples_cache)
    
    def get_examples_by_intent(self, intent: IntentType) -> list[FewShotExample]:
        """
        Filtra exemplos por intenção específica.
        
        Args:
            intent: Intenção a filtrar
        
        Returns:
            list[FewShotExample]: Exemplos da intenção especificada
        """
        if self._examples_cache is None:
            logger.warning("Cache vazio. Carregue os exemplos primeiro.")
            return []
        
        filtered = [ex for ex in self._examples_cache if ex.intent == intent]
        logger.debug(f"Encontrados {len(filtered)} exemplos para intent '{intent.value}'")
        return filtered
