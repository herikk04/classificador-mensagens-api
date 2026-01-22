from abc import ABC, abstractmethod
from typing import Any

from src.domain.models import ClassificationResult, FewShotExample


class ILLMProvider(ABC):
  
    
    @abstractmethod
    async def generate_completion(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 512,
        **kwargs: Any
    ) -> str:
        """
        Gera uma completion baseada no prompt fornecido.
        
        Args:
            prompt: Texto do prompt para o LLM
            temperature: Temperatura para geração (0.0 = determinístico, 2.0 = criativo)
            max_tokens: Número máximo de tokens na resposta
            **kwargs: Parâmetros adicionais específicos do provider
        
        Returns:
            str: Texto gerado pelo LLM
        
        Raises:
            LLMProviderException: Se houver erro na comunicação com o LLM
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verifica se o provider está funcionando corretamente.
        
        Returns:
            bool: True se o provider está operacional
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Retorna o nome do modelo sendo utilizado.
        
        Returns:
            str: Nome do modelo (ex: "gemini-2.5-flash")
        """
        pass


class IPromptManager(ABC):
    """
    Interface para gerenciamento de prompts few-shot.
    Responsável por carregar exemplos e construir prompts estruturados.
    """
    
    @abstractmethod
    async def load_examples(self) -> list[FewShotExample]:
        """
        Carrega os exemplos few-shot do repositório (arquivo, DB, etc.).
        
        Returns:
            list[FewShotExample]: Lista de exemplos carregados
        
        Raises:
            ExamplesLoadException: Se houver erro ao carregar exemplos
        """
        pass
    
    @abstractmethod
    async def build_prompt(
        self,
        user_input: str,
        examples: list[FewShotExample] | None = None,
        max_examples: int = 5
    ) -> str:
        """
        Constrói o prompt few-shot para classificação de intenção.
        
        Args:
            user_input: Texto do usuário a ser classificado
            examples: Exemplos a serem incluídos (None = carrega automaticamente)
            max_examples: Número máximo de exemplos a incluir
        
        Returns:
            str: Prompt formatado pronto para envio ao LLM
        
        Raises:
            PromptBuildException: Se houver erro ao construir o prompt
        """
        pass
    
    @abstractmethod
    def get_system_instruction(self) -> str:
        """
        Retorna a instrução do sistema para o LLM.
        
        Returns:
            str: Instrução base do sistema
        """
        pass


class IIntentClassifier(ABC):
    """
    Interface para o serviço de classificação de intenção.
    Define o contrato principal da aplicação.
    """
    
    @abstractmethod
    async def classify(
        self,
        user_input: str,
        request_id: str | None = None
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
        pass
    
    @abstractmethod
    async def classify_batch(
        self,
        user_inputs: list[str],
        request_id: str | None = None
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
        pass


class IExamplesRepository(ABC):
    """
    Interface para repositório de exemplos few-shot.
    Abstrai a fonte de dados (arquivo JSON, DB, API, etc.).
    """
    
    @abstractmethod
    async def get_all(self) -> list[FewShotExample]:
        """
        Recupera todos os exemplos disponíveis.
        
        Returns:
            list[FewShotExample]: Lista de todos os exemplos
        
        Raises:
            ExamplesLoadException: Se houver erro ao recuperar exemplos
        """
        pass
    
    @abstractmethod
    async def get_by_intent(self, intent: str) -> list[FewShotExample]:
        """
        Recupera exemplos filtrados por intenção específica.
        
        Args:
            intent: Nome da intenção a filtrar
        
        Returns:
            list[FewShotExample]: Exemplos da intenção especificada
        """
        pass
    
    @abstractmethod
    async def add_example(self, example: FewShotExample) -> bool:
        """
        Adiciona um novo exemplo ao repositório.
        
        Args:
            example: Exemplo a ser adicionado
        
        Returns:
            bool: True se adicionado com sucesso
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """
        Conta o número total de exemplos disponíveis.
        
        Returns:
            int: Número de exemplos
        """
        pass
