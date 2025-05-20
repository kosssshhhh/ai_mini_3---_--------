from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json


class KeywordExtractionAgent:
    """기술 트렌드 분석을 위한 키워드 추출 에이전트"""

    # 프롬프트 템플릿 분리
    KEYWORD_SYSTEM_PROMPT = """당신은 기술 트렌드 분석 전문가입니다. 
    주어진 대주제와 관련된 현재 가장 중요한 키워드들을 추출해야 합니다.
    각 키워드는 다음 조건을 만족해야 합니다:
    1. 현재 기술 트렌드에서 중요한 위치를 차지하고 있는 키워드
    2. 향후 5년간 발전 가능성이 높은 기술 분야
    3. 구체적인 기술이나 방법론을 나타내는 키워드
    
    응답은 반드시 다음과 같은 JSON 형식이어야 합니다:
    {
        "keywords": [
            "키워드1",
            "키워드2",
            "키워드3"
        ]
    }
    
    - 키워드는 영문으로 작성해주세요.
    - 각 키워드는 검색 가능한 형태여야 합니다.
    - 3개 키워드를 추출해주세요.
    - 일반적인 용어보다는 구체적인 기술 용어를 사용해주세요.
    """

    KEYWORD_HUMAN_PROMPT_TEMPLATE = """다음 대주제에 대한 주요 키워드들을 추출해주세요: {main_topic}

예시 응답 형식:
{{
    "keywords": [
        "Large Language Models",
        "Computer Vision",
        "Reinforcement Learning"
    ]
}}"""

    DESCRIPTION_SYSTEM_PROMPT = """각 기술 키워드에 대해 간단하고 명확한 설명을 제공해주세요.
    설명은 다음 사항을 포함해야 합니다:
    1. 기술의 정의
    2. 주요 특징
    3. 현재 발전 단계
    
    응답은 반드시 다음과 같은 JSON 형식이어야 합니다:
    {
        "키워드1": "설명1",
        "키워드2": "설명2"
    }
    """

    DESCRIPTION_HUMAN_PROMPT_TEMPLATE = """다음 키워드들에 대한 설명을 제공해주세요:

{keywords}

각 설명은 2-3문장으로 간단히 작성해주세요."""

    def __init__(self, model_name="gpt-4-turbo-preview"):
        """
        키워드 추출 에이전트 초기화

        Args:
            model_name: 사용할 OpenAI 모델 이름
        """
        self.llm = ChatOpenAI(model_name=model_name)
        self.logger = None

    def set_logger(self, logger):
        """워크플로우 로거 설정"""
        self.logger = logger

    def _log(self, level: str, message: str, keyword: str = None):
        """로깅 헬퍼 함수"""
        if self.logger and hasattr(self.logger, level):
            log_method = getattr(self.logger, level)
            if keyword:
                log_method(message, keyword)
            else:
                log_method(message)
        else:
            # 로거가 없거나 해당 레벨이 없는 경우 기본 출력
            print(f"[{level.upper()}] {message}")

    def _parse_json_response(self, response_text: str, fallback_handler=None) -> Any:
        """
        LLM 응답을 JSON으로 파싱

        Args:
            response_text: LLM 응답 텍스트
            fallback_handler: JSON 파싱 실패 시 대체 처리 함수

        Returns:
            파싱된 데이터
        """
        try:
            if response_text.startswith("{"):
                return json.loads(response_text)
            elif fallback_handler:
                return fallback_handler(response_text)
            else:
                raise ValueError("응답이 JSON 형식이 아닙니다")
        except json.JSONDecodeError as e:
            self._log("error", f"JSON 파싱 오류: {str(e)}")
            if fallback_handler:
                return fallback_handler(response_text)
            raise

    def _invoke_llm(self, system_prompt: str, human_prompt: str) -> str:
        """
        LLM 호출 공통 함수

        Args:
            system_prompt: 시스템 프롬프트
            human_prompt: 사용자 프롬프트

        Returns:
            LLM 응답 텍스트
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            self._log("error", f"LLM 호출 중 오류: {str(e)}")
            raise

    def _parse_keywords_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 키워드 추출 (JSON 파싱 실패 시 대체 처리)

        Args:
            text: 파싱할 텍스트

        Returns:
            추출된 키워드 목록
        """
        lines = text.split("\n")
        keywords = []
        for line in lines:
            line = line.strip()
            if line.startswith('"') or line.startswith("-"):
                keyword = line.strip('"- ').strip()
                if keyword:
                    keywords.append(keyword)
        return keywords

    def _parse_descriptions_from_text(
        self, text: str, keywords: List[str]
    ) -> Dict[str, str]:
        """
        텍스트에서 키워드 설명 추출 (JSON 파싱 실패 시 대체 처리)

        Args:
            text: 파싱할 텍스트
            keywords: 키워드 목록

        Returns:
            키워드: 설명 형태의 딕셔너리
        """
        descriptions = {}
        current_keyword = None
        current_description = []

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line in keywords:
                if current_keyword and current_description:
                    joined = " ".join(current_description)
                    descriptions[current_keyword] = joined
                current_keyword = line
                current_description = []
            elif current_keyword and line:
                current_description.append(line)

        if current_keyword and current_description:
            joined = " ".join(current_description)
            descriptions[current_keyword] = joined

        return descriptions

    def _create_description_handler(self, keywords: List[str]):
        """
        키워드 설명 처리를 위한 핸들러 함수 생성

        Args:
            keywords: 키워드 목록

        Returns:
            텍스트 처리 함수
        """

        # 키워드 설명 파싱 함수 반환
        def handler(text: str) -> Dict[str, str]:
            return self._parse_descriptions_from_text(text, keywords)

        return handler

    def extract_keywords(self, main_topic: str) -> List[str]:
        """
        주요 주제에서 관련된 키워드들을 추출합니다.

        Args:
            main_topic: 키워드를 추출할 주제

        Returns:
            추출된 키워드 목록
        """
        if self.logger:
            self._log("info", f"키워드 추출 시작: {main_topic}")

        try:
            # 프롬프트 준비
            human_prompt = self.KEYWORD_HUMAN_PROMPT_TEMPLATE.format(
                main_topic=main_topic
            )
            # LLM 호출
            response_text = self._invoke_llm(self.KEYWORD_SYSTEM_PROMPT, human_prompt)

            # 응답 파싱
            handler = self._parse_keywords_from_text
            keywords_data = self._parse_json_response(response_text, handler)

            # 키워드 추출
            if isinstance(keywords_data, dict) and "keywords" in keywords_data:
                keywords = keywords_data["keywords"]
            else:
                keywords = keywords_data

            # 결과 로깅
            if self.logger:
                extracted = ", ".join(keywords)
                self._log("success", f"키워드 추출 완료: {extracted}")

            return keywords
        except Exception as e:
            error_msg = f"키워드 추출 중 오류: {str(e)}"
            self._log("error", error_msg)
            return []

    def get_keyword_descriptions(self, keywords: List[str]) -> Dict[str, str]:
        """
        추출된 키워드들에 대한 간단한 설명을 생성합니다.

        Args:
            keywords: 설명을 생성할 키워드 목록

        Returns:
            키워드: 설명 형태의 딕셔너리
        """
        if not keywords:
            return {}

        if self.logger:
            kw_str = ", ".join(keywords)
            self._log("info", f"키워드 설명 생성 시작: {kw_str}")

        try:
            # 키워드 포맷팅
            formatted_keywords = json.dumps(keywords, indent=2, ensure_ascii=False)
            human_prompt = self.DESCRIPTION_HUMAN_PROMPT_TEMPLATE.format(
                keywords=formatted_keywords
            )
            response_text = self._invoke_llm(
                self.DESCRIPTION_SYSTEM_PROMPT, human_prompt
            )

            handler = self._create_description_handler(keywords)
            descriptions = self._parse_json_response(response_text, handler)

            if self.logger:
                count = len(descriptions)
                self._log("success", f"키워드 설명 생성 완료: {count}개")

            return descriptions
        except Exception as e:
            error_msg = f"키워드 설명 생성 중 오류: {str(e)}"
            self._log("error", error_msg)
            return {}
