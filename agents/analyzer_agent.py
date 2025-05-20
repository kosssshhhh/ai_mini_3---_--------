from typing import Dict, List, Union
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class TrendAnalysis(BaseModel):
    """기술 트렌드 분석 결과 모델"""

    trend_summary: str = Field(description="기술 트렌드 요약")
    impact_score: int = Field(description="영향력 점수 (1-10)")
    key_applications: List[str] = Field(description="주요 응용 분야")
    risks: List[str] = Field(description="잠재적 위험 요소")
    opportunities: List[str] = Field(description="사업 기회")
    date_range: Dict[str, Union[str, int]] = Field(
        description="데이터 수집 기간 정보 (start: 시작일, end: 종료일, days: 일수)",
        default_factory=lambda: {"start": "", "end": "", "days": 0},
    )
    paper_count: int = Field(description="수집된 논문 수")
    news_analysis: Dict = Field(
        description="뉴스 데이터 분석 결과", default_factory=dict
    )


class AnalyzerAgent:
    """AI 기술 트렌드 분석을 수행하는 에이전트"""

    # 클래스 상수 정의
    DEFAULT_TEMPERATURE = 0
    TREND_WINDOW_WEEKS = 26  # 최근 26주(약 6개월)
    MAX_PAPERS_DETAIL = 10  # 상세 분석을 위한 최대 논문 수

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        """
        분석 에이전트 초기화

        Args:
            temperature: LLM 생성 다양성 제어 파라미터 (0: 결정적, 1: 다양)
        """
        self.llm = ChatOpenAI(temperature=temperature)
        self.logger = None  # 워크플로우에서 주입
        self.parser = PydanticOutputParser(pydantic_object=TrendAnalysis)

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

    def _calculate_trend_score(self, trends: Dict) -> float:
        """
        Google Trends 데이터로부터 트렌드 점수 계산

        Args:
            trends: Google Trends 데이터

        Returns:
            최근 트렌드 평균 점수
        """
        try:
            if not trends or "trend_scores" not in trends:
                return 0.0

            # 최근 6개월의 트렌드 점수 평균 계산
            weeks = self.TREND_WINDOW_WEEKS
            recent_scores = trends["trend_scores"][-weeks:]
            if not recent_scores:
                return 0.0

            return sum(recent_scores) / len(recent_scores)

        except Exception as e:
            self._log("warning", f"트렌드 점수 계산 실패: {str(e)}")
            return 0.0

    def _format_first_paper(self, paper: Dict) -> str:
        """
        첫 번째 논문 상세 정보 포맷팅

        Args:
            paper: 논문 데이터

        Returns:
            포맷된 논문 정보 문자열
        """
        first_paper_text = (
            f"제목: {paper.get('title', '제목 없음')}\n"
            f"카테고리: {paper.get('category', '분류 없음')}\n"
            f"날짜: {paper.get('date', '날짜 없음')}"
        )

        if "summary" in paper and paper["summary"]:
            first_paper_text += f"\n요약: {paper['summary']}"

        return first_paper_text

    def _format_paper_titles(self, papers: List[Dict], limit: int = 9) -> str:
        """
        논문 제목 목록 포맷팅 (첫 번째 논문 제외)

        Args:
            papers: 논문 목록
            limit: 표시할 최대 논문 수

        Returns:
            포맷된 논문 제목 목록 문자열
        """
        if not papers or len(papers) <= 1:
            return "추가 논문 없음"

        # 2번째부터 limit+1번째 논문까지만 표시
        remaining = papers[1 : min(limit + 1, len(papers))]
        titles = []

        for p in remaining:
            date = p.get("date", "날짜 없음")
            category = p.get("category", "분류 없음")
            title = p.get("title", "제목 없음")
            titles.append(f"- {date} | {category} | {title}")

        return "\n".join(titles)

    def _calculate_date_range(self, papers: List[Dict]) -> Dict[str, Union[str, int]]:
        """
        논문 날짜 기반 데이터 수집 기간 계산

        Args:
            papers: 논문 목록

        Returns:
            기간 정보 딕셔너리 (시작일, 종료일, 일수)
        """
        # 날짜 추출
        paper_dates = [p.get("date") for p in papers if p.get("date")]

        if not paper_dates:
            self._log("warning", "유효한 논문 날짜가 없습니다.")
            return {"start": "", "end": "", "days": 0}

        # 시작일, 종료일, 기간 계산
        start_date = min(paper_dates)
        end_date = max(paper_dates)

        try:
            days = (
                datetime.strptime(end_date, "%Y-%m-%d")
                - datetime.strptime(start_date, "%Y-%m-%d")
            ).days + 1  # 당일 포함
        except ValueError as e:
            self._log("error", f"날짜 형식 오류: {str(e)}")
            days = 0

        return {"start": start_date, "end": end_date, "days": days}

    def _prepare_analysis_data(self, data: Dict) -> Dict:
        """
        분석에 필요한 데이터 준비 및 가공

        Args:
            data: 원본 데이터

        Returns:
            분석용 정제 데이터
        """
        papers = data.get("papers", [])
        trends = data.get("trends", {})
        keyword = data.get("keyword", "")

        # 날짜 범위 계산
        date_range = self._calculate_date_range(papers)
        days = date_range.get("days", 0)

        # 일평균 논문 수 계산
        papers_per_day = len(papers) / days if days > 0 else 0

        # 트렌드 점수 계산
        trend_score = self._calculate_trend_score(trends)

        # 최신 n개 논문만 선택
        recent_papers = papers[: self.MAX_PAPERS_DETAIL]

        return {
            "keyword": keyword,
            "papers": recent_papers,
            "date_range": date_range,
            "paper_count": len(papers),
            "papers_per_day": papers_per_day,
            "trend_score": trend_score,
        }

    def create_analysis_prompt(self, data: Dict) -> List:
        """
        분석 프롬프트 생성

        Args:
            data: 분석에 사용할 처리된 데이터

        Returns:
            LLM 입력용 포맷팅된 프롬프트
        """
        template = """다음 데이터를 기반으로 기술 트렌드를 분석해주세요:

키워드: {keyword}
수집 기간: {days}일 ({start_date} ~ {end_date})
수집된 논문 수: {paper_count}개
일평균 논문 수: {papers_per_day:.1f}개
트렌드 점수: {trend_score:.1f}

최신 논문 정보:
{first_paper}

전체 논문 제목 목록 (최신순):
{paper_titles}

다음 형식으로 분석 결과를 제공해주세요:
{format_instructions}
"""
        # 분석에 필요한 정보 추출
        papers = data.get("papers", [])
        if not papers:
            self._log("warning", "분석할 논문이 없습니다.")
            return []

        # 첫 번째 논문 상세 정보
        first_paper = self._format_first_paper(papers[0])

        # 나머지 논문들의 제목만 표시
        paper_titles = self._format_paper_titles(papers)

        # 프롬프트 생성
        return ChatPromptTemplate.from_template(template).format_messages(
            keyword=data.get("keyword", ""),
            days=data.get("date_range", {}).get("days", 0),
            start_date=data.get("date_range", {}).get("start", ""),
            end_date=data.get("date_range", {}).get("end", ""),
            paper_count=data.get("paper_count", 0),
            papers_per_day=data.get("papers_per_day", 0),
            trend_score=data.get("trend_score", 0),
            first_paper=first_paper,
            paper_titles=paper_titles,
            format_instructions=self.parser.get_format_instructions(),
        )

    def _parse_analysis_response(
        self, content: str, keyword: str = ""
    ) -> TrendAnalysis:
        """
        GPT 응답을 TrendAnalysis 객체로 파싱

        Args:
            content: GPT 응답 텍스트
            keyword: 분석 키워드 (로깅용)

        Returns:
            파싱된 TrendAnalysis 객체
        """
        try:
            analysis = self.parser.parse(content)
            self._log("success", "응답 파싱 성공", keyword)
            return analysis
        except Exception as e:
            error_msg = f"응답 파싱 실패: {str(e)}"
            self._log("error", error_msg, keyword)
            raise ValueError(error_msg)

    def analyze_trend(self, data: Dict) -> TrendAnalysis:
        """
        트렌드 분석 수행

        Args:
            data: 분석할 데이터 (키워드, 논문, 트렌드 정보, 뉴스)

        Returns:
            분석 결과 객체
        """
        keyword = data.get("keyword", "unknown")

        try:
            self._log("info", "데이터 분석 시작")

            # 추출된 데이터 검증
            papers = data.get("papers", [])
            trends_data = data.get("trends", {})
            news_data = data.get("news", [])

            if not papers and not trends_data:
                raise ValueError("분석할 데이터가 충분하지 않습니다")

            # 유효성 검증
            if not papers:
                error_msg = "분석할 논문 데이터가 없습니다."
                self._log("error", error_msg, keyword)
                raise ValueError(error_msg)

            if not trends_data:
                warn_msg = "트렌드 데이터가 없습니다. 제한된 분석이 수행됩니다."
                self._log("warning", warn_msg, keyword)

            # 논문 메타데이터 분석
            paper_analysis = self._analyze_papers(papers)

            # 트렌드 데이터 분석
            trends_analysis = self._analyze_trends(trends_data)

            # 뉴스 데이터 분석 추가
            news_analysis = self._analyze_news(news_data)

            # 분석 데이터 준비
            analysis_data = self._prepare_analysis_data(data)
            self._log("info", "분석 데이터 준비 완료", keyword)

            # 분석 프롬프트 생성
            prompt = self.create_analysis_prompt(analysis_data)
            self._log("info", "분석 프롬프트 생성 완료", keyword)

            # GPT를 통한 분석 수행
            response = self.llm.invoke(prompt)
            self._log("info", "GPT 분석 완료", keyword)

            # 응답 파싱
            analysis = self._parse_analysis_response(response.content, keyword)
            self._log("success", "분석 완료", keyword)

            # 날짜 및 논문 수 정보 업데이트
            analysis.date_range = analysis_data["date_range"]
            analysis.paper_count = analysis_data["paper_count"]

            # 뉴스 분석 결과 추가
            analysis.news_analysis = news_analysis

            self._log("success", "데이터 분석 완료")
            return analysis

        except Exception as e:
            error_msg = f"분석 실패: {str(e)}"
            self._log("error", error_msg, keyword)
            raise ValueError(f"분석 실패: {str(e)}")

    def _analyze_papers(self, papers: List[Dict]) -> Dict:
        """논문 메타데이터 분석"""
        self._log("info", "논문 데이터 분석 중")

        if not papers:
            return {
                "paper_count": 0,
                "date_range": {"start": "", "end": "", "days": 0},
                "top_categories": [],
                "top_keywords": [],
                "yearly_counts": "0건",
            }

        # 논문 개수
        paper_count = len(papers)

        # 날짜 범위 분석
        date_range = self._calculate_date_range(papers)

        # 카테고리 분석
        categories = {}
        for paper in papers:
            category = paper.get("category", "")
            if category:
                categories[category] = categories.get(category, 0) + 1

        top_categories = [
            cat
            for cat, count in sorted(
                categories.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

        # 키워드 추출 (간단하게 논문 제목에서 추출)
        keywords = {}
        stopwords = ["the", "a", "an", "and", "of", "in", "for", "with", "on", "by"]
        for paper in papers:
            title = paper.get("title", "")
            if title:  # None 체크 추가
                for word in title.lower().split():
                    if len(word) > 3 and word not in stopwords:
                        keywords[word] = keywords.get(word, 0) + 1

        top_keywords = [
            kw
            for kw, count in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
        ]

        # 연도별 논문 수 집계
        yearly_counts = {}
        for paper in papers:
            date = paper.get("date", "")
            if date:
                year = date.split("-")[0]
                yearly_counts[year] = yearly_counts.get(year, 0) + 1

        yearly_counts_str = ", ".join(
            [f"{year}: {count}건" for year, count in sorted(yearly_counts.items())]
        )

        return {
            "paper_count": paper_count,
            "date_range": date_range,
            "top_categories": top_categories,
            "top_keywords": top_keywords,
            "yearly_counts": yearly_counts_str if yearly_counts else "0건",
        }

    def _analyze_trends(self, trends_data: Dict) -> Dict:
        """트렌드 데이터 분석"""
        self._log("info", "트렌드 데이터 분석 중")

        if (
            not trends_data
            or "trend_scores" not in trends_data
            or not trends_data["trend_scores"]
        ):
            return {
                "trend_description": "데이터 없음",
                "change_rate": 0,
                "recent_interest": "낮음",
                "related_queries": [],
            }

        scores = trends_data.get("trend_scores", [])

        # 관심도 변화율 계산 (최근 6개월과 이전 6개월 비교)
        if len(scores) >= 52:  # 최소 1년치 데이터 필요
            recent = scores[-26:]  # 최근 6개월
            previous = scores[-52:-26]  # 이전 6개월

            recent_avg = sum(recent) / len(recent) if recent else 0
            previous_avg = sum(previous) / len(previous) if previous else 0

            if previous_avg > 0:
                change_rate = round((recent_avg - previous_avg) / previous_avg * 100)
            else:
                change_rate = 0
        else:
            change_rate = 0

        # 최근 관심도 평가
        recent_scores = scores[-26:] if len(scores) >= 26 else scores
        recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0

        if recent_avg > 70:
            interest_level = "매우 높음"
        elif recent_avg > 50:
            interest_level = "높음"
        elif recent_avg > 30:
            interest_level = "중간"
        elif recent_avg > 10:
            interest_level = "낮음"
        else:
            interest_level = "매우 낮음"

        # 트렌드 설명
        if change_rate > 50:
            trend_desc = "급격히 상승 중"
        elif change_rate > 20:
            trend_desc = "상승 중"
        elif change_rate > -20:
            trend_desc = "안정적"
        elif change_rate > -50:
            trend_desc = "하락 중"
        else:
            trend_desc = "급격히 하락 중"

        # 관련 쿼리 (일반적으로 API에서 제공하지만 여기서는 더미 데이터 사용)
        related_queries = ["관련 검색어 정보 없음"]

        return {
            "trend_description": trend_desc,
            "change_rate": change_rate,
            "recent_interest": interest_level,
            "related_queries": related_queries,
        }

    def _prepare_analysis_prompt(
        self,
        keyword: str,
        paper_analysis: Dict,
        trends_analysis: Dict,
        news_analysis: Dict,
    ) -> str:
        """LLM 분석을 위한 프롬프트 준비"""
        prompt = f"""
        다음 데이터를 기반으로 '{keyword}' 기술의 트렌드를 분석해주세요:
        
        1. 논문 데이터:
        - 총 논문 수: {paper_analysis['paper_count']}개
        - 수집 기간: {paper_analysis['date_range']['start']} ~ {paper_analysis['date_range']['end']}
        - 주요 카테고리: {', '.join(paper_analysis['top_categories'])}
        - 주요 키워드: {', '.join(paper_analysis['top_keywords'])}
        - 연간 논문 수 추이: {paper_analysis['yearly_counts']}
        
        2. 구글 트렌드 데이터:
        - 관심도 추세: {trends_analysis['trend_description']}
        - 관심도 변화율: {trends_analysis['change_rate']}%
        - 최근 관심도: {trends_analysis['recent_interest']}
        - 관련 검색어: {', '.join(trends_analysis['related_queries'])}
        
        3. 뉴스 데이터:
        - 최근 {news_analysis['timeframe']}일 내 뉴스 기사 수: {news_analysis['article_count']}개
        - 주요 출처: {', '.join(news_analysis['top_sources'])}
        - 핵심 토픽: {', '.join(news_analysis['key_topics'])}
        - 감성 분석: 긍정({news_analysis['sentiment']['positive']}%), 중립({news_analysis['sentiment']['neutral']}%), 부정({news_analysis['sentiment']['negative']}%)
        - 최근 주요 이벤트: {news_analysis['recent_events']}
        
        다음 항목을 분석해주세요:
        1. 트렌드 요약: 이 기술의 현재 동향과 미래 방향성에 대한 250자 내외 요약
        2. 영향력 점수: 향후 5년간 이 기술의 영향력을 1-10 점수로 평가
        3. 주요 응용 분야: 이 기술이 가장 큰 영향을 미칠 5개 산업 영역 나열
        4. 잠재적 위험 요소: 이 기술의 발전을 방해할 수 있는 5가지 위험 요소
        5. 사업적 기회: 이 기술과 관련된 5가지 유망한 사업 기회 나열
        
        결과는 JSON 형식으로 반환해주세요.
        """
        return prompt

    def _run_llm_analysis(self, prompt: str) -> TrendAnalysis:
        """LLM을 이용한 분석 실행"""
        self._log("info", "LLM 분석 실행 중")

        # ChatPromptTemplate 생성
        prompt_template = ChatPromptTemplate.from_template(prompt)

        # LLM 체인 구성
        chain = prompt_template | self.llm | self.parser

        # 분석 실행
        try:
            analysis = chain.invoke({})
            self._log("success", "LLM 분석 완료")
            return analysis
        except Exception as e:
            self._log("error", f"LLM 분석 실패: {str(e)}")
            raise ValueError(f"LLM 분석 중 오류: {str(e)}")

    def _analyze_news(self, news_data: List[Dict]) -> Dict:
        """뉴스 데이터 분석"""
        self._log("info", "뉴스 데이터 분석 중")

        if not news_data:
            return {
                "timeframe": 30,
                "article_count": 0,
                "top_sources": [],
                "key_topics": [],
                "sentiment": {"positive": 0, "neutral": 0, "negative": 0},
                "recent_events": "관련 뉴스 데이터 없음",
            }

        # 뉴스 개수
        article_count = len(news_data)

        # 주요 출처 분석
        sources = {}
        for article in news_data:
            source = article.get("source", "")
            if source:
                sources[source] = sources.get(source, 0) + 1

        top_sources = [
            source
            for source, count in sorted(
                sources.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

        # 간단한 키워드 추출 (제목 기반)
        keywords = {}
        all_titles = []
        for article in news_data:
            title = article.get("title", "")
            if title:  # None 체크 추가
                all_titles.append(title)

        # 제목이 있는 경우에만 처리
        if all_titles:
            all_titles_text = " ".join(all_titles)

            # 불용어 제거 및 빈도수 계산
            stopwords = [
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "by",
            ]
            for word in all_titles_text.lower().split():
                if len(word) > 3 and word not in stopwords:
                    keywords[word] = keywords.get(word, 0) + 1

        key_topics = (
            [
                topic
                for topic, count in sorted(
                    keywords.items(), key=lambda x: x[1], reverse=True
                )[:10]
            ]
            if keywords
            else []
        )

        # 간단한 감성 분석 (실제로는 NLP 라이브러리 사용 권장)
        positive_words = ["growth", "innovation", "success", "breakthrough", "positive"]
        negative_words = ["risk", "challenges", "threat", "problem", "issue", "crisis"]

        sentiment = {"positive": 0, "neutral": 0, "negative": 0}
        for article in news_data:
            title = article.get("title", "")
            desc = article.get("description", "")

            # None 체크 추가
            content = ""
            if title:
                content += title.lower()
            if desc:
                content += " " + desc.lower()

            is_positive = any(word in content for word in positive_words)
            is_negative = any(word in content for word in negative_words)

            if is_positive and not is_negative:
                sentiment["positive"] += 1
            elif is_negative and not is_positive:
                sentiment["negative"] += 1
            else:
                sentiment["neutral"] += 1

        # 백분율로 변환
        total = article_count
        if total > 0:
            sentiment["positive"] = round(sentiment["positive"] / total * 100)
            sentiment["neutral"] = round(sentiment["neutral"] / total * 100)
            sentiment["negative"] = round(sentiment["negative"] / total * 100)

        # 최근 주요 이벤트 (가장 최근 3개 뉴스 헤드라인)
        sorted_articles = sorted(
            news_data, key=lambda x: x.get("published_at", ""), reverse=True
        )

        # None 체크 추가
        recent_headlines = []
        for article in sorted_articles[:3]:
            title = article.get("title", "")
            if title:
                recent_headlines.append(title)

        recent_events = (
            " / ".join(recent_headlines)
            if recent_headlines
            else "주요 이벤트 정보 없음"
        )

        return {
            "timeframe": 30,
            "article_count": article_count,
            "top_sources": top_sources,
            "key_topics": key_topics,
            "sentiment": sentiment,
            "recent_events": recent_events,
        }


if __name__ == "__main__":
    # 테스트 코드
    test_data = {
        "keyword": "Quantum Computing",
        "paper_count": 15,
        "date_range": {"start": "2024-01-01", "end": "2024-03-01", "days": 60},
        "daily_counts": {"2024-01-01": 5, "2024-02-01": 5, "2024-03-01": 5},
        "trend_score": 75.5,
        "papers": [
            {
                "title": "Recent Advances in Quantum Computing",
                "category": "quant-ph",
                "date": "2024-03-01",
                "summary": "This paper discusses the latest developments...",
            }
        ],
        "google_trends": {"dates": ["2023-01-01", "2023-01-02"]},
    }

    analyzer = AnalyzerAgent()
    result = analyzer.analyze_trend(test_data)
    print("\n분석 결과:")
    print(f"트렌드 요약: {result.trend_summary}")
    print(f"영향력 점수: {result.impact_score}")
    print(f"수집 기간: {result.date_range['start']} ~ {result.date_range['end']}")
    print(f"논문 수: {result.paper_count}개")
