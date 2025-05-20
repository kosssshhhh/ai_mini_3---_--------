import arxiv
from pytrends.request import TrendReq
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import feedparser
import os

load_dotenv()

# pandas 경고 설정
pd.set_option("future.no_silent_downcasting", True)


class CollectorAgent:
    """데이터 수집을 담당하는 에이전트"""

    # 클래스 상수 정의
    DEFAULT_MAX_WORKERS = 3
    DEFAULT_BATCH_SIZE = 50
    DEFAULT_MAX_RESULTS = 200
    DEFAULT_TIME_FRAME = "today 5-y"
    RECENT_WEEKS = 26  # 최근 트렌드 계산에 사용할 주 수
    PAPER_PERIOD_DAYS = 180  # 최근 6개월
    API_BATCH_SIZE = 100
    API_BATCH_DELAY = 1  # 초
    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    NEWS_API_KEY = os.getenv(
        "NEWS_API_KEY", "YOUR_NEWS_API_KEY"
    )  # .env 파일에서 키 불러오기

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS):
        """
        데이터 수집 에이전트 초기화

        Args:
            max_workers: 병렬 처리시 최대 작업자 수
        """
        self.pytrends = TrendReq(hl="en-US", tz=360)
        self.logger = None  # 워크플로우에서 주입
        self.arxiv_client = arxiv.Client()
        self.max_workers = max_workers

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

    def _optimize_search_keyword(self, keyword: str) -> str:
        """
        트렌드 검색을 위한 키워드 최적화

        Args:
            keyword: 원본 키워드

        Returns:
            최적화된 검색어
        """
        # 특수 케이스 처리
        if keyword == "GPT-4":
            return '"ChatGPT" "GPT-4"'  # GPT-4의 경우 더 일반적인 검색어 추가

        # 기본적으로 정확한 매칭을 위해 따옴표 추가
        return f'"{keyword}"'

    def _process_arxiv_result(
        self, result: arxiv.Result, paper_count: int, keyword: str
    ) -> Dict:
        """
        arXiv 검색 결과 처리

        Args:
            result: 논문 결과
            paper_count: 현재까지 수집된 논문 수
            keyword: 검색 키워드

        Returns:
            처리된 논문 데이터
        """
        pub_date = result.published.replace(tzinfo=None)
        paper_data = {
            "title": result.title,
            "date": pub_date.strftime("%Y-%m-%d"),
            "category": result.primary_category,
        }

        # 첫 번째 논문은 상세 정보 포함
        if paper_count == 0 and self.logger:
            # 논문 요약 준비
            summary = result.summary[:100] + "..." if result.summary else ""

            paper_data.update(
                {
                    "url": result.entry_id,
                    "summary": summary,
                }
            )
            self._log(
                "info",
                f"첫 번째 논문: {paper_data['title']} "
                f"({paper_data['category']}, {paper_data['date']})",
                keyword,
            )

        return paper_data

    def _log_batch_progress(
        self, batch_count: int, paper_count: int, last_date: str, keyword: str
    ):
        """배치 진행상황 로깅"""
        self._log(
            "info",
            f"배치 {batch_count} 완료: {paper_count}개 수집 "
            f"(최근 날짜: {last_date})",
            keyword,
        )

    def _log_collection_result(
        self, papers: List[Dict], paper_count: int, keyword: str
    ):
        """논문 수집 결과 로깅"""
        if papers:
            self._log(
                "success",
                f"수집 완료: {paper_count}개 논문, "
                f"기간: {papers[-1]['date']} ~ {papers[0]['date']}",
                keyword,
            )
        else:
            self._log("warning", "수집된 논문이 없습니다.", keyword)

    def collect_papers(
        self,
        keyword: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> List[Dict]:
        """
        arXiv에서 논문 수집 (최근 6개월)

        Args:
            keyword: 검색 키워드
            batch_size: 로깅 배치 크기
            max_results: 최대 검색 결과 수

        Returns:
            수집된 논문 목록
        """
        try:
            self._log(
                "info",
                f"'{keyword}' 관련 논문 검색 시작 (최대 {max_results}개)",
                keyword,
            )

            search = arxiv.Search(
                query=keyword,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            papers = []
            period_days = self.PAPER_PERIOD_DAYS
            six_months_ago = datetime.now() - timedelta(days=period_days)
            paper_count = 0
            batch_count = 0

            # Client.results() 사용
            for result in self.arxiv_client.results(search):
                # API 요청 간 딜레이 추가
                if paper_count > 0 and paper_count % self.API_BATCH_SIZE == 0:
                    time.sleep(self.API_BATCH_DELAY)

                # 기간 내 논문만 수집
                pub_date = result.published.replace(tzinfo=None)
                if pub_date > six_months_ago:
                    paper_data = self._process_arxiv_result(
                        result, paper_count, keyword
                    )
                    papers.append(paper_data)
                    paper_count += 1

                    # 배치 단위로 진행상황 기록
                    if paper_count % batch_size == 0:
                        batch_count += 1
                        self._log_batch_progress(
                            batch_count, paper_count, papers[-1]["date"], keyword
                        )
                else:
                    # 이전 논문이면 중단 (날짜순 정렬)
                    break

            self._log_collection_result(papers, paper_count, keyword)
            return papers

        except Exception as e:
            self._log("error", f"수집 실패: {str(e)}", keyword)
            return []

    def _validate_trend_data(
        self, trend_scores: List[float], dates: List[str], keyword: str
    ) -> bool:
        """
        트렌드 데이터 유효성 검증

        Args:
            trend_scores: 트렌드 점수 목록
            dates: 날짜 목록
            keyword: 검색 키워드

        Returns:
            데이터 유효 여부
        """
        # 데이터 존재 여부 확인
        if not trend_scores:
            self._log("warning", "트렌드 점수가 없습니다.", keyword)
            return False

        # 최근 데이터 검증
        recent_scores = trend_scores[-self.RECENT_WEEKS :]
        no_recent_data = not recent_scores or all(score == 0 for score in recent_scores)

        if no_recent_data:
            self._log(
                "warning",
                f"최근 트렌드 데이터가 없거나 모두 0입니다.",
                keyword,
            )
            return False

        return True

    def _log_trend_stats(
        self, trend_scores: List[float], dates: List[str], keyword: str
    ):
        """트렌드 데이터 통계 로깅"""
        if not trend_scores:
            return

        recent_scores = trend_scores[-self.RECENT_WEEKS :]
        avg_score = sum(trend_scores) / len(trend_scores)
        max_score = max(trend_scores)
        recent_avg = sum(recent_scores) / len(recent_scores)

        stats_msg = (
            f"데이터 통계:\n"
            f"- 전체 포인트: {len(trend_scores)}개\n"
            f"- 전체 평균: {avg_score:.1f}\n"
            f"- 최대값: {max_score}\n"
            f"- 최근 {self.RECENT_WEEKS}주 평균: {recent_avg:.1f}\n"
            f"- 기간: {dates[0]} ~ {dates[-1]}"
        )
        self._log("info", stats_msg, keyword)

    def _handle_trends_request(
        self, search_keyword: str, timeframe: str, keyword: str, retry_delay: int
    ) -> Tuple[List[float], List[str]]:
        """
        트렌드 API 요청 및 결과 처리

        Args:
            search_keyword: 검색어
            timeframe: 검색 기간
            keyword: 원본 키워드
            retry_delay: 재시도 대기 시간

        Returns:
            트렌드 점수와 날짜 tuple
        """
        # API 요청 간 대기 시간
        time.sleep(retry_delay)
        self.pytrends.build_payload([search_keyword], timeframe=timeframe)
        time.sleep(retry_delay)
        interest_over_time_df = self.pytrends.interest_over_time()

        if interest_over_time_df.empty:
            self._log("warning", "트렌드 데이터가 없습니다.", keyword)
            return [], []

        # 데이터 검증 및 디버깅
        self._log(
            "info",
            f"수집된 데이터 샘플:\n{interest_over_time_df.head()}",
            keyword,
        )

        dates = interest_over_time_df.index.strftime("%Y-%m-%d")
        trend_scores = interest_over_time_df[search_keyword].tolist()

        return trend_scores, dates

    def get_google_trends(
        self, keyword: str, timeframe: str = DEFAULT_TIME_FRAME
    ) -> Dict:
        """
        Google Trends 데이터 수집

        Args:
            keyword: 검색 키워드
            timeframe: 검색 기간

        Returns:
            트렌드 데이터
        """
        max_retries = 3
        retry_delay = 3

        # 키워드 검색어 최적화
        search_keyword = self._optimize_search_keyword(keyword)

        for attempt in range(max_retries):
            try:
                self._log(
                    "info",
                    f"트렌드 데이터 수집 시작 (검색어: {search_keyword})",
                    keyword,
                )

                # API 요청 및 결과 처리
                trend_scores, dates = self._handle_trends_request(
                    search_keyword, timeframe, keyword, retry_delay
                )

                # 데이터 유효성 검증
                if not self._validate_trend_data(trend_scores, dates, keyword):
                    return {"trend_scores": [], "dates": []}

                # 데이터 통계 로깅
                self._log_trend_stats(trend_scores, dates, keyword)

                # 결과 반환
                result = {
                    "trend_scores": trend_scores,
                    "dates": dates.tolist() if hasattr(dates, "tolist") else dates,
                }

                self._log("success", "트렌드 데이터 수집 완료", keyword)
                return result

            except Exception as e:
                msg = (
                    f"트렌드 데이터 수집 시도 {attempt + 1}/{max_retries} "
                    f"실패: {str(e)}"
                )
                self._log("warning", msg, keyword)

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    self._log("error", f"트렌드 데이터 수집 실패: {str(e)}", keyword)
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    return {"trend_scores": [], "dates": [current_date]}

    def _calculate_date_range(self, papers: List[Dict]) -> Dict:
        """
        논문 날짜 범위 계산

        Args:
            papers: 수집된 논문 목록

        Returns:
            날짜 범위 정보
        """
        if not papers:
            return {"start": None, "end": None, "days": 0}

        paper_dates = [datetime.strptime(paper["date"], "%Y-%m-%d") for paper in papers]
        start_date = min(paper_dates)
        end_date = max(paper_dates)

        return {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "days": (end_date - start_date).days + 1,  # 당일 포함
        }

    def _calculate_daily_counts(self, papers: List[Dict]) -> Dict:
        """
        일별 논문 수 집계

        Args:
            papers: 수집된 논문 목록

        Returns:
            일별 논문 수
        """
        daily_counts = {}
        for paper in papers:
            day = paper["date"]
            daily_counts[day] = daily_counts.get(day, 0) + 1
        return daily_counts

    def _calculate_trend_score(self, trend_scores: List[float], keyword: str) -> float:
        """
        트렌드 점수 계산

        Args:
            trend_scores: 트렌드 점수 목록
            keyword: 검색 키워드

        Returns:
            최근 트렌드 점수
        """
        if not trend_scores:
            return 0

        recent_scores = trend_scores[-self.RECENT_WEEKS :]
        if not recent_scores or all(score == 0 for score in recent_scores):
            return 0

        recent_trend_score = sum(recent_scores) / len(recent_scores)

        msg = (
            f"트렌드 점수 계산 결과:\n"
            f"- 최근 {self.RECENT_WEEKS}주 평균: {recent_trend_score:.1f}"
        )
        self._log("info", msg, keyword)

        return recent_trend_score

    def analyze_trend_metrics(self, keyword: str) -> Dict:
        """
        트렌드 메트릭스 분석

        Args:
            keyword: 검색 키워드

        Returns:
            분석 결과
        """
        papers = self.collect_papers(keyword)
        trends = self.get_google_trends(keyword)

        # 수집된 논문이 없는 경우 처리
        if not papers:
            return {
                "keyword": keyword,
                "paper_count": 0,
                "date_range": {"start": None, "end": None, "days": 0},
                "monthly_counts": {},
                "trend_score": 0,
                "papers": [],
                "google_trends": trends,
            }

        # 분석 수행
        date_range = self._calculate_date_range(papers)
        daily_counts = self._calculate_daily_counts(papers)
        trend_score = self._calculate_trend_score(trends["trend_scores"], keyword)

        return {
            "keyword": keyword,
            "paper_count": len(papers),
            "date_range": date_range,
            "daily_counts": daily_counts,
            "trend_score": trend_score,
            "papers": papers,
            "google_trends": trends,
        }

    def collect(self, keyword: str) -> Dict:
        """
        단일 키워드에 대한 데이터 수집

        Args:
            keyword: 검색 키워드

        Returns:
            수집 결과
        """
        try:
            papers = self.collect_papers(keyword)
            trends = self.get_google_trends(keyword)

            return {
                "papers": papers,
                "trends": trends,
                "status": "success",
                "error": None,
            }
        except Exception as e:
            error_msg = f"데이터 수집 실패: {str(e)}"
            self._log("error", error_msg, keyword)
            return {
                "papers": None,
                "trends": None,
                "status": "failed",
                "error": error_msg,
            }

    def collect_parallel(self, keywords: List[str]) -> Dict[str, Dict]:
        """
        여러 키워드에 대한 병렬 데이터 수집

        Args:
            keywords: 검색 키워드 목록

        Returns:
            키워드별 수집 결과
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 키워드에 대한 future 생성
            future_to_keyword = {
                executor.submit(self.collect, keyword): keyword for keyword in keywords
            }

            # 완료된 작업 처리
            for future in as_completed(future_to_keyword):
                keyword = future_to_keyword[future]
                try:
                    result = future.result()
                    results[keyword] = result
                except Exception as e:
                    error_msg = f"데이터 수집 실패: {str(e)}"
                    self._log("error", error_msg, keyword)
                    results[keyword] = {
                        "papers": None,
                        "trends": None,
                        "status": "failed",
                        "error": error_msg,
                    }

        return results

    def collect_news(self, keyword: str, days: int = 30) -> List[Dict]:
        """NewsAPI를 통한 뉴스 기사 수집"""
        self._log("info", f"뉴스 데이터 수집 시작: {keyword}", keyword)

        # 날짜 범위 설정 (최근 N일)
        today = datetime.now()
        from_date = today - timedelta(days=days)

        # API 요청 파라미터 설정
        params = {
            "q": keyword,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 100,
            "apiKey": self.NEWS_API_KEY,
        }

        try:
            # API 호출
            response = requests.get(self.NEWS_API_URL, params=params)
            response.raise_for_status()  # 오류 발생 시 예외 발생
            data = response.json()

            # 응답 데이터 확인
            if data["status"] != "ok":
                self._log(
                    "error",
                    f"뉴스 API 오류: {data.get('message', '알 수 없는 오류')}",
                    keyword,
                )
                return []

            if data["totalResults"] == 0:
                self._log("warning", f"뉴스 검색 결과 없음: {keyword}", keyword)
                return []

            # 뉴스 데이터 추출 및 정리
            articles = []
            for article in data["articles"]:
                news_item = {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "published_at": article.get("publishedAt", ""),
                    "author": article.get("author", ""),
                }
                articles.append(news_item)

            self._log("success", f"{len(articles)}개 뉴스 기사 수집 완료", keyword)
            return articles

        except requests.exceptions.RequestException as e:
            self._log("error", f"뉴스 API 요청 실패: {str(e)}", keyword)
            raise ValueError(f"NewsAPI 오류: {str(e)}")
        except Exception as e:
            self._log("error", f"뉴스 데이터 처리 중 오류: {str(e)}", keyword)
            raise ValueError(f"뉴스 데이터 처리 오류: {str(e)}")

    def _process_related_topics(self, related_topics_data):
        """관련 주제 데이터 처리"""
        result = {"rising": [], "top": []}

        for topic_type in ["rising", "top"]:
            if (
                topic_type in related_topics_data
                and not related_topics_data[topic_type].empty
            ):
                df = related_topics_data[topic_type]
                topics = []

                for _, row in df.iterrows():
                    topic = {
                        "title": row.get("topic_title", ""),
                        "type": row.get("topic_type", ""),
                        "value": float(row.get("value", 0)),
                    }
                    topics.append(topic)

                result[topic_type] = topics

        return result

    def _process_related_queries(self, related_queries_data):
        """관련 쿼리 데이터 처리"""
        result = {"rising": [], "top": []}

        for query_type in ["rising", "top"]:
            if (
                query_type in related_queries_data
                and not related_queries_data[query_type].empty
            ):
                df = related_queries_data[query_type]
                queries = []

                for _, row in df.iterrows():
                    query = {
                        "query": row.get("query", ""),
                        "value": float(row.get("value", 0)),
                    }
                    queries.append(query)

                result[query_type] = queries

        return result


if __name__ == "__main__":
    # 테스트 코드
    agent = CollectorAgent()
    result = agent.analyze_trend_metrics("GPT-4")
    print(f"Recent papers: {result['paper_count']}")
    print(f"Trend score: {result['trend_score']}")
