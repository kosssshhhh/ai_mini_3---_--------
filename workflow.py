from typing import Dict, List, TypedDict, Optional
from langgraph.graph import StateGraph, END, START
from datetime import datetime
from agents.collector_agent import CollectorAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.report_agent import ReportAgent
from agents.keyword_extraction_agent import KeywordExtractionAgent
from utils.logger import Logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class KeywordResult(TypedDict):
    keyword: str
    status: str
    error: Optional[str]
    papers: Optional[List[Dict]]
    trends: Optional[Dict]
    news: Optional[List[Dict]]


class AnalysisResult(TypedDict):
    keyword: str
    status: str
    error: Optional[str]
    analysis: Optional[Dict]


class WorkflowState(TypedDict):
    main_topic: Optional[str]
    keywords: List[str]
    start_time: datetime
    status: str
    errors: List[str]
    collection_results: Optional[List[KeywordResult]]
    analysis_results: Optional[List[AnalysisResult]]
    report: Optional[str]
    report_path: Optional[str]
    failed_keywords: Optional[List[str]]
    extracted_keywords: Optional[Dict[str, str]]


class TrendAnalysisWorkflow:
    def __init__(self, max_workers: int = 3):
        self.graph = StateGraph(state_schema=WorkflowState)
        self.logger = Logger()

        # 에이전트 초기화 및 로거 주입
        self.keyword_extractor = KeywordExtractionAgent()
        self.collector = CollectorAgent()
        self.analyzer = AnalyzerAgent()
        self.reporter = ReportAgent()

        self._inject_logger_to_agents()
        self.max_workers = max_workers

    def _inject_logger_to_agents(self):
        """에이전트들에게 로거 주입"""
        agents = [self.collector, self.analyzer, self.reporter]
        for agent in agents:
            agent.set_logger(self.logger)

    def create_workflow(self):
        """워크플로우 그래프 생성"""
        # 노드 정의
        nodes = {
            "extract_keywords": self.extract_keywords,
            "collect_data": self.collect_data_parallel,
            "analyze_data": self.analyze_data_parallel,
            "generate_report": self.generate_report,
        }

        for name, func in nodes.items():
            self.graph.add_node(name, func)

        # 엣지 정의
        self.graph.add_edge(START, "extract_keywords")
        self.graph.add_edge("extract_keywords", "collect_data")
        self.graph.add_edge("collect_data", "analyze_data")
        self.graph.add_edge("analyze_data", "generate_report")
        self.graph.add_edge("generate_report", END)

        return self.graph.compile()

    def extract_keywords(self, state: WorkflowState) -> WorkflowState:
        """대주제에서 키워드 추출"""
        try:
            main_topic = state["main_topic"]
            self.logger.info(f"키워드 추출 시작: {main_topic}")
            keywords = self.keyword_extractor.extract_keywords(main_topic)
            descriptions = self.keyword_extractor.get_keyword_descriptions(keywords)

            state["keywords"] = keywords
            state["extracted_keywords"] = descriptions
            self.logger.success(f"키워드 추출 완료: {', '.join(keywords)}")

        except Exception as e:
            error_msg = f"키워드 추출 실패: {str(e)}"
            self.logger.error(error_msg)
            state["errors"].append(error_msg)
            state["keywords"] = []
            state["extracted_keywords"] = {}

        return state

    def collect_data_for_keyword(self, keyword: str) -> KeywordResult:
        """단일 키워드에 대한 데이터 수집"""
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                self.logger.info("처리 시작", keyword)
                papers = self.collector.collect_papers(keyword)
                trends = self.collector.get_google_trends(keyword)

                # 뉴스 데이터 수집 추가
                news = self.collector.collect_news(keyword, days=30)

                self.logger.success("처리 완료", keyword)

                return {
                    "keyword": keyword,
                    "papers": papers,
                    "trends": trends,
                    "news": news,  # 뉴스 데이터 추가
                    "status": "success",
                    "error": None,
                }
            except Exception as e:
                error_msg = f"{str(e)} (시도 {attempt + 1}/{max_retries})"
                self.logger.error(f"처리 실패: {error_msg}", keyword)

                if attempt < max_retries - 1:
                    self.logger.info(f"{retry_delay}초 후 재시도", keyword)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 지수 백오프
                else:
                    return self._create_failure_result(keyword, error_msg)

    def _create_failure_result(self, keyword: str, error_msg: str) -> KeywordResult:
        """실패 결과 생성 헬퍼 함수"""
        return {
            "keyword": keyword,
            "papers": None,
            "trends": None,
            "news": None,
            "status": "failed",
            "error": error_msg,
        }

    def _execute_parallel(self, items, worker_func, max_workers=None):
        """병렬 작업 실행을 위한 헬퍼 함수"""
        results = []
        workers = max_workers or self.max_workers

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker_func, item): item for item in items}
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def collect_data_parallel(self, state: WorkflowState) -> WorkflowState:
        """병렬 데이터 수집"""
        state["collection_results"] = self._execute_parallel(
            state["keywords"], self.collect_data_for_keyword
        )
        return state

    def analyze_single_keyword(self, data: KeywordResult) -> AnalysisResult:
        """단일 키워드 분석"""
        try:
            keyword = data["keyword"]
            self.logger.info("분석 시작", keyword)

            # 데이터 유효성 확인
            # 데이터 상태 확인
            status_failed = data["status"] == "failed"
            no_papers = not data["papers"]
            no_trends = not data["trends"]
            has_data_issues = status_failed or no_papers or no_trends
            if has_data_issues:
                error_msg = data.get("error", "데이터 수집 실패")
                return {
                    "keyword": keyword,
                    "status": "failed",
                    "error": error_msg,
                    "analysis": None,
                }

            try:
                analysis = self.analyzer.analyze_trend(data)
                self.logger.success("분석 완료", keyword)
                return {
                    "keyword": keyword,
                    "analysis": analysis,
                    "status": "success",
                    "error": None,
                }
            except Exception as analysis_error:
                error_msg = f"분석 중 오류: {str(analysis_error)}"
                self.logger.error(f"분석 실패: {error_msg}", keyword)
                return {
                    "keyword": keyword,
                    "status": "failed",
                    "error": error_msg,
                    "analysis": None,
                }

        except Exception as e:
            keyword = data["keyword"]
            error_msg = f"예상치 못한 오류: {str(e)}"
            self.logger.error(f"분석 실패: {error_msg}", keyword)
            return {
                "keyword": keyword,
                "status": "failed",
                "error": error_msg,
                "analysis": None,
            }

    def analyze_data_parallel(self, state: WorkflowState) -> WorkflowState:
        """병렬 데이터 분석"""
        state["analysis_results"] = self._execute_parallel(
            state["collection_results"], self.analyze_single_keyword
        )
        return state

    def _validate_analysis_fields(self, analysis):
        """분석 결과의 필수 필드 검증"""
        required_fields = [
            "trend_summary",
            "impact_score",
            "key_applications",
            "risks",
            "opportunities",
            "date_range",
            "paper_count",
        ]

        missing_fields = [
            field for field in required_fields if not hasattr(analysis, field)
        ]

        if missing_fields:
            fields_str = ", ".join(missing_fields)
            raise ValueError(f"분석 결과에 필수 필드가 누락됨: {fields_str}")

        return True

    def _convert_analysis_to_dict(self, keyword: str, analysis) -> Dict:
        """분석 결과 객체를 딕셔너리로 변환"""
        result = {
            "keyword": keyword,
            "trend_summary": analysis.trend_summary,
            "impact_score": analysis.impact_score,
            "key_applications": analysis.key_applications,
            "risks": analysis.risks,
            "opportunities": analysis.opportunities,
            "date_range": analysis.date_range,
            "paper_count": analysis.paper_count,
        }

        # 뉴스 분석 결과가 있는 경우 추가
        if hasattr(analysis, "news_analysis"):
            result["news_analysis"] = analysis.news_analysis

        return result

    def _process_analysis_results(self, results) -> tuple:
        """분석 결과 처리 및 변환"""
        successful_analyses = []
        failed_keywords = []

        for result in results:
            keyword = result["keyword"]
            if result["status"] == "success" and result["analysis"]:
                try:
                    self._validate_analysis_fields(result["analysis"])
                    analysis = result["analysis"]
                    analysis_dict = self._convert_analysis_to_dict(keyword, analysis)
                    successful_analyses.append(analysis_dict)
                    self.logger.success("분석 결과 처리 완료", keyword)
                except Exception as e:
                    error_msg = f"분석 결과 처리 중 오류: {str(e)}"
                    failed_keywords.append(keyword)
                    self.logger.error(f"분석 결과 처리 실패: {error_msg}", keyword)
            else:
                failed_keywords.append(keyword)
                error = result.get("error", "알 수 없는 오류")
                self.logger.warning(f"분석 결과 없음: {error}", keyword)

        return successful_analyses, failed_keywords

    def generate_report(self, state: WorkflowState) -> WorkflowState:
        """보고서 생성"""
        try:
            self.logger.info("보고서 생성 시작")

            # 분석 결과 처리
            analysis_data = self._process_analysis_results(state["analysis_results"])
            successful_analyses, failed_keywords = analysis_data
            state["failed_keywords"] = failed_keywords

            if not successful_analyses and not failed_keywords:
                raise ValueError("분석 결과가 없습니다.")

            if successful_analyses:
                count = len(successful_analyses)
                self.logger.info(f"성공한 분석 결과 수: {count}")
                report = self.reporter.generate_report(successful_analyses)
                state["report"] = report

                try:
                    filepath = self.reporter.save_report(report)
                    state["report_path"] = filepath
                    self.logger.success(f"보고서 저장 성공: {filepath}")
                except Exception as save_error:
                    error_msg = f"보고서 저장 실패: {str(save_error)}"
                    self.logger.error(error_msg)
                    state["errors"].append(error_msg)
                    state["report_path"] = None
            else:
                error_msg = "모든 키워드 분석이 실패했습니다."
                self.logger.error(error_msg)
                state["report"] = error_msg
                state["report_path"] = None

        except Exception as e:
            error_msg = f"보고서 생성 오류: {str(e)}"
            self.logger.error(error_msg)
            state["errors"].append(error_msg)
            state["report"] = f"보고서 생성 실패: {error_msg}"
            state["report_path"] = None
            if "failed_keywords" not in state:
                state["failed_keywords"] = state["keywords"].copy()

        return state

    def run(self, main_topic: str) -> WorkflowState:
        """워크플로우 실행"""
        self.logger.info("=== 워크플로우 실행 시작 ===")
        self.logger.info(f"분석할 대주제: {main_topic}")
        self.logger.info(f"최대 동시 작업 수: {self.max_workers}")

        workflow = self.create_workflow()
        start_time = datetime.now()

        initial_state: WorkflowState = {
            "main_topic": main_topic,
            "keywords": [],
            "start_time": start_time,
            "status": "started",
            "errors": [],
            "collection_results": [],
            "analysis_results": [],
            "report": None,
            "report_path": None,
            "failed_keywords": [],
            "extracted_keywords": {},
        }

        try:
            self.logger.info("워크플로우 시작")
            final_state = workflow.invoke(initial_state)

            # 실행 결과 분석 및 요약
            self._summarize_workflow_results(final_state, start_time)
            final_state["status"] = "completed"

        except Exception as e:
            error_msg = f"워크플로우 실행 중 오류 발생: {str(e)}"
            self.logger.error(error_msg)

            final_state = {
                **initial_state,
                "status": "failed",
                "errors": [error_msg],
                "failed_keywords": [],
            }

        return final_state

    def _summarize_workflow_results(self, state: WorkflowState, start_time: datetime):
        """워크플로우 실행 결과 요약"""
        # 실행 통계 계산
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 성공한 분석 결과 카운트
        results = state["analysis_results"]
        success_results = [r for r in results if r["status"] == "success"]
        success_count = len(success_results)
        fail_count = len(state["failed_keywords"])

        self.logger.success("=== 워크플로우 실행 완료 ===")
        self.logger.info(f"총 소요 시간: {duration:.1f}초")
        self.logger.info(f"성공: {success_count}개 키워드")
        self.logger.info(f"실패: {fail_count}개 키워드")

        if state["report_path"]:
            self.logger.success(f"보고서 저장: {state['report_path']}")
