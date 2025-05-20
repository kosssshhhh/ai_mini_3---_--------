from typing import Dict, List, Any, Tuple
import markdown
from datetime import datetime
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.logger import Logger
import weasyprint


class ReportAgent:
    """보고서 생성을 담당하는 에이전트"""

    # 프롬프트 템플릿 상수
    SUMMARY_PROMPT_TEMPLATE = """
        다음 기술 트렌드 분석 보고서를 간결하게 요약해 주세요.
        
        분석된 키워드: {keywords}
        
        영향력 높은 기술들:
        {high_impact_techs}
        
        보고서 전체 내용:
        {report_content}
        
        요약은 다음 형식을 따라 마크다운으로 작성해 주세요:
        1. ## SUMMARY
        2. 전체 트렌드 핵심 요약 (3-5문장)
        3. 주요 발견점 (3-5개의 글머리 기호)
        4. 핵심 통찰 (1-2문장)
        5. 주목할 기업들 (2-3개 글머리 기호)
        
        최대 200단어 내외로 간결하게 작성해 주세요.
        반드시 한국어로 작성해 주세요.
        """

    COMPANY_PROMPT_TEMPLATE = """
        다음 기술 분야의 주요 기업들과 유망 스타트업에 대한 정보를 제공해주세요.
        
        기술 분야: {keyword}
        기술 요약: {summary}
        주요 응용 분야: {applications}
        기회 요소: {opportunities}
        
        다음 내용을 포함하여 한국어로 마크다운 형식으로 작성해 주세요:
        1. ### 주요 선도 기업 (3-5개)
        - 각 기업별 기술 분야 내 주요 제품/서비스와 시장 포지션
        - 왜 이 기업이 이 기술 분야에서 중요한지 간략히 설명
        
        2. ### 유망 스타트업 (3-5개)
        - 각 스타트업의 혁신적 접근법
        - 투자 현황이나 성장 가능성에 대한 간단한 평가
        
        3. ### 기업 생태계 트렌드
        - 이 기술 분야의 투자/인수 동향
        - 기업들이 취하고 있는 전략적 방향
        
        글로벌 기업과 함께 한국의 관련 기업이나 스타트업도 포함해주세요.
        특히 최근 2년 내 주목받는 기업들에 집중해 주세요.
        실제 존재하는 회사 정보만 제공하고, 가상의 회사는 만들지 마세요.
        """

    TREND_PROMPT_TEMPLATE = """
        기술 트렌드 분석 보고서의 세부 섹션을 작성해 주세요.
        
        주제: {keyword}
        영향력 점수: {impact_score}/10
        트렌드 요약: {trend_summary}
        주요 응용 분야: {applications}
        잠재적 위험 요소: {risks}
        사업 기회: {opportunities}
        수집 기간: {date_info}
        총 논문 수: {paper_count}개
        일평균 논문 수: {papers_per_day}개
        
        다음 내용을 포함하여 한국어로 마크다운 형식으로 작성해 주세요:
        1. 현재 기술 상태 분석 (1-2 문단)
        2. 주요 발전 방향 (1 문단)
        3. 산업에 미치는 영향 (1 문단)
        4. 전문가의 시각 (인사이트를 제공하는 1 문단)
        
        전체적으로 정확하고 객관적이면서도 통찰력 있는 내용으로 작성해 주세요.
        반드시 한국어로 응답해 주세요.
        """

    CONCLUSION_PROMPT_TEMPLATE = """
        기술 트렌드 분석 보고서의 결론 부분을 작성해 주세요.
        
        분석된 키워드(영향력 순): {keywords}
        
        각 키워드별 요약:
        {summaries}
        
        확인된 기회 요소들:
        {opportunities}
        
        다음 형식을 따라 한국어로 마크다운으로 작성해 주세요:
        1. ## 결론
        2. ### 종합 분석 (전체 트렌드를 종합적으로 분석한 2-3 문단)
        3. ### 주요 기회 영역 (가장 중요한 5개 기회 영역 추출)
        4. ### 투자 유망 분야 (각 기술 분야별 투자가 유망한 세부 영역 3-5개)
        5. ### 미래 전망 (향후 5년간의 전망을 1-2 문단으로 서술)
        6. ### 제언 (기업과 연구자들을 위한 실용적 조언 3-5개)
        
        통찰력 있고 실용적인 내용으로 작성하되, 마크다운 형식을 정확히 따라주세요.
        반드시 한국어로 작성해 주세요.
        """

    # 기타 상수
    DEFAULT_FORMAT = "md"
    SUPPORTED_FORMATS = ["md", "html", "pdf"]
    MAX_CONTENT_LENGTH = 6000
    SECTION_PREVIEW_LENGTH = 500
    MAX_OPPORTUNITIES = 15

    def __init__(self, output_dir: str = "./outputs"):
        """ReportAgent 초기화"""
        self.output_dir = output_dir
        self.llm = ChatOpenAI(
            temperature=0.7
        )  # 더 창의적인 내용을 위해 temperature 조정
        self.logger = None  # 워크플로우에서 주입
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def set_logger(self, logger: Logger) -> None:
        """워크플로우 로거 설정"""
        self.logger = logger

    def log(self, level: str, message: str) -> None:
        """통합 로깅 메서드"""
        if self.logger:
            getattr(self.logger, level)(message)

    def generate_report(self, analysis_results: List[Dict[str, Any]]) -> str:
        """분석 결과를 바탕으로 마크다운 형식의 보고서 생성"""
        try:
            self.log("info", "보고서 생성 시작")

            # 각 섹션 생성
            header = self._create_report_header()
            trend_sections = self._generate_trend_sections(analysis_results)
            footer = self._create_report_footer(analysis_results)

            # 각 섹션을 결합하여 전체 보고서 내용 생성
            full_content = header + trend_sections + footer

            # 보고서 요약 생성 후 삽입
            summary = self._create_summary(analysis_results, full_content)
            final_report = header + summary + trend_sections + footer

            self.log("success", "보고서 생성 완료")
            return final_report

        except Exception as e:
            self.log("error", f"보고서 생성 실패: {str(e)}")
            raise

    def _generate_trend_sections(self, analysis_results: List[Dict[str, Any]]) -> str:
        """모든 트렌드 섹션을 생성하는 메서드"""
        trend_sections = ""
        for result in analysis_results:
            # 관련 회사 정보 생성
            company_info = self._create_company_section(result)
            # 트렌드 섹션에 회사 정보 포함
            trend_sections += self._create_trend_section(result, company_info)
        return trend_sections

    def _create_report_header(self) -> str:
        """보고서 헤더 생성"""
        return f"""# 기술 트렌드 분석 보고서
생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 개요
본 보고서는 최신 기술 동향을 분석하고 향후 5년 이내의 주요 트렌드를 예측한 결과입니다.

"""

    def _create_summary(
        self, analysis_results: List[Dict[str, Any]], full_content: str
    ) -> str:
        """보고서 요약 섹션 생성"""
        self.log("info", "보고서 요약 생성 중")

        # 키워드 목록 추출
        keywords = [result["keyword"] for result in analysis_results]

        # 영향력이 높은 키워드 추출 (상위 3개)
        high_impact_keywords = self._get_high_impact_keywords(analysis_results)

        # 보고서 내용이 너무 길면 중요 부분만 추출
        truncated_content = self._truncate_long_content(full_content)

        # 고영향력 키워드 정보
        high_impact_info = self._format_high_impact_info(high_impact_keywords)

        # LLM 호출 및 응답 처리
        summary_prompt = ChatPromptTemplate.from_template(self.SUMMARY_PROMPT_TEMPLATE)
        chain = summary_prompt | self.llm | StrOutputParser()
        summary_content = chain.invoke(
            {
                "keywords": ", ".join(keywords),
                "high_impact_techs": "\n".join(high_impact_info),
                "report_content": truncated_content,
            }
        )

        return f"{summary_content}\n\n"

    def _get_high_impact_keywords(
        self, analysis_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """영향력이 높은 키워드 추출 (상위 3개)"""
        return sorted(analysis_results, key=lambda x: x["impact_score"], reverse=True)[
            :3
        ]

    def _truncate_long_content(self, content: str) -> str:
        """긴 보고서 내용을 요약하는 메서드"""
        if len(content) <= self.MAX_CONTENT_LENGTH:
            return content

        # 각 섹션의 첫 부분과 결론 부분만 포함
        sections = content.split("##")
        important_parts = [sections[0]]  # 헤더

        for section in sections[1:]:
            if section.strip():
                # 각 섹션의 처음 500자 정도만 포함
                important_parts.append(section[: self.SECTION_PREVIEW_LENGTH] + "...")

        return "##".join(important_parts)

    def _format_high_impact_info(
        self, high_impact_keywords: List[Dict[str, Any]]
    ) -> List[str]:
        """고영향력 키워드 정보 포맷팅"""
        high_impact_info = []
        for item in high_impact_keywords:
            keyword = item["keyword"]
            impact = item["impact_score"]
            summary = item["trend_summary"][:100]

            info = f"{keyword} (영향력: {impact}/10): {summary}..."
            high_impact_info.append(info)
        return high_impact_info

    def _create_company_section(self, result: Dict[str, Any]) -> str:
        """기술 분야 관련 주요 기업 및 유망 스타트업 정보 생성"""
        self.log("info", f"관련 기업 정보 생성 중 - {result['keyword']}")

        # LLM 프롬프트 작성
        company_prompt = ChatPromptTemplate.from_template(self.COMPANY_PROMPT_TEMPLATE)

        # LLM 호출 및 응답 처리
        chain = company_prompt | self.llm | StrOutputParser()
        company_content = chain.invoke(
            {
                "keyword": result["keyword"],
                "summary": result["trend_summary"],
                "applications": ", ".join(result["key_applications"]),
                "opportunities": ", ".join(result["opportunities"]),
            }
        )

        return company_content

    def _create_trend_section(
        self, result: Dict[str, Any], company_info: str = ""
    ) -> str:
        """LLM을 활용한 트렌드 섹션 생성"""
        # 기본 정보 구성
        date_info = self._format_date_info(result["date_range"])
        days = result["date_range"].get("days", 0)
        papers_per_day = self._calculate_papers_per_day(result["paper_count"], days)

        # 뉴스 섹션 정보 생성
        news_content = ""
        if "news_analysis" in result and result["news_analysis"]:
            news_content = self._create_news_section(result["news_analysis"])

        # LLM 프롬프트 작성
        trend_prompt = ChatPromptTemplate.from_template(self.TREND_PROMPT_TEMPLATE)

        # LLM 호출 및 응답 처리
        chain = trend_prompt | self.llm | StrOutputParser()
        llm_content = chain.invoke(
            {
                "keyword": result["keyword"],
                "impact_score": result["impact_score"],
                "trend_summary": result["trend_summary"],
                "applications": ", ".join(result["key_applications"]),
                "risks": ", ".join(result["risks"]),
                "opportunities": ", ".join(result["opportunities"]),
                "date_info": date_info,
                "paper_count": result["paper_count"],
                "papers_per_day": f"{papers_per_day:.1f}",
            }
        )

        # 기본 정보와 LLM 생성 내용 결합
        section = self._format_trend_section(
            result, date_info, papers_per_day, llm_content, company_info, news_content
        )
        return section

    def _create_news_section(self, news_analysis: Dict[str, Any]) -> str:
        """뉴스 분석 결과를 바탕으로 뉴스 섹션 생성"""
        if not news_analysis or news_analysis.get("article_count", 0) == 0:
            return ""

        # 뉴스 분석 결과로 LLM 프롬프트 작성
        news_prompt = ChatPromptTemplate.from_template(
            """
            다음 뉴스 분석 데이터를 바탕으로 최신 미디어 동향 섹션을 작성해 주세요:
            
            - 분석 기간: 최근 {timeframe}일
            - 관련 뉴스 기사 수: {article_count}개
            - 주요 뉴스 출처: {sources}
            - 주요 키워드: {topics}
            - 감성 분석: 긍정({positive}%), 중립({neutral}%), 부정({negative}%)
            - 최근 주요 이벤트: {events}
            
            다음 내용을 포함하여 한국어로 마크다운 형식으로 작성해 주세요:
            1. ### 최근 미디어 동향
            2. 미디어에서 다루는 주요 이슈와 트렌드 (2-3 문장)
            3. 뉴스 감성 분석 결과와 시사점 (1-2 문장)
            4. 주목할만한 최근 이벤트 (1-2 문장)
            
            객관적이고 통찰력 있는 내용으로 작성해 주세요.
            결론이나 의견은 데이터에 기반하여 제시해 주세요.
            """
        )

        # LLM 호출 및 응답 처리
        chain = news_prompt | self.llm | StrOutputParser()
        news_content = chain.invoke(
            {
                "timeframe": news_analysis.get("timeframe", 30),
                "article_count": news_analysis.get("article_count", 0),
                "sources": ", ".join(news_analysis.get("top_sources", [])),
                "topics": ", ".join(news_analysis.get("key_topics", [])),
                "positive": news_analysis.get("sentiment", {}).get("positive", 0),
                "neutral": news_analysis.get("sentiment", {}).get("neutral", 0),
                "negative": news_analysis.get("sentiment", {}).get("negative", 0),
                "events": news_analysis.get("recent_events", "정보 없음"),
            }
        )

        return news_content

    def _format_date_info(self, date_range: Dict[str, Any]) -> str:
        """날짜 정보 포맷팅"""
        days = date_range.get("days", 0)
        return f"{days}일 ({date_range['start']} ~ {date_range['end']})"

    def _calculate_papers_per_day(self, paper_count: int, days: int) -> float:
        """일평균 논문 수 계산"""
        return paper_count / days if days > 0 else 0

    def _format_trend_section(
        self,
        result: Dict[str, Any],
        date_info: str,
        papers_per_day: float,
        llm_content: str,
        company_info: str,
        news_content: str = "",
    ) -> str:
        """트렌드 섹션 포맷팅"""
        return f"""## {result['keyword']} 분석

### 데이터 수집 정보
- 수집 기간: {date_info}
- 총 논문 수: {result['paper_count']}개
- 일평균 논문 수: {papers_per_day:.1f}개

### 영향력 평가
- 영향력 점수: {result['impact_score']}/10

{llm_content}

{news_content}

{company_info}

"""

    def _create_report_footer(self, analysis_results: List[Dict[str, Any]]) -> str:
        """LLM을 활용한 결론 생성"""
        # 영향력 점수가 높은 순으로 정렬
        sorted_results = sorted(
            analysis_results, key=lambda x: x["impact_score"], reverse=True
        )

        # LLM에 전달할 데이터 구성
        keywords, summaries, opportunities = self._prepare_footer_data(sorted_results)

        # LLM 프롬프트 작성
        conclusion_prompt = ChatPromptTemplate.from_template(
            self.CONCLUSION_PROMPT_TEMPLATE
        )

        # LLM 호출 및 응답 처리
        chain = conclusion_prompt | self.llm | StrOutputParser()
        conclusion = chain.invoke(
            {
                "keywords": ", ".join(keywords),
                "summaries": "\n".join(summaries),
                "opportunities": "\n".join(
                    opportunities[: self.MAX_OPPORTUNITIES]
                ),  # 너무 많지 않게 제한
            }
        )

        return conclusion

    def _prepare_footer_data(
        self, sorted_results: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """결론을 위한 데이터 준비"""
        # 키워드와 영향력 점수 포맷팅
        keywords = []
        for r in sorted_results:
            keyword_info = f"{r['keyword']}(영향력: {r['impact_score']}/10)"
            keywords.append(keyword_info)

        # 각 키워드별 요약 포맷팅
        summaries = []
        for r in sorted_results:
            summary = f"{r['keyword']}: {r['trend_summary']}"
            summaries.append(summary)

        # 기회 요소 포맷팅
        opportunities = []
        for result in sorted_results:
            for opp in result["opportunities"]:
                opp_info = f"{result['keyword']}: {opp}"
                opportunities.append(opp_info)

        return keywords, summaries, opportunities

    def save_report(self, report: str, format: str = DEFAULT_FORMAT) -> str:
        """보고서를 파일로 저장"""
        try:
            self.log("info", "보고서 저장 시작")

            # 형식 검증
            if format not in self.SUPPORTED_FORMATS:
                format = self.DEFAULT_FORMAT
                self.log(
                    "warning",
                    f"지원하지 않는 형식입니다. 기본값({format})으로 저장합니다.",
                )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trend_report_{timestamp}.{format}"
            filepath = os.path.join(self.output_dir, filename)

            self._print_save_info(filepath, report)

            # 형식에 따라 저장 방법 선택
            if format == "pdf":
                pdf_filepath = self._save_as_pdf(report, timestamp)

                # PDF 생성 후 다른 형식의 중간 파일 삭제
                md_filename = f"trend_report_{timestamp}.md"
                md_filepath = os.path.join(self.output_dir, md_filename)
                if os.path.exists(md_filepath):
                    os.remove(md_filepath)
                    self.log("info", f"중간 마크다운 파일 삭제: {md_filepath}")

                self.log("success", f"PDF 보고서 저장 완료: {pdf_filepath}")
                return pdf_filepath
            else:
                self._write_report_to_file(filepath, report, format)
                self.log("success", f"보고서 저장 완료: {filepath}")
                return filepath

        except Exception as e:
            self.log("error", f"보고서 저장 실패: {str(e)}")
            raise ValueError(f"보고서 저장 중 오류 발생: {str(e)}")

    def _save_as_pdf(self, report: str, timestamp: str) -> str:
        """마크다운 보고서를 PDF로 변환하여 저장"""
        try:
            # 먼저 HTML로 변환
            html_content = markdown.markdown(report, extensions=["extra", "codehilite"])

            # CSS 스타일 추가
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>기술 트렌드 분석 보고서</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 40px;
                        color: #333;
                    }}
                    h1 {{
                        color: #E60012;
                        border-bottom: 2px solid #E60012;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #E60012;
                        margin-top: 30px;
                    }}
                    h3 {{
                        color: #E60012;
                        margin-top: 20px;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    code {{
                        background-color: #f5f5f5;
                        padding: 2px 5px;
                        border-radius: 3px;
                    }}
                    blockquote {{
                        border-left: 4px solid #E60012;
                        padding-left: 15px;
                        color: #666;
                    }}
                    .date {{
                        color: #666;
                        font-style: italic;
                    }}
                    .summary {{
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 5px solid #E60012;
                    }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            # HTML 임시 파일 저장
            html_filepath = os.path.join(
                self.output_dir, f"trend_report_{timestamp}.html"
            )
            with open(html_filepath, "w", encoding="utf-8") as f:
                f.write(styled_html)

            # HTML을 PDF로 변환
            pdf_filepath = os.path.join(
                self.output_dir, f"trend_report_{timestamp}.pdf"
            )
            html = weasyprint.HTML(filename=html_filepath)
            html.write_pdf(pdf_filepath)

            # HTML 임시 파일 삭제
            if os.path.exists(html_filepath):
                os.remove(html_filepath)
                self.log("info", f"HTML 임시 파일 삭제: {html_filepath}")

            return pdf_filepath

        except ImportError:
            self.log(
                "warning",
                "weasyprint 라이브러리가 설치되어 있지 않습니다. 마크다운 형식으로 저장합니다.",
            )
            filepath = os.path.join(self.output_dir, f"trend_report_{timestamp}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)
            return filepath

        except Exception as e:
            self.log("error", f"PDF 변환 중 오류 발생: {str(e)}")
            filepath = os.path.join(self.output_dir, f"trend_report_{timestamp}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)
            self.log(
                "warning", f"PDF 변환에 실패하여 마크다운으로 저장했습니다: {filepath}"
            )
            return filepath

    def _print_save_info(self, filepath: str, report: str) -> None:
        """저장 정보 출력"""
        print("\n[보고서] 저장 시작")
        print(f"- 저장 위치: {filepath}")
        print(f"- 보고서 크기: {len(report)} 문자")

    def _write_report_to_file(self, filepath: str, report: str, format: str) -> None:
        """파일에 보고서 작성"""
        # 디렉토리 존재 확인 및 생성
        os.makedirs(self.output_dir, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if format == "html":
                f.write(markdown.markdown(report))
            else:
                f.write(report)


if __name__ == "__main__":
    # 테스트 코드
    test_results = [
        {
            "keyword": "Large Language Models",
            "trend_summary": "LLM 기술의 급속한 발전...",
            "impact_score": 9,
            "key_applications": ["기업 자동화", "교육"],
            "risks": ["윤리적 문제", "데이터 보안"],
            "opportunities": ["새로운 비즈니스 모델", "생산성 향상"],
            "date_range": {"days": 365, "start": "2023-01-01", "end": "2023-12-31"},
            "paper_count": 1200,
        }
    ]

    agent = ReportAgent()
    report = agent.generate_report(test_results)
    filepath = agent.save_report(report)
    print(f"Report saved to: {filepath}")
