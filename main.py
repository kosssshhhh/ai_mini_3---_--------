import argparse
import sys
from workflow import TrendAnalysisWorkflow


def parse_args():
    """명령행 인자 처리"""
    parser = argparse.ArgumentParser(description="기술 트렌드 분석기")
    parser.add_argument(
        "--keywords",
        "-k",
        nargs="+",
        required=True,
        help="분석할 키워드 (여러 개 입력 가능)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=3, help="병렬 처리 워커 수 (기본값: 3)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./outputs",
        help="출력 파일 디렉토리 (기본값: ./outputs)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["md", "html", "pdf"],
        default="md",
        help="출력 형식 (md, html, pdf 중 선택, 기본값: md)",
    )

    return parser.parse_args()


def run_workflow(args):
    """워크플로우 실행"""
    print(f"\n분석 시작: {', '.join(args.keywords)}")
    print(f"병렬 처리 워커 수: {args.workers}")

    workflow = TrendAnalysisWorkflow(max_workers=args.workers)

    try:
        # 단일 키워드 또는 첫 번째 키워드만 사용
        main_topic = args.keywords[0] if len(args.keywords) == 1 else args.keywords

        # 워크플로우 실행
        result = workflow.run(main_topic)

        # 출력 형식 설정
        output_format = args.format

        # 결과 처리
        print("\n분석 완료!")

        # 보고서 저장
        if result["report"]:
            # 이미 저장된 보고서 확인
            if result["report_path"]:
                # 원하는 형식이 아니면 다시 저장
                if args.format != "md":
                    try:
                        filepath = workflow.reporter.save_report(
                            result["report"], format=output_format
                        )
                        print(f"보고서 저장 위치: {filepath}")
                    except Exception as e:
                        print(f"{args.format} 형식 저장 실패: {str(e)}")
                        print(f"마크다운 보고서 위치: {result['report_path']}")
                else:
                    print(f"보고서 저장 위치: {result['report_path']}")
            else:
                # 직접 저장
                try:
                    filepath = workflow.reporter.save_report(
                        result["report"], format=output_format
                    )
                    print(f"보고서 저장 위치: {filepath}")
                except Exception as e:
                    print(f"보고서 저장 실패: {str(e)}")

        # 실패한 키워드 표시
        if result["failed_keywords"]:
            print("\n실패한 키워드:")
            for kw in result["failed_keywords"]:
                print(f"- {kw}")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return 1

    return 0


def main():
    """메인 함수"""
    args = parse_args()
    return run_workflow(args)


if __name__ == "__main__":
    sys.exit(main())
