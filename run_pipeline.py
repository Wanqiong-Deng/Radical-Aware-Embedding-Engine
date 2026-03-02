"""
run_pipeline.py — 一键运行完整分析管道

用法：
  python run_pipeline.py              # 全量运行（有 embedding 缓存则跳过 encode）
  python run_pipeline.py --force      # 强制重新 encode（数据有大改动时用）
  python run_pipeline.py --step embed # 只跑某步（preprocess / embed / analyze / visualize）
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEPS = ["preprocess", "embed", "analyze", "visualize"]


def main():
    parser = argparse.ArgumentParser(description="部首语义指南针 · 分析管道")
    parser.add_argument("--step",  choices=STEPS, default=None,
                        help="只执行某一步，不填则全量运行")
    parser.add_argument("--force", action="store_true",
                        help="强制重新 encode embeddings")
    args = parser.parse_args()

    steps_to_run = [args.step] if args.step else STEPS

    for step in steps_to_run:
        print(f"\n{'='*50}")
        print(f"  步骤: {step.upper()}")
        print(f"{'='*50}")

        if step == "preprocess":
            from src.preprocess import run
            run()

        elif step == "embed":
            from src.embed import run
            run(force_encode=args.force)

        elif step == "analyze":
            from src.analyze import run
            run()

        elif step == "visualize":
            from src.visualize import run
            run()

    print("\n\n🎉 管道运行完成！")
    print(f"   输出目录: {os.path.join(os.path.dirname(__file__), 'data', 'processed')}")


if __name__ == "__main__":
    main()