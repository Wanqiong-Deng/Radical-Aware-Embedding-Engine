"""
run_pipeline.py — 一键运行完整分析管道

用法：
  python run_pipeline.py                        # 全量运行
  python run_pipeline.py --step embed           # 只跑某步
  python run_pipeline.py --step embed --force   # 强制重新 encode
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEPS = ["preprocess", "embed", "analyze", "visualize", "radical_vectors", "index"]

STEP_DESC = {
    "preprocess":      "合并 3 个 Excel → cleaned.csv",
    "embed":           "BGE encode → embed_index.npz + coords.csv",
    "analyze":         "偏移向量 + Rayleigh 检验 → metrics.csv + report",
    "visualize":       "生成所有图表",
    "radical_vectors": "768维偏移向量 → radical_shift_vectors.npz",
    "index":           "声旁倒排索引 → phonetic_index.json",
}


def main():
    parser = argparse.ArgumentParser(description="GlyphDrift · 分析管道")
    parser.add_argument("--step", choices=STEPS, default=None,
                        help="只执行某步，不填则全量运行")
    parser.add_argument("--force", action="store_true",
                        help="强制重新 encode（embed 步骤用）")
    args = parser.parse_args()

    steps_to_run = [args.step] if args.step else STEPS

    print("🧭 GlyphDrift Pipeline")
    print("=" * 55)

    for step in steps_to_run:
        print(f"\n▶  {step.upper()}: {STEP_DESC.get(step, '')}")
        print("-" * 40)

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

        elif step == "radical_vectors":
            from src.radical_vectors import run
            run()

        elif step == "index":
            from src.phonetic_index import run
            run()

    print("\n" + "=" * 55)
    print("🎉 Pipeline 完成！")
    print(f"   输出目录: {os.path.join(os.path.dirname(__file__), 'data', 'processed')}")
    print("\n下一步：")
    print("  streamlit run app.py          # 启动 Demo")
    print("  python -m src.predict 童 言部  # CLI 预测")
    print("  python -m src.llm_generate    # 测试 LLM API")


if __name__ == "__main__":
    main()
