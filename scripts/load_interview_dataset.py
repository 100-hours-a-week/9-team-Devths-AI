#!/usr/bin/env python3
"""
면접 데이터셋 로드 및 탐색 스크립트
Dataset: UICHEOL-HWANG/InterView_Datasets
"""

from datasets import load_dataset


def main():
    print("📥 면접 데이터셋 로드 중...")
    ds = load_dataset("UICHEOL-HWANG/InterView_Datasets")

    print("\n📊 데이터셋 구조:")
    print(ds)

    # 각 split 확인
    for split_name in ds:
        print(f"\n{'='*80}")
        print(f"Split: {split_name}")
        print(f"{'='*80}")
        split_data = ds[split_name]
        print(f"행 개수: {len(split_data)}")
        print("\n칼럼:")
        print(split_data.column_names)

        # 첫 번째 샘플 출력
        if len(split_data) > 0:
            print("\n첫 번째 샘플:")
            first_sample = split_data[0]
            for key, value in first_sample.items():
                print(
                    f"  {key}: {value[:200] if isinstance(value, str) and len(value) > 200 else value}"
                )

    # 데이터 저장
    print("\n💾 데이터셋을 JSON 파일로 저장 중...")
    for split_name in ds:
        output_file = f"data/interview_dataset_{split_name}.json"
        ds[split_name].to_json(output_file, orient="records", force_ascii=False)
        print(f"✅ {split_name} 저장 완료: {output_file}")

    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
