# 모드 차수 비교 분석 인수인계 — 먼저 읽기

작성일: 2026-09-13  
프로젝트 기준 경로: `ResonanceGUI/`  
목적: 다른 컴퓨터의 Codex가 이전 대화 기록 없이도 2026-09-11 모드 차수 비교와 2026-09-10 5차 추가 비교의 자료, 계산 범위, 결론을 이해하도록 한다.

## 가장 먼저 읽을 파일

1. [최종 분석 보고서](../20260913_모드차수_비교분석_20260911_vs_20260910_5thmode_220um.md)
2. [분석 자료 파일 지도](01_분석자료_파일지도.md)
3. [계산 재현 가이드](02_계산_재현가이드.md)

다른 Codex에는 다음처럼 요청하면 된다.

> `Particle_data/processed/mode_comparison_handoff_20260913/00_START_HERE_다른_Codex용.md`부터 읽고, 링크된 최종 보고서와 파일 지도를 확인한 뒤 작업을 이어가라. JSON 안의 `source` 절대경로가 현재 컴퓨터와 다르면 그 문자열을 그대로 믿지 말고 `Particle_data/raw/`에서 같은 basename의 파일을 찾아라.

## 한 문단 요약

분석한 집합은 2026-09-11의 2차 1세션, 3차 1세션, 4차 2세션, 5차 1세션과 2026-09-10의 `5thmode_220um` 1세션이다. 저장 후보 총 80개를 대상으로 원시 주파수 기록에서 국소 노이즈를 같은 조건으로 다시 계산했다. 주 조건은 이벤트 중심 전후 각각 0.5초의 구간, 중심 ±0.25초 제외, 각 측면 개별 1차 드리프트 제거, `ddof=0` 표준편차, 저장된 `abs(depth_hz)` 사용이다. 이 조건에서 9/11 5차는 평균 SNR 21.88, 최저 18.85, 10σ 통과 11/11로 가장 좋았다. 따라서 현재 자료에서는 5차를 우선 사용하고 9/11 5차의 실험 상태를 재현하는 것이 추천된다. 다만 모드 자체와 날짜·입자·유동·공진/제어 상태가 분리된 통제 실험이 아니므로 “5차는 항상 우수하다” 또는 “목표 BW를 3 Hz로 두면 개선된다”고 일반화하면 안 된다.

## 비교 대상 6개 세션

| 짧은 이름 | 원시 기록 | 후보 폴더 | 후보 수 | 공통 조건 평균 SNR |
| --- | --- | --- | ---: | ---: |
| 9/11 2차 | `Particle_data/raw/raw_20260911_2ndmode_P.csv` | `Particle_data/processed/20260911/particle_graphs_20260911_2ndmode/` | 16 | 8.24 |
| 9/11 3차 | `Particle_data/raw/raw_20260911_3rdmode_P.csv` | `Particle_data/processed/20260911/particle_graphs_20260911_3rdmode/` | 14 | 6.43 |
| 9/11 4차 _2 | `Particle_data/raw/raw_20260911_4thmode_2.csv` | `Particle_data/processed/20260911/particle_graphs_20260911_4thmode_2/` | 8 | 8.69 |
| 9/11 4차 _3 | `Particle_data/raw/raw_20260911_4thmode_3.csv` | `Particle_data/processed/20260911/particle_graphs_20260911_4thmode_3/` | 11 | 9.29 |
| 9/11 5차 | `Particle_data/raw/raw_20260911_5thmode.csv` | `Particle_data/processed/20260911/particle_graphs_20260911_5thmode/` | 11 | 21.88 |
| 9/10 5차 220um | `Particle_data/raw/raw_20260910_5thmode_220um.csv` | `Particle_data/processed/20260910/particle_graphs_20260910_5thmode_220um/` | 20 | 9.68 |

## 결론을 읽을 때 반드시 유지할 제한

- 후보 80개는 독립 입자 80개나 전체 통과 수가 아니다. `retained_review.csv`에 저장된 후보 집합이다.
- 후보 상태가 `미검토`인 행도 포함되어 있다. 전체 후보를 수동으로 진위 판정한 분석이 아니다.
- 후보 깊이는 새로 검출하지 않고 `retained_review.csv`의 `depth_hz`를 사용했다.
- `SNR >= 10`은 이 비교에서 정한 품질 지표이지, 검출률·질량 정확도·특정 오검출 확률을 뜻하지 않는다.
- 9/10의 `220um`은 폴더 식별자다. 메타데이터에는 130–155 µm, 후처리 물성 설정에는 190–220 µm가 있어 실제 크기가 확정되지 않는다.
- 일부 `retained_review.json`의 `particle_property_settings.mode_number`가 폴더상 모드와 다르다. 그래서 기존 질량·밀도·등가지름은 모드 추천 근거로 쓰지 않았다.
- `PLL_target_bandwidth_Hz`는 저장된 목표값이다. 실제 폐루프 BW 실측값으로 해석하지 않는다.

## 다른 컴퓨터로 옮길 때

최소 분석 묶음은 다음이다.

- 이 인수인계 폴더의 Markdown 3개
- 최종 분석 보고서 1개
- 위 표의 원시 CSV 6개
- 각 후보 폴더의 `retained_review.csv`, `retained_review.json`
- 각 후보 폴더의 `noise_snr/snr_summary.csv`, `noise_snr/snr_analysis_settings.json`, `noise_snr/snr_report.md`
- 재계산용 `particle_review_core.py`, `particle_review_noise.py`

시각 검증까지 하려면 각 후보 폴더의 번호가 붙은 PNG와 `noise_snr/snr_context_overview*.png`도 함께 옮긴다. Git으로 전달할 때는 주의해야 한다. 현재 `.gitignore`가 `.codex_tmp/`, `raw_*.csv`, `particle_graphs_*/` 등을 제외하므로, 단순 커밋/클론만으로 데이터와 임시 비교 스크립트가 다른 컴퓨터에 전달된다고 가정하면 안 된다.

