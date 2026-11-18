# P5a–P8: 거리·신뢰도와 보수적 평면 방향 보정

**P5a/P5b/P5c 진단과 P6/P7/P8 보정을21개 기존 사례에서 완료했다. 이전8개 미지원 사례는 그대로 보존했다. 새 독립 장면 검증이 아닌, P4 결과를 이미 본 뒤 규칙을 사전에 고정한 후속 탐색 실험이다.**

## 사전 고정 조건

사용자가 P8까지 연속 실행을 요청했다. 새 GPU 추론·학습·마스크 변경 없이 기존 제공 K/pose·동일 영역·FIT/보류·RGB 기반 공통scale을 유지했다. VGGT-Ω 원본 depth와 depth_conf를 보존했다. confidence는 픽셀 깊이 점수이며 카메라 자세의 신뢰도나 보정된 확률이 아니다. 거리 구분은 GT가 아닌 scaled raw camera-Z의 순위 삼등분이다.

P6–P8의 모든 규칙은 P5의 GT 진단 전에 `protocol.json`으로 동결했다. P6는 near / high-confidence / 교집합의 세 고정 군이며 주 비교는 near다. P7는 같은 선택 규칙의 두 시점 법선을 raw bootstrap 안정성·공간분포·상대 confidence로 가중 통합하고 거리만 보정한다. 주 비교는 P7_NEAR 대 P6_NEAR다. P8는 전체 raw 평면 방향에서 최대1°만 움직이게 하고 FIT 재투영 손실을 최소화한다. 기존 B2/B3는 고정/자유 대조군이다. GT로 군이나 각도 한도를 선택하지 않았다.

## P5a: 가까운 점의 방향이 더 좋은가?

| 데이터셋 | 가까운 점 법선오차 평균(°) | 먼 점 | 가까운 쪽이 나은 사례 | 점수/공간 통제 |
|---|---:|---:|---:|---|
| kitti | 1.079 | 0.701 | 2/14 | 동일 개수 재표집20회; 아래 공간 한계 참조 |
| middlebury | 4.667 | 3.788 | 2/7 | 동일 개수 재표집20회; 아래 공간 한계 참조 |

이번 고정 ROI에서 raw 추정 가까운 구간만 쓰는 방향은 전체 점보다 일관되게 낫지 않았다. 그러나 near/mid/far는 영상의 서로 다른 위치에 놓이므로 거리 효과를 인과적으로 분리하지 못한다. 4×4 공통 cell에서 세 군의 개수와 분포를 동시에 맞추는 통제는 KITTI0/14, Middlebury0/7로 불가능했다. 이를 `NOT_IDENTIFIABLE_SPATIAL_CONFOUNDING`으로 남겼으며 “근거리일수록 본질적으로 덜 정확하다”로 일반화하지 않는다. GT도 local subset으로 평면을 따로 적합한 참고 각도를 기록했다.

## P5b: 높은 depth_conf의 방향이 더 좋은가?

| 데이터셋 | 높은 confidence 법선오차(°) | 낮은 confidence | 높은 쪽이 나은 사례 |
|---|---:|---:|---:|
| kitti | 0.976 | 0.687 | 3/14 |
| middlebury | 4.099 | 4.212 | 4/7 |

점 개수만 맞춘 비교와 공간 cell별 개수까지 맞춘 비교를 분리했다. 높은 깊이 confidence가 올바른 평면 방향의 신뢰도를 의미한다고 볼 수 없다. 이 점수가 모든 깊이/매칭 과제에서 무용하다는 결론도 아니다.

## P5c 및 공간 통제 상세

| 데이터셋 | 비교군 | 통제 | 지원 사례 | 군별 평균 법선오차(°) |
|---|---|---|---:|---|
| kitti | NEAR / MID / FAR | COUNT_MATCHED | 14/14 | NEAR=1.080, MID=0.689, FAR=0.701 |
| kitti | NEAR / MID / FAR | SPATIAL_CELL_MATCHED | 0/14 | 미분리 |
| kitti | LOW_CONF / MID_CONF / HIGH_CONF | COUNT_MATCHED | 14/14 | LOW_CONF=0.687, MID_CONF=0.698, HIGH_CONF=0.975 |
| kitti | LOW_CONF / MID_CONF / HIGH_CONF | SPATIAL_CELL_MATCHED | 13/14 | LOW_CONF=0.692, MID_CONF=0.726, HIGH_CONF=0.754 |
| kitti | ALL / NEAR / HIGH_CONF / NEAR_HIGH | COUNT_MATCHED | 14/14 | ALL=0.700, NEAR=1.078, HIGH_CONF=0.974, NEAR_HIGH=1.124 |
| kitti | ALL / NEAR / HIGH_CONF / NEAR_HIGH | SPATIAL_CELL_MATCHED | 14/14 | ALL=0.926, NEAR=1.036, HIGH_CONF=0.984, NEAR_HIGH=1.060 |
| middlebury | NEAR / MID / FAR | COUNT_MATCHED | 7/7 | NEAR=4.674, MID=4.073, FAR=3.789 |
| middlebury | NEAR / MID / FAR | SPATIAL_CELL_MATCHED | 0/7 | 미분리 |
| middlebury | LOW_CONF / MID_CONF / HIGH_CONF | COUNT_MATCHED | 7/7 | LOW_CONF=4.217, MID_CONF=4.311, HIGH_CONF=4.090 |
| middlebury | LOW_CONF / MID_CONF / HIGH_CONF | SPATIAL_CELL_MATCHED | 5/7 | LOW_CONF=4.426, MID_CONF=4.216, HIGH_CONF=4.405 |
| middlebury | ALL / NEAR / HIGH_CONF / NEAR_HIGH | COUNT_MATCHED | 7/7 | ALL=4.185, NEAR=4.682, HIGH_CONF=4.107, NEAR_HIGH=4.663 |
| middlebury | ALL / NEAR / HIGH_CONF / NEAR_HIGH | SPATIAL_CELL_MATCHED | 7/7 | ALL=4.301, NEAR=4.271, HIGH_CONF=4.274, NEAR_HIGH=4.311 |

각 표의 지원 사례 집합과 공통 cell 영역은 다를 수 있으므로 표 사이 숫자를 그대로 상대 성능으로 비교하지 않는다. count control은 각 군 최대1024점·20회, spatial control은 공통 cell마다 동일 개수·20회다. 표는 raw normal 지원 기준을 모든 군에서 통과한 paired 반복에 대한 장면별 평균의 평균이다. 같은 cell 안의 정확한 위치까지 같지는 않아 잔여 공간 혼재가 있다. 교집합 선택도 일관된 방향 개선을 보장하지 않았다.

## P6–P8 깊이 보정 결과

| 방법 | KITTI 평균 MAE(cm),14사례 | Middlebury 평균 MAE(cm),7사례 |
|---|---:|---:|
| B0 | 43.343 | 7.811 |
| B1 | 43.275 | 7.814 |
| B2 | 10.834 | 0.884 |
| B3 | 10.888 | 0.331 |
| P6_NEAR | 16.469 | 0.862 |
| P6_CONF | 14.917 | 0.828 |
| P6_NEAR_CONF | 16.877 | 0.864 |
| P7_NEAR | 16.235 | 0.828 |
| P7_NEAR_CONF | 16.442 | 0.836 |
| P8_LIMITED_RAW | 9.668 | 0.625 |

B0는 원본×고정 공통scale, B1은 기존 raw 평면 투영이다. P6/P7은 고른/통합한 방향을 고정하고 기존 FIT 대응으로 거리만 보정한다. 각 arm의 원본/비대상 픽셀·카메라·split은 동일하다.

### P6 먼 영역 전파

| 데이터셋 | B2 먼 구간 MAE(cm) | P6_NEAR | P6_CONF | P6_NEAR_CONF |
|---|---:|---:|---:|---:|
| kitti | 16.290 | 26.565 | 23.385 | 27.169 |
| middlebury | 0.865 | 0.904 | 0.834 | 0.907 |

가까운 점을 선택하면 먼 구간까지 더 정확해진다는 가설은 현재 규칙으로 일관되게 지지되지 않았다. 원인을 거리 자체로 단정하지 않으며, 좁아진 지지 영역·raw 편향·법선 추정 민감도가 함께 남는다.

### P7 두 시점 법선 통합

현재 지원 사례는 stereo 두 장이다. 5-view ETH3D는 이전부터 지원 부족이므로 이번에도 미검증이다. 각 시점의 raw 부분 평면을 제공 자세로 같은 좌표계에 표현하고, 16×16 공간 block bootstrap30회의 법선 변동·영상 hull·ROI 내 confidence percentile로 정한 가중치로 축 평균했다. 이는 raw 추정의 안정성 가중이며 실제 정확도의 보정된 확률이 아니다. P7는 KITTI에서 P6보다 평균 오차를 소폭 줄였지만 기존 B2를 넘지 못했다. Middlebury에서도 B3 수준에 도달하지 못했다. 카메라 자세 confidence를 사용하거나 최적화한 실험은 아니다.

### P8 방향 변화1° 제한

양의 깊이 선형 제약에 n0·p − cos(1°)||p|| ≥ 0의 해석미분 제약을 추가했다. B2/raw B1과 범위 내에 있을 때만 B3를 초기값으로 사용하고 FIT 손실만으로 선택했다. 이는 항상 방향을 고정하거나 완전히 자유롭게 두는 것 사이의 고정된 보수적 비교다. KITTI에서는 평균 개선, Middlebury에서는 자유 B3보다 큰 오차가 나타났다. 1°를 최적 한도로 주장하지 않으며 GT로 조절하지 않았다.

## Paired 변화와 불확실성

| 데이터셋 | 비교 | 평균 MAE 차이(cm) | ID bootstrap 명목95% 구간 | 개선 사례 |
|---|---|---:|---|---:|
| kitti | P6_NEAR_vs_B2 | +5.635 | [+3.502, +7.736] | 1/14 |
| kitti | P6_CONF_vs_B2 | +4.083 | [+2.156, +6.015] | 1/14 |
| kitti | P6_NEAR_CONF_vs_B2 | +6.043 | [+3.856, +8.246] | 1/14 |
| kitti | P7_NEAR_vs_P6_NEAR | -0.234 | [-0.607, +0.099] | 8/14 |
| kitti | P7_NEAR_CONF_vs_P6_NEAR_CONF | -0.435 | [-0.968, +0.052] | 7/14 |
| kitti | P8_LIMITED_RAW_vs_B2 | -1.165 | [-2.756, +0.684] | 11/14 |
| kitti | P8_LIMITED_RAW_vs_B3 | -1.220 | [-2.792, -0.025] | 7/14 |
| middlebury | P6_NEAR_vs_B2 | -0.021 | [-0.329, +0.177] | 2/7 |
| middlebury | P6_CONF_vs_B2 | -0.055 | [-0.274, +0.098] | 4/7 |
| middlebury | P6_NEAR_CONF_vs_B2 | -0.019 | [-0.333, +0.187] | 1/7 |
| middlebury | P7_NEAR_vs_P6_NEAR | -0.034 | [-0.082, +0.009] | 5/7 |
| middlebury | P7_NEAR_CONF_vs_P6_NEAR_CONF | -0.028 | [-0.075, +0.013] | 4/7 |
| middlebury | P8_LIMITED_RAW_vs_B2 | -0.259 | [-0.454, -0.109] | 7/7 |
| middlebury | P8_LIMITED_RAW_vs_B3 | +0.294 | [+0.030, +0.651] | 1/7 |

음수는 앞의 방법이 더 정확하다는 뜻이다. 기존 장면을 이미 분석했고, KITTI 연속ID·Middlebury 유사 장면 사이의 독립성이 보장되지 않아 이 구간을 독립 일반화 신뢰구간이나 유의성 입증으로 해석하지 않는다. 후속 규칙은 이미 관찰한 P4에서 아이디어를 얻었으므로 모든 결과가 탐색적이다.

## 정확도 악화와 기술 검증

- kitti P6_NEAR가 B2보다 악화: 000005, 000006, 000007, 000008, 000010, 000011, 000012, 000013, 000014, 000016, 000017, 000018, 000019.
- kitti P7_NEAR가 B2보다 악화: 000005, 000006, 000007, 000008, 000010, 000011, 000012, 000013, 000014, 000016, 000017, 000018, 000019.
- kitti P8_LIMITED_RAW가 B2보다 악화: 000006, 000007, 000014.
- middlebury P6_NEAR가 B2보다 악화: Piano, Playroom, Playtable, PlaytableP, Teddy.
- middlebury P7_NEAR가 B2보다 악화: Motorcycle, Piano, Playroom, PlaytableP, Teddy.
- middlebury P8_LIMITED_RAW가 B2보다 악화: 없음.

21개 모두6개 새 보정군 계산 완료. 각 군의 상태와 fallback은 개별 metric/solver log에 보존했다. P8 법선 변화 한도·양의깊이·동일 공유평면 교차·잔차복원·비대상 불변·raw/conf 캐시 일치·평가 전후 fit hash 불변을 검증했다. 별도 합성 실험에서 cone 제약 gradient와 내부 정답 복원, 바깥 정답에 대한1° 경계 수렴을 확인했다.

## 실험 이력·산출물

- canonical fit: `P5-P8fit-all-cases-20261117-002`. 초기 `P5-P8fit-all-cases-20261117-001`은 경로를 얻기 위해 GT 지표가 포함된 P4 JSON container를 읽는 행정 I/O가 있었다. 알고리즘은 경로 필드만 사용했고 GT를 최적화하지 않았지만, GT 미로딩 경계를 엄격히 유지하려고 보존 후 canonical 실행에서는 pre-GT P3 manifest에서만 경로를 구성해 재실행했다.
- `protocol.json`/config와 source snapshot: P5 GT 진단 전에 P6–P8 규칙 고정. 기존 결과와 별도 폴더, 원본 데이터/실행은 변경하지 않았다.
- 각 사례: `P5_diagnostics.json`, `P5_controls.json`, `metrics.json`, `P5_groups.png`, `P6_P8_depth_GT.png`, `ROI_pointmaps.npz`.
- fit 각 사례: `inputs.npz`(원본·depth_conf·기존scale·baseline·mask), 군별 잔차/보정depth·p, raw normal·선택 pixel IDs·가중치·solver log.
- `aggregate.json`, `per_case_metrics.csv`, 데이터셋별 요약 그림. RGB/GT/오차는 동일 ROI·공통 표시 범위를 사용하고 GT 결측은 회색이다.

## 결론

현재 데이터에서 “가깝고 신뢰도가 높으면 더 정확한 평면 방향”이라는 선택 규칙은 확인되지 않았다. 특히 거리와 영상 내 분포를 동시에 맞춘 비교가 불가능하므로 거리 자체의 효과는 `INCONCLUSIVE`다. 높은 depth_conf만으로 방향의 정확도를 보장할 수도 없다. 전체 점으로 얻은 방향을 중심으로 작은 범위의 관측 보정을 허용하는 P8은 KITTI에서 유망한 탐색 결과를 보였지만, 실내에서는 큰 방향 수정이 필요한 사례를 막는다. 독립 장면에서의 평가·카메라 신뢰도·다중시점 ETH3D·장기 SLAM·학습 모델은 이번 단계의 검증 범위 밖이다.

## 감사 완료 보충

`audit.json`에서21개 사례·126개 새 보정군의 원본/신뢰도 캐시, rank thirds 선택ID, GT MAE 재계산, 공유평면 깊이·원본·비대상 불변·부모 fit hash 불변을 확인했다. P8의 최대 방향 변화는1.0000000001733942°로 수치 허용오차 이내다. `selection_ranges.json`에 각 군의 추정 거리 범위·confidence 범위·고유 점수 수를 남겨 rank 분할의 동점 여부를 추적할 수 있다. GT 참조 방향은 기존 P4의 공간 bootstrap 분석과 함께 해석해야 하며, 새 모든 군의 GT 참조 불확실성 전파를 새로 계산한 것은 아니다.

KITTI P8−B2 평균 MAE 변화는−1.165cm지만 ID bootstrap 명목95% 구간 [−2.756,+0.684]cm가0을 포함한다. 따라서 평균 개선이 곧 일반적인 우월성 입증은 아니다. Middlebury의 P8은 B2보다 평균 개선됐지만 B3보다 평균0.294cm 더 크다.

행정 경로 로딩 경계를 엄격히 한 canonical 재실행은 초기 실행의126개 보정군과 p·잔차·보정 깊이가 bitwise 동일했다(`routing_rerun_equivalence.json`). 성능에 따라 재실행/해를 고른 것이 아니다.
