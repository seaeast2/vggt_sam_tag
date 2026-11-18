# P1c 코너 검출·LK 추적 관측 지원 진단

학습 모델·GT·raw depth를 사용하지 않고 승인한 source polygon 내부의 Shi–Tomasi 코너를 원본 RGB(긴변 최대1024)에서 추적했다. source→target→source 왕복오차0.5px, 양방향 epipolar1px(기존512처리격자), 영상 내 유효성 및 target2px cell 중복을 검사했다. 설정은 실행 전에 config로 저장했다.

| 후보 | FIT | 보류 | 면적 비율 | 시차(°) | 판정 |
|---|---:|---:|---:|---:|---|
| K01 | 47 | 21 | 0.259 | 3.56205864384887 | OBSERVATION_GATE_PASSED_TARGET_MASK_REVIEW_PENDING |
| P01 | 0 | 0 | 0.000 | None | INSUFFICIENT_SUPPORT |
| P02 | 0 | 0 | 0.000 | None | INSUFFICIENT_SUPPORT |
| M01 | 15 | 2 | 0.359 | 4.939373178331488 | INSUFFICIENT_SUPPORT |

FIT/보류는 source 좌표를 기반으로 한 track ID hash/seed0으로 분리했다. 새로운 특징 검출기로 얻은 보류 집합이므로 SIFT 대비 정확도 개선을 직접 주장하지 않는다. 가까운 코너의 패치는 상관될 수 있어 대응 개수를 독립 물리 표본 수로 해석하지 않는다. LK 패치는 평면 경계를 넘을 수 있으며 target mask 검토와 추가 퇴화 검사가 남아 있다. source mask만 사용자 승인 상태이고 target polygon은 FIT 호모그래피로 생성한 검토 후보다. 관측 지원 gate 통과가 실제 깊이 정확도나 동일 표면 소속의 검증은 아니다.

산출물: `results/planar_residual_depth/P1c-approved-planes-20261117-001/`.
