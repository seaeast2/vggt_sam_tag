# P2 관측 기반 평면 깊이 보정

`solver.py`는 GT를 입력받지 않는 공통 평면 solver다. FIT 양방향 재투영 오차에 radial Huber(1px)를 적용한다. B2는 B1 plane의 배율만 바꿔 법선을 유지하고 B3는 세 계수를 갱신한다. 모든 source/target polygon 꼭짓점에서 양의 깊이를 만족하는 선형 제약을 적용한다. 초기화 raw와 FIT linear 중 성공한 낮은 FIT objective만 선택한다.

```bash
.venvs/p1/bin/python experiments/planar_residual_depth/P2/check_solver.py
.venvs/p1/bin/python experiments/planar_residual_depth/P2/fit.py
```

실데이터 실행에는 `results/planar_residual_depth/P1c-approved-planes-20261117-001/artifacts/K01/target1_approval.json`의 실제 사용자 검토 기록이 필요하다. 이 기록이 없으면 실행을 중단한다. 사용자 확인 없이 파일을 생성하거나 승인 상태를 추정하지 않는다.

원본 모델·mask·K/pose는 고정한다. 새 LK 보류 대응의 각 시점 3px 이내에 있는 기존 scale-FIT SIFT track을 제외한 뒤, FIT 삼각측량/원본 depth의 median scale 하나를 전 비교군에 공통 적용한다. 이전 P1b의 scale과 수치가 달라질 수 있으므로 각 run의 scale을 반드시 병기한다.

출력은 별도 P2 run이며 GT 평가는 완료된 후보를 대상으로 별도 실행해야 한다. 보류 대응은 보고에만 사용하고 solver·초기화·모델 선택에 사용하지 않는다. M01/P01/P02는 현재 관측 지원 기준 미달로 이 P2 실행 대상에 포함하지 않는다.
