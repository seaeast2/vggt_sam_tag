# P0 합성 평면 기하 검증

실행: 프로젝트 루트에서 아래 명령을 사용한다.

```bash
MPLCONFIGDIR=/tmp/p0-mpl .venvs/p0/bin/python experiments/planar_residual_depth/P0/run.py
```

필요 환경은 `requirements.lock.txt`에 고정했다. CPU 전용이며 GPU·VGGT 추론·실제 RGB는 사용하지 않는다. 출력은 새 run ID로 분리하며 기존 실행을 덮어쓰지 않는다. 3방향 × 7조건 × B0–B3 = 84개 지표 묶음을 생성한다. 정상 12조건에서 B3 복원 기준을 검사한다. 순수 회전·공선은 거절된 제안도 저장하며 오매칭은 진단 조건이다. GT는 합성 장면·관측 생성과 평가에만 사용하고 solver에는 FIT 관측·카메라·raw 초기값만 전달한다.

현재 canonical은 `P0-synthetic-planes-20261117-002`다. `-001` 결과도 유지한다. `-002`는 같은 실험에 해석/수치 Jacobian 검증을 추가한 실행이며 성능 설정은 변경하지 않았다. 실행 보고서는 `docs/experiments/planar_residual_depth/P0/`에 있다.
