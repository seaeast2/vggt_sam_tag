# 승인 마스크 동결·원본 RGB 해상도 매칭 진단

M01 v004는 사용자 승인 상태로 동결했다. K01/P01/P02의 기존 검토 기록을 유지한다. 원래 P1 지원 부족 판정은 변경하지 않았다. 별도 P1a에서 매칭 입력만 원본 RGB 긴 변 최대1024로 변경했다. VGGT 깊이·카메라·마스크·ratio·처리격자 기준 epipolar 허용값은 유지했다. GT와 raw depth는 읽지 않았다.

| 후보 | P1 FIT/보류 | P1a FIT/보류 | 면적 비율 | 시차(°) | 판정 |
|---|---|---|---:|---:|---|
| K01 | {'fit': 13, 'heldout': 2} | {'fit': 10, 'heldout': 3} | 0.229 | 3.608996676954649 | INSUFFICIENT_SUPPORT |
| P01 | {'fit': 0, 'heldout': 0} | {'fit': 0, 'heldout': 0} | 0.000 | None | INSUFFICIENT_SUPPORT |
| P02 | {'fit': 0, 'heldout': 0} | {'fit': 0, 'heldout': 0} | 0.000 | None | INSUFFICIENT_SUPPORT |
| M01 | {'fit': 5, 'heldout': 2} | {'fit': 4, 'heldout': 5} | 0.090 | 4.933883258088638 | INSUFFICIENT_SUPPORT |

고해상도에서 새로 추출한 track의 hash 기반 split이며 기존 track과 일대일 동일한 보류 집합은 아니다. 따라서 지원 수 진단이며 성능 개선 비교로 해석하지 않는다. 호모그래피 전이는 FIT track만 사용한 후보이며 다른 시점의 평면 마스크는 사용자 검토 전이다. 최소 지원 기준을 낮추지 않았다. P2 실행 조건이 충족되지 않은 영역은 보정을 수행하지 않는다.

산출물: `results/planar_residual_depth/P1a-approved-planes-20261117-001/`.
