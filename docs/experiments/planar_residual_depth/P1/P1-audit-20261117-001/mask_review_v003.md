# P1 마스크 사용자 검토 및 M01 수정

K01·P01·P02는 사용자가 대체로 정확하다고 확인했다. 정밀 경계 GT나 방향 GT로 해석하지 않는다. M01은 책 옆면을 포함했다는 사용자 지적에 따라 윗면 안쪽으로 좁혔다. 수정된 M01은 재검토 대기이며 원본 후보는 보존한다.

수정 좌표: [[86, 206], [127, 192], [165, 207], [101, 223]]

- K01: FIT/보류 {'fit': 13, 'heldout': 2} → {'fit': 13, 'heldout': 2}; INSUFFICIENT_SUPPORT
- P01: FIT/보류 {'fit': 0, 'heldout': 0} → {'fit': 0, 'heldout': 0}; INSUFFICIENT_SUPPORT
- P02: FIT/보류 {'fit': 0, 'heldout': 0} → {'fit': 0, 'heldout': 0}; INSUFFICIENT_SUPPORT
- M01: FIT/보류 {'fit': 6, 'heldout': 2} → {'fit': 5, 'heldout': 2}; INSUFFICIENT_SUPPORT

수정은 RGB만 사용했다. raw·GT·solver 결과는 선택에 사용하지 않았다. 다른 시점 마스크와 평면별 관측 지원·scale·GT 격자 검증은 여전히 미완료다. 깊이 추론과 보정을 재실행하지 않았다.

산출물: `results/planar_residual_depth/review/P1-candidates-v003/`.
