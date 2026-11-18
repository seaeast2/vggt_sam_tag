# VGGT-Ω 원본 깊이와 GT 전체 화면 비교

산출물: `results/planar_residual_depth/visualizations/raw-vggt-GT-v002/`.

- Middlebury Adirondack와 KITTI 2015 000000의 왼쪽 이미지, 실제 VGGT-Ω 추론 캐시를 사용했다. 픽셀별 깊이 보정·추론 재실행은 하지 않았다.
- 비교에 사용한 깊이는 원본 출력에 P1b에서 동결한 FIT RGB 삼각측량 기반 공통 scale을 곱한 것이다. GT 기반 scale/shift 적합은 하지 않았다. 모델 원래 단위 배열과 PNG도 별도 저장했다.
- 각 장면의 VGGT/GT는 같은 좌표·색 범위를 쓴다. 전체 화면 표시 범위는 유효 GT의 1–99 percentile이며 범위 밖 값은 끝 색으로 포화된다. 포화된 색만으로 평면성·깊이 변화 유무를 판단하지 않는다. 책 확대는 예측과 GT 양쪽을 포함한 공통 표시 범위를 사용한다.
- GT 결측은 회색이다. GT hole filling이나 smoothing을 하지 않았다. 오차는 유효 GT에서만 계산했다. 이 valid mask는 시각화용으로, 독립 heldout 평가 mask가 아니다.
- signed error = 예측 − GT: 빨강은 더 멀게, 파랑은 더 가깝게 추정한 것이다. 절대 오차(m) 범위는 장면별로 다르고 상대 오차(%)는 ±25%로 같다.
- ETH3D는 원본 왜곡 GT와 undistorted RGB의 격자 대응 검증이 남아 이 비교에서 제외했다.
- 입력 파일 hash·scale·표시 범위·GT 유효 픽셀 수는 산출물 `provenance.json`에 저장했다.
