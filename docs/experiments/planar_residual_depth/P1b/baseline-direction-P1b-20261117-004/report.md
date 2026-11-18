# P1b 승인 마스크의 원본 방향·평면화 진단

주 보정 B2/B3는 관측 지원 부족으로 미실행이다. GT를 평가에만 읽어 원본 방향·B1 평면화의 영향을 진단했다. scale은 장면 전체 FIT RGB track의 삼각측량 깊이/원본 깊이 median으로 고정했다. 다른 시점 평면 마스크를 확정한 주 실험이나 독립 보류 정확도 평가가 아니다.

| 후보 | raw 법선 오차(°) | GT plane RMSE(mm) | bootstrap 방향 p95(°) | B0 MAE(cm) | B1 MAE(cm) |
|---|---:|---:|---:|---:|---:|
| K01 | 0.4378 | 16.082 | 0.0212 | 63.719 | 62.741 |
| M01 | 7.3347 | 0.541 | 0.0683 | 28.024 | 28.008 |

Middlebury PFM은 공식 규약에 따라 scale 절댓값을 무시한다. 공식 SDK의 수정하지 않은 ReadFilePFM C++ 함수 출력과 전체 배열 동일성을 검사했다. OpenCV reader는 scale magnitude를 적용하므로 Middlebury 시차의 독립 기준으로 사용하지 않았다. 이전 -002의 Middlebury 수치는 잘못된 scale 적용으로 무효이며 보존만 한다. [공식 규약](https://vision.middlebury.edu/stereo/submit3/upload-format.html)

GT 평면은10회 robust weighted SVD로 적합했다. bootstrap50회는 지점 재표본의 조건부 안정성이며 센서·등록의 체계적 오차는 포함하지 않는다. Middlebury/KITTI는 원본 GT 격자에 역변환하여 최근접 유효값을 읽었다. metric depth는 fx*baseline/(disparity+cx_right-cx_left)이며 baseline은 m다. GT-camera 광선 왕복검사는 좌표 변환의 대수적 일치 검사이며 외부 센서 정합 검증은 아니다. ETH3D는 distortion/depth 격자 검증 미완료로 평가하지 않았다.

이 단계는 기존 실패 원인을 확정하지 않는다. GT 평면을 solver에 넣거나 raw 방향을 GT에 맞춰 수정하지 않았다. 같은 source-view 마스크의 유효GT 전체를 평가했으며 test holdout 주장 없이 진단으로 제한한다.
