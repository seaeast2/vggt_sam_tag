# P11 KITTI 1초 간격·프레임별 도로 마스크 재실험

1초 간격16장의 모델 추론과 사후 LiDAR 깊이 평가를 완료했다. 도로 영역의 시간 대응점 지원이 부족해 B2/B3/P8은 실행하지 않고 원본을 유지했다. 보정 성공으로 세지 않는다.

## 이전 실험과 바뀐 조건

- 동일 KITTI drive0005의 전체15초 구간에서0,1,…15s에 가장 가까운 실제 영상16장을 선택했다. source frame [0, 10, 19, 29, 39, 48, 58, 68, 77, 87, 97, 106, 116, 126, 136, 145].
- 실제 인접 간격 0.928–1.034초. 1Hz 영상이며 임의의 동일 이미지 복제나 중간 보간이 아니다.
- 384×192 입력16장을 VGGT-Ω에 공동 추론했다. peak allocated5071418368B, host GPU outside sandbox. GPU 추론 stdout는 /tmp/p11_gpu.log에서 별도 보존했다.
- 이전 화면 고정 polygon 대신 각 프레임 RGB에서 보이는 도로의 안전 영역을 개별 지정했다. 차량·자전거·보도·기둥을 제외하려고 했다. masks001의8–11번 주변 curb 혼입을 RGB 검사로 발견해 masks002로1회 수정했고 두 이력을 보존했다. 새 raw depth/GT를 보기 전 수정했다.
- per-frame polygon은 도로 후보 영역이며 동일 물리 patch를 보장하지 않는다. 시간에 따라 같은 도로가 보인다는 것과 같은 물리점이 남는다는 것을 구분한다.
- 실제 이미지 비교는 source ROI를 target에 투영한 footprint에서 수행하도록 시각화했다. target에 고정 box를 복사해서 이동이 없다고 가정하지 않는다. 예측 footprint 자체가 맞는지는 별도 검증이 필요하다.
- 시간 간격 외에 clip 시작/길이, 마스크, 대응점 방법도 바뀌므로 이전 P10과의 차이는 간격만의 통제 효과가 아니다. 거의 동일한 입력이 이전 성능 저하의 원인이라는 주장을 검증한 결과도 아니다.

## 대응점과 보정 실행 여부

1초 간격의 상호 SIFT ratio0.75 및 모델 pose의 epipolar1px 필터를 거친 track에서 양쪽 관측 모두 per-view road mask 안에 있는 것만 선택했다. 평면 track은 FIT2개/heldout0개다. inherited 기준 FIT12/heldout6, 최대 per-view hull10%, median parallax1°, rank를 유지했으며 모든 prefix2/4/8/16이 미지원이다. SIFT는 초기/긴 간격 대응점 진단으로 실행한 대안이며 P10의 LK와 같은 matching 조건이 아니다.

기존 temporal LK 방식도 별도로 확인했고 유지된 평면 track이0개였다. 이는 오직 도로 질감 부족의 증거는 아니다. 이동/FOV, 양방향 추적, ROI 겹침, 모델 pose에 의한 epipolar filtering이 함께 작용한다. 기준을 낮추거나 GT를 보고 mask/track을 다시 선택하지 않았다. 실제 다중시점 평면 보정의 개선/실패 깊이 성능을 평가할 만큼의 관측이 없었다.

## 깊이 평가

RGB 추정 K/pose를 사용했고 제공 pose/LiDAR는 보정 입력에 넣지 않았다. 평가 단계에서 첫 프레임 LiDAR GT291픽셀의16px 공간 block에서 scalar 하나를 계산했다. 나머지1166픽셀을 첫 프레임 test로 사용했다. 이 scalar를 전체16장과 모든 arm에 고정 적용했다. 절대 scale 회복 성능을 뜻하지 않는다.

| 첫 프레임 도로 ROI | MAE | 상태 |
|---|---:|---|
| B0 원본 | 8.37cm | 원본 |
| B1 원본 평면화 | 7.52cm | 초기 plane 투영 |
| B2/B3/P8 | 8.37cm | 미지원으로 원본 유지, 보정 미실행 |

원본 depth의 프레임별 오차는 다음과 같다. 첫 프레임은 calibration 제외, 이후 프레임은 ROI의 모든 유효 GT 픽셀로 평가했다. GT는 희소 Velodyne minimum-Z projection, no fill/no motion compensation이다. 원본 depth의 clip 전역 scalar 정렬 후 오차이며 시점별로 다시 scale을 맞추지 않았다.

| 프레임 | 시각(s) | GT 픽셀 | 원본 MAE(cm) |
|---|---:|---:|---:|
| 0 | 0.000 | 1166 | 8.37 |
| 1 | 1.033 | 1455 | 45.01 |
| 2 | 1.963 | 1505 | 7.51 |
| 3 | 2.995 | 1321 | 44.45 |
| 4 | 4.028 | 1184 | 53.54 |
| 5 | 4.957 | 756 | 58.48 |
| 6 | 5.989 | 1296 | 48.49 |
| 7 | 7.022 | 1919 | 60.01 |
| 8 | 7.954 | 1256 | 85.99 |
| 9 | 8.988 | 994 | 88.43 |
| 10 | 10.021 | 308 | 97.41 |
| 11 | 10.950 | 542 | 98.10 |
| 12 | 11.982 | 911 | 81.42 |
| 13 | 13.014 | 690 | 39.63 |
| 14 | 14.045 | 626 | 100.52 |
| 15 | 14.973 | 703 | 127.18 |

16개 프레임 MAE의 단순 평균은65.28cm이며 전체 유효 픽셀 가중 MAE는57.95cm다. ROI/거리/조명이 프레임별로 다르고 sensor projection도 근사다. source0 8.37cm만으로 clip 전체 정확도를 대표하지 않는다.

## 실제 이미지와 이동한 투영 영역

[프레임별 road mask 전체](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P11masks-kitti-1s-20261118-002/outdoor/mask_contact.png) · [RGB/raw/GT/오차 영상](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P11eval-kitti-1s-20261118-001/RGB_raw_GT_comparison.mp4)

[1초 간격2→3 실제 투영](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P11eval-kitti-1s-20261118-001/projection_02_to_03.png) · [6→7](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P11eval-kitti-1s-20261118-001/projection_06_to_07.png) · [13→14](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P11eval-kitti-1s-20261118-001/projection_13_to_14.png)

빨간 polygon은 각 프레임의 도로 후보이며, 청록 윤곽은 source mask의 예측 투영 footprint다. 2→3에서는6043픽셀 footprint 중3682개가 target road mask와 겹친다. 6→7/13→14에서는 초기 B1 평면·예측 pose로 계산한 footprint가 target 화면에 남지 않는다. 검은 투영은 렌더링을 숨긴 것이 아니라 이 조건의 유효 source ROI coverage가0이라는 뜻이다. B1 footprint는 GT나 확인된 대응점이 아니며, 후반의 raw plane 적용/pose/가시성 가정을 진단하는 시각화다.

## 실행 이력·감사

input001 → raw P10raw-three-sequences-20261118-001(실제는 KITTI1개만) → masks001/002 → fit001; 별도 LKobs001 → eval001. generic P10 infer의 상태 문자열 ALL_THREE_RAW_CACHES_COMPLETED는 고정 라벨이고 실제 manifest sequences는 outdoor1개다. raw 숫자나 checkpoint는 수정하지 않았다.

fit 이전 LiDAR 값은 읽지 않았다. 평가 전후 fit 전체 hash 불변,20개 arm/prefix의 residual 항등식/ROI 밖 불변/양의 유한 깊이,16프레임 영상 전체 decode를 확인했다. 보정 미지원도 보존했다. 모든 코드는 experiments/planar_residual_depth/P11 아래에 있다. 실험마다 새 ID를 쓰며 기존 P10은 덮어쓰지 않았다.
