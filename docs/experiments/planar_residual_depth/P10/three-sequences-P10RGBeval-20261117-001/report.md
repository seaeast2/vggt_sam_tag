# P10 최종 결과: 실내·실외·드론 연속 이미지 실험

세 시퀀스의 VGGT-Ω 추론, 연속 관측 기반 평면 보정, 센서 기준 평가, 비교 영상 생성을 완료했다. 이번 조건에서는 RGB 추정 자세로 평면 보정을 수행해도 깊이 정확도가 일관되게 좋아지지 않았다.

## 데이터와 실행 범위

- 실내: TUM RGB-D fr3_structure_texture_far.
- 실외: KITTI Raw 2011_09_26_drive_0005_sync, image02 frame50–88 step2.
- 드론: EuRoC MAV V1_01_easy cam0. 실내 비행이며 흑백 영상을 3채널로 입력했다.
- 각 약4초, 5Hz20장 중16장 mapping/4장 future. 제공 자세 통제에서는 future 평가를 했고, RGB 추정 자세 조건은 mapping16장만 평가했다.
- 모델 입력16장은 공동 추론했다. 보정 관측을2/4/8/16장으로 늘리는 실험과, 모델 자체 입력을2/4/8/16장으로 바꾸는 실험을 분리했다. 실시간 인과적 처리나 장시간 SLAM/loop closure 검증은 아니다.
- 원본 raw depth는 보존하고 수정은 별도 residual로 저장했다. GPU는 샌드박스 밖에서 실행했다.

## RGB 추정 카메라를 사용한 결과

보정에는 VGGT가 예측한 K/pose를 사용했다. 깊이 GT와 제공 궤적은 보정에 넣지 않았다. 다만 제공 보정값으로 사전 왜곡 제거한 데이터이며, 카메라 보정까지 필요 없는 시스템을 검증한 것은 아니다.

단안 scale 모호성을 분리하기 위해 source0 GT의16px 공간 블록 약20%에서 평가용 scalar 하나를 계산했다. 나머지 GT 픽셀만 점수를 냈다. 동일 clip의 모든 보정 arm/prefix에 같은 scalar를 적용했고 fit 파일을 변경하지 않았다. **GT 없이 절대 미터 scale을 추정했다는 결과가 아니다.** 아래는 source0 평면 ROI의 N16 MAE(cm)다.

| 시퀀스 | B0 원본 | B1 평면화 | B2 거리만 | B3 방향·거리 | P8 방향1° 제한 |
|---|---:|---:|---:|---:|---:|
| indoor | 0.762 | 0.727 | 0.748 | 3.053 | 1.043 |
| outdoor | 12.603 | 12.101 | 274.427 | 349.947 | 332.955 |
| drone | 1.340 | 1.423 | 6.208 | 2.620 | 5.183 |

실내는 B2가 원본과 거의 같았고 B3/P8은 악화됐다. 실외는 모든 관측 기반 보정이 크게 악화됐다. 드론은 B3의 법선 오차가4.27°→2.61°로 줄었지만 깊이 MAE는1.34→2.62cm로 늘었다. 방향 개선과 깊이 정확도 개선은 같은 결과가 아니다. 원본 scalar를 GT로 맞춘 조건이므로, 이 표는 metric scale 회복 능력보다 동일 gauge에서의 국소 깊이 형태와 보정 편향을 평가한다.

## 모델 입력 장수만 바꾼 원본 깊이

각 입력 장수는 별도 RGB 추론이다. 각 출력의 단안 gauge는 동일 GT calibration 픽셀로 별도 정렬한 뒤 같은 test 픽셀에서 평가했다. 보정은 적용하지 않았다.

| 시퀀스 | RGB2 | RGB4 | RGB8 | RGB16 |
|---|---:|---:|---:|---:|
| indoor | 1.159 | 0.921 | 0.941 | 0.762 |
| outdoor | 12.799 | 15.766 | 12.155 | 12.603 |
| drone | 1.730 | 1.119 | 1.634 | 1.340 |

실내는2장보다16장이 좋았지만, 세 시퀀스 모두 장수가 늘 때마다 단조롭게 좋아지지는 않았다. 시퀀스당 한 구간이므로 입력 수 효과의 일반화나 유의성을 주장하지 않는다.

## 실제 영상

아래 RGB 자세 조건 영상은16장 입력 구간 전체를 보여준다. 각 영상은 RGB/원본 depth/보정 depth/센서 참조/보정 오차 순서다. 색 범위는 clip 내 고정이며 회색은 ROI 밖 또는 참조 결측이다. 센서 기준과 scale은 평가·표시 전용이다.

| 구간 | 원본 영상20장 | B3 비교16장 | P8 비교16장 |
|---|---|---|---|
| indoor | [원본](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10input-three-sequences-20261117-001/indoor/source_clip.mp4) | [B3](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10RGBeval-three-sequences-20261117-001/indoor/B3_RGB_pose_comparison.mp4) | [P8](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10RGBeval-three-sequences-20261117-001/indoor/P8_RGB_pose_comparison.mp4) |
| outdoor | [원본](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10input-three-sequences-20261117-001/outdoor/source_clip.mp4) | [B3](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10RGBeval-three-sequences-20261117-001/outdoor/B3_RGB_pose_comparison.mp4) | [P8](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10RGBeval-three-sequences-20261117-001/outdoor/P8_RGB_pose_comparison.mp4) |
| drone | [원본](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10input-three-sequences-20261117-001/drone/source_clip.mp4) | [B3](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10RGBeval-three-sequences-20261117-001/drone/B3_RGB_pose_comparison.mp4) | [P8](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10RGBeval-three-sequences-20261117-001/drone/P8_RGB_pose_comparison.mp4) |

## 제공 자세 통제와의 관계

[제공 자세 통제 보고서](../three-sequences-P10eval-20261117-002/report.md)는 depth GT로 scale을 정하지 않고 제공 궤적의 FIT 삼각측량으로 metric scale을 정했다. RGB 자세 표와 평가 pixel/scale 조건이 다르므로 숫자의 차이를 자세 오차만의 인과 효과로 해석하지 않는다. 그 조건에서도 실외 보정은 악화됐으며, 이미지 누적 자체가 정확도 향상을 보장하지 않았다.

## 한계와 이력

- 동일 평면 ROI는 RGB에서 수동 polygon 방식으로 지정했다. SAM 자동 분할이나 학습 모델을 이번 단계에서 평가하지 않았다.
- 최초 source0 hull support gate로 실외가 모두 실패한 결과를 보존했다. GT 평가 전 다른 관측 시점의 hull도 허용하는 gate로 바꿨다.
- 첫 GT/영상 감사 후 드론 후반 ROI의 물체 혼입을 확인해 RGB polygon을1회 수정했다. 이후 추가 수정은 없다. 이 실험은 blind test가 아닌 후행 탐색이다.
- 수정 후에도 드론 후반 참조의 국소 평면 RMS가 약4.6–8.6cm여서 가림·ROI·실제 비평면·투영 GT 문제가 남는다. source0의 좋은 평면성과 후반의 평면성을 구분한다.
- KITTI와 EuRoC 기준은 점군의 최소Z 투영이며 완벽한 dense/occlusion GT가 아니다. 결측을 채우지 않았다.
- 제공 자세 조건의 미래 heldout track 쌍은 실내0/실외1/드론202개다. 앞의 두 구간에서 미래 정합 일반화를 주장할 수 없다. RGB 조건은 미래 자세를 추론하지 않았다.
- pose는 고정한 채 평면만 보정했다. 추정 pose/K 오차, 매칭 편향, 평면 가정 위반이 원인 후보이며 이번 결과만으로 원인을 확정하지 않는다.
- P8의1° 제한은 예측 K가 정의한 보정 좌표계 기준이다. 물리 K로 사후 평가한 법선 변화의1° 제한을 뜻하지 않는다.

## 재현과 보존

[실행 코드·절차](/home/hlee221212/works/gradresearch/vggt_sam_tag/experiments/planar_residual_depth/P10/README.md). 원본 실행과 실패는 별도 run ID로 보존했다. canonical 제공 자세 fit003/eval002, RGB 자세 fit004/RGBeval001, 모델 입력 수 ablation context001이다. fit004의 숫자는 변경하지 않고 잘못된 provided-pose metadata만 평가 전에 정정했으며 원래 metadata 파일도 보존했다.

[최종 감사](/home/hlee221212/works/gradresearch/vggt_sam_tag/results/planar_residual_depth/P10RGBeval-three-sequences-20261117-001/final_audit.json)에서 두 카메라 조건의3시퀀스×4prefix×5arm=120개 출력에 대해 raw 동일성, raw×scale+residual 항등식, ROI 밖 불변, 미사용 미래 프레임 불변, 유한·양의 깊이를 확인했다. 12개 비교 영상의 모든 프레임을 실제 decode했다. RGB 평가 전후 fit hash도 동일하다.

데이터 공식 설명: [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php), [EuRoC](https://projects.asl.ethz.ch/datasets/euroc-mav/). EuRoC mirror pin과 SHA, 원본 이력은 제공 자세 통제 보고서 및 input manifest에 기록했다.
