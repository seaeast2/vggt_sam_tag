# P10 실내·실외·드론 연속 영상: 다중시점 관측 누적 파일럿

**실내/실외/드론 각1개 실제 연속 시퀀스의 짧은 구간에서 VGGT-Ω 추론·평면 보정·센서 기준 평가·영상 출력을 완료했다. 총60개 RGB 샘플(시퀀스당20개),48개 mapping 이미지(각16),12개 미래 보류 이미지(각4)다. 카메라 자세는 제공 궤적으로 고정한 통제 조건이며 RGB만으로 자세까지 추정하는 배포/장기 SLAM 검증은 아니다.**

## 데이터와 시간 분할

- 실내: TUM RGB-D `fr3_structure_texture_far`, 첫 RGB 이후2s에서5Hz 근처20장, 약3.8s. 등록된 Kinect depth를 사후 기준으로 사용했다.
- 실외: KITTI Raw `2011_09_26_drive_0005_sync`, image02의 frame50–88 step2, 약3.9s. GPS/IMU OXTS·공식 외부보정으로 카메라를 구성하고 동기 Velodyne을 사후 투영했다.
- 드론: EuRoC MAV `V1_01_easy`의 cam0, 카메라/pose 공통 시작 이후10s에서5Hz20장. 실내 MAV 촬영·흑백 카메라를 RGB3채널로 복제했다. radial-tangential 왜곡을 제공 K로 제거했고 Leica MS50 room scan을 사후 투영했다. 야외 드론·자율비행 성능을 입증한 샘플로 표현하지 않는다.

EuRoC 기존 공식 서버 이동/접속 실패로 공개 mirror revision `d05bd6a92a5f456562153f83d65805fea27cb57d`를 사용했고 archive SHA256 `a920fe5b5e69a6ad19b32f1cfaf90ac2f59d45dfd1ca18cc9722e07684ba45fa`를 확인했다. 모든 원본 archive/source hash는 P10input manifest에 있다.

## 통제 조건과 실행 이력

VGGT-Ω 원본 checkpoint로 mapping16장만 공동 추론하고 future4장은 입력하지 않았다. 384 long-side, 공식 max_size 전처리, 실제 host GPU에서 수행했다. peak allocated VRAM은 실내5310002176B, 실외5071418368B, 드론5200589824B로 최대4.95GiB였다. 각 prefix의 raw는 같은16-frame cache이므로 N2도 이미16개 mapping RGB 문맥을 가진 오프라인 보정 비교다. 실시간/인과적 네트워크 추론 누적 실험으로 부르지 않는다.

제공 K/pose를 camera-error control로 사용했다. 모델 입력은 RGB뿐이지만 보정은 외부 제공 자세를 쓰므로 전체 파이프라인이 RGB-only SLAM이라고 주장하지 않는다. depth PNG/Velodyne/room scan 값은 첫 fit 이후에만 평가에 사용했다. 각 clip scale은 mapping0–15의 FIT SIFT 삼각측량과 raw depth 비율 중앙값 하나로 고정했다. future16–19나 depth GT로 scale을 정하지 않았다.

관측은 원본 undistorted RGB crop을 긴변768 이하로 두고, global SIFT mutual ratio0.75·epipolar1px로 scale을 구했다. 평면 보정에는 temporal Shi–Tomasi/LK와 FB0.5px·epipolar1px를 사용했다. 매핑 프레임에서만 재검출하고 future에서는 새 track을 만들지 않았다. Track seed ID hash로 FIT/보류를 고정했다. Support는 FIT12/보류6 track, 면적10%, 중앙 시차1°, Jacobian rank를 유지했다. 같은 plane의 모든 관측쌍에 기존 양방향 Huber1px 목적함수를 적용했다. 같은 track 내 시간 관측은 상관되며 독립 표본으로 세지 않는다.

1. `P10fit-001`: 첫 프레임의 FIT hull을 요구한 원래 gate. 실외 source0 추적 전달이 없어 전 prefix 미지원으로 보존했다.
2. `P10fit-002`: 깊이 GT를 읽기 전에, 같은 물리 평면의 다른 시점에서도 면적10%를 충족할 수 있게 최대 per-view hull gate로 바꿨다. 숫자 기준·frames·mask·raw·solver는 유지했고 원래 실패도 보존했다. 실외 N16에서 처음 support를 확보했다. 첫 GT 평가는 `P10eval-001`이다.
3. 첫 GT/영상 감사에서 드론 후반 fixed-screen ROI가 보이는 계단형 비평면 물체를 포함함을 확인했다. 원래 마스크/결과를 보존하고 드론 frame1–19 polygon만 RGB에서 물체를 피해 움직이도록1회 수정했다. source0 polygon은 동일하다. `P10masks-002`→`P10obs-002`→`P10fit-003`→현재 `P10eval-002`가 mask-audited canonical이다. 이 수정은 GT를 본 뒤의 후행 수정이므로 blind/independent test로 표현하지 않는다. 재검출 track과 scale의3px 보류 제외가 함께 달라질 수 있어 파생 scale값도 별도 기록했다. 수정 후 GT 결과를 보고 다시 영역을 조절하지 않았다.

이번 단계는 관측 누적 효과를 기존 B2/B3/P8 feature 목적함수로 통제한 P10이다. P9의 direct RGB 광도 목적함수를20-view로 확장하거나 카메라 자세를 최적화한 실험은 아니다.

## N16 고정 source0 ROI 결과

| 시퀀스 | 유효 센서 기준 픽셀 | B0 원본×scale MAE(cm) | B1 | B2 거리만 | B3 방향·거리 | P8 방향1° 제한 |
|---|---:|---:|---:|---:|---:|---:|
| indoor | 6626 | 1.110 | 1.060 | 0.725 | 1.565 | 0.984 |
| outdoor | 946 | 27.281 | 27.252 | 100.592 | 67.780 | 30.765 |
| drone | 3748 | 22.886 | 22.775 | 15.648 | 14.057 | 14.878 |

세 데이터셋의 오차를 하나로 평균하지 않았다. GT는 시퀀스별 센서 기준이며, KITTI/EuRoC는 camera pixel에 최소Z로 투영한 근사 참조다. 원래 카메라 시점의 occlusion-perfect dense depth나 공식 벤치마크 점수로 표현하지 않는다. KITTI motion compensation은 수행하지 않았고 GT 결측은 채우지 않았다.

## 관측2→4→8→16장 변화

| 시퀀스 | N | B2 MAE(cm) | B3 MAE(cm) | P8 MAE(cm) | support | FIT/보류 track | 중앙 시차(°) |
|---|---:|---:|---:|---:|---|---:|---:|
| indoor | 2 | 1.599 | 1.703 | 1.682 | passed | 92/36 | 1.0142394477880217 |
| indoor | 4 | 3.172 | 3.095 | 3.097 | passed | 137/55 | 1.705931684993203 |
| indoor | 8 | 1.006 | 1.141 | 1.133 | passed | 226/100 | 2.6318107541077365 |
| indoor | 16 | 0.725 | 1.565 | 0.984 | passed | 476/197 | 2.6055910819402675 |
| outdoor | 2 | 27.281 | 27.281 | 27.281 | raw retained | 0/0 | — |
| outdoor | 4 | 27.281 | 27.281 | 27.281 | raw retained | 0/0 | — |
| outdoor | 8 | 27.281 | 27.281 | 27.281 | raw retained | 3/3 | 2.33727533200478 |
| outdoor | 16 | 100.592 | 67.780 | 30.765 | passed | 62/23 | 1.8874800223162058 |
| drone | 2 | 11.492 | 12.101 | 11.564 | passed | 42/22 | 1.243172535318119 |
| drone | 4 | 17.813 | 16.096 | 17.658 | passed | 255/109 | 1.0706211500924774 |
| drone | 8 | 13.777 | 12.833 | 13.645 | passed | 383/165 | 2.1418098335420286 |
| drone | 16 | 15.648 | 14.057 | 14.878 | passed | 659/259 | 3.224717196821685 |

**사진/관측 수가 증가할수록 오차가 단조롭게 줄지는 않았다.** 실외2/4/8장은 미지원이라 raw 유지이며, 그 수치를 작은 prefix의 보정 성공으로 세지 않는다. 실내/드론에서도 prefix 변화가 비단조적이다. 시간 간격·시차·같은 물리점의 생존·재검출·조명·제공 자세 오차·실제 평면 가정이 함께 작용하므로 개수만의 인과 효과를 확정하지 않는다.

## 미래4개 시점 평가

| 시퀀스 | 방법 N16 | 미래4 ROI 평균 MAE(cm) | 미래 track-heldout 쌍 수 | 미래 재투영 median(px) |
|---|---|---:|---:|---:|
| indoor | B1 | 0.7688509508379571 | 0 | — |
| indoor | B2 | 1.4900743374890741 | 0 | — |
| indoor | B3 | 1.2923896515879 | 0 | — |
| indoor | P8 | 1.1268819568347537 | 0 | — |
| outdoor | B1 | 245.28588666497618 | 1 | 6.555734951003343 |
| outdoor | B2 | 116.05414182235619 | 1 | 1.577774283031598 |
| outdoor | B3 | 35.090482021894395 | 1 | 1.0008878869460396 |
| outdoor | P8 | 98.07603739200437 | 1 | 1.5153600602527464 |
| drone | B1 | 19.070216157914572 | 202 | 1.173327319762412 |
| drone | B2 | 10.845912493127887 | 202 | 1.0640420165765971 |
| drone | B3 | 15.912617619022823 | 202 | 1.1179692679327697 |
| drone | P8 | 13.677719721236759 | 202 | 1.0887470690400949 |

future RGB4장은 VGGT와 plane fit에 쓰지 않았다. 보정한 공유 평면을 제공 future camera에 투영해 sensor reference와 비교했다. future raw inference는 없으므로 B0 future depth 비교는 없다. 미래 RGB track-heldout은 mapping15→future로 고정했다. 실내는0쌍, 실외는1쌍으로 이 재투영만으로 일반화를 주장할 수 없다. 센서 depth 비교와 RGB 대응 일치를 구분했다.

## GT 평면 가정·방향 불확실성

| 시퀀스 | source0 GT plane RMS(cm) | GT법선 block bootstrap p95(°) | sampled 전체 world GT plane RMS(cm) |
|---|---:|---:|---:|
| indoor | 0.467 | 0.4634851510576412 | 0.632 |
| outdoor | 1.204 | 0.2878121418499862 | 3.667 |
| drone | 0.263 | 0.1596765522647152 | 4.949 |

Source0 GT 기준 법선은 같은 ROI에 robust plane을 적합하고16px 공간 block bootstrap50회로 변동을 기록했다. 전체 plane 진단은 frames0/4/8/12/15/19의 센서 기준점을 제공 자세로 묶은 후 적합했다. 이 RMS는 실제 비평면성·마스크 가림·센서 투영·자세 오차를 분리하지 못한다. GT 평면에 맞지 않는 사례나 점을 사후 삭제하지 않았다. 최초 잘못 포함된 드론 객체의 진단은 이전 run에 남아 있다.

## 입력 영상·결과 영상과 pointmap


- **indoor**: source RGB `results/planar_residual_depth/P10input-three-sequences-20261117-001/indoor/source_clip.mp4`.
  비교영상: `results/planar_residual_depth/P10eval-three-sequences-20261117-002/indoor/B3_N16_comparison.mp4`, `results/planar_residual_depth/P10eval-three-sequences-20261117-002/indoor/P8_N16_comparison.mp4`; GIF `results/planar_residual_depth/P10eval-three-sequences-20261117-002/indoor/P8_RGB_corrected_sensor.gif`.
  그림: `results/planar_residual_depth/P10eval-three-sequences-20261117-002/indoor/source0_depth_GT_errors.png`, `results/planar_residual_depth/P10eval-three-sequences-20261117-002/indoor/prefix_curves.png`.

- **outdoor**: source RGB `results/planar_residual_depth/P10input-three-sequences-20261117-001/outdoor/source_clip.mp4`.
  비교영상: `results/planar_residual_depth/P10eval-three-sequences-20261117-002/outdoor/B3_N16_comparison.mp4`, `results/planar_residual_depth/P10eval-three-sequences-20261117-002/outdoor/P8_N16_comparison.mp4`; GIF `results/planar_residual_depth/P10eval-three-sequences-20261117-002/outdoor/P8_RGB_corrected_sensor.gif`.
  그림: `results/planar_residual_depth/P10eval-three-sequences-20261117-002/outdoor/source0_depth_GT_errors.png`, `results/planar_residual_depth/P10eval-three-sequences-20261117-002/outdoor/prefix_curves.png`.

- **drone**: source RGB `results/planar_residual_depth/P10input-three-sequences-20261117-001/drone/source_clip.mp4`.
  비교영상: `results/planar_residual_depth/P10eval-three-sequences-20261117-002/drone/B3_N16_comparison.mp4`, `results/planar_residual_depth/P10eval-three-sequences-20261117-002/drone/P8_N16_comparison.mp4`; GIF `results/planar_residual_depth/P10eval-three-sequences-20261117-002/drone/P8_RGB_corrected_sensor.gif`.
  그림: `results/planar_residual_depth/P10eval-three-sequences-20261117-002/drone/source0_depth_GT_errors.png`, `results/planar_residual_depth/P10eval-three-sequences-20261117-002/drone/prefix_curves.png`.

비교영상은 실제 sampled RGB, 원본×scale ROI depth, 보정/외삽 depth, 센서 기준 depth, signed error를 공통 색 범위로 보여준다. 회색은 ROI 밖/결측/원본 미추론이다. 마지막4장은 raw 미추론을 명시하고 plane extrapolation만 표시한다. `fused_ROI_pointmaps/*.ply`는16장 유한 ROI의 파생점이며 RGB·pixel_u/v·frame_id·공유plane/카메라 계보를 보존했다. 전체 지도 정확도·메시·SLAM 성공을 뜻하지 않는다.

## 결론과 남은 한계

실제 연속 영상3개를 처리했고, 초기/후반 support 전환과 관측 누적의 비단조성을 확인했다. 실내 source0는 거리만 보정한 B2가 가장 작고, 드론은 원본 대비 보정 이득이 있으나 관측을 더 넣을수록 항상 좋아지지는 않는다. 실외는 관측 support가 생긴 N16에서도 source0 깊이가 악화해 현재 규칙의 적용 한계를 드러냈다. GT로 scene별 최적 arm이나 최적 prefix를 골라 실제 성능으로 제시하지 않는다.

카메라 제공 자세를 RGB 추정 자세로 바꾸는 조건, 긴 시간/루프 폐쇄, 야외 드론, 동적 객체 분할, plane ID 자동 유지, P9 광도 loss의 시간 확장, 독립 시퀀스 반복은 아직 수행하지 않았다. 이번 결과는 카메라 오차를 통제한 약4초 pilot이며 end-to-end RGB-only 배포 검증이 아니다.
