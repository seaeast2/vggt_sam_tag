# P9a–P9d: 원본 RGB 직접 정렬에 의한 평면 깊이 보정

**기존21개 사례에서 P9a–P9d를 처리했다. RGB 보정은 KITTI13개/Middlebury3개에서 실행됐고, 나머지5개는 영상 블록 지원 부족으로 B2를 유지했다. 이전8개 미지원 사례는 별도 보존했다. 같은 장면을 이미 분석한 후속 탐색이며 독립 검증이 아니다.**

## 사전 고정 방법과 GT 경계

모든 규칙을 fit 전에 `protocol.json`으로 고정했다. 기존 source/target 평면 mask, RGB, 제공 K/pose, raw depth, 공통 FIT-RGB scale을 유지했다. P9b/c는 고정 LK 대응점 잔차를 사용하지 않고 source0→target1 평면 warp의 RGB 차이를 최소화했다. RGB[0,1] sRGB에 radial Huber0.03, 전역 scalar gain[0.7,1.3]·bias[−0.15,0.15]를 사용했다. 광도 모델은 실제 카메라 response나 반사 모델을 완전히 설명하지 못한다.

P9b는 기존 B1의 법선 방향 고정·거리만 보정, P9c는 방향·거리 공동 보정이다. Gaussian sigma1.0/0.5/0.0에서 같은 픽셀 support를 사용했다. 양의 깊이와 기존 B2 대비 source polygon corner의 target warp 각 좌표±3px를 제약했다. 최종 dense source ROI에서도 한도를 검증했다. Feature same-trust 대조군은 기존 LK FIT 목적함수를 같은 제약에서 최적화해 탐색 범위 변경 효과를 분리했다.

Source mask4px·target mask9px 침식 후 B2와B3에서 모두 target 내부인 source 점들을 고정했다. 32px 영상 tile의 경계8px를 제외하고 tile hash seed0으로 FIT/보류70/30을 나눴다. FIT64px·2tiles, 보류32px·1tile, FIT hull10% 기준을 낮추지 않았다. 최적화 중 현재 잔차에 따라 픽셀을 삭제하거나 mask를 바꾸지 않았다. photometric 보류는 P9 갱신에 쓰지 않은 픽셀이지만 원래 RGB/LK 기반 초기값이 이미 같은 관측을 반영했을 수 있어 독립 테스트로 부르지 않는다.

GT는 모든 fit 및 pre-GT 기술 검사가 끝난 뒤 별도 프로세스에서 읽었다. GT 참조 평면의 warp는 사후 oracle 진단만 하며 초기화·support·해 선택에 전달하지 않았다. Oracle의 gain/bias만 고정된 FIT 픽셀에서 적합했다. GT 참조 평면도 센서 GT의 근사 평면이며 실제 노면 굴곡·참조 불확실성은 기존 P4 분석과 함께 해석한다.

## P9a: 영상 정렬 오차는 깊이 오차와 연결되는가?

| 데이터셋 | 보정 지원 사례 | B2/B3의 RGB 보류 순위와 ROI GT 깊이 순위 일치 | 같은 RGB 보류 위치의 GT로 비교 | GT참조 plane의 RGB 보류 손실이 B2보다 작은 사례 |
|---|---:|---:|---:|---:|
| kitti | 13 | 10/13 | 9/12 | 9/13 |
| middlebury | 3 | 3/3 | 3/3 | 3/3 |

대부분의 지원 사례에서 정렬 점수와 깊이 순위가 일치했지만 예외가 있다. GT 참조 평면이 항상 RGB 오차를 최소화하지도 않는다. 밝기·가림·반사·무늬·GT 평면 근사·카메라 보정 잔차·리샘플링 오차 등은 가능한 요인이며 이번 관측만으로 개별 원인을 확정하지 않는다.

## P9b/c: 동일 지원 사례의 paired 정확도

| 방법 | KITTI 평균 MAE(cm),13개 | Middlebury 평균 MAE(cm),3개 |
|---|---:|---:|
| C_B2 | 10.670 | 1.117 |
| C_B3 | 10.870 | 0.192 |
| F9C_MATCH_TRUST | 10.870 | 0.192 |
| P9B_PHOTO_FIXED | 10.958 | 1.168 |
| P9C_PHOTO_FREE | 8.998 | 0.366 |

C_B2/C_B3는 기존 기하를 그대로 사용하고 gain/bias만 FIT에 맞춘 RGB 진단 대조군이다. Feature same-trust의 깊이 평균은 기존 B3와 사실상 같아 이번 지원 사례에서 기하 탐색 범위 제한 자체가 결과 개선의 주된 차이로 보이지 않는다. 이는 RGB 목적함수·새 픽셀 sampling·밝기 nuisance가 함께 바뀐 비교이며 단일 원인의 인과 입증은 아니다.

| 방법 | KITTI 평균 법선오차(°) | Middlebury 평균 법선오차(°) |
|---|---:|---:|
| C_B2 | 0.692 | 3.300 |
| C_B3 | 0.755 | 0.389 |
| F9C_MATCH_TRUST | 0.755 | 0.389 |
| P9B_PHOTO_FIXED | 0.692 | 3.300 |
| P9C_PHOTO_FREE | 0.814 | 0.657 |

KITTI P9c는 평균 깊이 오차가 감소했지만 평균 법선 오차는 개선하지 않았다. 이미지 정렬/깊이 개선을 평면 방향 정확도 개선과 동일하게 해석하지 않는다. Middlebury의 지원 사례는 Playtable/PlaytableP/Vintage의3개 ID로, 앞의 둘은 유사 장면이므로 독립3장면 일반화로 볼 수 없다.

## B2 fallback을 포함한 전체21개 결과

| 방법 | KITTI14개 평균 MAE(cm) | Middlebury7개 평균 MAE(cm) |
|---|---:|---:|
| C_B2 | 10.834 | 0.884 |
| C_B3 | 10.888 | 0.331 |
| F9C_MATCH_TRUST | 10.888 | 0.331 |
| P9B_PHOTO_FIXED | 11.101 | 0.905 |
| P9C_PHOTO_FREE | 9.282 | 0.562 |

이 표에서 P9b/c의 미지원5개는 원래 B2를 유지했다. 따라서 전체 평균과 위의 paired 지원 평균을 혼동하지 않는다. 실패 시 GT로 B2/B3 중 더 좋은 것을 골라 유지하지 않았다.

## Paired 변화·보류 RGB 손실

| 데이터셋 | 비교 | 평균 MAE 차이(cm) | ID bootstrap 명목95% 구간 | 깊이 개선 | RGB 보류 개선 | 깊이·방향 개선+RGB 보류 비악화 |
|---|---|---:|---|---:|---:|---:|
| kitti | P9B_PHOTO_FIXED_vs_C_B2 | +0.288 | [-1.886,+2.412] | 5/13 | 7/13 | 0/13 |
| kitti | P9C_PHOTO_FREE_vs_C_B3 | -1.872 | [-6.246,+1.547] | 9/13 | 7/13 | 2/13 |
| kitti | P9C_PHOTO_FREE_vs_P9B_PHOTO_FIXED | -1.959 | [-3.472,-0.030] | 10/13 | 7/13 | 5/13 |
| kitti | P9C_PHOTO_FREE_vs_F9C_MATCH_TRUST | -1.872 | [-6.246,+1.547] | 9/13 | 7/13 | 2/13 |
| middlebury | P9B_PHOTO_FIXED_vs_C_B2 | +0.051 | [-0.037,+0.198] | 2/3 | 2/3 | 0/3 |
| middlebury | P9C_PHOTO_FREE_vs_C_B3 | +0.174 | [-0.055,+0.477] | 1/3 | 3/3 | 0/3 |
| middlebury | P9C_PHOTO_FREE_vs_P9B_PHOTO_FIXED | -0.802 | [-1.639,-0.212] | 3/3 | 3/3 | 3/3 |
| middlebury | P9C_PHOTO_FREE_vs_F9C_MATCH_TRUST | +0.174 | [-0.055,+0.477] | 1/3 | 3/3 | 0/3 |

방향 개선 판정은1e-6°보다 큰 감소만 센다. 방향 고정군의 부동소수점 반올림 차이를 방향 개선으로 세지 않는다. 평균은 장면 ROI MAE의 동일 가중 평균이며 단위가 다른 데이터셋을 합치지 않았다. ID bootstrap10000회 seed0은 기술 통계다. 이미 본 데이터·연속 KITTI ID·동일 물리 장면 변형이 포함돼 독립 일반화 신뢰구간이나 유의성 입증으로 해석하지 않는다.

## P9d: 밝기·무텍스처·가림 관련 진단

| 데이터셋 | 군 | 밝기 파라미터 한도 도달 | warp 한도 도달 | 보류 고잔차 비율 평균 | 보류 RGB MAE: 보정없음→gain/bias |
|---|---|---:|---:|---:|---|
| kitti | C_B2 | 0 | 0 | 0.322 | 0.0207→0.0156 |
| kitti | C_B3 | 1 | 0 | 0.333 | 0.0209→0.0159 |
| kitti | F9C_MATCH_TRUST | 1 | 0 | 0.333 | 0.0209→0.0159 |
| kitti | P9B_PHOTO_FIXED | 0 | 0 | 0.321 | 0.0206→0.0155 |
| kitti | P9C_PHOTO_FREE | 0 | 0 | 0.320 | 0.0203→0.0155 |
| middlebury | C_B2 | 0 | 0 | 0.238 | 0.0172→0.0171 |
| middlebury | C_B3 | 0 | 0 | 0.164 | 0.0118→0.0114 |
| middlebury | F9C_MATCH_TRUST | 0 | 0 | 0.164 | 0.0118→0.0114 |
| middlebury | P9B_PHOTO_FIXED | 0 | 0 | 0.213 | 0.0169→0.0166 |
| middlebury | P9C_PHOTO_FREE | 0 | 0 | 0.144 | 0.0114→0.0108 |

고잔차는 RGB residual norm>0.03의 비율이다. 실제 가림 GT가 없으므로 가림 검출 정확도라고 표현하지 않는다. source gradient, 밝기 자유도를 제거한 plane Jacobian singular ratio, gain/bias 한도, trust 한도를 보존했다. 원래 pixel support가 충분한16개에서 low-texture gate는 통과했으나 이는 깊이 정확도 보장이 아니다.

| 미지원 사례 | FIT/보류 픽셀 | FIT/보류 tile | 이유 |
|---|---:|---:|---|
| kitti/000019 | 997/29 | 6/1 | INSUFFICIENT_SPATIAL_PHOTO_SUPPORT; B2 retained |
| middlebury/Motorcycle | 0/21 | 0/1 | INSUFFICIENT_SPATIAL_PHOTO_SUPPORT; B2 retained |
| middlebury/Piano | 42/0 | 1/0 | INSUFFICIENT_SPATIAL_PHOTO_SUPPORT; B2 retained |
| middlebury/Playroom | 182/0 | 1/0 | INSUFFICIENT_SPATIAL_PHOTO_SUPPORT; B2 retained |
| middlebury/Teddy | 768/0 | 4/0 | INSUFFICIENT_SPATIAL_PHOTO_SUPPORT; B2 retained |

작은 평면 ROI에서 공간적으로 분리된 학습/보류 관측을 만들 수 없는 사례는 영역을 늘리거나 검증 기준을 낮춰 통과시키지 않았다. 이전8개 미지원 사례(ETH3D 두 장면 포함)는 현재 비교 범위 밖에 그대로 남겼다.

## 사례별 악화와 시각 산출물

- kitti P9B_PHOTO_FIXED가 C_B2보다 악화: 000005, 000006, 000008, 000009, 000012, 000014, 000016, 000017.
- kitti P9C_PHOTO_FREE가 C_B3보다 악화: 000006, 000008, 000017, 000018.
- middlebury P9B_PHOTO_FIXED가 C_B2보다 악화: Playtable.
- middlebury P9C_PHOTO_FREE가 C_B3보다 악화: Playtable, PlaytableP.

- `RGB_warp_alignment.png`: source RGB, target warp,8px checkerboard, 공통 RGB error 범위. Oracle GT plane 열은 사후 진단 전용이다.
- `depth_GT_errors.png`: 동일 ROI/카메라/scale의 깊이와 GT 오차. GT 결측은 회색이다.
- `ROI_pointmaps.npz`: 각 군의 source0 유한 ROI pointmap과 동일 RGB·pixel_uv·frame_id. 전체 장면이나 원래 VGGT 예측 pose pointmap과 구분한다.
- 각 사례 `metrics.json`, `oracle_GT_plane_warp.json`, support/solver log, source snapshot·원본/잔차·GT 출처, `per_case_metrics.csv`, `aggregate.json`.

## 기술 검증과 결론

합성 정답 평면에서 RGB warp의 해석미분을 수치미분과 비교했고, 자유/방향고정 광도 보정으로 정답 warp를 복원했다. 무텍스처 Jacobian 퇴화와 합성 수정 패치의 고잔차 반응도 확인했다. 실제 사례는 원본 캐시·잔차복원·비대상 불변·양의깊이·corner/dense warp trust·평가 전후 fit hash 불변을 검사했다. P9b scalar 관측 가능성은 `preGT_observability_check.json`에서 GT 평가 전에 별도로 검증했다.

RGB 직접 정렬은 기존 고정 대응점과 다른 관측을 사용해 KITTI에서 깊이 개선을 보였지만, 법선 정확도와 실내 성능의 일관된 개선은 확인하지 못했다. 광도 일치가 곧 기하 정답이라는 가정은 성립하지 않는 사례가 있다. 현재는 연구 방향의 가능성과 한계를 함께 보인 탐색 결과이며 새 독립 장면·노출 변화·실제 가림 GT·다중시점/장기 SLAM 검증은 남는다. 이번에도 기존 평면 내부를 대상으로 했고 RGB 경계 기반 재분할이나 물체 경계 복원은 실험하지 않았다.

Canonical fit: `P9fit-all-cases-20261117-001`. 평가·그림·보고서: `P9eval-all-cases-20261117-001`.

## 최종 감사 보충

`audit.json`에서21개 사례의 원본/신뢰도 캐시 일치, FIT/보류 tile 분리와 guard/hash 규약, 공유평면 교차, 양의 깊이·잔차복원·비대상 불변·P9b 방향 고정·dense warp 한도, RGB/GT MAE 재계산, 기존 fit/새 fit hash 불변을 확인했다. `FIT_photo_descent_check.json`에서16개 실제 보정 사례 모두 P9b/c의 FIT 광도 목적함수가 원래 B2보다 악화되지 않았음을 확인했다. 보류 RGB 손실이나 GT 오차는 해 선택에 사용하지 않았다.

KITTI P9c−B3 평균 MAE 변화는−1.872cm지만 ID bootstrap 명목95% 구간 [−6.246,+1.547]cm가0을 포함한다. 평균 개선만으로 일반적 우월성을 확정하지 않는다. 각도 개선 판정의 부동소수점 tolerance를 추가하기 전 보고서는 `report_before_direction_tolerance.md`로 보존했다. 기하·광도 fit 결과는 변경하지 않았다.
