# P10 연속 영상 파일럿 실행 기록

최종 결과는 [통합 보고서](../../../docs/experiments/planar_residual_depth/P10/P10RGBeval-three-sequences-20261117-001/report.md)에 있다. 실내 TUM/실외 KITTI/실내 드론 EuRoC 각 한 구간에서 실행했다. raw/residual/카메라/관측/GT 평가/영상을 별도 run 폴더에 보존한다.

## 실행 환경

작업 루트에서 `.venvs/p1/bin/python`을 사용한다. CPU 단계는 일반 환경, GPU 단계는 반드시 host에서 `P10_HOST_GPU=1`로 실행한다. 데이터는 `data/planar_residual_depth/P10/`이며 archive/source 경로와 hash는 input manifest가 기준이다. checkpoint는 원본 `vggt_omega_1b_512.pt`다. 추론 입력384 long-side, 수정되지 않은 checkpoint, 최대 peak allocated 약4.95GiB(16장 기본 추론)다. context ablation은 별도 environment/manifest를 확인한다.

## 실제 완료한 단계

1. `prepare.py` → P10input001: 원본60장과 제공 K/pose, 전처리 이미지, source_clip.mp4.
2. `infer.py` → P10raw001: RGB16장씩만 GPU 추론. GT 깊이는 읽지 않는다.
3. `freeze_masks.py` → P10masks001; `observe.py --masks P10masks-three-sequences-20261117-001` → P10obs001.
4. `fit.py --observations ...P10obs-...001` → fit001(엄격 gate), gate 변경 후 fit002. `evaluate.py --fit P10fit-three-sequences-20261117-002` → eval001.
5. `revise_masks.py` → masks002: 첫 평가 후 드론 후반 ROI를 RGB로1회 수정. post-hoc임을 기록했다. `observe.py --masks P10masks-three-sequences-20261117-002` → obs002.
6. `fit.py --observations P10obs-three-sequences-20261117-002` → fit003. `evaluate.py --fit P10fit-three-sequences-20261117-003 --reuse-GT P10eval-three-sequences-20261117-001` → eval002. `report.py` → 제공 자세 통제 보고서·20프레임 비교 영상·PLY.
7. `prepare_rgb_pose.py` → RGBinput001. `observe.py --masks P10RGBinput-three-sequences-20261117-001` → obs003. `fit.py --observations P10obs-three-sequences-20261117-003` → fit004. 보정에는 모델 K/pose만 사용했다. 초기 metadata의 provided-pose 오표기만 정정하고 원본 metadata를 보존했다.
8. `infer_contexts.py` → context001: RGB 입력2/4/8장 별도 추론,16장은 raw001을 재사용. GPU host 실행.
9. `evaluate_rgb.py` → RGBeval001: 평가용 scalar calibration/test 분리, N2/4/8/16 및 B0/B1/B2/B3/P8 평가, RGB 추정 자세16프레임 영상, native-gauge pointmaps.

숫자 suffix는 실제 실행 이력이다. 일부 스크립트는 날짜/기존 부모 run을 고정 참조하며 재실행 명령을 무조건 연결하는 범용 pipeline은 아니다. 재실행 시 원본 source snapshot과 config_resolved.json을 확인하고 새 run ID/부모를 지정해야 한다. 기존 run을 덮어쓰지 않는다. 특히 `prepare_rgb_pose.py`/`infer_contexts.py`는 기존 출력이 있으면 중단한다.

## 판정 범위

GT를 보정에 사용하지 않았으나 RGB 조건의 평가 scalar에는 GT 일부가 필요하다. 절대 scale 추정 성공이 아니다. 제공 자세 통제와 RGB 자세 조건은 평가 scale/pixel 정의가 달라 숫자를 자세 오차의 순수 효과로 비교하지 않는다. 실내 드론은 야외 드론 검증이 아니다. 짧은 오프라인 관측 누적 실험이며 장기 SLAM이 아니다.

최종 감사: RGBeval001/final_audit.json. 두 카메라 조건120개 prefix/arm 출력의 raw 불변/잔차 항등식/ROI 밖 불변/유한 양의 깊이 및12개 비교 영상 전체 프레임 decode를 확인했다. 평가 전후 fit004 hash 불변은 fit_hashes.json과 manifest에 있다.
