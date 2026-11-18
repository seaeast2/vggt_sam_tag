# P16 깊이맵 잔차 보정 실행

[모델 설계서](../../../docs/depthmap_refinement_model_design.md)

## 데이터

데이터 manifest는 아래 형식이다. 캐시 경로는 manifest 기준 상대 경로나 절대 경로를 사용한다. 같은 물리 공간이나 세션을 나타내는 동일한 scene ID는 하나의 split에만 배정한다. 배치 크기가 2 이상이면 한 배치에 들어가는 래스터 크기를 맞춘다.

```json
{
  "clips": [
    {"scene":"room-a","split":"train","cache":"room-a.npz"},
    {"scene":"room-b","split":"val","cache":"room-b.npz"},
    {"scene":"room-c","split":"test","cache":"room-c.npz"}
  ]
}
```

| NPZ 키 | 형상 및 의미 |
|---|---|
| rgb | N×H×W×3 uint8, VGGT 입력과 같은 RGB 래스터 |
| raw, confidence, sam | N×H×W. 양의 camera-Z, 원본 confidence, SAM 경계 거리 0–1 |
| model_K, model_pose | N×3×3 및 N×4×4, 월드에서 카메라로의 변환 |
| neighbors, eval_neighbors | N×2 정수. 0 기반 프레임 번호, 빈 항목 −1. 재투영 학습과 평가에서 서로 다른 쌍 사용 |
| gt | N×H×W 미터 깊이. 학습과 평가에 사용. 결측 0 또는 NaN |
| provided_K, provided_pose | 제공 카메라 통제 조건용. pose 이동량은 미터 |
| provided_scale | 제공 자세 비용 볼륨의 GT 없는 추론에 필요한 외부 미터 환산 계수 |
| static | 선택. N×H×W 정적 영역 마스크. 없으면 장면을 정적으로 가정 |
| wall_floor | 선택. 독립 검토한 경계 부근에서 벽 쪽 1, 바닥 쪽 2, 평가 제외 0 |
| rule_depth | 선택. 동일 원본과 단위의 P13 비교 깊이 |

TUM Freiburg3는 `pack_tum.py`로 기존 VGGT 예측과 SAM 마스크를 묶는다. 프레임 번호에 맞춰 SAM NPZ를 제공한다. 각 NPZ의 masks는 M×H×W 불리언 배열이다. `--timestamps`에는 VGGT에 입력한 TUM RGB 타임스탬프 문자열 배열을 전달한다. 타임스탬프 파일이 없으면 offset과 interval로 원래 선택을 재현한다. 동일 종횡비 리사이즈 입력을 대상으로 한다. `--verify-input-manifest`는 RGB 해시가 기록된 원본 추론 manifest에 사용한다. 초기 캐시에 해시가 없는 경우 캐시 생성 이력의 프레임 순서를 확인한다.

## 실행

저장소 루트에서 실행한다. 아래 예시의 결과 경로는 새 run마다 바꾼다. CUDA 학습은 호스트에서 실행한다.

```bash
p16=experiments/planar_residual_depth/P16
P10_HOST_GPU=1 .venvs/p1/bin/python "$p16/run.py" train --manifest dataset.json \
  --out results/planar_residual_depth/P16-sensor-001 --device cuda --amp --epochs 5 --lambda-rgb 0
P10_HOST_GPU=1 .venvs/p1/bin/python "$p16/run.py" train --manifest dataset.json \
  --out results/planar_residual_depth/P16-multiview-001 --device cuda --amp --epochs 5 --lambda-rgb 0.1
.venvs/p1/bin/python "$p16/run.py" evaluate --manifest dataset.json \
  --checkpoint results/planar_residual_depth/P16-multiview-001/best.pt \
  --out results/planar_residual_depth/P16-evaluation-001
.venvs/p1/bin/python "$p16/run.py" infer --manifest dataset.json \
  --checkpoint results/planar_residual_depth/P16-multiview-001/best.pt \
  --out results/planar_residual_depth/P16-inference-001
```

`--features rgb-depth|confidence|sam`은 입력 단계다. confidence 단계부터 학습 자료의 70분위수로 깊이 수정을 제한하므로 첫 입력 비교는 confidence 입력과 보호 조건의 결합 효과다. `--camera provided`는 포즈 통제에, `--volume`은 9개 깊이 후보 정합 비용을 모델 입력으로 넣는 실험에 사용한다. 비용 볼륨은 일반 이웃을 사용하고 평가 지표는 별도로 지정한 평가 이웃을 사용한다.

학습은 검증 MAE로 체크포인트를 선택한다. 평가의 카메라 깊이는 클립별 원본으로 정한 스케일 하나로 정렬한다. 보정 후 PLY는 VGGT-Ω 예측 카메라 좌표계에 저장한다. 각 결과 NPZ에 원본, 보정 깊이, 잔차, gate, 적용 픽셀과 카메라 행렬이 있다. 전체·신뢰도별·깊이 경계별 오차와 재투영 지지율은 metrics.json에 기록된다. 실제 장면의 바닥–벽 지표에는 독립 경계 주석이 필요하다.

## 구현 확인 이력

합성 장면 세 개로 독립 train/val/test 경로를 확인했다. 센서 감독, RGB 재투영 감독, 제공 자세 및 비용 볼륨, CUDA 혼합 정밀도 학습을 각각 한 epoch 실행했다. 기존 TUM 2초 간격 **초기 8장**과 SAM 캐시 또한 연결해 추론과 평가 파일을 생성했다. 합성 검증용 체크포인트의 TUM 출력은 파일 및 데이터 연결 확인용이다. 실제 학습 장면과 독립 평가 장면을 사용한 성능 실험은 별도로 수행한다.

## 세 영상 파일럿

`real_inputs.py`는 P10 준비 결과의 각 20장 영상에서 인덱스 0–5, 7–12, 14–19를 배정한다. 각 블록마다 VGGT-Ω를 독립 실행하고 SAM 경계 지도를 저장한다. `pack_real.py`는 등록된 TUM 센서 깊이, KITTI Velodyne 투영, EuRoC Leica 방 스캔 투영을 캐시에 묶는다. EuRoC 참조에는 데이터셋 자세를 사용해 점군을 픽셀로 투영한다. 이는 깊이 감독의 생성 과정이며 모델은 제공 자세를 입력으로 받지 않는다.

`P16-real-20261118-001/dataset.json`은 `within_scene_temporal_pilot`를 명시한 예외다. 이 모드에서는 같은 영상의 블록을 나누되 영상 ID와 프레임 ID가 중복되면 중단한다. 이 파일럿으로는 새 장면에 대한 일반화 성능을 판단할 수 없다. 동적 객체 주석이 없어 재투영 가시성과 시차로 유효 관측을 걸렀다. EuRoC는 실내 드론의 흑백 영상을 3채널로 사용했다.

같은 시드로 센서 손실과 센서 + RGB 재투영 손실을 각각 5 epoch 학습한다. 두 모델의 체크포인트는 검증 MAE로 각각 고르고, 테스트에서 영상별 원본과 비교한다. 테스트의 스케일은 블록으로 분리한 센서 픽셀에서 원본 기준으로 한 번 정렬한다.
