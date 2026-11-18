# P17 장면 확장과 조기 종료

기존 P16 모델을 새로운 초기 가중치로 학습한다. 최초 실행은 기존 Middlebury 평가 자료에 새로 다운로드한 Backpack, Bicycle1, Couch, Mask, Storage, Umbrella를 추가했다. 이전 KITTI·EuRoC 학습 클립도 포함했다. Middlebury의 조명·노출 변형은 같은 장면 ID로 묶었다. 검증은 Piano와 Recycle, 테스트는 Playroom·Playtable·Teddy와 기존 TUM 클립이다. 추가 TUM 파일을 받을 경우 prepare.py의 --include-tum으로 영상 클립을 확장할 수 있다.

```bash
p=results/planar_residual_depth
prepared="$p/P17-input-20261118-001"
run="$p/P17-expanded-20261118-001"
v16=experiments/planar_residual_depth/P16
v17=experiments/planar_residual_depth/P17
.venvs/p1/bin/python "$v17/prepare.py" "$prepared" --include-stereo Backpack Couch Storage Bicycle1 Mask Umbrella
P10_HOST_GPU=1 .venvs/p1/bin/python "$v16/real_inputs.py" "$prepared" "$run" vggt --whole-clips
P10_HOST_GPU=1 .venvs/p1/bin/python "$v16/real_inputs.py" "$prepared" "$run" sam --whole-clips
.venvs/p1/bin/python "$v17/pack.py" "$prepared" "$run"
P10_HOST_GPU=1 .venvs/p1/bin/python "$v16/run.py" train --manifest "$run/dataset.json" \
  --out "$run/sensor" --device cuda --amp --epochs 30 --patience 5 --selection-group low_gt_edge
P10_HOST_GPU=1 .venvs/p1/bin/python "$v16/run.py" train --manifest "$run/dataset.json" \
  --out "$run/multiview" --device cuda --amp --epochs 30 --patience 5 --selection-group low_gt_edge --lambda-rgb .1
for arm in sensor multiview; do
  for test in stereo tum; do
    .venvs/p1/bin/python "$v16/run.py" evaluate --manifest "$run/test-$test.json" \
      --checkpoint "$run/$arm/best.pt" --out "$run/eval-$arm-$test"
  done
done
.venvs/p1/bin/python "$v17/summarize.py" "$run"
```

GPU 단계는 호스트에서 실행한다. SAM 단계에는 공식 segment-anything과 PyTorch에 맞는 torchvision이 필요하다. 새 결과에는 고유 준비 및 실행 폴더를 지정한다.

검증 체크포인트는 센서 깊이 단차 주변 저신뢰 픽셀의 MAE로 고른다. 정확히 같은 값은 개선으로 세지 않는다. 다섯 epoch 연속 개선되지 않으면 즉시 종료한다. 동일한 평가 픽셀과 원본 기준 스케일로 두 모델을 비교한다. Middlebury 테스트는 이 프로젝트의 장면 분할이며 공식 벤치마크 점수가 아니다.
