# A100에서 실제 1-epoch 학습 검증

## 목적

GitHub Private 저장소 업로드 전에 6개 원본 training runner가 실제 데이터 로딩,
GPU forward/backward, 2-fold held-out 예측 저장까지 정상 수행하는지 확인합니다.
**이 테스트는 논문 원본 학습(8 stages × 3 seeds × 5 random + 7 LOSO)의 재현이 아니며,
원본 Random/LOSO reference folds를 검증하지 않습니다.**

## 준비

- venv 내 `torch`, `torch_geometric`, `rdkit`, `pandas`, `sklearn` 설치
- **실제 원본** matched Intersection-set manifest: 3,480행 (435 systems × 8 stages)
- 314열 **원본** descriptor CSV (E/Z 테스트에 필요)
- manifest의 `input_xyz` 절대 경로가 A100의 실제 XYZ 파일에 연결되어 있어야 함
- 테스트 출력은 Git에서 제외되는 `run_outputs/smoke_training/`에 저장

기존 실제 예측 결과 디렉터리를 덮어쓰지 않도록 테스트는 독립된 디렉터리에서 수행합니다.

## 1. 파일 경로 찾기

```bash
cd /home/solvation_mrl_project/baseline_mrl/strict_experiments
find . -type f -name '*intersection*manifest*.csv' | head -30
find . -type f -name '*contact*descriptor*.csv' | head -30
```

**Available-set manifest(시스템 수 435~557 가변)는 사용하지 마세요.**
실제 435-system all8 manifest와 descriptor CSV를 구별해서 선택합니다.

## 2. 런타임 준비와 기본 검증

```bash
cd /path/to/GNN-Geometry-Optimization
python - <<'PY'
import torch, torch_geometric, rdkit
print('torch', torch.__version__, 'PyG', torch_geometric.__version__)
print('CUDA', torch.cuda.is_available())
assert torch.cuda.is_available()
PY
sha256sum -c SOURCE_SHA256.txt
python -m unittest discover -s tests -v
```

## 3. Dry-run

```bash
export MANIFEST=/ABSOLUTE/PATH/organic_GNN_intersection_manifest.csv
export DESC=/ABSOLUTE/PATH/organic_contact_descriptors_intersection_all8.csv

python scripts/smoke_train.py --manifest "$MANIFEST" \
  --descriptors "$DESC" --models B3D BDim
```

Dry-run은 실제 파일을 읽지 않고 정확한 실행 명령만 보여줍니다.

## 4. 실제 GPU 학습 (먼저 2개)

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/smoke_train.py \
  --manifest "$MANIFEST" --descriptors "$DESC" \
  --models B3D BDim --execute \
  --outroot run_outputs/smoke_training
```

이 테스트는 **Initial 1개 stage, seed 0, Random 2-fold, 최대 1 epoch**입니다.
DimeNet++는 2-fold 테스트에서 논문 5-fold reference CSV와 일치할 수 없기 때문에
**이 테스트에 한해** `--reference-folds none`을 넘깁니다. 논문 실험을 돌릴 때는
반드시 원본 `reference_outer_folds_B_E.csv`를 사용하세요.

`PASS B3D`와 `PASS BDim`이 출력되면 전체 6개로 확대합니다.

## 5. 6개 모델 전체 확인

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/smoke_train.py \
  --manifest "$MANIFEST" --descriptors "$DESC" \
  --models B3D E3D Z3D BDim EDim ZDim --execute \
  --outroot run_outputs/smoke_all_six
```

각 모델에서 435개 held-out 예측, 2개 fold 결과, 유한한 숫자 값,
설정 JSON 및 예측 파일 저장을 확인합니다. 로그는 각 모델의 `smoke.log`에 남습니다.

```bash
cat run_outputs/smoke_all_six/SMOKE_PASS_summary.json
find run_outputs/smoke_all_six -name 'smoke.log' -print
```

## 6. 실제 기존 예측 파일을 이용한 논문 표 검증 (별도 필수)

분석용 `analysis/prediction_paths.json`에 6개 모델의 **matched-set** 예측 CSV 경로를
연결한 후 수행합니다.

```bash
python analysis/verify_matched_predictions.py \
  --inputs analysis/prediction_paths.json \
  --report run_outputs/original_96_RMSE_audit.json
python analysis/bootstrap_from_predictions.py \
  --inputs analysis/prediction_paths.json \
  --outdir run_outputs/bootstrap_original
```

엄격 검증은 96개 RMSE가 Tables 3--4와 ±0.00011 이내에서 일치하는지 확인합니다.
Bootstrap 신뢰구간은 Table S5와 직접 비교하고, 난수 추출 방식이 다르면 방법을 통일합니다.

## 공개 기준

- 본 스모크 테스트 6개 모델 PASS
- 원본 125,280 held-out 예측과 논문 96개 RMSE 일치
- Table S5 bootstrap 재검증
- 실제 3,480개 XYZ의 128-neighbor cap 검사
- Figshare 데이터·descriptor·reference folds 공개, KRICT 승인 후 LICENSE 지정

이 체크리스트를 채우기 전에는 GitHub를 **Private**로 유지하는 것을 권장합니다.
