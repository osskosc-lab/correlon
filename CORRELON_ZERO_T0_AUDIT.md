# Correlon Zero T0 Repository Audit

**監査日:** 2026-08-22 (Asia/Tokyo)  
**repository:** `osskosc-lab/correlon`  
**監査ブランチ:** `agent/correlon-zero-adversarial-invariance`  
**監査基点:** `a727f11e95e40d7b0919dc87dd9730bfc16abc69` (`origin/agent/timeless-correlon-phase-t0-t3`)  
**監査方針:** T0 中は既存コードを変更せず、remote clone 直後の全 ref、履歴、追跡ファイルを読み取った。

## 1. 結論

現在リポジトリで再現可能な最新の Correlon 演算子は、Phase 4 から継続している窓別 whitened cross-covariance

\[
Q_t=(C_{xx,t}+\epsilon I)^{-1/2}C_{xy,t}(C_{yy,t}+\epsilon I)^{-1/2}
\]

と、その特異値・左右特異ベクトルの時間列である。既存の scalar readout は `T_iso`、`T_pers`、`T_floor` の3個で、単一の総合スコアは凍結されていない。

監査で得た最も重要な制約は次の3点である。

1. **Phase 4 の `rank1_persistent` と `common_driver` は同一実装である。** `make_data` の同じ分岐で同じ latent AR(1) `z` を `X` と `Y` に加えている。両クラスの違いはラベルだけであり、Phase 4 は direct mechanism と common driver を生成上区別していない。
2. **最新の Timeless T3 は、この演算子の rank-k 再構成が当該実装で CCA と数値的に同一であることを確認した。** 記録値は最大差 `0.0`、判定は `CORRELON_EQUALS_CCA_FOR_THIS_OPERATOR` である。
3. **Phase 5C は Correlon 固有の interventional utility を支持しなかった。** Confirmation 200 seeds で Correlon gain は約 `1.43%`、PCA gain は約 `1.88%`、specificity contrast は負であり、判定は `C_PLACEBO_EQUIVALENT_NO_GO` である。

したがって T1 へ渡せる操作的候補は、「因果機構」でも「Correlon 固有の表現」でもなく、**CCA-equivalent な normalized cross-space mode に Phase-4 型の時間連続性 readout を付加した記述統計**である。Correlon Zero はこの候補が generic correlation、low rank、spectrum、persistence、common driver、predictive dependence から分離できるかを反証的に検査しなければならない。

## 2. ブランチ監査と連続性判断

clone 直後に local/remote の全ブランチを確認した。主要 ref は以下の通りである。

| ref | commit | commit date | 内容 |
|---|---|---|---|
| `origin/main` | `07fecd1` | 2026-08-08 | README のみの初期状態 |
| `origin/agent/phase4-correlon-core-persistence` | `6686384` | 2026-08-08 | Phase 4 実装・要約 |
| `origin/agent/phase5c-interventional-representation-falsification` | `c470b02` | 2026-08-11 | Phase 5C preregistration・集計結果 |
| `origin/agent/timeless-correlon-phase-t0-t3` | `a727f11` | 2026-08-16 | Phase 5C を完全包含する最新 continuation |

`a727f11` は `c470b02` の直系子孫であり、Phase 4/5C の追跡ファイルを変更せず Timeless T0-T3 のコード・raw data・報告を追加している。このため、README-only main や古い Phase 5C tip ではなく `a727f11` を新規ブランチの基点とした。

## 3. 既存 operational Correlon の系譜

### 3.1 Phase 4: persistent normalized cross-space mode

実装: `experiments/Phase4_correlon_core_persistence.py`

凍結されていた設定:

| item | value |
|---|---:|
| time points | `T=900` |
| dimensions | `d=q=8` |
| signal strength | `1.2` |
| channel noise | AR(1), `rho=0.7` |
| shared scalar mode | AR(1), `rho=0.92` |
| window / step | `140 / 40` |
| covariance inverse-square-root regularizer | `epsilon=1e-5` |
| persistence lags | `1..3` |
| matched-null replicates | `6` circular shifts |
| Phase-4 run seeds | `100` per scenario; scenario offset `10000` |

窓ごとに `Q_t` を SVD し、最大・第2特異値 `s1_t,s2_t` と先頭左右 mode `u_t,v_t` を得る。既存 readout は以下である。

- `strength = median(s1_t)`
- `gap = median(s1_t-s2_t)`
- `rel_gap = median((s1_t-s2_t)/s1_t)`
- `persistence_mean`: lag 1..3 の左右 squared projection overlap の幾何平均を lag 間で平均
- `persistence_q10`: 隣接窓 joint projection overlap の10th percentile
- `persistence_min`: 同 overlap の最小値
- `T_iso = observed gap - circular-shift-null median gap`
- `T_pers = observed persistence_mean - circular-shift-null median`
- `T_floor = observed persistence_q10 - circular-shift-null median`

Phase 4 の最終文書は `T_iso` と continuity-floor diagnostics を併用しているが、単一の aggregation formula、positive threshold、preserve/destroy similarity は定義していない。

実装識別子:

- Git blob: `bf80ad90beecbbcc8c3638e3250e57a2b56eded4`
- file SHA-256: `faa6e602fe507eba37fb902ff191a0bcf348b225766ff4ed676e37438e8fc8cb`

### 3.2 Phase 5C: intervention-coordinate readout

文書上は Phase-4 operator の leading X-side singular vectorを `c=u_C^T deltaX` という intervention coordinate にし、RBF kernel ridge を用いる。Validation は baseline と mixture weight を選び、Confirmation は seeds `10000..10199` の200件で一度だけ実行された。

凍結選択:

- strongest baseline: `ew_nonlinear`
- Correlon weight: `0.25`
- strongest placebo: PCA
- placebo weight: `0.25`
- configuration SHA-256: `71b546652dc41fe2eca419b69534ac44a68a96e482461d1e4cf163612f830e39`

結果:

| metric | Correlon | PCA placebo |
|---|---:|---:|
| confirmation gain | `0.014303` | `0.018784` |
| specificity `G_C-G_P` | `-0.004481` | — |
| specificity CI95 | `[-0.005691,-0.003286]` | — |

Phase 5C は Correlon を distinct interventional representation として棄却し、descriptive persistent relation-mode detector のみを残した。

**再現性制約:** Phase 5C branch に追跡されているのは preregistration、aggregate decision JSON、validator JSON、validation CSV、results report だけである。SCM/fit/confirmation 実装、seed-level confirmation raw、bootstrap script は追跡されていない。そのため、集計値の相互整合性は監査できるが、repository-only で Phase 5C を再実行することはできない。

### 3.3 Timeless T3: CCA identity

`experiments/Timeless_T3_relation_state_bridge.py` は sample covariance を block 分割し、同じ

`Q = invsqrt(Cxx) @ Cxy @ invsqrt(Cyy)`

を SVD して rank-k cross block を再構成する。`cca_equivalent` は同じ reconstruction の copy として明示実装され、confirmation の差は `0.0` だった。Validation が選んだ rank は4、最強 admissible baseline は raw `Cxy` SVD rank 4。Confirmation 7,200 paired cases で Correlon mean generator error `0.088297` は raw-SVD baseline `0.087572` より悪く、relative gain CI95 は `[-0.007102,-0.006611]` だった。

この結果は現在の operator の novelty claim を閉じる。Correlon Zero で CCA/equivalent representation similarity を baseline から外してはならない。

## 4. 既存 generator / null / control inventory

| series | implemented worlds / controls | status and relevance |
|---|---|---|
| Phase 2A synthetic calibration | smooth continuum、square-root cusp、delta+continuum (`Z=.10,.25,.50`)、finite comb (`n=8,32`)、near-degenerate doublet | spectral pole classifier の校正。finite-comb false positive と delta/cusp confounding を既に確認 |
| Phase 4 time series | independent AR(1) null、rank-1 persistent、rank-2 degenerate、rank-1 switching、transient burst、common driver | Correlon Zero への直接の legacy generator 群。ただし rank-1 persistent と common driver は生成上同一 |
| Phase 4 matched null | `Y` の大きな circular time shiftを6回、各 space 内の autocorrelation/spectrum を保持 | `T_iso/T_pers/T_floor` の null subtraction に使用 |
| Phase 5C SCM (documented only) | nonlinear delayed direct SCM、approximately matched latent common-driver SCM、sham、PCA/circular-shift/random representations | 実装・raw data は repository にないため再利用不能。設計と aggregate verdict のみ継承可能 |
| Timeless T0 | stationary OU: same population covariance / different rotational dynamics | equal-time covarianceだけでは dynamics を同定できない no-go control |
| Timeless T1 | random SPD Gaussian-Gibbs covariance、known/unknown symplectic structure controls | covariance-to-flow の限定 positive control と additional-algebra ablation |
| Timeless T2 | Gibbs state、unrelated non-Gibbs、eigenvector-rotated Gibbs、diagonal-algebra restriction | modular-flow theorem reproduction と negative controls |
| Timeless T3 | coupled Gaussian Gibbs precision/covariance、ranks `{1,2,4}`、strengths `{.1,.25,.5,1}` | CCA identity、block diagonal、PCA、random rank、raw SVD、full-state oracle comparisons |

Phase 1/2B の Hubbard/J1-J2 exact-diagonalization は物理 model simulation であり、Correlon Zero の multivariate time-series generator として直接再利用できない。ただし「強い spectral feature は Correlon を意味しない」という既存の falsification evidence を与える。

## 5. 既存 baseline と threshold

### Phase 4

- Baseline/null: circular-shifted `Y` の同一 Correlon readout。
- Formal positive threshold はない。report は target/null の CI と相対比較を使用。
- Mean persistence alone は sufficient discriminator として明示的に棄却済み。

### Phase 5C

- Candidate baselines: rank-1 ARX、Volterra rank-1、nonlinear reduced-rank regression、local nonlinear RRR、exponentially weighted nonlinear regression、full-vector RBF、nonlinear state-space。
- Placebos: PCA、circular shift、random mode。
- Strong GO thresholds: `CI95_low(G_C)>0.10`、common/sham harm upper `<=0.02`、`CI95_low(G_C-G_P*)>0.05`。
- Result: gain gate と specificity gate が失敗。

### Timeless T3

- Baselines: block diagonal、PCA rank matched、random rank matched、raw cross-covariance SVD。CCA-equivalent rowは identity check、full covariance は oracle。
- Specificity threshold: lower CI of relative gain `>0.10`。
- Result: raw SVD が Correlon を上回り、CCA identity gate が成立。

## 6. 追跡 artifact と再現性ギャップ

ユーザーが期待した4 artifact はすべて最新基点に存在する。

- `experiments/Phase4_correlon_core_persistence.py`
- `PHASE4_RESULTS.md`
- `PHASE5C_PREREGISTRATION.md`
- `RESULTS_PHASE5C.md`

ただし以下は追跡されていない。

- `results/Phase4_correlon_core_100seeds_raw.csv`（Phase 4 script は生成するが branch にない）
- Phase 5C の実行コード
- Phase 5C の development/validation/confirmation seed-level raw
- Phase 5C の independent bootstrap 実装

Timeless series は実行コード、stage別 raw CSV、frozen config、decision、independent validator、figures を追跡しており、Phase 5C より repository-only reproducibility が高い。

## 7. T1 への拘束条件

1. Primary candidate は Phase-4 `Q_t` と既存 readout から選び、operator 自体を tuned replacement にしない。
2. 複数 component を使う場合、単一 scalar aggregation とすべての scale/regularization を confirmatory 前に明記する。
3. CCA を「optional」扱いにせず、現 operator と同一であることを baseline interpretation に反映する。
4. Phase-4 `rank1_persistent` を direct-coupling ground truth として再利用しない。direct coupling は明示的 SCM で新規実装し、common driver と construction-level に区別する。
5. Common-driver、matched low-rank、matched spectrum、matched autocorrelation を inconvenient null として除外しない。
6. Phase 5C の aggregate result は prior evidence として保持するが、raw/code 不在のため Correlon Zero の confirmatory data と混ぜない。
7. Positive threshold は pilot で尺度確認後に一度だけ凍結し、confirmatory 後は変更しない。

## 8. T0 verdict

**T0_AUDIT_COMPLETE_WITH_REPRODUCIBILITY_GAPS**

Frozen v1 candidate の出発点は存在するが、既存 evidence は uniqueness を既に強く否定している。特に common-driver construction identity、PCA placebo superiority、CCA identity、raw-SVD baseline superiority は、Correlon Zero が再検証すべき既知の falsification targets であり、支持証拠として再解釈してはならない。
