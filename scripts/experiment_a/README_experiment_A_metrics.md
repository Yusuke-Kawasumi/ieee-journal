了解、学生くん。
以下、**README / methods_note / 作業スレッドにそのまま貼れる DRAM/LTS proxy 整理メモ**として作るね。

---

# DRAM / LTS Proxy Metric Handling Memo

## 実験Aにおけるメモリトラフィック指標の扱い

## 1. 背景

実験Aでは、本来は Nsight Compute の direct DRAM byte counter を用いて、層ごとの実測DRAMトラフィックを取得し、Measured OI を計算する予定だった。

内部計画では、Measured OI は以下のように定義されている。

```text
Measured OI = FLOPs / measured_DRAM_traffic
measured_DRAM_traffic = Nsight Compute の dram__bytes.sum
```

また、`dram__bytes.sum` が取れない場合は、`dram__bytes_read.sum + dram__bytes_write.sum` や L2 miss bytes などの fallback を使う方針になっている。

しかし、今回の Jetson Orin Nano 環境では、以下の direct DRAM metrics は取得できなかった。

```text
dram__bytes.sum
dram__bytes_read.sum
dram__bytes_write.sum
```

そのため、実験Aでは全層で以下を代替指標として使用した。

```text
lts__t_bytes.sum
```

これは direct DRAM traffic ではなく、**L2 / LTS traffic に基づく memory traffic proxy** として扱う。ResNet18 / EfficientNet-B7 の全測定行で `measured_dram_metric_used = lts__t_bytes.sum`、`is_proxy_metric=True` として記録済み。

---

## 2. 採用する用語

以後、図表・README・Methods メモでは、以下の用語を使う。

### 推奨用語

```text
L2/LTS traffic proxy
L2/LTS-proxy measured OI
L2/LTS proxy bandwidth ratio
memory-system pressure proxy
```

日本語では：

```text
L2/LTSトラフィックproxy
L2/LTS proxyに基づくMeasured OI
L2/LTS proxy帯域比
メモリシステム圧力のproxy
```

---

## 3. 避けるべき表記

以下の表記は、物理DRAMを直接測ったように見えるため、論文・図表では避ける。

```text
DRAM traffic
Measured DRAM traffic
DRAM bandwidth utilization
Measured OI
```

ただし、CSV列名として既に `measured_DRAM_traffic` や `dram_bandwidth_utilization_pct` が存在する場合は、READMEで必ず以下を明記する。

```text
These columns are proxy-derived values based on lts__t_bytes.sum, not direct DRAM counters.
```

---

## 4. 今回の定義

### 4.1 Memory traffic proxy

```text
memory_traffic_proxy_bytes = lts__t_bytes.sum
```

より明示的には：

```text
lts_l2_traffic_proxy_bytes = lts__t_bytes.sum
```

意味：

```text
L2/LTS階層で観測されたトラフィック量を、メモリトラフィックのproxyとして用いる。
これは物理DRAM trafficそのものではない。
```

---

### 4.2 L2/LTS-proxy measured OI

```text
L2/LTS-proxy measured OI
= theoretical_FLOPs / lts__t_bytes.sum
```

今回の測定では、`theoretical_FLOPs` は batch size 8 を含む総FLOPsとして計算されている。したがって、Measured OI proxy も batch size 8 の単一層実行から得られた L2/LTS traffic proxy で割って算出する。

```text
L2/LTS-proxy measured OI
= theoretical_FLOPs(N=8) / L2_LTS_traffic_proxy_bytes(N=8)
```

---

### 4.3 L2/LTS proxy bandwidth ratio

既存の `dram_bandwidth_utilization_pct` に相当する値は、厳密には DRAM bandwidth utilization ではない。

計算としては：

```text
proxy_bandwidth_ratio_pct
= lts__t_bytes.sum / kernel_time_seconds / measured_peak_DRAM_bandwidth * 100
```

今回の measured peak DRAM bandwidth は：

```text
25.77 GB/s
```

したがって：

```text
proxy_bandwidth_ratio_pct
= lts__t_bytes.sum / kernel_time_seconds / 25.77e9 * 100
```

ただし、この値は L2/LTS traffic を peak DRAM bandwidth で正規化したものであり、物理DRAM帯域利用率ではない。

そのため、100%を超える場合がある。これは異常値ではなく、**分子がDRAM trafficではなくL2/LTS traffic proxyであることに由来する**。実験Aの測定まとめでも、LTS proxy bandwidth utilization が100%を超える層があるため、物理DRAM帯域利用率ではなく memory-system pressure proxy として解釈する必要があると整理している。

---

## 5. CSV列名の推奨整理

既存CSVに以下の列がある場合：

```text
measured_DRAM_traffic_bytes
dram_bandwidth_utilization_pct
measured_OI_FLOPs_per_byte
```

論文用・プロット用の統合CSVでは、可能なら以下の列名を追加する。

```text
memory_traffic_proxy_bytes
memory_traffic_proxy_metric_used
is_proxy_metric

measured_OI_proxy_FLOPs_per_byte
proxy_bandwidth_ratio_pct
proxy_bandwidth_denominator
```

値の対応：

```text
memory_traffic_proxy_bytes
= measured_DRAM_traffic_bytes

memory_traffic_proxy_metric_used
= lts__t_bytes.sum

is_proxy_metric
= True

measured_OI_proxy_FLOPs_per_byte
= measured_OI_FLOPs_per_byte

proxy_bandwidth_ratio_pct
= dram_bandwidth_utilization_pct

proxy_bandwidth_denominator
= measured_peak_DRAM_bandwidth_25.77_GB_per_s
```

既存列を削除する必要はない。
ただし、READMEで意味を明記する。

---

## 6. 図表での推奨軸ラベル

### Theoretical OI vs SM utilization

```text
x-axis: Theoretical OI [FLOPs/byte]
y-axis: SM utilization [%]
```

この図は問題なし。

---

### Theoretical OI vs memory-system pressure

避ける：

```text
DRAM bandwidth utilization [%]
```

推奨：

```text
L2/LTS proxy bandwidth ratio [%]
```

または：

```text
Memory-system pressure proxy [%]
```

---

### Theoretical OI vs measured OI

避ける：

```text
Measured OI [FLOPs/byte]
```

推奨：

```text
L2/LTS-proxy measured OI [FLOPs/byte]
```

または：

```text
Measured OI based on L2/LTS traffic proxy [FLOPs/byte]
```

---

### Roofline plot

Roofline 図では、横軸は原則として：

```text
Theoretical OI [FLOPs/byte]
```

を使う。

L2/LTS-proxy measured OI は補助図に回すのが安全。

---

## 7. Methods 用 英語メモ

そのまま Methods / README に使える文章：

```text
Direct DRAM byte counters such as dram__bytes.sum, dram__bytes_read.sum, and dram__bytes_write.sum were unavailable in the Jetson Orin Nano Nsight Compute environment. Therefore, lts__t_bytes.sum was used as a proxy for memory traffic. Metrics derived from this counter, including the measured OI proxy and the proxy bandwidth ratio, should be interpreted as L2/LTS traffic-based proxies rather than physical DRAM traffic or DRAM bandwidth utilization.
```

もう少し短い版：

```text
Because direct DRAM byte counters were unavailable on the Jetson Orin Nano, lts__t_bytes.sum was used as an L2/LTS traffic proxy. Proxy-derived OI and bandwidth ratios are interpreted as memory-system pressure indicators rather than physical DRAM utilization.
```

---

## 8. 日本語メモ

```text
Jetson Orin Nano の Nsight Compute 環境では、dram__bytes.sum などの direct DRAM byte counter が取得できなかった。そのため、本実験では lts__t_bytes.sum をメモリトラフィックの proxy として使用した。これに基づく Measured OI や帯域比は、物理DRAMトラフィックやDRAM帯域利用率ではなく、L2/LTSトラフィックに基づくメモリシステム圧力の proxy として解釈する。
```

---

## 9. 図キャプション例

### Theoretical OI vs SM utilization

```text
Layer-level Theoretical OI versus SM utilization. Layer types are distinguished by markers. Theoretical OI was computed from layer FLOPs and theoretical FP32 memory bytes.
```

### Theoretical OI vs L2/LTS proxy bandwidth ratio

```text
Layer-level Theoretical OI versus L2/LTS proxy bandwidth ratio. Since direct DRAM byte counters were unavailable on the Jetson Orin Nano, lts__t_bytes.sum was used as a memory-traffic proxy. Values above 100% indicate that the proxy traffic normalized by peak DRAM bandwidth exceeds the measured DRAM peak and should not be interpreted as physical DRAM utilization.
```

### Theoretical OI vs L2/LTS-proxy measured OI

```text
Comparison between Theoretical OI and L2/LTS-proxy measured OI. The measured OI proxy was computed by dividing theoretical layer FLOPs by lts__t_bytes.sum.
```

---

## 10. 最終方針

今回の実験Aでは、以下のように扱う。

```text
direct DRAM traffic:
  unavailable

actual metric used:
  lts__t_bytes.sum

interpretation:
  L2/LTS traffic proxy

Measured OI:
  L2/LTS-proxy measured OI

Bandwidth utilization:
  not physical DRAM bandwidth utilization
  but L2/LTS proxy bandwidth ratio

Main conclusion:
  use proxy metrics for relative layer-wise comparison,
  not as absolute physical DRAM traffic measurements.
```

---

## 11. チェックリスト

プロット作成前に確認すること。

```text
[ ] README / methods_note に proxy metric の説明を追加した
[ ] 図の軸名に "DRAM bandwidth utilization" と書いていない
[ ] "Measured OI" 単独表記を避けている
[ ] "L2/LTS-proxy measured OI" と明記している
[ ] 100%超えの帯域比を異常値扱いしていない
[ ] CSVに is_proxy_metric=True が残っている
[ ] metric_used列に lts__t_bytes.sum が残っている
```

---

このメモを `methods_note_dram_lts_proxy.txt` か `README_experiment_A_metrics.md` として保存しておくといい。
次は、この命名に合わせて **プロット用CSVの列名整理**をやるのが安全。
