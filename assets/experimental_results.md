# Results

These are the result tables of the paper and its Supporting Information.
The table numbers are the same as in the paper.
D<sub>s</sub> is the dataset built at the downscaling factor *s*.

In Tables 3–7, S4–S6, and S8–S11, every metric is computed once over the cells within the area of interest (AOI) of all samples of a validation split or a test set, at every output time step.
It is not averaged over the samples or over time.
Lower values of the RMSE and higher values of the other five metrics indicate better downscaling performance.

## Main text

### Table 3

Performance of TG-PFT trained on D<sub>8</sub> for each flood variable, with the bicubic interpolation of the coarse-grid flood map in brackets. Results are computed over all cells within the AOI in the upper part and within each flooding severity level in the middle and lower parts. Sign is the sign agreement of the flow velocities.

| Evaluation set | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | |
| Validation split | 0.054 (0.411) | 0.977 (−0.344) | 0.988 (0.348) | 0.970 (0.396) | 0.933 (0.555) | 0.907 (0.301) |
| 2-year | 0.066 (0.279) | 0.939 (−0.104) | 0.970 (0.249) | 0.933 (0.359) | 0.913 (0.455) | 0.856 (0.251) |
| 100-year | 0.062 (0.590) | 0.982 (−0.597) | 0.991 (0.335) | 0.970 (0.423) | 0.929 (0.625) | 0.903 (0.337) |
| 1000-year | 0.068 (0.705) | 0.985 (−0.616) | 0.993 (0.400) | 0.970 (0.457) | 0.934 (0.719) | 0.908 (0.388) |
| **Flow velocity in the *x* direction** (m/s) | | | | | | |
| Validation split | 0.035 (0.162) | 0.917 (−0.821) | 0.958 (0.166) | 0.915 (0.252) | 0.877 (0.307) | 0.811 (0.160) |
| 2-year | 0.033 (0.128) | 0.884 (−0.792) | 0.941 (0.056) | 0.872 (0.168) | 0.857 (0.138) | 0.761 (0.082) |
| 100-year | 0.047 (0.220) | 0.912 (−0.959) | 0.955 (0.174) | 0.914 (0.262) | 0.864 (0.370) | 0.799 (0.181) |
| 1000-year | 0.056 (0.254) | 0.910 (−0.868) | 0.954 (0.248) | 0.919 (0.312) | 0.867 (0.521) | 0.806 (0.243) |
| **Flow velocity in the *y* direction** (m/s) | | | | | | |
| Validation split | 0.038 (0.167) | 0.925 (−0.454) | 0.962 (0.221) | 0.916 (0.288) | 0.877 (0.337) | 0.812 (0.184) |
| 2-year | 0.037 (0.132) | 0.882 (−0.483) | 0.940 (0.082) | 0.866 (0.198) | 0.856 (0.157) | 0.755 (0.096) |
| 100-year | 0.050 (0.225) | 0.925 (−0.517) | 0.962 (0.233) | 0.919 (0.296) | 0.872 (0.402) | 0.810 (0.206) |
| 1000-year | 0.061 (0.259) | 0.922 (−0.422) | 0.960 (0.328) | 0.923 (0.359) | 0.875 (0.569) | 0.815 (0.282) |

| Evaluation set | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | | | | |
| Validation split | 0.870 | 0.795 | 0.135 | 0.815 | 0.819 | 0.205 | 0.957 | 0.940 | 0.417 |
| 2-year | 0.810 | 0.744 | 0.212 | 0.723 | 0.737 | 0.314 | 0.882 | 0.905 | 0.599 |
| 100-year | 0.808 | 0.738 | 0.151 | 0.751 | 0.759 | 0.217 | 0.963 | 0.929 | 0.362 |
| 1000-year | 0.794 | 0.725 | 0.159 | 0.750 | 0.751 | 0.225 | 0.964 | 0.942 | 0.346 |

| Evaluation set | Slow flow<br>Precision | Slow flow<br>Recall | Slow flow<br>RMSE | Moderate flow<br>Precision | Moderate flow<br>Recall | Moderate flow<br>RMSE | Fast flow<br>Precision | Fast flow<br>Recall | Fast flow<br>RMSE | Sign |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Flow velocity in the *x* direction** (m/s) | | | | | | | | | | |
| Validation split | 0.756 | 0.733 | 0.139 | 0.743 | 0.717 | 0.240 | 0.915 | 0.846 | 0.454 | 0.997 |
| 2-year | 0.675 | 0.666 | 0.169 | 0.672 | 0.661 | 0.269 | 0.852 | 0.825 | 0.461 | 0.997 |
| 100-year | 0.734 | 0.701 | 0.152 | 0.718 | 0.688 | 0.257 | 0.915 | 0.834 | 0.468 | 0.997 |
| 1000-year | 0.748 | 0.714 | 0.151 | 0.713 | 0.686 | 0.262 | 0.916 | 0.831 | 0.494 | 0.996 |
| **Flow velocity in the *y* direction** (m/s) | | | | | | | | | | |
| Validation split | 0.736 | 0.720 | 0.150 | 0.721 | 0.686 | 0.262 | 0.918 | 0.855 | 0.440 | 0.997 |
| 2-year | 0.650 | 0.647 | 0.180 | 0.646 | 0.639 | 0.287 | 0.850 | 0.830 | 0.470 | 0.997 |
| 100-year | 0.728 | 0.704 | 0.156 | 0.700 | 0.667 | 0.266 | 0.925 | 0.848 | 0.448 | 0.997 |
| 1000-year | 0.727 | 0.710 | 0.161 | 0.699 | 0.657 | 0.274 | 0.922 | 0.841 | 0.474 | 0.996 |

### Table 4

Performance of TG-PFT and of UNet (He et al., 2023), SwinFlood (Song et al., 2025), and RSwinUNet (Song & Guan, 2026) in downscaling the water depth on the 100-year test set, all trained on D<sub>8</sub>. L<sub>2</sub> and L<sub>1</sub> are the loss functions of the three models in their papers, and our loss is the loss function of Section 3.3 of the paper. In each column, the best value is shown in bold, the second best is underlined, and the worst, that of the bicubic interpolation of the coarse-grid flood map, is shown in italics.

| Model | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (ours) | **0.062** | **0.982** | **0.991** | **0.970** | **0.929** | **0.903** |
| Bicubic interpolation | *0.590* | *−0.597* | *0.335* | *0.423* | *0.625* | *0.337* |
| UNet, L<sub>2</sub> | <ins>0.087</ins> | <ins>0.965</ins> | <ins>0.983</ins> | 0.899 | 0.915 | 0.830 |
| UNet, our loss | 0.091 | 0.962 | 0.981 | <ins>0.945</ins> | 0.902 | <ins>0.858</ins> |
| SwinFlood, L<sub>2</sub> | 0.104 | 0.950 | 0.975 | 0.848 | <ins>0.916</ins> | 0.787 |
| SwinFlood, our loss | 0.115 | 0.939 | 0.969 | 0.885 | 0.882 | 0.791 |
| RSwinUNet, L<sub>1</sub> | 0.163 | 0.878 | 0.938 | 0.903 | 0.803 | 0.740 |
| RSwinUNet, our loss | 0.158 | 0.886 | 0.942 | 0.901 | 0.830 | 0.761 |

| Model | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (ours) | **0.808** | **0.738** | **0.151** | **0.751** | **0.759** | **0.217** | **0.963** | **0.929** | **0.362** |
| Bicubic interpolation | *0.143* | *0.274* | *1.081* | *0.126* | *0.208* | *1.249* | *0.340* | *0.383* | *2.443* |
| UNet, L<sub>2</sub> | 0.593 | <ins>0.666</ins> | <ins>0.182</ins> | 0.631 | 0.654 | 0.270 | 0.944 | 0.889 | <ins>0.528</ins> |
| UNet, our loss | <ins>0.721</ins> | 0.653 | 0.205 | <ins>0.676</ins> | 0.684 | 0.297 | 0.939 | <ins>0.904</ins> | 0.542 |
| SwinFlood, L<sub>2</sub> | 0.514 | 0.649 | 0.194 | 0.567 | <ins>0.685</ins> | <ins>0.258</ins> | 0.954 | 0.867 | 0.644 |
| SwinFlood, our loss | 0.555 | 0.633 | 0.192 | 0.610 | 0.646 | 0.304 | <ins>0.955</ins> | 0.838 | 0.730 |
| RSwinUNet, L<sub>1</sub> | 0.577 | 0.467 | 0.331 | 0.546 | 0.500 | 0.500 | 0.884 | 0.823 | 0.983 |
| RSwinUNet, our loss | 0.577 | 0.509 | 0.326 | 0.553 | 0.515 | 0.486 | 0.885 | 0.834 | 0.944 |

### Table 5

Performance of TG-PFT for the three flood variables and of UNet, SwinFlood, and RSwinUNet with their own loss functions for the water depth on Cyclone Gabrielle, all trained on D<sub>8</sub>. For each flood variable, the best value in each column is shown in bold and the worst, that of the bicubic interpolation of the coarse-grid flood map, in italics, and for the water depth, the second best is underlined.

| Model | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | |
| TG-PFT (ours) | **0.154** | **0.804** | **0.905** | **0.857** | **0.878** | **0.766** |
| Bicubic interpolation | *0.619* | *−2.191* | *0.227* | *0.309* | *0.595* | *0.255* |
| UNet, L<sub>2</sub> | <ins>0.168</ins> | <ins>0.765</ins> | <ins>0.887</ins> | <ins>0.773</ins> | <ins>0.872</ins> | <ins>0.694</ins> |
| SwinFlood, L<sub>2</sub> | 0.174 | 0.747 | 0.879 | 0.736 | <ins>0.872</ins> | 0.665 |
| RSwinUNet, L<sub>1</sub> | 0.238 | 0.527 | 0.798 | 0.752 | 0.784 | 0.623 |
| **Flow velocity in the *x* direction** (m/s) | | | | | | |
| TG-PFT (ours) | **0.059** | **0.719** | **0.858** | **0.742** | **0.783** | **0.615** |
| Bicubic interpolation | *0.196* | *−2.163* | *0.114* | *0.165* | *0.337* | *0.124* |
| **Flow velocity in the *y* direction** (m/s) | | | | | | |
| TG-PFT (ours) | **0.070** | **0.684** | **0.844** | **0.721** | **0.783** | **0.602** |
| Bicubic interpolation | *0.195* | *−1.434* | *0.144* | *0.182* | *0.357* | *0.137* |

### Table 6

Ablation studies of TG-PFT trained on D<sub>8</sub> for the water depth on the 100-year test set. PFT is TG-PFT without the multi-scale topographic guidance and the global residual connection. The other rows remove the component they specify from TG-PFT, except that the PFT blocks are replaced by residual convolutions. In each column, the best value is shown in bold, the second best is underlined, and the worst is displayed in italics.

| Configuration | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (full) | **0.062** | **0.982** | **0.991** | **0.970** | **0.929** | **0.903** |
| **Architecture** | | | | | | |
| PFT | 0.094 | 0.959 | 0.980 | 0.940 | 0.896 | 0.848 |
| w/o PFT blocks | *0.142* | *0.907* | *0.953* | *0.915* | *0.848* | *0.786* |
| w/o multi-scale guidance | 0.079 | 0.971 | 0.986 | 0.958 | 0.909 | 0.874 |
| w/o residual connection | 0.071 | <ins>0.977</ins> | 0.988 | <ins>0.964</ins> | 0.916 | <ins>0.886</ins> |
| w/o AOI gate | <ins>0.070</ins> | <ins>0.977</ins> | <ins>0.989</ins> | 0.962 | <ins>0.918</ins> | 0.885 |
| **Loss function and sampling** | | | | | | |
| w/o classification term | 0.078 | 0.972 | 0.986 | 0.961 | 0.905 | 0.873 |
| w/o balanced sampling | 0.076 | 0.973 | 0.987 | 0.958 | 0.904 | 0.870 |

| Configuration | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (full) | **0.808** | **0.738** | **0.151** | **0.751** | **0.759** | **0.217** | **0.963** | **0.929** | **0.362** |
| **Architecture** | | | | | | | | | |
| PFT | 0.681 | 0.634 | 0.199 | 0.630 | 0.667 | 0.290 | 0.948 | 0.876 | 0.582 |
| w/o PFT blocks | *0.617* | *0.535* | *0.288* | *0.567* | *0.548* | *0.427* | *0.894* | *0.849* | *0.869* |
| w/o multi-scale guidance | 0.748 | 0.678 | 0.184 | 0.692 | 0.699 | 0.269 | 0.952 | 0.906 | 0.473 |
| w/o residual connection | <ins>0.784</ins> | <ins>0.704</ins> | <ins>0.164</ins> | <ins>0.725</ins> | 0.732 | 0.242 | 0.956 | <ins>0.919</ins> | 0.424 |
| w/o AOI gate | 0.775 | <ins>0.704</ins> | 0.165 | 0.723 | <ins>0.733</ins> | <ins>0.241</ins> | <ins>0.957</ins> | <ins>0.919</ins> | <ins>0.418</ins> |
| **Loss function and sampling** | | | | | | | | | |
| w/o classification term | 0.773 | 0.679 | 0.177 | 0.719 | 0.721 | 0.263 | 0.954 | 0.914 | 0.466 |
| w/o balanced sampling | 0.745 | 0.657 | 0.177 | 0.686 | 0.692 | 0.263 | 0.950 | 0.910 | 0.454 |

### Table 7

Performance of TG-PFT trained on D<sub>2</sub>, D<sub>4</sub>, D<sub>8</sub>, and D<sub>16</sub> at the downscaling factors of 2, 4, 8, and 16 for the water depth on the 100-year test set. The brackets after each downscaling factor give the resolution of the coarse grid, and those in the other columns give the bicubic interpolation of the coarse-grid flood map for comparison.

| Downscaling factor | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| *s* = 2 (16 m) | 0.074 (0.157) | 0.975 (0.886) | 0.987 (0.942) | 0.963 (0.787) | 0.930 (0.890) | 0.898 (0.717) |
| *s* = 4 (32 m) | 0.074 (0.346) | 0.975 (0.452) | 0.987 (0.733) | 0.967 (0.642) | 0.919 (0.802) | 0.890 (0.554) |
| *s* = 8 (64 m) | 0.062 (0.590) | 0.982 (−0.597) | 0.991 (0.335) | 0.970 (0.423) | 0.929 (0.625) | 0.903 (0.337) |
| *s* = 16 (128 m) | 0.053 (0.671) | 0.987 (−1.066) | 0.994 (0.152) | 0.974 (0.179) | 0.941 (0.566) | 0.917 (0.158) |

### Table 8

Computational cost of TG-PFT, UNet, SwinFlood, and RSwinUNet at *s* = 8 on the 100-year test set, against the fine-grid simulation. MACs and inference are for the whole test set. Coarse grid and fine-grid simulation are the wall-clock times of the BG-Flood simulations on the 64 m and 8 m grids, total is the coarse grid plus the inference, and speed-up is the fine-grid simulation divided by the total.

| Model | Parameters (M) | MACs (T) | Inference (s) | Coarse grid (s) | Total (s) | Speed-up |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (ours) | 3.270 | 611 | 265.1 | 1330 | 1,595 | 47.79 |
| UNet | 8.053 | 1537 | 235.8 | 1330 | 1,566 | 48.69 |
| SwinFlood | 3.720 | 275 | 128.0 | 1330 | 1,458 | 52.29 |
| RSwinUNet | 36.960 | 423 | 126.5 | 1330 | 1,457 | 52.34 |
| Fine-grid simulation | – | – | – | – | 76,234 | 1.00 |

## Supporting Information

### Table S4

Performance of TG-PFT trained on D<sub>8</sub> for each flood variable on the validation split and the six test sets, with the bicubic interpolation of the coarse-grid flood map in brackets. Results are computed over all cells within the AOI.

| Evaluation set | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | |
| Validation split | 0.054 (0.411) | 0.977 (−0.344) | 0.988 (0.348) | 0.970 (0.396) | 0.933 (0.555) | 0.907 (0.301) |
| 2-year | 0.066 (0.279) | 0.939 (−0.104) | 0.970 (0.249) | 0.933 (0.359) | 0.913 (0.455) | 0.856 (0.251) |
| 5-year | 0.058 (0.351) | 0.967 (−0.223) | 0.983 (0.276) | 0.945 (0.368) | 0.926 (0.496) | 0.879 (0.268) |
| 10-year | 0.056 (0.405) | 0.975 (−0.316) | 0.987 (0.294) | 0.957 (0.380) | 0.926 (0.528) | 0.889 (0.283) |
| 50-year | 0.063 (0.537) | 0.979 (−0.532) | 0.989 (0.322) | 0.963 (0.411) | 0.926 (0.595) | 0.894 (0.321) |
| 100-year | 0.062 (0.590) | 0.982 (−0.597) | 0.991 (0.335) | 0.970 (0.423) | 0.929 (0.625) | 0.903 (0.337) |
| 1000-year | 0.068 (0.705) | 0.985 (−0.616) | 0.993 (0.400) | 0.970 (0.457) | 0.934 (0.719) | 0.908 (0.388) |
| **Flow velocity in the *x* direction** (m/s) | | | | | | |
| Validation split | 0.035 (0.162) | 0.917 (−0.821) | 0.958 (0.166) | 0.915 (0.252) | 0.877 (0.307) | 0.811 (0.160) |
| 2-year | 0.033 (0.128) | 0.884 (−0.792) | 0.941 (0.056) | 0.872 (0.168) | 0.857 (0.138) | 0.761 (0.082) |
| 5-year | 0.035 (0.154) | 0.905 (−0.879) | 0.951 (0.088) | 0.900 (0.199) | 0.861 (0.193) | 0.786 (0.108) |
| 10-year | 0.038 (0.172) | 0.907 (−0.925) | 0.952 (0.108) | 0.907 (0.215) | 0.860 (0.231) | 0.790 (0.125) |
| 50-year | 0.044 (0.207) | 0.911 (−0.972) | 0.955 (0.153) | 0.908 (0.247) | 0.861 (0.324) | 0.792 (0.163) |
| 100-year | 0.047 (0.220) | 0.912 (−0.959) | 0.955 (0.174) | 0.914 (0.262) | 0.864 (0.370) | 0.799 (0.181) |
| 1000-year | 0.056 (0.254) | 0.910 (−0.868) | 0.954 (0.248) | 0.919 (0.312) | 0.867 (0.521) | 0.806 (0.243) |
| **Flow velocity in the *y* direction** (m/s) | | | | | | |
| Validation split | 0.038 (0.167) | 0.925 (−0.454) | 0.962 (0.221) | 0.916 (0.288) | 0.877 (0.337) | 0.812 (0.184) |
| 2-year | 0.037 (0.132) | 0.882 (−0.483) | 0.940 (0.082) | 0.866 (0.198) | 0.856 (0.157) | 0.755 (0.096) |
| 5-year | 0.038 (0.159) | 0.912 (−0.536) | 0.955 (0.120) | 0.897 (0.224) | 0.867 (0.209) | 0.788 (0.121) |
| 10-year | 0.041 (0.175) | 0.916 (−0.534) | 0.957 (0.149) | 0.906 (0.239) | 0.864 (0.247) | 0.793 (0.138) |
| 50-year | 0.048 (0.212) | 0.921 (−0.541) | 0.960 (0.206) | 0.909 (0.277) | 0.862 (0.350) | 0.793 (0.183) |
| 100-year | 0.050 (0.225) | 0.925 (−0.517) | 0.962 (0.233) | 0.919 (0.296) | 0.872 (0.402) | 0.810 (0.206) |
| 1000-year | 0.061 (0.259) | 0.922 (−0.422) | 0.960 (0.328) | 0.923 (0.359) | 0.875 (0.569) | 0.815 (0.282) |

### Table S5

Performance of TG-PFT at each flooding severity level on the validation split and the six test sets for each flood variable, with the bicubic interpolation of the coarse-grid flood map in brackets. Sign is the sign agreement of the flow velocities.

| Evaluation set | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | | | | |
| Validation split | 0.870 (0.149) | 0.795 (0.275) | 0.135 (0.803) | 0.815 (0.123) | 0.819 (0.177) | 0.205 (1.021) | 0.957 (0.354) | 0.940 (0.346) | 0.417 (2.480) |
| 2-year | 0.810 (0.158) | 0.744 (0.278) | 0.212 (0.445) | 0.723 (0.126) | 0.737 (0.151) | 0.314 (0.674) | 0.882 (0.278) | 0.905 (0.189) | 0.599 (2.664) |
| 5-year | 0.814 (0.140) | 0.757 (0.255) | 0.163 (0.618) | 0.754 (0.111) | 0.770 (0.148) | 0.241 (0.787) | 0.920 (0.274) | 0.926 (0.240) | 0.456 (2.565) |
| 10-year | 0.822 (0.140) | 0.754 (0.255) | 0.152 (0.738) | 0.752 (0.112) | 0.767 (0.164) | 0.228 (0.897) | 0.937 (0.277) | 0.926 (0.267) | 0.408 (2.500) |
| 50-year | 0.801 (0.145) | 0.735 (0.269) | 0.160 (0.985) | 0.740 (0.123) | 0.754 (0.195) | 0.233 (1.150) | 0.954 (0.315) | 0.924 (0.342) | 0.386 (2.446) |
| 100-year | 0.808 (0.143) | 0.738 (0.274) | 0.151 (1.081) | 0.751 (0.126) | 0.759 (0.208) | 0.217 (1.249) | 0.963 (0.340) | 0.929 (0.383) | 0.362 (2.443) |
| 1000-year | 0.794 (0.143) | 0.725 (0.284) | 0.159 (1.295) | 0.750 (0.129) | 0.751 (0.231) | 0.225 (1.459) | 0.964 (0.418) | 0.942 (0.532) | 0.346 (2.426) |

| Evaluation set | Slow flow<br>Precision | Slow flow<br>Recall | Slow flow<br>RMSE | Moderate flow<br>Precision | Moderate flow<br>Recall | Moderate flow<br>RMSE | Fast flow<br>Precision | Fast flow<br>Recall | Fast flow<br>RMSE | Sign |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Flow velocity in the *x* direction** (m/s) | | | | | | | | | | |
| Validation split | 0.756 (0.125) | 0.733 (0.203) | 0.139 (0.310) | 0.743 (0.101) | 0.717 (0.089) | 0.240 (0.668) | 0.915 (0.157) | 0.846 (0.114) | 0.454 (1.616) | 0.997 |
| 2-year | 0.675 (0.089) | 0.666 (0.104) | 0.169 (0.283) | 0.672 (0.046) | 0.661 (0.021) | 0.269 (0.703) | 0.852 (0.017) | 0.825 (0.008) | 0.461 (1.665) | 0.997 |
| 5-year | 0.705 (0.099) | 0.681 (0.136) | 0.157 (0.287) | 0.711 (0.075) | 0.685 (0.045) | 0.254 (0.690) | 0.893 (0.052) | 0.826 (0.029) | 0.454 (1.651) | 0.997 |
| 10-year | 0.713 (0.105) | 0.681 (0.160) | 0.157 (0.292) | 0.718 (0.089) | 0.689 (0.061) | 0.256 (0.682) | 0.902 (0.085) | 0.829 (0.052) | 0.460 (1.643) | 0.997 |
| 50-year | 0.725 (0.118) | 0.693 (0.215) | 0.153 (0.302) | 0.719 (0.109) | 0.694 (0.098) | 0.253 (0.661) | 0.912 (0.154) | 0.835 (0.110) | 0.461 (1.614) | 0.997 |
| 100-year | 0.734 (0.128) | 0.701 (0.251) | 0.152 (0.310) | 0.718 (0.116) | 0.688 (0.117) | 0.257 (0.651) | 0.915 (0.176) | 0.834 (0.133) | 0.468 (1.598) | 0.997 |
| 1000-year | 0.748 (0.159) | 0.714 (0.349) | 0.151 (0.350) | 0.713 (0.129) | 0.686 (0.177) | 0.262 (0.628) | 0.916 (0.231) | 0.831 (0.220) | 0.494 (1.534) | 0.996 |
| **Flow velocity in the *y* direction** (m/s) | | | | | | | | | | |
| Validation split | 0.736 (0.135) | 0.720 (0.221) | 0.150 (0.313) | 0.721 (0.104) | 0.686 (0.093) | 0.262 (0.667) | 0.918 (0.238) | 0.855 (0.148) | 0.440 (1.668) | 0.997 |
| 2-year | 0.650 (0.087) | 0.647 (0.106) | 0.180 (0.281) | 0.646 (0.060) | 0.639 (0.028) | 0.287 (0.703) | 0.850 (0.034) | 0.830 (0.012) | 0.470 (1.730) | 0.997 |
| 5-year | 0.686 (0.097) | 0.674 (0.139) | 0.165 (0.283) | 0.687 (0.080) | 0.660 (0.049) | 0.271 (0.692) | 0.891 (0.100) | 0.841 (0.044) | 0.449 (1.712) | 0.997 |
| 10-year | 0.695 (0.105) | 0.674 (0.165) | 0.165 (0.288) | 0.693 (0.090) | 0.665 (0.064) | 0.271 (0.685) | 0.909 (0.148) | 0.841 (0.072) | 0.448 (1.697) | 0.997 |
| 50-year | 0.710 (0.133) | 0.680 (0.247) | 0.161 (0.296) | 0.689 (0.109) | 0.664 (0.102) | 0.270 (0.663) | 0.920 (0.240) | 0.846 (0.145) | 0.447 (1.659) | 0.997 |
| 100-year | 0.728 (0.146) | 0.704 (0.286) | 0.156 (0.305) | 0.700 (0.114) | 0.667 (0.120) | 0.266 (0.651) | 0.925 (0.268) | 0.848 (0.177) | 0.448 (1.639) | 0.997 |
| 1000-year | 0.727 (0.166) | 0.710 (0.358) | 0.161 (0.364) | 0.699 (0.138) | 0.657 (0.195) | 0.274 (0.614) | 0.922 (0.321) | 0.841 (0.285) | 0.474 (1.552) | 0.996 |

### Table S6

Mean and sample standard deviation of the performance of TG-PFT for the water depth on the validation split and the 100-year test set over three trainings with random seeds of 61, 1202, and 2027. Table 3 reports the results of the training with random seed 61.

| Evaluation set | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Validation split | 0.055 ± 0.001 | 0.976 ± 0.001 | 0.988 ± 0.000 | 0.969 ± 0.001 | 0.931 ± 0.002 | 0.904 ± 0.003 |
| 100-year | 0.063 ± 0.002 | 0.982 ± 0.001 | 0.991 ± 0.000 | 0.969 ± 0.001 | 0.929 ± 0.000 | 0.902 ± 0.001 |

| Evaluation set | Nonflooding<br>Precision | Nonflooding<br>Recall | Nonflooding<br>RMSE | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Validation split | 0.998 ± 0.000 | 0.999 ± 0.000 | 0.017 ± 0.000 | 0.861 ± 0.008 | 0.786 ± 0.008 | 0.141 ± 0.005 |
| 100-year | 0.997 ± 0.000 | 0.999 ± 0.000 | 0.021 ± 0.002 | 0.807 ± 0.001 | 0.737 ± 0.001 | 0.153 ± 0.003 |

| Evaluation set | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Validation split | 0.804 ± 0.010 | 0.805 ± 0.012 | 0.213 ± 0.008 | 0.953 ± 0.003 | 0.937 ± 0.004 | 0.424 ± 0.008 |
| 100-year | 0.750 ± 0.001 | 0.760 ± 0.001 | 0.219 ± 0.004 | 0.961 ± 0.001 | 0.930 ± 0.001 | 0.365 ± 0.006 |

### Table S7

Diagnostics of the flood maps of the 100-year test set at 28 h over the Gisborne catchments and in the valley, floodplain, and urban windows of Figure 5 of the paper. The columns are defined in Text S7 of the Supporting Information.

| Region | Flooded or flowing area (%)<br>Coarse | Flooded or flowing area (%)<br>Simulated | Flooded or flowing area (%)<br>Predicted | Mean simulated | CSI | RMSE | RMSE\* |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | | |
| Gisborne catchments | 7.7 | 7.9 | 7.6 | 1.72 | 0.928 | 0.069 | 0.244 |
| Valley | 3.7 | 2.3 | 2.2 | 2.47 | 0.875 | 0.092 | 0.592 |
| Floodplain | 16.7 | 41.2 | 38.9 | 1.98 | 0.939 | 0.128 | 0.199 |
| Urban | 33.2 | 39.0 | 37.8 | 1.45 | 0.961 | 0.069 | 0.111 |
| **Flow velocity in the *x* direction** (m/s) | | | | | | | |
| Gisborne catchments | 3.8 | 5.0 | 4.6 | 0.76 | 0.830 | 0.061 | 0.264 |
| Valley | 1.1 | 2.0 | 1.9 | 1.34 | 0.798 | 0.091 | 0.615 |
| Floodplain | 6.3 | 26.6 | 23.5 | 0.58 | 0.823 | 0.090 | 0.171 |
| Urban | 14.0 | 12.8 | 11.9 | 0.63 | 0.818 | 0.045 | 0.119 |
| **Flow velocity in the *y* direction** (m/s) | | | | | | | |
| Gisborne catchments | 4.1 | 5.5 | 5.1 | 0.88 | 0.852 | 0.064 | 0.265 |
| Valley | 0.9 | 2.0 | 1.9 | 1.31 | 0.799 | 0.090 | 0.607 |
| Floodplain | 6.4 | 29.8 | 26.8 | 0.85 | 0.861 | 0.096 | 0.174 |
| Urban | 18.4 | 15.6 | 13.7 | 0.69 | 0.829 | 0.050 | 0.122 |

| Region | Squared error (%)<br>Both | Squared error (%)<br>Simulated | Squared error (%)<br>Predicted | Within (%)<br>0.2 | Within (%)<br>0.5 | Bias (%) | Sign (%) | Flooded or flowing<br>cells (%)<br>Edge | Flooded or flowing<br>cells (%)<br>Near threshold |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | | | | |
| Gisborne catchments | 81 | 14 | 5 | 79.6 | 95.8 | −4.0 | – | 21.4 | 8.6 |
| Valley | 86 | 11 | 3 | 36.9 | 69.2 | −6.3 | – | 59.4 | 5.5 |
| Floodplain | 96 | 4 | 0 | 75.2 | 97.3 | −6.5 | – | 6.3 | 8.5 |
| Urban | 89 | 9 | 2 | 95.3 | 99.1 | −2.8 | – | 9.4 | 7.0 |
| **Flow velocity in the *x* direction** (m/s) | | | | | | | | | |
| Gisborne catchments | 54 | 39 | 6 | 79.5 | 94.9 | −5.9 | 99.7 | 30.7 | 18.0 |
| Valley | 70 | 26 | 4 | 42.5 | 76.0 | −8.0 | 97.9 | 63.7 | 8.9 |
| Floodplain | 44 | 51 | 4 | 85.8 | 97.6 | −7.7 | 99.8 | 12.1 | 19.5 |
| Urban | 60 | 27 | 6 | 92.7 | 99.2 | −4.6 | 99.9 | 17.6 | 31.9 |
| **Flow velocity in the *y* direction** (m/s) | | | | | | | | | |
| Gisborne catchments | 58 | 36 | 6 | 79.0 | 94.9 | −6.5 | 99.8 | 26.7 | 15.8 |
| Valley | 69 | 27 | 4 | 43.6 | 75.1 | −9.5 | 97.6 | 64.8 | 9.2 |
| Floodplain | 63 | 33 | 3 | 84.7 | 98.0 | −8.4 | 99.9 | 9.2 | 15.2 |
| Urban | 65 | 26 | 4 | 92.5 | 99.2 | −6.2 | 100.0 | 14.1 | 34.6 |

### Table S8

As Table 4, but on the validation split.

| Model | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (ours) | **0.054** | **0.977** | **0.988** | **0.970** | **0.933** | **0.907** |
| Bicubic interpolation | *0.411* | *−0.344* | *0.348* | *0.396* | *0.555* | *0.301* |
| UNet, L<sub>2</sub> | <ins>0.080</ins> | <ins>0.950</ins> | <ins>0.975</ins> | 0.859 | <ins>0.911</ins> | 0.793 |
| UNet, our loss | 0.082 | 0.947 | 0.973 | <ins>0.946</ins> | 0.892 | <ins>0.849</ins> |
| SwinFlood, L<sub>2</sub> | 0.088 | 0.939 | 0.969 | 0.829 | 0.910 | 0.766 |
| SwinFlood, our loss | 0.098 | 0.924 | 0.961 | 0.874 | 0.875 | 0.777 |
| RSwinUNet, L<sub>1</sub> | 0.135 | 0.854 | 0.925 | 0.909 | 0.794 | 0.736 |
| RSwinUNet, our loss | 0.130 | 0.866 | 0.931 | 0.905 | 0.823 | 0.758 |

| Model | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (ours) | **0.870** | **0.795** | **0.135** | **0.815** | **0.819** | **0.205** | **0.957** | **0.940** | **0.417** |
| Bicubic interpolation | *0.149* | *0.275* | *0.803* | *0.123* | *0.177* | *1.021* | *0.354* | *0.346* | *2.480* |
| UNet, L<sub>2</sub> | 0.580 | <ins>0.692</ins> | <ins>0.176</ins> | 0.644 | 0.674 | 0.270 | 0.928 | 0.877 | <ins>0.632</ins> |
| UNet, our loss | <ins>0.767</ins> | 0.690 | 0.187 | <ins>0.705</ins> | <ins>0.704</ins> | 0.291 | 0.930 | <ins>0.887</ins> | 0.657 |
| SwinFlood, L<sub>2</sub> | 0.543 | 0.681 | 0.187 | 0.588 | 0.700 | <ins>0.260</ins> | 0.938 | 0.850 | 0.703 |
| SwinFlood, our loss | 0.594 | 0.678 | 0.179 | 0.639 | 0.658 | 0.306 | <ins>0.944</ins> | 0.811 | 0.805 |
| RSwinUNet, L<sub>1</sub> | 0.654 | 0.542 | 0.268 | 0.587 | 0.541 | 0.443 | 0.883 | 0.784 | 1.108 |
| RSwinUNet, our loss | 0.642 | 0.576 | 0.265 | 0.593 | 0.559 | 0.433 | 0.883 | 0.799 | 1.054 |

### Table S9

As Table 5, with the three models also trained with our loss and with the results at the three flooding severity levels.

| Model | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | |
| TG-PFT (ours) | **0.154** | **0.804** | **0.905** | **0.857** | **0.878** | **0.766** |
| Bicubic interpolation | *0.619* | *−2.191* | *0.227* | *0.309* | *0.595* | *0.255* |
| UNet, L<sub>2</sub> | 0.168 | 0.765 | 0.887 | 0.773 | <ins>0.872</ins> | 0.694 |
| UNet, our loss | <ins>0.163</ins> | <ins>0.780</ins> | <ins>0.894</ins> | <ins>0.836</ins> | 0.859 | <ins>0.735</ins> |
| SwinFlood, L<sub>2</sub> | 0.174 | 0.747 | 0.879 | 0.736 | <ins>0.872</ins> | 0.665 |
| SwinFlood, our loss | 0.179 | 0.733 | 0.869 | 0.774 | 0.841 | 0.675 |
| RSwinUNet, L<sub>1</sub> | 0.238 | 0.527 | 0.798 | 0.752 | 0.784 | 0.623 |
| RSwinUNet, our loss | 0.234 | 0.546 | 0.808 | 0.748 | 0.806 | 0.634 |
| **Flow velocity in the *x* direction** (m/s) | | | | | | |
| TG-PFT (ours) | **0.059** | **0.719** | **0.858** | **0.742** | **0.783** | **0.615** |
| Bicubic interpolation | *0.196* | *−2.163* | *0.114* | *0.165* | *0.337* | *0.124* |
| **Flow velocity in the *y* direction** (m/s) | | | | | | |
| TG-PFT (ours) | **0.070** | **0.684** | **0.844** | **0.721** | **0.783** | **0.602** |
| Bicubic interpolation | *0.195* | *−1.434* | *0.144* | *0.182* | *0.357* | *0.137* |

| Model | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Water depth** (m) | | | | | | | | | |
| TG-PFT (ours) | **0.568** | **0.569** | **0.413** | **0.455** | **0.510** | **0.616** | **0.803** | **0.802** | **1.056** |
| Bicubic interpolation | *0.111* | *0.253* | *1.248* | *0.103* | *0.206* | *1.470* | *0.204* | *0.331* | *2.482* |
| UNet, L<sub>2</sub> | 0.409 | <ins>0.520</ins> | 0.451 | 0.380 | 0.454 | 0.666 | 0.776 | 0.770 | 1.149 |
| UNet, our loss | <ins>0.521</ins> | 0.517 | <ins>0.437</ins> | <ins>0.417</ins> | <ins>0.478</ins> | <ins>0.643</ins> | 0.784 | <ins>0.784</ins> | <ins>1.122</ins> |
| SwinFlood, L<sub>2</sub> | 0.358 | 0.510 | 0.453 | 0.346 | 0.462 | 0.654 | 0.785 | 0.737 | 1.229 |
| SwinFlood, our loss | 0.394 | 0.506 | 0.450 | 0.372 | 0.441 | 0.673 | <ins>0.791</ins> | 0.705 | 1.280 |
| RSwinUNet, L<sub>1</sub> | 0.402 | 0.393 | 0.666 | 0.323 | 0.356 | 0.952 | 0.662 | 0.705 | 1.559 |
| RSwinUNet, our loss | 0.397 | 0.422 | 0.664 | 0.326 | 0.361 | 0.948 | 0.668 | 0.717 | 1.517 |

| Model | Slow flow<br>Precision | Slow flow<br>Recall | Slow flow<br>RMSE | Moderate flow<br>Precision | Moderate flow<br>Recall | Moderate flow<br>RMSE | Fast flow<br>Precision | Fast flow<br>Recall | Fast flow<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Flow velocity in the *x* direction** (m/s) | | | | | | | | | |
| TG-PFT (ours) | **0.494** | **0.521** | **0.274** | **0.450** | **0.489** | **0.437** | **0.701** | **0.713** | **0.698** |
| Bicubic interpolation | *0.079* | *0.213* | *0.315* | *0.085* | *0.124* | *0.649* | *0.082* | *0.104* | *1.617* |
| **Flow velocity in the *y* direction** (m/s) | | | | | | | | | |
| TG-PFT (ours) | **0.465** | **0.509** | **0.307** | **0.408** | **0.452** | **0.502** | **0.684** | **0.720** | **0.749** |
| Bicubic interpolation | *0.085* | *0.232* | *0.311* | *0.092* | *0.136* | *0.646* | *0.122* | *0.129* | *1.664* |

### Table S10

As Table 6, but on the validation split.

| Configuration | RMSE | NSE | Correlation ρ | Precision | Recall | CSI |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (full) | **0.054** | **0.977** | **0.988** | **0.970** | **0.933** | **0.907** |
| **Architecture** | | | | | | |
| PFT | 0.081 | 0.947 | 0.973 | 0.934 | 0.892 | 0.839 |
| w/o PFT blocks | *0.122* | *0.881* | *0.939* | *0.916* | *0.835* | *0.776* |
| w/o multi-scale guidance | 0.071 | 0.960 | 0.980 | 0.953 | 0.906 | 0.867 |
| w/o residual connection | 0.067 | 0.964 | 0.982 | <ins>0.961</ins> | 0.913 | <ins>0.881</ins> |
| w/o AOI gate | <ins>0.066</ins> | <ins>0.966</ins> | <ins>0.983</ins> | 0.959 | <ins>0.915</ins> | <ins>0.881</ins> |
| **Loss function and sampling** | | | | | | |
| w/o classification term | 0.073 | 0.957 | 0.979 | 0.959 | 0.897 | 0.864 |
| w/o balanced sampling | 0.073 | 0.957 | 0.978 | 0.954 | 0.898 | 0.860 |

| Configuration | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| TG-PFT (full) | **0.870** | **0.795** | **0.135** | **0.815** | **0.819** | **0.205** | **0.957** | **0.940** | **0.417** |
| **Architecture** | | | | | | | | | |
| PFT | 0.726 | 0.678 | 0.192 | 0.663 | 0.689 | 0.295 | 0.931 | 0.865 | 0.653 |
| w/o PFT blocks | *0.671* | *0.590* | *0.243* | *0.597* | *0.582* | *0.400* | *0.890* | *0.810* | *1.001* |
| w/o multi-scale guidance | 0.791 | 0.720 | 0.176 | 0.726 | 0.725 | 0.272 | 0.936 | 0.900 | 0.560 |
| w/o residual connection | <ins>0.829</ins> | 0.746 | 0.163 | <ins>0.763</ins> | <ins>0.763</ins> | <ins>0.253</ins> | <ins>0.943</ins> | 0.915 | 0.529 |
| w/o AOI gate | 0.821 | <ins>0.747</ins> | <ins>0.161</ins> | 0.762 | 0.758 | <ins>0.253</ins> | 0.942 | <ins>0.917</ins> | <ins>0.515</ins> |
| **Loss function and sampling** | | | | | | | | | |
| w/o classification term | 0.816 | 0.718 | 0.171 | 0.754 | 0.740 | 0.272 | 0.940 | 0.904 | 0.581 |
| w/o balanced sampling | 0.786 | 0.697 | 0.180 | 0.714 | 0.709 | 0.280 | 0.932 | 0.900 | 0.574 |

### Table S11

Performance of TG-PFT at each flooding severity level (upper part) and its computational cost (lower part) at the downscaling factors of 2, 4, 8, and 16 for the water depth on the 100-year test set. The brackets give the resolution of the coarse grid.

| Downscaling factor | Slight flooding<br>Precision | Slight flooding<br>Recall | Slight flooding<br>RMSE | Severe flooding<br>Precision | Severe flooding<br>Recall | Severe flooding<br>RMSE | Extreme flooding<br>Precision | Extreme flooding<br>Recall | Extreme flooding<br>RMSE |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| *s* = 2 (16 m) | 0.793 | 0.722 | 0.159 | 0.730 | 0.746 | 0.252 | 0.948 | 0.927 | 0.454 |
| *s* = 4 (32 m) | 0.799 | 0.719 | 0.161 | 0.751 | 0.750 | 0.246 | 0.958 | 0.924 | 0.450 |
| *s* = 8 (64 m) | 0.808 | 0.738 | 0.151 | 0.751 | 0.759 | 0.217 | 0.963 | 0.929 | 0.362 |
| *s* = 16 (128 m) | 0.837 | 0.774 | 0.133 | 0.786 | 0.790 | 0.189 | 0.965 | 0.943 | 0.299 |

| Downscaling factor | Parameters (M) | MACs (T) | Inference (s) | Coarse grid (s) | Total (s) | Speed-up |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| *s* = 2 (16 m) | 2.523 | 1679 | 3116.7 | 24,315 | 27,432 | 2.78 |
| *s* = 4 (32 m) | 2.864 | 788 | 845.2 | 4,701 | 5,546 | 13.75 |
| *s* = 8 (64 m) | 3.270 | 611 | 265.1 | 1,330 | 1,595 | 47.79 |
| *s* = 16 (128 m) | 3.533 | 601 | 102.5 | 424 | 527 | 144.79 |
| Fine-grid simulation | – | – | – | – | 76,234 | 1.00 |

### Table S12

Comparison of the coarse-grid flood maps with the simulated fine-grid flood map for the water depth on the 100-year test set. The columns are defined in the caption of Table S12 of the Supporting Information.

| Downscaling factor | Precision | Recall | CSI | RMSE (m) | Mean depth on the<br>coarse grid (m)<br>Simulated | Mean depth on the<br>coarse grid (m)<br>Aggregated | Flooded-cell ratio |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| *s* = 2 (16 m) | 0.891 | 0.848 | 0.768 | 0.141 | 0.0662 | 0.0658 | 0.952 |
| *s* = 4 (32 m) | 0.843 | 0.718 | 0.633 | 0.335 | 0.0708 | 0.0658 | 0.851 |
| *s* = 8 (64 m) | 0.717 | 0.521 | 0.432 | 0.590 | 0.0774 | 0.0658 | 0.726 |
| *s* = 16 (128 m) | 0.417 | 0.451 | 0.277 | 0.677 | 0.1085 | 0.0658 | 1.081 |

| Downscaling factor | RMSE (m)<br>Resolution only | RMSE (m)<br>Bicubic | RMSE (m)<br>TG-PFT | CSI<br>Resolution only | CSI<br>Bicubic | CSI<br>TG-PFT |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| *s* = 2 (16 m) | 0.100 | 0.157 | 0.074 | 0.847 | 0.717 | 0.898 |
| *s* = 4 (32 m) | 0.159 | 0.346 | 0.074 | 0.702 | 0.554 | 0.890 |
| *s* = 8 (64 m) | 0.221 | 0.590 | 0.062 | 0.566 | 0.337 | 0.903 |
| *s* = 16 (128 m) | 0.280 | 0.671 | 0.053 | 0.466 | 0.158 | 0.917 |
