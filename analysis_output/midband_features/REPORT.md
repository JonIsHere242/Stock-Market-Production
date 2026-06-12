# Mid-band feature EDA + transform search

Panel: `Data/XGBPipeline/PreparedData/train.parquet`  (463,824 rows, 2330 tickers, 567 days)

## Health / distribution / raw signal

```
                              pct_nonnull  pct_zero  n_inf     mean      std     skew   kurtosis     p01      p50      p99    maxabs  tail_mass_99  between_ticker_var_share  lag1_autocorr  monthly_mean_cv  rankIC  rankIC_IR  rankIC_t  rankIC_pos  pearsonIC  n_days  decile_monotonicity  topqIC_raw  topqIR_raw  topqIC_tsz  topqIR_tsz  max_abs_corr_top  most_correlated_with  mean_abs_corr_top best_transform  best_IR  best_rankIC
feature                                                                                                                                                                                                                                                                                                                                                                                                                                      
dollar_volume_ratio_252d            100.0    0.0000      0   1.0814   1.0324  33.8275  3778.0269  0.2886   0.8809   4.2060  186.8274        1.0002                    0.2253         0.3670           0.2827  0.0026     0.0346    0.6592      0.5138    -0.0023     362              -0.9152      0.0391      0.6801      0.0476      0.9297            0.9292  dollar_volume_zscore             0.2251    winsor_1_99   0.0349       0.0026
atr_percentile_rank                 100.0    1.5756      0  47.0629  29.7913   0.0634    -1.2378  0.0000  46.4286  99.6032  100.0000        0.8732                    0.3579         0.8833           0.5762  0.0001     0.0017    0.0319      0.5166    -0.0023     362              -0.6364      0.0014      0.0210      0.0326      0.4866            0.7396       atr_regime_high             0.2169        ts_z252   0.0779       0.0064
price_position_50d                  100.0    0.0990      0   0.5939   0.2876  -0.4018    -1.0568  0.0161   0.6461   0.9950    1.0000        1.0002                    0.2506         0.8623           0.2488 -0.0113    -0.0787   -1.4973      0.4779    -0.0116     362              -0.7091     -0.0150     -0.1455     -0.0080     -0.0819            0.2108      high_close_ratio             0.1280        ts_z252  -0.1578      -0.0213
sustained_volume_burst_count        100.0   95.8676      0   0.0422   0.2051   4.8504    23.3605  0.0000   0.0000   1.0000    3.0000        0.0826                    0.1765         0.4598           1.4647 -0.0036    -0.0924   -1.7554      0.4765    -0.0030     361              -0.5152      0.0156      0.4360     -0.0010     -0.0164            0.3970               Volume%             0.1131    winsor_1_99  -0.0929      -0.0036
volume_spike_ratio                  100.0    0.0000      0   1.0396   0.7532  21.1866  1602.9448  0.3112   0.8965   3.4944   86.9979        1.0002                    0.0496         0.2290           0.1556  0.0008     0.0155    0.2956      0.5110    -0.0018     362              -0.6485      0.0371      0.7796      0.0359      0.7525            0.9309               Volume%             0.2182        ts_z252  -0.0172      -0.0009
```

## IC by transform

```
                     feature           transform  rankIC      IR       t
    dollar_volume_ratio_252d                 raw  0.0026  0.0346  0.6592
    dollar_volume_ratio_252d          signed_log  0.0026  0.0346  0.6592
    dollar_volume_ratio_252d         winsor_1_99  0.0026  0.0349  0.6634
    dollar_volume_ratio_252d             clip_z4  0.0026  0.0348  0.6626
    dollar_volume_ratio_252d             xs_rank  0.0026  0.0346  0.6592
    dollar_volume_ratio_252d                xs_z  0.0026  0.0346  0.6592
    dollar_volume_ratio_252d             ts_z252  0.0001  0.0013  0.0227
    dollar_volume_ratio_252d       log_then_xs_z  0.0026  0.0346  0.6592
    dollar_volume_ratio_252d winsor_then_xs_rank  0.0026  0.0349  0.6634
         atr_percentile_rank                 raw  0.0001  0.0017  0.0319
         atr_percentile_rank          signed_log  0.0001  0.0017  0.0319
         atr_percentile_rank         winsor_1_99 -0.0000 -0.0001 -0.0028
         atr_percentile_rank             clip_z4  0.0001  0.0017  0.0319
         atr_percentile_rank             xs_rank  0.0001  0.0017  0.0319
         atr_percentile_rank                xs_z  0.0001  0.0017  0.0319
         atr_percentile_rank             ts_z252  0.0064  0.0779  1.3546
         atr_percentile_rank       log_then_xs_z  0.0001  0.0017  0.0319
         atr_percentile_rank winsor_then_xs_rank -0.0000 -0.0001 -0.0028
          price_position_50d                 raw -0.0113 -0.0787 -1.4973
          price_position_50d          signed_log -0.0113 -0.0787 -1.4973
          price_position_50d         winsor_1_99 -0.0113 -0.0785 -1.4927
          price_position_50d             clip_z4 -0.0113 -0.0787 -1.4973
          price_position_50d             xs_rank -0.0113 -0.0787 -1.4973
          price_position_50d                xs_z -0.0113 -0.0787 -1.4973
          price_position_50d             ts_z252 -0.0213 -0.1578 -2.7416
          price_position_50d       log_then_xs_z -0.0113 -0.0787 -1.4973
          price_position_50d winsor_then_xs_rank -0.0113 -0.0785 -1.4927
sustained_volume_burst_count                 raw -0.0036 -0.0924 -1.7554
sustained_volume_burst_count          signed_log -0.0036 -0.0924 -1.7554
sustained_volume_burst_count         winsor_1_99 -0.0036 -0.0929 -1.7643
sustained_volume_burst_count             clip_z4 -0.0036 -0.0929 -1.7643
sustained_volume_burst_count             xs_rank -0.0036 -0.0924 -1.7554
sustained_volume_burst_count                xs_z -0.0036 -0.0924 -1.7554
sustained_volume_burst_count             ts_z252  0.0035  0.0410  0.7125
sustained_volume_burst_count       log_then_xs_z -0.0036 -0.0924 -1.7554
sustained_volume_burst_count winsor_then_xs_rank -0.0036 -0.0929 -1.7643
          volume_spike_ratio                 raw  0.0008  0.0155  0.2956
          volume_spike_ratio          signed_log  0.0008  0.0155  0.2956
          volume_spike_ratio         winsor_1_99  0.0009  0.0170  0.3228
          volume_spike_ratio             clip_z4  0.0008  0.0161  0.3071
          volume_spike_ratio             xs_rank  0.0008  0.0155  0.2956
          volume_spike_ratio                xs_z  0.0008  0.0155  0.2956
          volume_spike_ratio             ts_z252 -0.0009 -0.0172 -0.2983
          volume_spike_ratio       log_then_xs_z  0.0008  0.0155  0.2956
          volume_spike_ratio winsor_then_xs_rank  0.0009  0.0170  0.3228
```