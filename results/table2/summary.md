Ngưỡng: KHỚP = trùng 3 chữ số; ĐẠT = |ΔAbsRel| và |Δδ1| ≤ 0.005; còn lại KHÔNG ĐẠT. DDAD nằm ngoài phạm vi.

## Trạng thái (cấu hình gần paper nhất mỗi ô)

| Encoder | KITTI | NYUv2 | Sintel | ETH3D | DIODE |
|---|---|---|---|---|---|
| vits | 0.080 / 0.936 — KHỚP* (annotated|crop=none,cap=80,clamp=none) | 0.053 / 0.972 — KHỚP (midas) | 0.450 / 0.723 — KHÔNG ĐẠT* (final-capsweep|cap=150,mode=gtclip,clamp=cap) | 0.126 / 0.883 — ĐẠT* (official-fullres|lin/m>=0.644) | 0.075 / 0.943 — ĐẠT* (moge|crop=none,cap=None,clamp=none) |
| vitb | 0.080 / 0.939 — KHỚP* (annotated|crop=none,cap=80,clamp=none) | 0.046 / 0.979 — KHỚP (midas) | 0.432 / 0.752 — ĐẠT* (final-capsweep|cap=150,mode=gtclip,clamp=cap) | 0.129 / 0.876 — KHÔNG ĐẠT* (official-fullres|lin/m>=0.64) | 0.069 / 0.949 — ĐẠT* (moge|crop=none,cap=None,clamp=none) |
| vitl | 0.076 / 0.947 — KHỚP* (annotated|crop=none,cap=80,clamp=cap) | 0.044 / 0.980 — ĐẠT (midas) | 0.466 / 0.754 — KHÔNG ĐẠT* (final-capsweep|cap=120,mode=gtclip,clamp=none) | 0.125 / 0.886 — ĐẠT* (official-fullres|lin/m>=0.64) | 0.065 / 0.953 — ĐẠT* (moge-mask-392) |

\* protocol chưa được tác giả xác nhận (giả thuyết khớp số).

## Protocol chung cho cả 3 encoder

Mỗi dataset: cấu hình có độ lệch lớn nhất qua ViT-S/B/L nhỏ nhất (một protocol duy nhất, khó "ăn may" hơn chọn riêng từng ô).

| Dataset | Cấu hình | ViT-S | ViT-B | ViT-L | Kết luận chung |
|---|---|---|---|---|---|
| KITTI | annotated|crop=none,cap=80,clamp=none | 0.080 / 0.936 (KHỚP) | 0.080 / 0.939 (KHỚP) | 0.076 / 0.946 (ĐẠT) | ĐẠT |
| NYUv2 | midas | 0.053 / 0.972 (KHỚP) | 0.046 / 0.979 (KHỚP) | 0.044 / 0.980 (ĐẠT) | ĐẠT |
| Sintel | final-capsweep|cap=175,mode=gtclip,clamp=cap | 0.473 / 0.718 (KHÔNG ĐẠT) | 0.453 / 0.749 (KHÔNG ĐẠT) | 0.427 / 0.766 (KHÔNG ĐẠT) | KHÔNG ĐẠT |
| ETH3D | official-fullres|lin/m>=0.642 | 0.133 / 0.864 (KHÔNG ĐẠT) | 0.128 / 0.876 (KHÔNG ĐẠT) | 0.125 / 0.886 (ĐẠT) | KHÔNG ĐẠT |
| DIODE | moge|crop=none,cap=None,clamp=none | 0.075 / 0.943 (ĐẠT) | 0.069 / 0.949 (ĐẠT) | 0.065 / 0.954 (ĐẠT) | ĐẠT |

## Các cấu hình gần paper nhất (tối đa 8 dòng mỗi ô; đủ trong summary.json)

| Dataset | Encoder | Cấu hình | Mô tả | AbsRel | δ1 | Paper | ΔAbsRel | Δδ1 | Kết luận |
|---|---|---|---|---:|---:|---|---:|---:|---|
| KITTI | vits | annotated|crop=none,cap=80,clamp=none ★ | Eigen test 652, depth annotated; crop=none,cap=80,clamp=none | 0.080 | 0.936 | 0.080 / 0.936 | +0.000 | +0.000 | KHỚP |
| KITTI | vits | annotated|crop=none,cap=None,clamp=none | Eigen test 652, depth annotated; crop=none,cap=None,clamp=none | 0.080 | 0.936 | 0.080 / 0.936 | +0.000 | +0.000 | KHỚP |
| KITTI | vits | annotated|crop=none,cap=80,clamp=cap | Eigen test 652, depth annotated; crop=none,cap=80,clamp=cap | 0.079 | 0.937 | 0.080 / 0.936 | -0.001 | +0.001 | ĐẠT |
| KITTI | vits | annotated|crop=none,cap=None,clamp=cap | Eigen test 652, depth annotated; crop=none,cap=None,clamp=cap | 0.079 | 0.937 | 0.080 / 0.936 | -0.001 | +0.001 | ĐẠT |
| KITTI | vits | moge|crop=none,cap=80,clamp=cap | gói đánh giá MoGe; crop=none,cap=80,clamp=cap | 0.079 | 0.937 | 0.080 / 0.936 | -0.001 | +0.001 | ĐẠT |
| KITTI | vits | moge|crop=none,cap=80,clamp=none | gói đánh giá MoGe; crop=none,cap=80,clamp=none | 0.079 | 0.936 | 0.080 / 0.936 | -0.001 | +0.000 | ĐẠT |
| KITTI | vits | moge|crop=none,cap=None,clamp=cap | gói đánh giá MoGe; crop=none,cap=None,clamp=cap | 0.079 | 0.937 | 0.080 / 0.936 | -0.001 | +0.001 | ĐẠT |
| KITTI | vits | moge|crop=none,cap=None,clamp=none | gói đánh giá MoGe; crop=none,cap=None,clamp=none | 0.079 | 0.936 | 0.080 / 0.936 | -0.001 | +0.000 | ĐẠT |
| KITTI | vitb | annotated|crop=none,cap=80,clamp=none ★ | Eigen test 652, depth annotated; crop=none,cap=80,clamp=none | 0.080 | 0.939 | 0.080 / 0.939 | +0.000 | +0.000 | KHỚP |
| KITTI | vitb | annotated|crop=none,cap=None,clamp=none | Eigen test 652, depth annotated; crop=none,cap=None,clamp=none | 0.080 | 0.939 | 0.080 / 0.939 | +0.000 | +0.000 | KHỚP |
| KITTI | vitb | annotated|crop=none,cap=80,clamp=cap | Eigen test 652, depth annotated; crop=none,cap=80,clamp=cap | 0.079 | 0.940 | 0.080 / 0.939 | -0.001 | +0.001 | ĐẠT |
| KITTI | vitb | annotated|crop=none,cap=None,clamp=cap | Eigen test 652, depth annotated; crop=none,cap=None,clamp=cap | 0.079 | 0.940 | 0.080 / 0.939 | -0.001 | +0.001 | ĐẠT |
| KITTI | vitb | annotated|crop=eigen,cap=80,clamp=cap | Eigen test 652, depth annotated; crop=eigen,cap=80,clamp=cap | 0.081 | 0.938 | 0.080 / 0.939 | +0.001 | -0.001 | ĐẠT |
| KITTI | vitb | annotated|crop=eigen,cap=None,clamp=cap | Eigen test 652, depth annotated; crop=eigen,cap=None,clamp=cap | 0.081 | 0.938 | 0.080 / 0.939 | +0.001 | -0.001 | ĐẠT |
| KITTI | vitb | annotated|crop=eigen,cap=80,clamp=none | Eigen test 652, depth annotated; crop=eigen,cap=80,clamp=none | 0.081 | 0.937 | 0.080 / 0.939 | +0.001 | -0.002 | ĐẠT |
| KITTI | vitb | annotated|crop=eigen,cap=None,clamp=none | Eigen test 652, depth annotated; crop=eigen,cap=None,clamp=none | 0.081 | 0.937 | 0.080 / 0.939 | +0.001 | -0.002 | ĐẠT |
| KITTI | vitl | annotated|crop=none,cap=80,clamp=cap ★ | Eigen test 652, depth annotated; crop=none,cap=80,clamp=cap | 0.076 | 0.947 | 0.076 / 0.947 | +0.000 | +0.000 | KHỚP |
| KITTI | vitl | annotated|crop=none,cap=None,clamp=cap | Eigen test 652, depth annotated; crop=none,cap=None,clamp=cap | 0.076 | 0.947 | 0.076 / 0.947 | +0.000 | +0.000 | KHỚP |
| KITTI | vitl | annotated|crop=none,cap=80,clamp=none | Eigen test 652, depth annotated; crop=none,cap=80,clamp=none | 0.076 | 0.946 | 0.076 / 0.947 | +0.000 | -0.001 | ĐẠT |
| KITTI | vitl | annotated|crop=none,cap=None,clamp=none | Eigen test 652, depth annotated; crop=none,cap=None,clamp=none | 0.076 | 0.946 | 0.076 / 0.947 | +0.000 | -0.001 | ĐẠT |
| KITTI | vitl | annotated|crop=eigen,cap=80,clamp=cap | Eigen test 652, depth annotated; crop=eigen,cap=80,clamp=cap | 0.078 | 0.945 | 0.076 / 0.947 | +0.002 | -0.002 | ĐẠT |
| KITTI | vitl | annotated|crop=eigen,cap=None,clamp=cap | Eigen test 652, depth annotated; crop=eigen,cap=None,clamp=cap | 0.078 | 0.945 | 0.076 / 0.947 | +0.002 | -0.002 | ĐẠT |
| KITTI | vitl | annotated|crop=eigen,cap=80,clamp=none | Eigen test 652, depth annotated; crop=eigen,cap=80,clamp=none | 0.078 | 0.944 | 0.076 / 0.947 | +0.002 | -0.003 | ĐẠT |
| KITTI | vitl | annotated|crop=eigen,cap=None,clamp=none | Eigen test 652, depth annotated; crop=eigen,cap=None,clamp=none | 0.078 | 0.944 | 0.076 / 0.947 | +0.002 | -0.003 | ĐẠT |
| NYUv2 | vits | midas ★ | rawDepths, mask 0 < d < 10 m toàn ảnh, fit + chấm disparity, không flip | 0.053 | 0.972 | 0.053 / 0.972 | +0.000 | +0.000 | KHỚP |
| NYUv2 | vitb | midas ★ | rawDepths, mask 0 < d < 10 m toàn ảnh, fit + chấm disparity, không flip | 0.046 | 0.979 | 0.046 / 0.979 | +0.000 | +0.000 | KHỚP |
| NYUv2 | vitl | midas ★ | rawDepths, mask 0 < d < 10 m toàn ảnh, fit + chấm disparity, không flip | 0.044 | 0.980 | 0.043 / 0.981 | +0.001 | -0.001 | ĐẠT |
| NYUv2 | vitl | filled-eigen | đối chứng: depths đã inpaint + Eigen crop, chấm disparity | 0.053 | 0.965 | 0.043 / 0.981 | +0.010 | -0.016 | KHÔNG ĐẠT |
| Sintel | vits | final-capsweep|cap=150,mode=gtclip,clamp=cap ★ | pass final; cap=150,mode=gtclip,clamp=cap (cap được CHỈNH để khớp paper) | 0.450 | 0.723 | 0.464 / 0.739 | -0.014 | -0.016 | KHÔNG ĐẠT |
| Sintel | vits | final|crop=none,cap=80,clamp=none,mode=gtclip | pass final, depth .dpt chính thức; crop=none,cap=80,clamp=none,mode=gtclip | 0.450 | 0.718 | 0.464 / 0.739 | -0.014 | -0.021 | KHÔNG ĐẠT |
| Sintel | vits | final-capsweep|cap=80,mode=gtclip,clamp=none | pass final; cap=80,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.450 | 0.718 | 0.464 / 0.739 | -0.014 | -0.021 | KHÔNG ĐẠT |
| Sintel | vits | final-capsweep|cap=175,mode=gtclip,clamp=cap | pass final; cap=175,mode=gtclip,clamp=cap (cap được CHỈNH để khớp paper) | 0.473 | 0.718 | 0.464 / 0.739 | +0.009 | -0.021 | KHÔNG ĐẠT |
| Sintel | vits | final-capsweep|cap=90,mode=gtclip,clamp=none | pass final; cap=90,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.471 | 0.717 | 0.464 / 0.739 | +0.007 | -0.022 | KHÔNG ĐẠT |
| Sintel | vits | final-capsweep|cap=76,mode=gtclip,clamp=none | pass final; cap=76,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.441 | 0.719 | 0.464 / 0.739 | -0.023 | -0.020 | KHÔNG ĐẠT |
| Sintel | vits | final|crop=none,cap=100,clamp=none,mode=gtclip | pass final, depth .dpt chính thức; crop=none,cap=100,clamp=none,mode=gtclip | 0.491 | 0.715 | 0.464 / 0.739 | +0.027 | -0.024 | KHÔNG ĐẠT |
| Sintel | vits | final-capsweep|cap=100,mode=gtclip,clamp=none | pass final; cap=100,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.491 | 0.715 | 0.464 / 0.739 | +0.027 | -0.024 | KHÔNG ĐẠT |
| Sintel | vitb | final-capsweep|cap=150,mode=gtclip,clamp=cap ★ | pass final; cap=150,mode=gtclip,clamp=cap (cap được CHỈNH để khớp paper) | 0.432 | 0.752 | 0.432 / 0.756 | +0.000 | -0.004 | ĐẠT |
| Sintel | vitb | final|crop=none,cap=72,clamp=none,mode=gtclip | pass final, depth .dpt chính thức; crop=none,cap=72,clamp=none,mode=gtclip | 0.427 | 0.744 | 0.432 / 0.756 | -0.005 | -0.012 | KHÔNG ĐẠT |
| Sintel | vitb | final-capsweep|cap=72,mode=gtclip,clamp=none | pass final; cap=72,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.427 | 0.744 | 0.432 / 0.756 | -0.005 | -0.012 | KHÔNG ĐẠT |
| Sintel | vitb | final-capsweep|cap=76,mode=gtclip,clamp=none | pass final; cap=76,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.436 | 0.744 | 0.432 / 0.756 | +0.004 | -0.012 | KHÔNG ĐẠT |
| Sintel | vitb | final-capsweep|cap=250,mode=exclude,clamp=none | pass final; cap=250,mode=exclude,clamp=none (cap được CHỈNH để khớp paper) | 0.426 | 0.744 | 0.432 / 0.756 | -0.006 | -0.012 | KHÔNG ĐẠT |
| Sintel | vitb | final|crop=none,cap=80,clamp=none,mode=gtclip | pass final, depth .dpt chính thức; crop=none,cap=80,clamp=none,mode=gtclip | 0.444 | 0.742 | 0.432 / 0.756 | +0.012 | -0.014 | KHÔNG ĐẠT |
| Sintel | vitb | final-capsweep|cap=80,mode=gtclip,clamp=none | pass final; cap=80,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.444 | 0.742 | 0.432 / 0.756 | +0.012 | -0.014 | KHÔNG ĐẠT |
| Sintel | vitb | final-capsweep|cap=68,mode=gtclip,clamp=none | pass final; cap=68,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.417 | 0.745 | 0.432 / 0.756 | -0.015 | -0.011 | KHÔNG ĐẠT |
| Sintel | vitl | final-capsweep|cap=120,mode=gtclip,clamp=none ★ | pass final; cap=120,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.466 | 0.754 | 0.458 / 0.760 | +0.008 | -0.006 | KHÔNG ĐẠT |
| Sintel | vitl | final|crop=none,cap=200,clamp=cap,mode=gtclip | pass final, depth .dpt chính thức; crop=none,cap=200,clamp=cap,mode=gtclip | 0.449 | 0.763 | 0.458 / 0.760 | -0.009 | +0.003 | KHÔNG ĐẠT |
| Sintel | vitl | final-capsweep|cap=200,mode=gtclip,clamp=cap | pass final; cap=200,mode=gtclip,clamp=cap (cap được CHỈNH để khớp paper) | 0.449 | 0.763 | 0.458 / 0.760 | -0.009 | +0.003 | KHÔNG ĐẠT |
| Sintel | vitl | final|crop=none,cap=500,clamp=cap | pass final, depth .dpt chính thức; crop=none,cap=500,clamp=cap | 0.480 | 0.747 | 0.458 / 0.760 | +0.022 | -0.013 | KHÔNG ĐẠT |
| Sintel | vitl | final|crop=none,cap=100,clamp=none,mode=gtclip | pass final, depth .dpt chính thức; crop=none,cap=100,clamp=none,mode=gtclip | 0.433 | 0.759 | 0.458 / 0.760 | -0.025 | -0.001 | KHÔNG ĐẠT |
| Sintel | vitl | final-capsweep|cap=100,mode=gtclip,clamp=none | pass final; cap=100,mode=gtclip,clamp=none (cap được CHỈNH để khớp paper) | 0.433 | 0.759 | 0.458 / 0.760 | -0.025 | -0.001 | KHÔNG ĐẠT |
| Sintel | vitl | final-capsweep|cap=175,mode=gtclip,clamp=cap | pass final; cap=175,mode=gtclip,clamp=cap (cap được CHỈNH để khớp paper) | 0.427 | 0.766 | 0.458 / 0.760 | -0.031 | +0.006 | KHÔNG ĐẠT |
| Sintel | vitl | final-capsweep|cap=250,mode=gtclip,clamp=cap | pass final; cap=250,mode=gtclip,clamp=cap (cap được CHỈNH để khớp paper) | 0.493 | 0.760 | 0.458 / 0.760 | +0.035 | +0.000 | KHÔNG ĐẠT |
| ETH3D | vits | official-fullres|lin/m>=0.644 ★ | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.644 (ngưỡng được CHỈNH để khớp paper) | 0.126 | 0.883 | 0.127 / 0.885 | -0.001 | -0.002 | ĐẠT |
| ETH3D | vits | official-fullres|lin/m>=0.645 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.645 (ngưỡng được CHỈNH để khớp paper) | 0.126 | 0.883 | 0.127 / 0.885 | -0.001 | -0.002 | ĐẠT |
| ETH3D | vits | official-fullres|lin/m>=0.646 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.646 (ngưỡng được CHỈNH để khớp paper) | 0.126 | 0.883 | 0.127 / 0.885 | -0.001 | -0.002 | ĐẠT |
| ETH3D | vits | official-fullres|lin/m>=0.647 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.647 (ngưỡng được CHỈNH để khớp paper) | 0.126 | 0.883 | 0.127 / 0.885 | -0.001 | -0.002 | ĐẠT |
| ETH3D | vits | official-fullres|lin/m>=0.643 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.643 (ngưỡng được CHỈNH để khớp paper) | 0.127 | 0.882 | 0.127 / 0.885 | +0.000 | -0.003 | ĐẠT |
| ETH3D | vits | official-fullres|lin/m>=0.648 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.648 (ngưỡng được CHỈNH để khớp paper) | 0.124 | 0.888 | 0.127 / 0.885 | -0.003 | +0.003 | ĐẠT |
| ETH3D | vits | official-fullres|lin/m>=0.649 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.649 (ngưỡng được CHỈNH để khớp paper) | 0.124 | 0.888 | 0.127 / 0.885 | -0.003 | +0.003 | ĐẠT |
| ETH3D | vits | official-fullres|lin/m>=0.65 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.65 (ngưỡng được CHỈNH để khớp paper) | 0.124 | 0.889 | 0.127 / 0.885 | -0.003 | +0.004 | ĐẠT |
| ETH3D | vitb | official-fullres|lin/m>=0.64 ★ | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.64 (ngưỡng được CHỈNH để khớp paper) | 0.129 | 0.876 | 0.126 / 0.884 | +0.003 | -0.008 | KHÔNG ĐẠT |
| ETH3D | vitb | official-fullres|lin/m>=0.641 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.641 (ngưỡng được CHỈNH để khớp paper) | 0.129 | 0.876 | 0.126 / 0.884 | +0.003 | -0.008 | KHÔNG ĐẠT |
| ETH3D | vitb | official-fullres|lin/m>=0.642 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.642 (ngưỡng được CHỈNH để khớp paper) | 0.128 | 0.876 | 0.126 / 0.884 | +0.002 | -0.008 | KHÔNG ĐẠT |
| ETH3D | vitb | official-fullres|lin/m>=0.643 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.643 (ngưỡng được CHỈNH để khớp paper) | 0.122 | 0.894 | 0.126 / 0.884 | -0.004 | +0.010 | KHÔNG ĐẠT |
| ETH3D | vitb | official-fullres|lin/m>=0.644 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.644 (ngưỡng được CHỈNH để khớp paper) | 0.122 | 0.894 | 0.126 / 0.884 | -0.004 | +0.010 | KHÔNG ĐẠT |
| ETH3D | vitb | official-fullres|lin/m>=0.645 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.645 (ngưỡng được CHỈNH để khớp paper) | 0.122 | 0.894 | 0.126 / 0.884 | -0.004 | +0.010 | KHÔNG ĐẠT |
| ETH3D | vitb | official-fullres|lin/m>=0.646 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.646 (ngưỡng được CHỈNH để khớp paper) | 0.122 | 0.894 | 0.126 / 0.884 | -0.004 | +0.010 | KHÔNG ĐẠT |
| ETH3D | vitb | official-fullres|lin/m>=0.647 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.647 (ngưỡng được CHỈNH để khớp paper) | 0.122 | 0.894 | 0.126 / 0.884 | -0.004 | +0.010 | KHÔNG ĐẠT |
| ETH3D | vitl | official-fullres|lin/m>=0.64 ★ | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.64 (ngưỡng được CHỈNH để khớp paper) | 0.125 | 0.886 | 0.127 / 0.882 | -0.002 | +0.004 | ĐẠT |
| ETH3D | vitl | official-fullres|lin/m>=0.641 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.641 (ngưỡng được CHỈNH để khớp paper) | 0.125 | 0.886 | 0.127 / 0.882 | -0.002 | +0.004 | ĐẠT |
| ETH3D | vitl | official-fullres|lin/m>=0.642 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.642 (ngưỡng được CHỈNH để khớp paper) | 0.125 | 0.886 | 0.127 / 0.882 | -0.002 | +0.004 | ĐẠT |
| ETH3D | vitl | official-fullres|lin/m>=0.63 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.63 (ngưỡng được CHỈNH để khớp paper) | 0.131 | 0.868 | 0.127 / 0.882 | +0.004 | -0.014 | KHÔNG ĐẠT |
| ETH3D | vitl | official-fullres|lin/m>=0.643 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.643 (ngưỡng được CHỈNH để khớp paper) | 0.119 | 0.904 | 0.127 / 0.882 | -0.008 | +0.022 | KHÔNG ĐẠT |
| ETH3D | vitl | official-fullres|lin/m>=0.62 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.62 (ngưỡng được CHỈNH để khớp paper) | 0.134 | 0.859 | 0.127 / 0.882 | +0.007 | -0.023 | KHÔNG ĐẠT |
| ETH3D | vitl | official-fullres|lin/m>=0.644 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.644 (ngưỡng được CHỈNH để khớp paper) | 0.119 | 0.905 | 0.127 / 0.882 | -0.008 | +0.023 | KHÔNG ĐẠT |
| ETH3D | vitl | official-fullres|lin/m>=0.645 | depth chính thức thu nhỏ bilinear về kích thước mạng, ngưỡng mask =0.645 (ngưỡng được CHỈNH để khớp paper) | 0.119 | 0.905 | 0.127 / 0.882 | -0.008 | +0.023 | KHÔNG ĐẠT |
| DIODE | vits | moge|crop=none,cap=None,clamp=none ★ | gói đánh giá MoGe; crop=none,cap=None,clamp=none | 0.075 | 0.943 | 0.076 / 0.939 | -0.001 | +0.004 | ĐẠT |
| DIODE | vits | moge-mask | giao mask gốc với mask MoGe, disparity, không flip | 0.074 | 0.944 | 0.076 / 0.939 | -0.002 | +0.005 | ĐẠT |
| DIODE | vits | moge-mask|crop=none,cap=None,clamp=none | moge-mask; crop=none,cap=None,clamp=none | 0.074 | 0.944 | 0.076 / 0.939 | -0.002 | +0.005 | ĐẠT |
| DIODE | vits | moge|crop=none,cap=80,clamp=none | gói đánh giá MoGe; crop=none,cap=80,clamp=none | 0.070 | 0.945 | 0.076 / 0.939 | -0.006 | +0.006 | KHÔNG ĐẠT |
| DIODE | vits | moge-mask|crop=none,cap=80,clamp=none | moge-mask; crop=none,cap=80,clamp=none | 0.069 | 0.946 | 0.076 / 0.939 | -0.007 | +0.007 | KHÔNG ĐẠT |
| DIODE | vits | moge|crop=none,cap=None,clamp=cap | gói đánh giá MoGe; crop=none,cap=None,clamp=cap | 0.071 | 0.947 | 0.076 / 0.939 | -0.005 | +0.008 | KHÔNG ĐẠT |
| DIODE | vits | moge|crop=none,cap=80,clamp=cap | gói đánh giá MoGe; crop=none,cap=80,clamp=cap | 0.068 | 0.946 | 0.076 / 0.939 | -0.008 | +0.007 | KHÔNG ĐẠT |
| DIODE | vits | moge-mask|crop=none,cap=None,clamp=cap | moge-mask; crop=none,cap=None,clamp=cap | 0.070 | 0.948 | 0.076 / 0.939 | -0.006 | +0.009 | KHÔNG ĐẠT |
| DIODE | vitb | moge|crop=none,cap=None,clamp=none ★ | gói đánh giá MoGe; crop=none,cap=None,clamp=none | 0.069 | 0.949 | 0.069 / 0.946 | +0.000 | +0.003 | ĐẠT |
| DIODE | vitb | moge-mask | giao mask gốc với mask MoGe, disparity, không flip | 0.067 | 0.951 | 0.069 / 0.946 | -0.002 | +0.005 | ĐẠT |
| DIODE | vitb | moge|crop=none,cap=80,clamp=none | gói đánh giá MoGe; crop=none,cap=80,clamp=none | 0.064 | 0.951 | 0.069 / 0.946 | -0.005 | +0.005 | ĐẠT |
| DIODE | vitb | moge|crop=none,cap=None,clamp=cap | gói đánh giá MoGe; crop=none,cap=None,clamp=cap | 0.065 | 0.953 | 0.069 / 0.946 | -0.004 | +0.007 | KHÔNG ĐẠT |
| DIODE | vitb | moge|crop=none,cap=80,clamp=cap | gói đánh giá MoGe; crop=none,cap=80,clamp=cap | 0.061 | 0.952 | 0.069 / 0.946 | -0.008 | +0.006 | KHÔNG ĐẠT |
| DIODE | vitb | native | mask cảm biến gốc, toàn ảnh, fit + chấm disparity, không flip | 0.271 | 0.750 | 0.069 / 0.946 | +0.202 | -0.196 | KHÔNG ĐẠT |
| DIODE | vitl | moge-mask-392 ★ | mask MoGe, cạnh ngắn 392 | 0.065 | 0.953 | 0.066 / 0.952 | -0.001 | +0.001 | ĐẠT |
| DIODE | vitl | moge|crop=none,cap=None,clamp=none | gói đánh giá MoGe; crop=none,cap=None,clamp=none | 0.065 | 0.954 | 0.066 / 0.952 | -0.001 | +0.002 | ĐẠT |
| DIODE | vitl | moge-mask | giao mask gốc với mask MoGe (lọc biên/điểm xa), disparity, không flip | 0.063 | 0.955 | 0.066 / 0.952 | -0.003 | +0.003 | ĐẠT |
| DIODE | vitl | moge-mask-flip | mask MoGe + flip-TTA | 0.062 | 0.956 | 0.066 / 0.952 | -0.004 | +0.004 | ĐẠT |
| DIODE | vitl | moge|crop=none,cap=None,clamp=cap | gói đánh giá MoGe; crop=none,cap=None,clamp=cap | 0.062 | 0.957 | 0.066 / 0.952 | -0.004 | +0.005 | ĐẠT |
| DIODE | vitl | moge|crop=none,cap=80,clamp=none | gói đánh giá MoGe; crop=none,cap=80,clamp=none | 0.060 | 0.955 | 0.066 / 0.952 | -0.006 | +0.003 | KHÔNG ĐẠT |
| DIODE | vitl | moge|crop=none,cap=80,clamp=cap | gói đánh giá MoGe; crop=none,cap=80,clamp=cap | 0.058 | 0.956 | 0.066 / 0.952 | -0.008 | +0.004 | KHÔNG ĐẠT |
| DIODE | vitl | native | mask cảm biến gốc, toàn ảnh, fit + chấm disparity, không flip | 0.268 | 0.754 | 0.066 / 0.952 | +0.202 | -0.198 | KHÔNG ĐẠT |

★ = cấu hình gần paper nhất của ô (dataset, encoder).
