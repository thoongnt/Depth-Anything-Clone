Ngưỡng: KHỚP = trùng 3 chữ số; ĐẠT = |ΔAbsRel| và |Δδ1| ≤ 0.005; còn lại KHÔNG ĐẠT.

## Kết quả chính (protocol của paper)

| Dataset | Run | AbsRel | δ1 | Paper AbsRel | Paper δ1 | ΔAbsRel | Δδ1 | Kết luận |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SUN RGB-D | sunrgbd_paper | 0.500 | 0.660 | 0.500 | 0.660 | +0.000 | +0.000 | KHỚP |
| iBims-1 | ibims | 0.150 | 0.714 | 0.150 | 0.714 | +0.000 | +0.000 | KHỚP |
| Virtual KITTI 2 | vkitti2_seed0 | 0.085 | 0.911 | 0.085 | 0.913 | +0.000 | -0.002 | ĐẠT |
| DIODE Outdoor | diode_outdoor | 0.794 | 0.288 | 0.794 | 0.288 | +0.000 | +0.000 | KHỚP |

## Quét cấu hình — SUN RGB-D (paper 0.500 / 0.660)

| Cấu hình | Mô tả | AbsRel | δ1 | ΔAbsRel | Δδ1 | Kết luận |
|---|---|---:|---:|---:|---:|---|
| base ★ | protocol gốc của evaluate.py | 0.500 | 0.660 | +0.000 | +0.000 | KHỚP |
| noflip | không flip-TTA (chỉ 1 forward) | 0.500 | 0.661 | +0.000 | +0.001 | ĐẠT |
| no_8m_cut | không cắt GT > 8 m, max_depth_eval = 10 | 0.513 | 0.660 | +0.013 | +0.000 | KHÔNG ĐẠT |
| no_eigen_crop | bỏ eigen crop | 0.546 | 0.687 | +0.046 | +0.027 | KHÔNG ĐẠT |
| gt_raw | GT: depth/ chính thức (chưa lấp lỗ) /10000 | 0.351 | 0.678 | -0.149 | +0.018 | KHÔNG ĐẠT |
| gt_mm_bfx | GT đúng thang: depth_bfx giải mã mm /1000 | 0.157 | 0.831 | -0.343 | +0.171 | KHÔNG ĐẠT |
| gt_mm | GT đúng thang: depth/ giải mã mm /1000 | 0.114 | 0.850 | -0.386 | +0.190 | KHÔNG ĐẠT |

## Quét cấu hình — iBims-1 (paper 0.150 / 0.714)

| Cấu hình | Mô tả | AbsRel | δ1 | ΔAbsRel | Δδ1 | Kết luận |
|---|---|---:|---:|---:|---:|---|
| base ★ | protocol gốc của evaluate.py | 0.150 | 0.714 | +0.000 | +0.000 | KHỚP |
| noflip | không flip-TTA (chỉ 1 forward) | 0.150 | 0.713 | +0.000 | -0.001 | ĐẠT |
| no_transp_mask | giữ pixel trong suốt (chỉ mask invalid) | 0.151 | 0.714 | +0.001 | +0.000 | ĐẠT |
| no_masks | không dùng mask invalid/trong suốt | 0.151 | 0.714 | +0.001 | +0.000 | ĐẠT |
| max_eval_50 | max_depth_eval = 50 (toàn dải iBims) | 0.151 | 0.711 | +0.001 | -0.003 | ĐẠT |
| no_eigen_crop | bỏ eigen crop | 0.168 | 0.648 | +0.018 | -0.066 | KHÔNG ĐẠT |

## Quét cấu hình — Virtual KITTI 2 (paper 0.085 / 0.913)

| Cấu hình | Mô tả | AbsRel | δ1 | ΔAbsRel | Δδ1 | Kết luận |
|---|---|---:|---:|---:|---:|---|
| vkitti2_seed4 ★ | split ngẫu nhiên seed 4 | 0.085 | 0.913 | +0.000 | +0.000 | KHỚP |
| vkitti2_seed1 | split ngẫu nhiên seed 1 | 0.084 | 0.913 | -0.001 | +0.000 | ĐẠT |
| vkitti2_seed2 | split ngẫu nhiên seed 2 | 0.084 | 0.914 | -0.001 | +0.001 | ĐẠT |
| seed_mean_5 | trung bình 5 split seed | 0.085 | 0.912 | +0.000 | -0.001 | ĐẠT |
| base | protocol gốc của evaluate.py | 0.085 | 0.911 | +0.000 | -0.002 | ĐẠT |
| vkitti2_seed3 | split ngẫu nhiên seed 3 | 0.086 | 0.910 | +0.001 | -0.003 | ĐẠT |
| noflip | không flip-TTA (chỉ 1 forward) | 0.087 | 0.907 | +0.002 | -0.006 | KHÔNG ĐẠT |
| eigen_crop | eigen crop KITTI thay cho garg crop | 0.099 | 0.876 | +0.014 | -0.037 | KHÔNG ĐẠT |
| no_crop | bỏ garg crop | 0.123 | 0.801 | +0.038 | -0.112 | KHÔNG ĐẠT |

## Quét cấu hình — DIODE Outdoor (paper 0.794 / 0.288)

| Cấu hình | Mô tả | AbsRel | δ1 | ΔAbsRel | Δδ1 | Kết luận |
|---|---|---:|---:|---:|---:|---|
| base ★ | protocol gốc của evaluate.py | 0.794 | 0.288 | +0.000 | +0.000 | KHỚP |
| noflip | không flip-TTA (chỉ 1 forward) | 0.794 | 0.289 | +0.000 | +0.001 | ĐẠT |
| max_eval_300 | max_depth_eval = 300 (giữ GT xa > 80 m) | 0.792 | 0.286 | -0.002 | -0.002 | ĐẠT |
| diode_outdoor_antialias | Resize antialias=True (mặc định torchvision ≥ 0.17) | 0.800 | 0.289 | +0.006 | +0.001 | KHÔNG ĐẠT |
| no_crop | bỏ garg crop | 0.867 | 0.316 | +0.073 | +0.028 | KHÔNG ĐẠT |
| eigen_crop | eigen crop KITTI thay cho garg crop | 0.867 | 0.292 | +0.073 | +0.004 | KHÔNG ĐẠT |
| valid_mask | áp depth_mask.npy (loader gốc bỏ qua) | 0.572 | 0.290 | -0.222 | +0.002 | KHÔNG ĐẠT |

Kiểm tra sweep `base` so với `evaluate.py`:

- SUN RGB-D: sweep base 0.500 / 0.660, evaluate.py 0.500 / 0.660 → trùng
- iBims-1: sweep base 0.150 / 0.714, evaluate.py 0.150 / 0.714 → trùng
- Virtual KITTI 2: sweep base 0.085 / 0.911, evaluate.py 0.085 / 0.911 → trùng
- DIODE Outdoor: sweep base 0.794 / 0.288, evaluate.py 0.794 / 0.288 → trùng

★ = cấu hình gần paper nhất (khoảng cách nhỏ nhất; hoà thì ưu tiên `base`).
