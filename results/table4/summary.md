Ngưỡng (AbsRel, δ1): KHỚP = trùng 3 chữ số; ĐẠT = |Δ| ≤ 0.005; còn lại KHÔNG ĐẠT.

| Lượt | Mô tả | δ1 | δ2 | δ3 | AbsRel | RMSE | RMSE log | Khớp | Kết luận |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| **Paper** | Ours, Table 4 | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 | | |
| evaluate.py | lệnh chính thức `evaluate.py -d kitti` | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 | 6/6 | KHỚP |
| sweep:base | protocol gốc của evaluate.py | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 | 6/6 | KHỚP |
| sweep:noflip | không flip-TTA (chỉ 1 forward) | 0.982 | 0.998 | 0.999 | 0.047 | 1.922 | 0.070 | 2/6 | ĐẠT |
| sweep:eigen_crop | eigen crop KITTI thay cho garg crop | 0.974 | 0.997 | 0.999 | 0.053 | 2.368 | 0.080 | 0/6 | KHÔNG ĐẠT |
| sweep:no_crop | bỏ garg crop | 0.975 | 0.997 | 0.999 | 0.051 | 2.260 | 0.078 | 0/6 | KHÔNG ĐẠT |
| sweep:max_eval_50 | max_depth_eval = 50 | 0.985 | 0.998 | 1.000 | 0.045 | 1.403 | 0.065 | 2/6 | ĐẠT |
| sweep:min_eval_1 | min_depth_eval = 1 m | 0.982 | 0.998 | 1.000 | 0.046 | 1.896 | 0.069 | 6/6 | KHỚP |
