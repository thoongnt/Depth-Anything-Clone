# Tái hiện Table 5 — zero-shot metric depth (Depth Anything V1)

Rà soát ngày 2026-11-28. Mục tiêu: **Table 5** bản CVPR 2024, hàng *Depth Anything*
(framework ZoeDepth + encoder Depth Anything ViT-L), trên 4 bộ: SUN RGB-D, iBims-1,
Virtual KITTI 2 và DIODE Outdoor. HyperSim không nằm trong phạm vi.

- Checkpoint **indoor** (fine-tune NYUv2) dùng cho SUN RGB-D và iBims-1.
- Checkpoint **outdoor** (fine-tune KITTI) dùng cho vKITTI2 và DIODE Outdoor.
- Chạy đúng lệnh của `metric_depth/README.md`:
  `evaluate.py -m zoedepth --pretrained_resource=local::<ckpt> -d <dataset>`.

Không thay metric, chọn subset hoặc chỉnh tham số theo số của paper. Mọi thay đổi
chỉ nhằm khớp **dữ liệu, protocol và môi trường** mà tác giả đã dùng.

## Ngưỡng đánh giá

So sánh dùng các giá trị 3 chữ số thập phân mà `evaluate.py` in ra. Đặt
d = max(|ΔAbsRel|, |Δδ₁|):

| Kết luận | Điều kiện |
|---|---|
| **KHỚP** | d = 0 (trùng paper ở 3 chữ số) |
| **ĐẠT** | d ≤ 0.005 |
| **KHÔNG ĐẠT** | d > 0.005 |

Đổi ngưỡng bằng `collect_table5.py --tol`.

## Kết quả đã kiểm chứng

| Dataset | Checkpoint | Số ảnh | AbsRel ↓ | δ₁ ↑ | Paper AbsRel | Paper δ₁ | Kết luận |
|---|---|---:|---:|---:|---:|---:|---|
| SUN RGB-D | indoor | 5050 | 0.500 | 0.660 | 0.500 | 0.660 | KHỚP |
| iBims-1 | indoor | 100 | 0.150 | 0.714 | 0.150 | 0.714 | KHỚP |
| Virtual KITTI 2 (split seed 0) | outdoor | 1701 | 0.085 | 0.911 | 0.085 | 0.913 | ĐẠT (Δδ₁ −0.002) |
| DIODE Outdoor (val) | outdoor | 446 | 0.794 | 0.288 | 0.794 | 0.288 | KHỚP |

Cả 4 bộ đều đạt ngưỡng. Ba bộ trùng paper ở 3 chữ số. Riêng vKITTI2 lệch δ₁ −0.002,
vì tác giả không công bố split test và split này được chia ngẫu nhiên (xem mục 4
bên dưới).

Nếu chọn seed gần paper nhất thì seed 4 trùng chính xác 0.085 / 0.913. Tuy vậy,
lựa chọn này được đưa ra **sau khi** đã biết kết quả. Vì vậy bảng chính vẫn giữ
seed 0, được cố định từ trước.

## Quét cấu hình

Mỗi lần quét chỉ đổi **một** yếu tố so với protocol gốc (`base`). Nhờ đó có thể
thấy yếu tố nào quyết định việc khớp với paper.

`scripts/sweep_table5.py` chạy inference một lần, giống hệt `infer()` của
`evaluate.py`, và giữ lại cả lượt forward không flip. Sau đó, cùng một dự đoán được
chấm bằng `compute_metrics` dưới nhiều cấu hình.

Hai thay đổi ở mức loader cần một lượt inference riêng qua `evaluate.py`:

- antialias của DIODE;
- split seed của vKITTI2.

Kiểm tra: `base` của sweep trùng kết quả `evaluate.py` trên cả 4 bộ. Bảng đầy đủ
nằm ở `results/table5/summary.md`. ★ đánh dấu cấu hình gần paper nhất.

### SUN RGB-D (paper 0.500 / 0.660)

| Cấu hình | AbsRel | δ₁ | Kết luận |
|---|---:|---:|---|
| base ★ — GT ahanda (= `depth_bfx`) / 10000, eigen crop, cắt GT > 8 m, flip-TTA | 0.500 | 0.660 | KHỚP |
| không flip-TTA | 0.500 | 0.661 | ĐẠT |
| không cắt GT > 8 m (`max_depth_eval = 10`) | 0.513 | 0.660 | KHÔNG ĐẠT |
| bỏ eigen crop | 0.546 | 0.687 | KHÔNG ĐẠT |
| GT `depth/` chính thức (chưa lấp lỗ) / 10000 | 0.351 | 0.678 | KHÔNG ĐẠT |
| GT đúng thang: `depth_bfx` giải mã mm / 1000 | 0.157 | 0.831 | KHÔNG ĐẠT |
| GT đúng thang: `depth/` giải mã mm / 1000 | 0.114 | 0.850 | KHÔNG ĐẠT |

Để khớp paper cần đủ ba điều kiện: GT là bản ahanda (`depth_bfx` đọc /10000, tức thang
0.8×), eigen crop, và loại GT > 8 m. Flip-TTA gần như không ảnh hưởng.

Mọi cách đọc GT đúng thang đều cho sai số thấp hơn paper nhiều (0.114 – 0.157).
Đây là bằng chứng rằng số 0.500 / 0.660 của paper đến từ GT bị lệch scale (mục 1),
không phải từ model.

### iBims-1 (paper 0.150 / 0.714)

| Cấu hình | AbsRel | δ₁ | Kết luận |
|---|---:|---:|---|
| base ★ — eigen crop, ≤ 10 m, mask invalid + trong suốt, flip-TTA | 0.150 | 0.714 | KHỚP |
| không flip-TTA | 0.150 | 0.713 | ĐẠT |
| giữ pixel trong suốt | 0.151 | 0.714 | ĐẠT |
| không dùng mask nào | 0.151 | 0.714 | ĐẠT |
| `max_depth_eval = 50` | 0.151 | 0.711 | ĐẠT |
| bỏ eigen crop | 0.168 | 0.648 | KHÔNG ĐẠT |

Yếu tố quyết định ở iBims là eigen crop. Các thay đổi khác chỉ lệch tối đa 0.003.

### Virtual KITTI 2 (paper 0.085 / 0.913)

| Cấu hình | AbsRel | δ₁ | Kết luận |
|---|---:|---:|---|
| split seed 4 ★ | 0.085 | 0.913 | KHỚP (chọn sau khi biết kết quả) |
| split seed 1 | 0.084 | 0.913 | ĐẠT |
| split seed 2 | 0.084 | 0.914 | ĐẠT |
| trung bình 5 seed | 0.085 | 0.912 | ĐẠT |
| base — split seed 0, KB crop, garg crop, ≤ 80 m, flip-TTA | 0.085 | 0.911 | ĐẠT |
| split seed 3 | 0.086 | 0.910 | ĐẠT |
| không flip-TTA | 0.087 | 0.907 | KHÔNG ĐẠT |
| eigen crop KITTI thay cho garg crop | 0.099 | 0.876 | KHÔNG ĐẠT |
| bỏ garg crop | 0.123 | 0.801 | KHÔNG ĐẠT |

Cả 5 split đều đạt ngưỡng. Muốn đạt thì cần đủ hai điều kiện: flip-TTA và garg crop.

### DIODE Outdoor (paper 0.794 / 0.288)

| Cấu hình | AbsRel | δ₁ | Kết luận |
|---|---:|---:|---|
| base ★ — resize không antialias, garg crop, ≤ 80 m, không dùng `depth_mask`, flip-TTA | 0.794 | 0.288 | KHỚP |
| không flip-TTA | 0.794 | 0.289 | ĐẠT |
| `max_depth_eval = 300` (giữ cả GT > 80 m) | 0.792 | 0.286 | ĐẠT |
| `Resize` có antialias (mặc định torchvision ≥ 0.17) | 0.800 | 0.289 | KHÔNG ĐẠT |
| eigen crop KITTI thay cho garg crop | 0.867 | 0.292 | KHÔNG ĐẠT |
| bỏ garg crop | 0.867 | 0.316 | KHÔNG ĐẠT |
| áp `depth_mask.npy` | 0.572 | 0.290 | KHÔNG ĐẠT |

Để khớp paper, cấu hình phải thoả cả ba điều kiện:

1. Resize không antialias, như môi trường torchvision 0.14 gốc.
2. Dùng garg crop của KITTI.
3. Không dùng `depth_mask.npy`. Loader của ZoeDepth có đọc file này nhưng không
   áp dụng nó. Nếu áp dụng, AbsRel giảm từ 0.794 xuống 0.572. Như vậy các pixel mà
   DIODE đánh dấu là không hợp lệ (nhưng vẫn mang giá trị depth) góp một phần lớn
   vào sai số mà paper báo cáo.

## Các phát hiện về protocol

### 1. SUN RGB-D: GT của paper bằng 0.8 × độ sâu thật

Loader `zoedepth/data/sun_rgbd_loader.py` được kế thừa nguyên từ
[ZoeDepth](https://raw.githubusercontent.com/isl-org/ZoeDepth/main/zoedepth/data/sun_rgbd_loader.py).

- Nó đọc `rgb/img-XXXXXX.jpg` cùng `depth/N.png`, chia PNG cho 10000 và gán −1
  cho các giá trị > 8.
- [config.py](https://raw.githubusercontent.com/isl-org/ZoeDepth/main/zoedepth/utils/config.py)
  bật eigen crop, đặt `max_depth_eval = 8` và focal 715.0873.
- Bộ SUN RGB-D chính thức (`SUNRGBD.zip`) không có layout này.

Layout trên trùng với bản test 5050 ảnh do Ankur Handa phát hành, được liên kết
từ [ankurhanda/sunrgbd-meta-data](https://github.com/ankurhanda/sunrgbd-meta-data).
Bản này gồm hai file:

- `SUNRGBD-test_images.tgz`
- `sunrgb_test_depth.tgz`

Kết quả kiểm tra trên bản này:

- **5050/5050** file `depth/N.png` của ahanda trùng **từng byte** với
  `depth_bfx/*.png` chính thức (bản đã lấp lỗ). Không file nào trùng với `depth/` thô.
- `img-i` tương ứng phần tử `alltest[i-1]` trong
  `SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat`.

PNG chính thức lưu độ sâu dạng xoay bit. Toolbox giải mã như sau:

```
mm = bitor(bitshift(D,-3), bitshift(D,16-3))
```

Vì 3 bit thấp gần như luôn bằng 0, D ≈ 8·mm. Suy ra:

```
D / 10000 ≈ 0.8 · (mm / 1000)
```

Nói cách khác, GT mà paper dùng bằng **0.8 lần độ sâu thật**. Khi dùng GT đúng
thang, cùng một model đạt 0.114 / 0.850, tốt hơn nhiều so với 0.500 / 0.660 trong
paper. Con số trong Table 5 vì vậy phản ánh protocol (GT bị lệch scale), không
phải sai số trên độ sâu thật.

Loader chính thức của ZoeDepth cũng chia 10000 trên cùng layout này. Tuy nhiên,
nhóm **chưa chạy lại** hàng ZoeDepth (0.520 / 0.545) để xác nhận nó dùng cùng dữ liệu.

### 2. iBims-1: chuyển `.mat` sang `ibims1_core_raw`

Bản iBims-1 hiện tại (`iBims1.zip`, tải qua form tại [TUM](https://www.asg.ed.tum.de/lmf/ibims1/))
chỉ chứa `ibims1_core_mat/*.mat`. Trong khi đó, loader `ibims.py` cần thư mục
`ibims1_core_raw`.

`scripts/prepare_ibims.py` lấy các trường `depth`, `mask_invalid`, `mask_transp` và
`rgb` từ file `.mat`, rồi ghi thành PNG:

- Depth được lưu dạng uint16 `round(depth·65535/50)`. Loader giải mã bằng
  `value·50/65535`, nên sai số lượng tử hoá không vượt quá 0.4 mm.

Kết quả trùng paper.

### 3. DIODE Outdoor: antialias của `transforms.Resize`

`diode.py` resize ảnh RGB về cạnh ngắn 480 bằng `transforms.Resize(480)` trên tensor.
Có hai hành vi khác nhau:

- `metric_depth/environment.yml` gốc dùng torchvision 0.14.1. Phiên bản này resize
  tensor **không antialias**.
- Từ torchvision 0.17, `Resize` trên tensor mặc định `antialias=True`.

Với môi trường hiện tại (torchvision 0.19), cách chạy nguyên bản cho 0.800 / 0.289.
Vì vậy loader nay mặc định `antialias=False`. Có thể đổi bằng cờ
`--diode_resize_antialias=True`. Với thiết lập này, kết quả trùng paper ở 0.794 / 0.288.

Lần thử trước đó của nhóm, chạy bằng evaluator riêng `evaluate_diode_metric.py`, ra
0.799 / 0.289. Con số này phù hợp với trường hợp có antialias. Lần thử đó có thêm
DIODE Indoor, một phép thử không có trong paper. Xem
[table5/diode-metric.md](table5/diode-metric.md); kết quả nằm ở `results/table5/diode-metric/`.

### 4. Virtual KITTI 2: split test ngẫu nhiên, không được công bố

Loader gốc tự tạo `train.txt` / `test.txt` trong lần chạy đầu: lấy 8% khung hình
cuối của mỗi scene sau `random.shuffle` **không có seed**. Thứ tự `glob` và thứ tự
duyệt `set` của các scene cũng không cố định. Tác giả không phát hành file split này.

Repo bổ sung `vkitti2_split_seed` (mặc định 0):

- Danh sách file và scene được sắp xếp.
- Mỗi seed dùng `random.Random(seed)` riêng.
- Kết quả ghi ra `data/vkitti2/{train,test}_seed{N}.txt`.

Cách chia vẫn giữ 8% mỗi scene, Camera_0, đủ 5 scene và 10 biến thể, tổng 1701 ảnh test.

Seed 0 đạt 0.085 / 0.911. Phần phương sai theo seed được trình bày ở mục
[Phương sai split vKITTI2](#phương-sai-split-vkitti2). Truyền `--vkitti2_split=all`
để đánh giá toàn bộ khung hình.

### 5. Sửa lỗi riêng trên Windows (không ảnh hưởng metric)

- Các dataset đánh giá giữ thuộc tính `lambda`. Thuộc tính này không pickle được
  khi DataLoader `spawn` worker, nên trên Windows dùng `EVAL_WORKERS = 0`.
- Trong `vkitti2.py`, lệnh `.replace("/rgb/", "/depth/")` hỏng với đường dẫn `\`.
  Lệnh này được thay bằng `rgb_to_depth_path`, dùng regex không phụ thuộc dấu phân cách.

## Phương sai split vKITTI2

Mỗi split gồm 1701 ảnh: 8% khung hình mỗi scene, Camera_0. Tất cả dùng checkpoint
outdoor và cùng lệnh, chỉ khác `--vkitti2_split_seed`.

| Split seed | AbsRel ↓ | δ₁ ↑ | RMSE ↓ |
|---:|---:|---:|---:|
| 0 (mặc định, báo cáo ở bảng chính) | 0.085 | 0.911 | 4.082 |
| 1 | 0.084 | 0.913 | 4.019 |
| 2 | 0.084 | 0.914 | 4.052 |
| 3 | 0.086 | 0.910 | 4.004 |
| 4 | 0.085 | 0.913 | 4.016 |
| Trung bình ± độ lệch chuẩn | 0.085 ± 0.001 | 0.912 ± 0.002 | 4.035 |
| Paper | 0.085 | 0.913 | — |

Chỉ riêng việc đổi split đã làm AbsRel dao động trong khoảng 0.084–0.086 và δ₁
trong khoảng 0.910–0.914. Số của paper nằm trong khoảng này.

Seed 4 tình cờ trùng đúng 0.085 / 0.913. Tuy vậy, nó **không** được chọn làm kết quả
chính, vì chọn seed theo số của paper chính là chọn subset. Bảng chính giữ seed 0,
được cố định từ trước khi chạy.

Kết luận: phần lệch −0.002 của δ₁ ở seed 0 nằm trong mức nhiễu do split. Sai lệch
này không cho thấy khác biệt nào về model hay protocol.

## Khó khăn trong quá trình thực nghiệm

Phần này ghi lại các vấn đề đã gặp, để lần chạy sau không mất thời gian lại.
Mỗi dòng gồm: khó khăn → dấu hiệu nhận biết → cách xử lý.

### Về dữ liệu

| Khó khăn | Dấu hiệu | Cách xử lý |
|---|---|---|
| SUN RGB-D: paper không nói dùng file GT nào. `SUNRGBD.zip` chính thức có layout khác loader. | Loader cần `rgb/img-XXXXXX.jpg` và `depth/N.png`. Mọi cách giải mã đúng thang từ bản chính thức đều **không** ra 0.500 / 0.660 (0.114 – 0.351). | Lần theo layout tới bản test của Ankur Handa. Xác nhận 5050/5050 file depth trùng byte với `depth_bfx` và thứ tự RGB trùng `alltest`. Biến thể đối chứng tạo bằng `prepare_sunrgbd.py`. |
| SUN RGB-D: GT của paper bằng 0.8 × độ sâu thật. | Số paper trông "tệ", trong khi GT đúng thang cho 0.114 / 0.850. | Giữ protocol paper làm kết quả chính. Ghi rõ GT bị lệch scale (mục 1). |
| iBims-1: phải tải qua form. Bản hiện tại chỉ có `ibims1_core_mat/*.mat`. | Loader không tìm thấy `ibims1_core_raw/`. | Dùng `prepare_ibims.py` để chuyển `.mat` sang PNG. |
| vKITTI2: 2 file tar 7.1 GB + 7.6 GB, giải nén tốn thêm cùng dung lượng. | Đầy ổ đĩa. | Chỉ giải nén `rgb` và `depth`. Có thể xoá tar sau khi giải nén. |
| vKITTI2: split test không được công bố. Loader gốc chia ngẫu nhiên, không seed, và tự ghi `train.txt` / `test.txt` ở lần chạy đầu. | Mỗi máy ra số khác nhau. | Thêm `vkitti2_split_seed` (file `test_seed{N}.txt`). Báo cáo cả 5 seed. |
| Checkpoint: commit của nhóm đổi chỗ đặt sang `<repo>/checkpoints/`. | `FileNotFoundError ... depth_anything_vitl14.pth` | Chuyển `metric_depth/checkpoints/` lên `checkpoints/`. |

### Môi trường và mã nguồn

| Khó khăn | Dấu hiệu | Cách xử lý |
|---|---|---|
| torchvision 0.19 khác 0.14 (môi trường gốc): `Resize` trên tensor mặc định bật antialias. | DIODE ra 0.800 / 0.289 thay vì 0.794 / 0.288. Các bộ khác không đổi. | Loader DIODE mặc định `antialias=False`. |
| `infer_type` của ZoeDepth không xử lý boolean. | `--diode_resize_antialias=False` vẫn bị hiểu là True, vì `bool("False")` là True. | Thêm `as_bool` trong `data_mono.py`. |
| `compute_metrics` không có nhánh "không crop". | `NameError: eval_mask` khi `garg_crop` và `eigen_crop` cùng False. | `sweep_table5.py` dùng hàm chấm riêng cho cấu hình không crop. |
| Loader DIODE đọc `depth_mask.npy` nhưng không dùng. | Không báo lỗi. Chỉ lộ ra khi quét cấu hình: áp mask thì AbsRel còn 0.572. | Giữ nguyên để khớp paper và ghi chú lại. |
| xformers in traceback `No module named 'triton'` lúc khởi động. | Trông như lỗi nhưng chương trình vẫn chạy tiếp. Script theo dõi log grep "Traceback" / "Error" bị bắt nhầm và kết thúc sớm. | Bỏ qua cảnh báo này. Khi chờ job, kiểm tra dòng metric hoặc exit code, đừng grep "Error". |
| xformers không chạy trên CPU. Đặt `CUDA_VISIBLE_DEVICES=""` làm hỏng import. | Không smoke test được trên CPU. | Smoke test trên GPU với `--limit N`. |

### Windows

| Khó khăn | Dấu hiệu | Cách xử lý |
|---|---|---|
| DataLoader `spawn` worker phải pickle dataset, mà dataset có thuộc tính `lambda`. | Lỗi `Can't pickle local object '<lambda>'`. | Dùng `EVAL_WORKERS = 0` trên Windows. |
| Đường dẫn dùng `\`. | vKITTI2 không tìm thấy file depth, vì `.replace("/rgb/", "/depth/")` không khớp. | Thêm `rgb_to_depth_path` dùng regex. |
| Console cp1252. | `UnicodeEncodeError` khi in ký tự "δ". | Gọi `sys.stdout.reconfigure(encoding="utf-8")`. |
| Sửa file `.sh` trong lúc bash đang chạy chính file đó. | Bash đọc script theo từng đoạn, nên có thể chạy nhầm lệnh. | Không sửa script đang chạy. Xếp hàng job bằng lệnh riêng. |

### Phần cứng và thời gian

| Khó khăn | Dấu hiệu | Cách xử lý |
|---|---|---|
| GPU 4 GB (RTX 3050 Laptop), mỗi lúc chỉ chạy được 1 job. | SUN mất ≈ 78 phút mỗi lượt, vKITTI2 ≈ 24 phút mỗi seed. Tổng cộng ≈ 6 giờ GPU. | Xếp job tuần tự. `eval_sunrgbd_variants.py` và `sweep_table5.py` chỉ inference một lần rồi chấm nhiều GT và cấu hình. |
| Lỗi I/O tạm thời trong lần chạy dài. | Sweep SUN dừng ở ảnh 1621/5050 sau 25 phút với `OSError: broken data stream when reading image file`. Đọc lại thì cả 4 × 5050 file đều bình thường. | Thêm `read_png`, thử đọc lại tối đa 5 lần. |

### Phương pháp

| Khó khăn | Dấu hiệu | Cách xử lý |
|---|---|---|
| Nguy cơ chỉnh theo số của paper. | vKITTI2 seed 4 trùng đúng số paper. | Công bố đủ 5 seed, giữ seed 0 làm kết quả chính. Mọi cấu hình đều có kết luận theo ngưỡng, kể cả cấu hình KHÔNG ĐẠT. |
| Bảng tóm tắt cũ đặt các biến thể đối chứng cạnh số paper. | Dễ đọc nhầm thành "lệch nhiều", ví dụ `sunrgbd_mm` −0.386. | `summary.md` tách riêng "Kết quả chính" và "Quét cấu hình". Mỗi dòng có cột kết luận. |

## Dữ liệu

Mặc định, `HOME_DIR` trỏ tới `<repo>/data`. Có thể ghi đè bằng biến môi trường
`DA_DATA_ROOT`.

| Dataset | Nguồn | Thư mục config |
|---|---|---|
| SUN RGB-D | [`SUNRGBD-test_images.tgz`](http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz), [`sunrgb_test_depth.tgz`](https://www.doc.ic.ac.uk/~ahanda/sunrgb_test_depth.tgz) | `sunrgbd/test_ahanda` |
| SUN RGB-D (đối chứng) | [`SUNRGBD.zip`](https://rgbd.cs.princeton.edu/data/SUNRGBD.zip), [`SUNRGBDtoolbox.zip`](http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip) | `sunrgbd/test_{raw,raw_bfx,mm,mm_bfx}` |
| iBims-1 | [TUM iBims-1](https://www.asg.ed.tum.de/lmf/ibims1/) (`iBims1.zip`) | `ibims/ibims1_core_raw` |
| vKITTI2 | [`vkitti_2.0.3_rgb.tar`](http://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar), [`vkitti_2.0.3_depth.tar`](http://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar) | `vkitti2/` |
| DIODE | [`val.tar.gz`](http://diode-dataset.s3.amazonaws.com/val.tar.gz) | `diode/val/outdoor/` |

Checkpoint đặt tại `<repo>/checkpoints/` (xem `download_checkpoints.sh`):

- `depth_anything_vitl14.pth`, từ
  [checkpoints](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints).
- `depth_anything_metric_depth_{indoor,outdoor}.pt`, từ
  [checkpoints_metric_depth](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth).

Chuẩn bị dữ liệu (Git Bash, tại thư mục gốc repo):

```bash
# SUN RGB-D theo protocol paper
cd data/sunrgbd && mkdir -p test_ahanda/rgb \
  && tar -xzf ahanda/SUNRGBD-test_images.tgz -C test_ahanda/rgb \
  && tar -xzf ahanda/sunrgb_test_depth.tgz -C test_ahanda \
  && mv test_ahanda/sunrgbd_test_depth test_ahanda/depth && cd ../..

# SUN RGB-D đối chứng, tạo từ bản chính thức (không bắt buộc)
.venv/Scripts/python.exe scripts/prepare_sunrgbd.py --sun-root data/sunrgbd \
  --split data/sunrgbd/toolbox_dl/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat \
  --out data/sunrgbd/test --variants raw raw_bfx mm mm_bfx

# iBims-1
unzip -q data/iBims1.zip -d data/ibims
.venv/Scripts/python.exe scripts/prepare_ibims.py --src data/ibims/iBims-1 --out data/ibims/ibims1_core_raw

# vKITTI2 và DIODE: giải nén trực tiếp
tar -xf data/vkitti2/vkitti_2.0.3_rgb.tar -C data/vkitti2
tar -xf data/vkitti2/vkitti_2.0.3_depth.tar -C data/vkitti2
tar -xzf data/diode/val.tar.gz -C data/diode
```

## Chạy lại

```bash
bash scripts/run_table5.sh                      # diode ibims vkitti2 sunrgbd
bash scripts/run_table5.sh vkitti2_seeds diode_antialias   # các dòng đối chứng
for d in ibims diode_outdoor vkitti2 sunrgbd; do              # quét cấu hình
  .venv/Scripts/python.exe scripts/sweep_table5.py --dataset $d --run $d
done
.venv/Scripts/python.exe scripts/collect_table5.py results/table5 --tol 0.005
```

### Theo dõi tiến độ

Cả `evaluate.py` lẫn các script trong `scripts/` đều in thanh tiến độ tqdm vào file
log. Lệnh dưới đây in dòng tiến độ mới nhất, cập nhật mỗi 5 giây; nhấn Ctrl+C để
thoát. Đổi tên file để theo dõi bộ khác, ví dụ `results/table5/vkitti2_seed0.log`.

PowerShell:

```powershell
while ($true) { $l = ((Get-Content results\table5\sweep\sunrgbd.txt -Raw) -split "`r|`n" | Select-String "\d+/\d+ \[" | Select-Object -Last 1).Line; Write-Host -NoNewline "`r$l   "; Start-Sleep 5 }
```

Git Bash:

```bash
while sleep 5; do printf '\r%s' "$(tr '\r' '\n' < results/table5/sweep/sunrgbd.txt | grep -E '[0-9]+/[0-9]+ \[' | tail -1)"; done
```

### Chạy riêng một bộ

Ví dụ:

```bash
cd metric_depth
../.venv/Scripts/python.exe evaluate.py -m zoedepth \
  --pretrained_resource="local::./checkpoints/depth_anything_metric_depth_indoor.pt" -d sunrgbd
```

Để chấm SUN RGB-D với GT đúng thang, thêm
`--sunrgbd_root=<repo>/data/sunrgbd/test_mm --sunrgbd_depth_scale=1000`.

`scripts/eval_sunrgbd_variants.py` chỉ chạy inference một lần (cùng hàm `infer`,
có flip-average như `evaluate.py`), rồi chấm cùng dự đoán đó trên nhiều biến thể GT
bằng `compute_metrics`. Kết quả tương đương chạy `evaluate.py` riêng cho từng biến thể.

Thời gian chạy trên RTX 3050 Laptop 4 GB (chỉ để tham khảo):

| Bộ dữ liệu | Thời gian |
|---|---|
| SUN RGB-D | ≈ 78 phút |
| vKITTI2 | ≈ 24 phút mỗi seed |
| DIODE Outdoor | ≈ 6.5 phút |
| iBims-1 | ≈ 1.5 phút |

## Thay đổi mã

| File | Thay đổi |
|---|---|
| `metric_depth/zoedepth/utils/config.py` | `HOME_DIR = <repo>/data` hoặc `DA_DATA_ROOT`; root của 4 bộ; SUN mặc định `test_ahanda` cùng `sunrgbd_depth_scale` |
| `metric_depth/zoedepth/data/data_mono.py` | Đặt `EVAL_WORKERS`; truyền `sunrgbd_depth_scale`, `diode_resize_antialias`, `vkitti2_split` và `vkitti2_split_seed` |
| `metric_depth/zoedepth/data/sun_rgbd_loader.py` | Thêm tham số `depth_scale` (mặc định 10000, giữ nguyên hành vi cũ) |
| `metric_depth/zoedepth/data/diode.py` | Thêm `antialias` (mặc định False, như torchvision 0.14) |
| `metric_depth/zoedepth/data/vkitti2.py` | Glob có sắp xếp; split theo seed hoặc `all`; `rgb_to_depth_path` |
| `scripts/prepare_ibims.py`, `scripts/prepare_sunrgbd.py` | Chuẩn bị dữ liệu |
| `scripts/eval_sunrgbd_variants.py`, `scripts/run_table5.sh`, `scripts/sweep_table5.py`, `scripts/collect_table5.py` | Chạy, quét cấu hình, tổng hợp và chấm KHỚP / ĐẠT / KHÔNG ĐẠT |

## Artifacts

- `results/table5/<run>.log`: log đầy đủ của `evaluate.py`. Dòng cuối là dict metric.
- `results/table5/sunrgbd_variants.json` và `.txt`: metric của 4 biến thể GT SUN RGB-D.
- `results/table5/sweep/<dataset>.json` và `.txt`: kết quả quét cấu hình.
- `results/table5/summary.md` và `.json`: kết quả chính và bảng quét cấu hình, có
  kết luận theo ngưỡng. File này được sinh bởi `collect_table5.py`.
