import json

notebook = {'cells': [{'cell_type': 'markdown',
            'metadata': {},
            'source': ['# THỰC NGHIỆM 1: ƯỚC LƯỢNG ĐỘ SÂU TƯƠNG ĐỐI ZERO-SHOT (ZERO-SHOT RELATIVE DEPTH '
                       'ESTIMATION)\n',
                       '\n',
                       '> **Công trình nghiên cứu:** *Depth Anything: Unleashing the Power of Large-Scale '
                       'Unlabeled Data* (Yang et al., **CVPR 2024**)\n',
                       '>\n',
                       '> **Bảng 2 trong bài báo gốc:** Đánh giá khả năng tổng quát hóa (Zero-shot) trên 6 '
                       'bộ dữ liệu benchmark chính thức:\n',
                       '> 1. **NYUv2**: 654 ảnh (Official Test Split)\n',
                       '> 2. **DIODE**: 771 ảnh (Validation Split: Indoor + Outdoor)\n',
                       '> 3. **KITTI**: 652 ảnh (Eigen Test Split)\n',
                       '> 4. **ETH3D**: 454 ảnh (Test Split)\n',
                       '> 5. **DDAD**: 3,950 ảnh (Validation Split)\n',
                       '> 6. **Sintel**: 1,064 ảnh (Test Split)\n',
                       '>\n',
                       '> Notebook này thực thi kiểm thử lần lượt từng bộ dữ liệu trong số 6 bộ thuộc Thực '
                       'nghiệm 1.\n',
                       '\n',
                       '**Trạng thái triển khai:** NYUv2 và DIODE dùng evaluator đã rà soát. Các cell '
                       'KITTI/ETH3D/DDAD/Sintel là nhánh cũ, chưa được xác nhận protocol/kết quả. Để chỉ '
                       'chạy DIODE, dùng notebook `thuc_nghiem_1_diode_relative_depth.ipynb` hoặc các cell '
                       'setup/model và DIODE bên dưới.\n']},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 1. Môi trường & Cấu hình PyTorch GPU\n',
                       'Kiểm tra môi trường tính toán và GPU CUDA (`RTX 3050 Laptop GPU`).']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ['import os\n',
                       'import sys\n',
                       'import glob\n',
                       'import numpy as np\n',
                       'import cv2\n',
                       'import torch\n',
                       'import torch.nn.functional as F\n',
                       'from tqdm.notebook import tqdm\n',
                       'import pandas as pd\n',
                       'from tabulate import tabulate\n',
                       '\n',
                       'sys.path.append(os.getcwd())\n',
                       '\n',
                       'from depth_anything.dpt import DepthAnything\n',
                       'from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet\n',
                       'from torchvision.transforms import Compose\n',
                       '\n',
                       "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                       'print(f"[Setup] Running on Device: {device}")\n',
                       "if device == 'cuda':\n",
                       '    print(f"[GPU Device] {torch.cuda.get_device_name(0)}")']},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 2. Căn chỉnh và đánh giá relative depth\n',
                       'NYUv2 dùng `evaluate_nyuv2.py`: đọc **rawDepths** (độ sâu đo được, chưa lấp lỗ), '
                       'mask toàn ảnh `0 < depth < 10 m` theo MiDaS; không dùng Eigen crop của benchmark '
                       'metric depth.\n',
                       'Fit affine least squares float64 giữa dự đoán và 1/GT; tính AbsRel và delta1 ngay '
                       'trong disparity space, không min-max về [0,1]. Giới hạn disparity dự đoán tối thiểu '
                       '1/10 theo depth cap của MiDaS.\n',
                       'Nguồn: [tác giả Depth Anything, issue '
                       '#174](https://github.com/LiheYoung/Depth-Anything/issues/174) và [mã đánh giá NYUv2 '
                       'của MiDaS](https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022). JSON '
                       'kết quả ghi toàn bộ protocol.\n']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ['def compute_scale_and_shift(prediction, target, mask):\n',
                       '    """Least squares affine alignment: min_{s, t} sum((s * pred + t - '
                       'target)^2)"""\n',
                       '    a, b = prediction[mask], target[mask]\n',
                       '    if len(a) == 0:\n',
                       '        return 1.0, 0.0\n',
                       '    system_mat = np.zeros((2, 2))\n',
                       '    system_mat[0, 0] = np.sum(a * a)\n',
                       '    system_mat[0, 1] = np.sum(a)\n',
                       '    system_mat[1, 0] = system_mat[0, 1]\n',
                       '    system_mat[1, 1] = np.sum(mask)\n',
                       '    rhs = np.zeros((2, 1))\n',
                       '    rhs[0, 0] = np.sum(a * b)\n',
                       '    rhs[1, 0] = np.sum(b)\n',
                       '    try:\n',
                       '        scale_shift = np.linalg.solve(system_mat, rhs)\n',
                       '        scale, shift = float(scale_shift[0, 0]), float(scale_shift[1, 0])\n',
                       '    except np.linalg.LinAlgError:\n',
                       '        scale, shift = 1.0, 0.0\n',
                       '    return scale, shift\n',
                       '\n',
                       'def compute_metrics(gt, pred, mask):\n',
                       '    """Compute AbsRel and Delta1 metrics"""\n',
                       '    gt_valid, pred_valid = gt[mask], pred[mask]\n',
                       '    if len(gt_valid) == 0:\n',
                       '        return 0.0, 0.0\n',
                       '    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)\n',
                       '    ratio = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)\n',
                       '    delta1 = np.mean(ratio < 1.25)\n',
                       '    return abs_rel, delta1']},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 3. Load Trọng số Mô hình Depth Anything (ViT-L)\n',
                       'Nạp checkpoint pre-trained `./checkpoints/depth_anything_vitl14.pth`.']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ["encoder = 'vitl'\n",
                       'model_configs = {\n',
                       "    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, "
                       '1024]},\n',
                       "    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, "
                       '768]},\n',
                       "    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, "
                       '384]}\n',
                       '}\n',
                       "checkpoint_path = f'./checkpoints/depth_anything_{encoder}14.pth'\n",
                       'print(f"[Model] Nạp checkpoint: {checkpoint_path}")\n',
                       'model = DepthAnything(model_configs[encoder])\n',
                       "model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))\n",
                       'model.to(device).eval()\n',
                       '\n',
                       'transform = Compose([\n',
                       '    Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,\n',
                       "           ensure_multiple_of=14, resize_method='lower_bound', "
                       'image_interpolation_method=cv2.INTER_CUBIC),\n',
                       '    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n',
                       '    PrepareForNet(),\n',
                       '])\n',
                       'print("[Model] Khởi tạo mô hình thành công!")\n',
                       '\n',
                       'results = []\n',
                       'paper_benchmarks = {\n',
                       "    'NYUv2': {'type': 'Indoor (Trong nhà)', 'target_samples': 654, 'paper_abs_rel': "
                       "0.043, 'paper_delta1': 0.981},\n",
                       "    'DIODE': {'type': 'Indoor + Outdoor', 'target_samples': 771, 'paper_abs_rel': "
                       "0.066, 'paper_delta1': 0.952},\n",
                       "    'KITTI': {'type': 'Urban Driving', 'target_samples': 652, 'paper_abs_rel': "
                       "0.076, 'paper_delta1': 0.947},\n",
                       "    'ETH3D': {'type': 'Indoor + Outdoor', 'target_samples': 454, 'paper_abs_rel': "
                       "0.127, 'paper_delta1': 0.882},\n",
                       "    'DDAD': {'type': 'Multi-camera Driving', 'target_samples': 3950, "
                       "'paper_abs_rel': 0.230, 'paper_delta1': 0.789},\n",
                       "    'Sintel': {'type': '3D CGI Animation', 'target_samples': 1064, 'paper_abs_rel': "
                       "0.458, 'paper_delta1': 0.760},\n",
                       '}']},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 4.1. Đánh giá NYUv2: 654 ảnh test, protocol MiDaS\n',
                       'Lần kiểm chứng CLI 2026-11-26: ViT-L, AbsRel **0.0436349930**, δ₁ **0.9803277444**; '
                       'paper **0.043 / 0.981**. Đã gần nhưng chưa trùng tuyệt đối. Xem [báo cáo '
                       'protocol](docs/nyuv2-reproduction.md). Cell dưới dùng lại evaluator và cache của '
                       'CLI.\n']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ['from evaluate_nyuv2 import evaluate_nyuv2\n',
                       '\n',
                       'nyu_summary = evaluate_nyuv2(\n',
                       "    encoder=encoder, data_dir='./data', checkpoint_dir='./checkpoints',\n",
                       '    model=model, transform=transform,\n',
                       ')\n',
                       '# Replace any earlier NYUv2 row when this cell is run again.\n',
                       "results = [row for row in results if row['Dataset'] != 'NYUv2']\n",
                       "bm_nyu = paper_benchmarks['NYUv2']\n",
                       'results.append({\n',
                       "    'Dataset': 'NYUv2', 'Loại môi trường': bm_nyu['type'],\n",
                       "    'Số ảnh đã chạy': nyu_summary['num_images'],\n",
                       '    \'Our AbsRel ↓\': f"{nyu_summary[\'metrics\'][\'abs_rel\']:.4f}",\n',
                       '    \'Paper AbsRel ↓\': f"{nyu_summary[\'paper_reference\'][\'abs_rel\']:.4f}",\n',
                       '    \'Our Delta1 ↑\': f"{nyu_summary[\'metrics\'][\'delta1\']:.4f}",\n',
                       '    \'Paper Delta1 ↑\': f"{nyu_summary[\'paper_reference\'][\'delta1\']:.4f}",\n',
                       "    'Trạng thái': 'Đã hoàn thành (disparity)',\n",
                       '})\n',
                       'nyu_summary\n']},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 4.2. DIODE native validation: 325 indoor + 446 outdoor\n',
                       'Dùng chung `evaluate_diode_relative.py`: mask chính thức, fit và chấm disparity, '
                       'không crop/depth cap. Mốc ViT-L Table 2: **0.066 / 0.952**; split/clipping/TTA của '
                       'paper chưa được xác nhận đầy đủ. Xem [protocol](docs/diode-relative-depth.md).\n',
                       '\n',
                       'Chạy cell này riêng sau phần setup/model để chỉ đánh giá DIODE. '
                       "`diode_subset='all'`, `diode_limit=None` chạy đủ 771 ảnh; limit là chẩn đoán. Cache "
                       'tự phục hồi khi chạy lại. NYUv2 không cần chạy trước.\n']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ['from evaluate_diode_relative import evaluate_diode\n',
                       '\n',
                       "diode_subset = 'all'\n",
                       'diode_limit = None\n',
                       'diode_flip = False\n',
                       '\n',
                       'diode_summary = evaluate_diode(\n',
                       "    encoder=encoder, data_dir=data_dir, checkpoint_dir='./checkpoints',\n",
                       '    subset=diode_subset, limit=diode_limit, flip=diode_flip,\n',
                       '    model=model, transform=transform,\n',
                       "    data_audit='results/diode-relative/audit/official-archive-verification.json',\n",
                       "    cache_dir='.cache/diode-audit/fresh-predictions',\n",
                       "    output_dir='results/diode-relative/audit-rerun',\n",
                       ')\n',
                       '# Thay dòng DIODE cũ khi chạy lại cell, tránh lặp kết quả.\n',
                       "results = [row for row in results if row['Dataset'] != 'DIODE']\n",
                       'results.append({\n',
                       "    'Dataset': 'DIODE', 'Loại môi trường': diode_subset,\n",
                       "    'Số ảnh đã chạy': diode_summary['num_images'],\n",
                       '    \'Our AbsRel ↓\': f"{diode_summary[\'metrics\'][\'abs_rel\']:.6f}",\n',
                       '    \'Paper AbsRel ↓\': f"{diode_summary[\'paper_reference\'][\'abs_rel\']:.3f}",\n',
                       '    \'Our Delta1 ↑\': f"{diode_summary[\'metrics\'][\'delta1\']:.6f}",\n',
                       '    \'Paper Delta1 ↑\': f"{diode_summary[\'paper_reference\'][\'delta1\']:.3f}",\n',
                       "    'Trạng thái': 'Native validation; đối chiếu paper còn phụ thuộc protocol',\n",
                       '})\n',
                       "pd.DataFrame([{'domain': d, 'num_images': v['num_images'], **v['metrics']}\n",
                       "              for d, v in diode_summary['by_domain'].items()])\n"]},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 4.3. Đánh giá Bộ dữ liệu 3/6: KITTI Eigen Test Split (652 ảnh)']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ["kitti_dir = os.path.join(data_dir, 'kitti')\n",
                       "kitti_gt_dir = os.path.join(kitti_dir, 'gt', 'val')\n",
                       "kitti_raw_dir = os.path.join(kitti_dir, 'raw')\n",
                       "kitti_split_file = os.path.join('metric_depth', 'train_test_inputs', "
                       "'kitti_eigen_test_files_with_gt.txt')\n",
                       "bm_kitti = paper_benchmarks['KITTI']\n",
                       '\n',
                       'if os.path.exists(kitti_gt_dir) and os.path.exists(kitti_raw_dir) and '
                       'os.path.exists(kitti_split_file):\n',
                       '    print("[3/6] Đang chạy thực nghiệm trên KITTI Eigen Test Split (652 ảnh)...")\n',
                       "    with open(kitti_split_file, 'r') as f:\n",
                       '        kitti_lines = [line.strip().split() for line in f if line.strip()]\n',
                       '    abs_rels, delta1s = [], []\n',
                       '    for line in tqdm(kitti_lines, desc="KITTI Eigen Test"):\n',
                       '        img_p = os.path.join(kitti_raw_dir, line[0])\n',
                       '        gt_p = os.path.join(kitti_gt_dir, line[1])\n',
                       '        if not (os.path.exists(img_p) and os.path.exists(gt_p)):\n',
                       '            continue\n',
                       '        raw_img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB) / 255.0\n',
                       '        gt_depth = cv2.imread(gt_p, cv2.IMREAD_ANYDEPTH) / 256.0\n',
                       '        h, w = raw_img.shape[:2]\n',
                       "        img_tensor = torch.from_numpy(transform({'image': "
                       "raw_img})['image']).unsqueeze(0).to(device)\n",
                       '        with torch.no_grad():\n',
                       '            pred = model(img_tensor)\n',
                       "            pred = F.interpolate(pred.unsqueeze(1), size=(h, w), mode='bilinear', "
                       'align_corners=False).squeeze().cpu().numpy()\n',
                       '        valid_mask = (gt_depth > 0.001) & (gt_depth < 80.0)\n',
                       '        if np.sum(valid_mask) < 10:\n',
                       '            continue\n',
                       '        scale, shift = compute_scale_and_shift(pred, gt_depth, valid_mask)\n',
                       '        pred_aligned = scale * pred + shift\n',
                       '        abs_r, d1 = compute_metrics(gt_depth, pred_aligned, valid_mask)\n',
                       '        abs_rels.append(abs_r)\n',
                       '        delta1s.append(d1)\n',
                       '    if len(abs_rels) > 0:\n',
                       '        res_absrel, res_d1 = float(np.mean(abs_rels)), float(np.mean(delta1s))\n',
                       '        results.append({\n',
                       "            'Dataset': 'KITTI', 'Loại môi trường': bm_kitti['type'], 'Số ảnh đã "
                       "chạy': len(abs_rels),\n",
                       '            \'Our AbsRel ↓\': f"{res_absrel:.4f}", \'Paper AbsRel ↓\': '
                       'f"{bm_kitti[\'paper_abs_rel\']:.4f}",\n',
                       '            \'Our Delta1 ↑\': f"{res_d1:.4f}", \'Paper Delta1 ↑\': '
                       'f"{bm_kitti[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Đã hoàn thành\'\n',
                       '        })\n',
                       '        print(f" -> KITTI KẾT QUẢ ({len(abs_rels)} ảnh): AbsRel = {res_absrel:.4f} '
                       "(Bài báo: {bm_kitti['paper_abs_rel']:.4f}), Delta1 = {res_d1:.4f} (Bài báo: "
                       '{bm_kitti[\'paper_delta1\']:.4f})")\n',
                       '    else:\n',
                       "        results.append({'Dataset': 'KITTI', 'Loại môi trường': bm_kitti['type'], 'Số "
                       "ảnh đã chạy': 0, 'Our AbsRel ↓': '-', 'Paper AbsRel ↓': "
                       'f"{bm_kitti[\'paper_abs_rel\']:.4f}", \'Our Delta1 ↑\': \'-\', \'Paper Delta1 ↑\': '
                       'f"{bm_kitti[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Bỏ qua (Chưa giải nén '
                       "GT)'})\n",
                       'else:\n',
                       '    print("[3/6] Bỏ qua KITTI: Chưa có dữ liệu đầy đủ tại ./data/kitti")\n',
                       "    results.append({'Dataset': 'KITTI', 'Loại môi trường': bm_kitti['type'], 'Số ảnh "
                       "đã chạy': 0, 'Our AbsRel ↓': '-', 'Paper AbsRel ↓': "
                       'f"{bm_kitti[\'paper_abs_rel\']:.4f}", \'Our Delta1 ↑\': \'-\', \'Paper Delta1 ↑\': '
                       'f"{bm_kitti[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Bỏ qua (Chưa có data)\'})']},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 4.4. Đánh giá Bộ dữ liệu 4/6: ETH3D Test Split (454 ảnh)']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ["eth3d_dir = os.path.join(data_dir, 'eth3d')\n",
                       "eth3d_img_dir = os.path.join(eth3d_dir, 'multi_view_training_dslr_undistorted')\n",
                       "eth3d_gt_dir = os.path.join(eth3d_dir, 'multi_view_training_dslr_scan_eval')\n",
                       "bm_eth3d = paper_benchmarks['ETH3D']\n",
                       '\n',
                       'if os.path.exists(eth3d_img_dir) and os.path.exists(eth3d_gt_dir):\n',
                       '    print("[4/6] Đang kiểm tra dữ liệu ETH3D Benchmark...")\n',
                       "    img_files = sorted(glob.glob(os.path.join(eth3d_img_dir, '**', '*.JPG'), "
                       'recursive=True))\n',
                       "    depth_files = sorted(glob.glob(os.path.join(eth3d_gt_dir, '**', '*.pfm'), "
                       'recursive=True))\n',
                       '    if len(img_files) > 0 and len(depth_files) > 0:\n',
                       '        pass\n',
                       '    else:\n',
                       '        print("[4/6] Bỏ qua ETH3D: Đã tìm thấy ảnh JPG nhưng chưa giải nén file '
                       'depth .pfm")\n',
                       "        results.append({'Dataset': 'ETH3D', 'Loại môi trường': bm_eth3d['type'], 'Số "
                       "ảnh đã chạy': 0, 'Our AbsRel ↓': '-', 'Paper AbsRel ↓': "
                       'f"{bm_eth3d[\'paper_abs_rel\']:.4f}", \'Our Delta1 ↑\': \'-\', \'Paper Delta1 ↑\': '
                       'f"{bm_eth3d[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Bỏ qua (Chưa có depth '
                       ".pfm)'})\n",
                       'else:\n',
                       '    print("[4/6] Bỏ qua ETH3D: Chưa có dữ liệu tại ./data/eth3d")\n',
                       "    results.append({'Dataset': 'ETH3D', 'Loại môi trường': bm_eth3d['type'], 'Số ảnh "
                       "đã chạy': 0, 'Our AbsRel ↓': '-', 'Paper AbsRel ↓': "
                       'f"{bm_eth3d[\'paper_abs_rel\']:.4f}", \'Our Delta1 ↑\': \'-\', \'Paper Delta1 ↑\': '
                       'f"{bm_eth3d[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Bỏ qua (Chưa có data)\'})']},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 4.5. Đánh giá Bộ dữ liệu 5/6: DDAD Validation Split (3,950 ảnh)']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ["ddad_dir = os.path.join(data_dir, 'ddad')\n",
                       "bm_ddad = paper_benchmarks['DDAD']\n",
                       'if os.path.exists(ddad_dir):\n',
                       '    print("[5/6] Đang kiểm tra dữ liệu DDAD Validation Split (3,950 ảnh)...")\n',
                       "    results.append({'Dataset': 'DDAD', 'Loại môi trường': bm_ddad['type'], 'Số ảnh "
                       "đã chạy': 0, 'Our AbsRel ↓': '-', 'Paper AbsRel ↓': "
                       'f"{bm_ddad[\'paper_abs_rel\']:.4f}", \'Our Delta1 ↑\': \'-\', \'Paper Delta1 ↑\': '
                       'f"{bm_ddad[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Bỏ qua (Chưa giải nén '
                       "tarball)'})\n",
                       'else:\n',
                       '    print("[5/6] Bỏ qua DDAD: Chưa có dữ liệu tại ./data/ddad")\n',
                       "    results.append({'Dataset': 'DDAD', 'Loại môi trường': bm_ddad['type'], 'Số ảnh "
                       "đã chạy': 0, 'Our AbsRel ↓': '-', 'Paper AbsRel ↓': "
                       'f"{bm_ddad[\'paper_abs_rel\']:.4f}", \'Our Delta1 ↑\': \'-\', \'Paper Delta1 ↑\': '
                       'f"{bm_ddad[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Bỏ qua (Chưa có data)\'})']},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 4.6. Đánh giá Bộ dữ liệu 6/6: Sintel Depth Benchmark (1,064 ảnh)']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ['def read_sintel_depth(filepath):\n',
                       "    with open(filepath, 'rb') as f:\n",
                       '        TAG_FLOAT = 202021.25\n',
                       '        tag = np.fromfile(f, np.float32, count=1)[0]\n',
                       '        if tag != TAG_FLOAT:\n',
                       '            raise ValueError(f"Invalid tag {tag}")\n',
                       '        width = np.fromfile(f, np.int32, count=1)[0]\n',
                       '        height = np.fromfile(f, np.int32, count=1)[0]\n',
                       '        data = np.fromfile(f, np.float32, count=width * height)\n',
                       '        return data.reshape((height, width))\n',
                       '\n',
                       "sintel_dir = os.path.join(data_dir, 'sintel')\n",
                       "sintel_depth_dir = os.path.join(sintel_dir, 'MPI-Sintel-training_extras', "
                       "'training', 'depth')\n",
                       "sintel_img_dir = os.path.join(sintel_dir, 'MPI-Sintel-training_images', 'training', "
                       "'final')\n",
                       "bm_sintel = paper_benchmarks['Sintel']\n",
                       '\n',
                       'if os.path.exists(sintel_depth_dir) and os.path.exists(sintel_img_dir):\n',
                       '    print("[6/6] Đang chạy thực nghiệm trên Sintel Depth Benchmark (1,064 '
                       'ảnh)...")\n',
                       "    depth_files = sorted(glob.glob(os.path.join(sintel_depth_dir, '**', '*.dpt'), "
                       'recursive=True))\n',
                       '    abs_rels, delta1s = [], []\n',
                       '    for dpt_p in tqdm(depth_files, desc="Sintel Benchmark"):\n',
                       "        img_p = dpt_p.replace('MPI-Sintel-training_extras', "
                       "'MPI-Sintel-training_images').replace('depth', 'final').replace('.dpt', '.png')\n",
                       '        if not os.path.exists(img_p):\n',
                       "            img_p = img_p.replace('final', 'clean')\n",
                       '        if not os.path.exists(img_p):\n',
                       '            continue\n',
                       '        raw_img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB) / 255.0\n',
                       '        h, w = raw_img.shape[:2]\n',
                       "        img_tensor = torch.from_numpy(transform({'image': "
                       "raw_img})['image']).unsqueeze(0).to(device)\n",
                       '        with torch.no_grad():\n',
                       '            pred = model(img_tensor)\n',
                       "            pred = F.interpolate(pred.unsqueeze(1), size=(h, w), mode='bilinear', "
                       'align_corners=False).squeeze().cpu().numpy()\n',
                       '        gt_depth = read_sintel_depth(dpt_p)\n',
                       '        valid_mask = (gt_depth > 0) & (gt_depth < 200.0)\n',
                       '        if np.sum(valid_mask) < 10:\n',
                       '            continue\n',
                       '        scale, shift = compute_scale_and_shift(pred, gt_depth, valid_mask)\n',
                       '        pred_aligned = scale * pred + shift\n',
                       '        abs_r, d1 = compute_metrics(gt_depth, pred_aligned, valid_mask)\n',
                       '        abs_rels.append(abs_r)\n',
                       '        delta1s.append(d1)\n',
                       '    if len(abs_rels) > 0:\n',
                       '        res_absrel, res_d1 = float(np.mean(abs_rels)), float(np.mean(delta1s))\n',
                       '        results.append({\n',
                       "            'Dataset': 'Sintel', 'Loại môi trường': bm_sintel['type'], 'Số ảnh đã "
                       "chạy': len(abs_rels),\n",
                       '            \'Our AbsRel ↓\': f"{res_absrel:.4f}", \'Paper AbsRel ↓\': '
                       'f"{bm_sintel[\'paper_abs_rel\']:.4f}",\n',
                       '            \'Our Delta1 ↑\': f"{res_d1:.4f}", \'Paper Delta1 ↑\': '
                       'f"{bm_sintel[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Đã hoàn thành\'\n',
                       '        })\n',
                       '        print(f" -> Sintel KẾT QUẢ ({len(abs_rels)} ảnh): AbsRel = {res_absrel:.4f} '
                       "(Bài báo: {bm_sintel['paper_abs_rel']:.4f}), Delta1 = {res_d1:.4f} (Bài báo: "
                       '{bm_sintel[\'paper_delta1\']:.4f})")\n',
                       '    else:\n',
                       "        results.append({'Dataset': 'Sintel', 'Loại môi trường': bm_sintel['type'], "
                       "'Số ảnh đã chạy': 0, 'Our AbsRel ↓': '-', 'Paper AbsRel ↓': "
                       'f"{bm_sintel[\'paper_abs_rel\']:.4f}", \'Our Delta1 ↑\': \'-\', \'Paper Delta1 ↑\': '
                       'f"{bm_sintel[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Bỏ qua (Chưa giải nén file '
                       "depth .dpt)'})\n",
                       'else:\n',
                       '    print("[6/6] Bỏ qua Sintel: Chưa giải nén thư mục depth .dpt tại '
                       './data/sintel")\n',
                       "    results.append({'Dataset': 'Sintel', 'Loại môi trường': bm_sintel['type'], 'Số "
                       "ảnh đã chạy': 0, 'Our AbsRel ↓': '-', 'Paper AbsRel ↓': "
                       'f"{bm_sintel[\'paper_abs_rel\']:.4f}", \'Our Delta1 ↑\': \'-\', \'Paper Delta1 ↑\': '
                       'f"{bm_sintel[\'paper_delta1\']:.4f}", \'Trạng thái\': \'Bỏ qua (Chưa giải nén file '
                       "depth .dpt)'})"]},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 5. Tổng hợp Bảng Kết quả Thực nghiệm 1 & So sánh Bài báo gốc (CVPR 2024 Table '
                       '2)']},
           {'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': ['df = pd.DataFrame(results)\n',
                       'print("\\n" + "="*90)\n',
                       'print(f"BẢNG KẾT QUẢ THỰC NGHIỆM 1: ZERO-SHOT RELATIVE DEPTH (Depth Anything '
                       '{encoder.upper()})")\n',
                       'print("="*90)\n',
                       "print(tabulate(df, headers='keys', tablefmt='github', showindex=False))\n",
                       'df']}],
 'metadata': {'kernelspec': {'display_name': '.venv', 'language': 'python', 'name': 'python3'},
              'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3},
                                'file_extension': '.py',
                                'mimetype': 'text/x-python',
                                'name': 'python',
                                'nbconvert_exporter': 'python',
                                'pygments_lexer': 'ipython3',
                                'version': '3.12.11'}},
 'nbformat': 4,
 'nbformat_minor': 2}

if __name__ == '__main__':
    with open('thuc_nghiem_1_relative_depth.ipynb', 'w', encoding='utf-8') as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
    print('Generated notebook with shared NYUv2 and DIODE evaluators.')
