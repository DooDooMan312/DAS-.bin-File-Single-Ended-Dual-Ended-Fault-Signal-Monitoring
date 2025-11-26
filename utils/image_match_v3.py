# -*- coding: utf-8 -*-

#  ------------------------------------------------------------------------------
#  Copyright (c) 2025 Chaos
#  All rights reserved.
#  #
#  This software is proprietary and confidential.
#  Licensed exclusively to Shineway Technologies, Inc for internal use only,
#  according to the NDA / agreement signed on 2025.11.26
#  Unauthorized redistribution or disclosure is prohibited.
#  ------------------------------------------------------------------------------
#
#

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


def advanced_similarity_calc(img1, img2, show=False):
    # 图像预处理
    img1 = cv2.resize(img1, (640, 640), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, (640, 640), interpolation=cv2.INTER_LINEAR)

    img1 *= 85
    img2 *= 85

    if img1 is None or img2 is None:
        raise ValueError("无法读取输入图像")

    def ensure_cv_compatible_gray(img, prefer_uint8=True):
        """把任意 dtype 的灰度图转换成 OpenCV 友好的类型（uint8 或 float32）"""
        arr = np.asarray(img)
        if prefer_uint8:
            # 优先用 uint8（大多数 OpenCV 算法默认假设 0..255）
            if arr.dtype == np.uint8:
                return arr
            # 若已经 0..255 但不是 uint8
            if np.issubdtype(arr.dtype, np.floating) and arr.min() >= 0 and arr.max() <= 255:
                return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
            # 若是 0..1 区间
            if np.issubdtype(arr.dtype, np.floating) and arr.min() >= 0 and arr.max() <= 1:
                return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
            # 其它情况线性拉伸
            m, M = float(arr.min()), float(arr.max())
            if M == m:
                return np.zeros_like(arr, dtype=np.uint8)
            arr = (arr - m) / (M - m) * 255.0
            return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        else:
            # 选择 float32 流程（假设值在 0..1，若不是则先归一化）
            arr = arr.astype(np.float32, copy=False)
            m, M = float(arr.min()), float(arr.max())
            if not (m >= 0.0 and M <= 1.0):
                if M == m:
                    return np.zeros_like(arr, dtype=np.float32)
                arr = (arr - m) / (M -m)
        return arr

    # 颜色直方图特征
    def compute_color_hist(img, bins=32):
        gray = ensure_cv_compatible_gray(img, prefer_uint8=True)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hist = cv2.calcHist([bgr], [0, 1, 2], None,
                            [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)  # 归一化（L2），类型为 float32
        return hist

    # 改进的SIFT检测器
    sift = cv2.SIFT_create(
        nfeatures=8000,
        contrastThreshold=0.01,
        edgeThreshold=15,
        sigma=1.2
    )

    # 多尺度金字塔构建
    def build_pyramid(img, max_levels=None, min_size=64):
        if max_levels is None:
            max_levels = int(np.log2(min(img.shape[:2]) / min_size)) + 1
        pyramid = [img.copy()]
        for _ in range(max_levels - 1):
            img = cv2.pyrDown(img)
            if img.shape[0] < min_size or img.shape[1] < min_size:
                break
            pyramid.append(img)
        return pyramid

    def to_sift_ready(img):
        if img is None or img.size == 0:
            raise ValueError("输入图像为空")
        if img.ndim == 3:  # BGR/BGRA -> 灰度
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 \
                else cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if img.dtype == np.float64 or img.dtype == np.float32:
            if np.nanmax(img) <= 4.0:
                img = img * 255.0
            img = img.astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img / 256.0).round().astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return np.ascontiguousarray(img)

    def safe_detect(sift, img):
        """永远返回 (kps, des)，其中 des 至少是 shape (0, 128) 的空数组，不会是 None。"""
        img = to_sift_ready(img)
        kps, des = sift.detectAndCompute(img, None)
        # kps： keypoints 图像中稳定的局部特征点；des: descriptors: 描述子，描绘kps附近的梯度信息        if des is None:
        des = np.empty((0, sift.descriptorSize()), dtype=np.float32)
        return kps, des

    # 特征处理流程（修正版）
    def process_pyramid(pyramid):
        """
        pyramid: list[np.ndarray]，从 640x640 开始，每层 /2 的降采样
        返回: list[ (scaled_kps, des) ]，其中 des 始终是 (N,128) 的 ndarray（可为 N=0）
        """
        sift = cv2.SIFT_create()  # 确保在定义前就已创建，避免 free variable 问题
        features = []

        for level, img in enumerate(pyramid):
            # 关键点与描述子
            kps, des = safe_detect(sift, img)

            # 把坐标/size 按 2**level 放回到基础尺度（level=0 不变）
            factor = 2 ** level
            if factor != 1:
                scaled_kps = [cv2.KeyPoint(
                    kp.pt[0] * factor,      # x
                    kp.pt[1] * factor,      # y
                    kp.size * factor,       # size
                    kp.angle,               # angle
                    kp.response,            # response
                    kp.octave,              # octave
                    kp.class_id             # class_id
                )for kp in kps]
            else:
                scaled_kps = kps  # 0 层无需拷贝

            features.append((scaled_kps, des))

        return features

    # 特征过滤
    def filter_features(kp, des, img_shape, max_per_region=150):
        h, w = img_shape[:2]
        grid_size = max(48, min(h, w) // 16)
        filtered_kp, filtered_des = [], []
        grid_counts = np.zeros((h // grid_size + 1, w // grid_size + 1), dtype=int)

        for kp_item, desc in zip(kp, des):
            x, y = map(int, kp_item.pt)
            gx, gy = x // grid_size, y // grid_size
            if grid_counts[gy, gx] < max_per_region:
                filtered_kp.append(kp_item)
                filtered_des.append(desc)
                grid_counts[gy, gx] += 1

        return filtered_kp, np.array(filtered_des)




    # 多尺度处理
    img1_pyramid = build_pyramid(img1)
    img2_pyramid = build_pyramid(img2)

    # 特征聚合与过滤（修正版）
    all_kp1, all_des1 = [], []
    for kp, des in process_pyramid(img1_pyramid):  # 正确调用
        filtered_kp, filtered_des = filter_features(kp, des, img1.shape)
        all_kp1.extend(filtered_kp)
        all_des1.extend(filtered_des.tolist())

    all_kp2, all_des2 = [], []
    for kp, des in process_pyramid(img2_pyramid):  # 正确调用
        filtered_kp, filtered_des = filter_features(kp, des, img2.shape)
        all_kp2.extend(filtered_kp)
        all_des2.extend(filtered_des.tolist())

    # # FLANN匹配
    # if not all_des1.size or not all_des2.size:
    #     return 0.0

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=300)

    all_des1 = np.array(all_des1, dtype=np.float32)  # shape:(1225, 128)
    all_des2 = np.array(all_des2, dtype=np.float32)  # shape:(1028, 128)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(all_des1, all_des2, k=2)

    # 双向匹配验证
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            back_match = flann.knnMatch(all_des2[m.trainIdx:m.trainIdx + 1],
                                        all_des1, k=1)
            if back_match and len(back_match[0]) == 1:
                if back_match[0][0].trainIdx == m.queryIdx:
                    good.append(m)

    # 空间验证
    inlier_ratio = 0.0
    if len(good) > 3:
        src_pts = np.float32([all_kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([all_kp2[m.trainIdx].pt for m in good])

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inlier_count = np.sum(mask)
        inlier_ratio = inlier_count / len(good) if good else 0

    # 颜色相似度
    color_sim = cv2.compareHist(compute_color_hist(img1),
                                compute_color_hist(img2),
                                cv2.HISTCMP_CORREL)

    # 综合相似度计算
    kp_ratio = min(len(all_kp1), len(all_kp2)) / max(len(all_kp1), len(all_kp2)) if max(len(all_kp1),
                                                                                        len(all_kp2)) else 0
    match_strength = len(good) / max(len(all_kp1), len(all_kp2)) if max(len(all_kp1), len(all_kp2)) else 0

    final_similarity = (
                               (inlier_ratio * 0.5) +
                               (match_strength * 0.3) +
                               (kp_ratio * 0.1) +
                               (color_sim * 0.08)
                       ) * 100

    # 可视化（可选）
    if show:
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=mask.ravel().tolist() if len(good) > 0 else [],
                           flags=2)
        result = cv2.drawMatches(img1, all_kp1, img2, all_kp2, good, None, **draw_params)
        cv2.imshow("Advanced Matches", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        pass

    return round(final_similarity, 3)


#======================
# SSIM + NCC + 直方图 为主的相似度计算
#======================
def _ncc_score(a, b):
    # 输入都先标准化到 uint8，再转 float32
    a8 = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b8 = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    a32 = a8.astype(np.float32);
    b32 = b8.astype(np.float32)
    am = (a32 - a32.mean()) / (a32.std() + 1e-6)
    bm = (b32 - b32.mean()) / (b32.std() + 1e-6)
    # 归一化互相关，范围约[-1,1]；映射到[0,1]
    c = float((am * bm).mean())
    return (c + 1.0) / 2.0

def _ensure_gray_u8(img):
    """确保是 2D uint8 灰度图"""
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # 统一类型
    if arr.dtype != np.float32:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    return arr

def compute_image_similarity_RP(img1, img2, *, w_ssim=0.5, w_ncc=0.3, w_hist=0.2, return_details=False):
    # 统一尺寸
    g1 = cv2.resize(img1, (640, 640), interpolation=cv2.INTER_LINEAR)
    g2 = cv2.resize(img2, (640, 640), interpolation=cv2.INTER_LINEAR)

    g1 = _ensure_gray_u8(g1)
    g2 = _ensure_gray_u8(g2)
    # SSIM（skimage 输出[-1,1]或[0,1]，此处保障到[0,1]）
    s, _ = ssim(
        cv2.normalize(g1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.normalize(g2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        full=True
    )
    ssim_score = float(np.clip(s, 0.0, 1.0))

    # NCC
    ncc = _ncc_score(g1, g2)  # [0,1]

    # 灰度直方图（单通道更合适）
    h1 = cv2.calcHist([g1], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [64], [0, 256])
    cv2.normalize(h1, h1);
    cv2.normalize(h2, h2)
    hist_intersection = float(np.minimum(h1, h2).sum() / np.maximum(h1, h2).sum())  # [0,1]

    final = (w_ssim * ssim_score + w_ncc * ncc + w_hist * hist_intersection) * 100.0
    if return_details:
        return final, {"ssim": ssim_score, "ncc": ncc, "hist": hist_intersection}
    return final


    #======================


if __name__ == '__main__':
    # 使用示例
    y = np.random.randn(2, 360, 360)
    similarity = advanced_similarity_calc(y[0], y[1], show=False)
    print(f"高级相似度：{similarity}%")