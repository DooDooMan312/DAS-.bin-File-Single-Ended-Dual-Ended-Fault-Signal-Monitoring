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

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import os

def numpy_array2RP(np_array, eps, steps, OUTDIR):
    """处理 NumPy 数组并生成递归图"""
    np_array = np_array.reshape(-1, 1)
    # print(f"{rec} array shape: {np_array.shape}")

    # 归一化处理
    scaler = StandardScaler()
    signal = scaler.fit_transform(np_array.reshape(-1, 1)).flatten()

    # print(f"Processed signal shape: {signal.shape}")
    rp = recurrence_plot(signal, eps, steps)

    # 生成并保存递归图
    # save_plot(rp, OUTDIR)
    return rp

@staticmethod
def recurrence_plot(signal, eps=0.05, steps=3):
    """生成递归图"""
    _2d_array = signal[:, None]
    distance = pdist(_2d_array)
    # distance = np.floor(distance / eps)  # 取整
    distance = distance / eps
    distance[distance > steps] = steps
    return np.nan_to_num(squareform(distance), nan=0.0, posinf=255.0, neginf=0.0)

def save_plot(rp, OUTDIR):
    """绘制并保存递归图"""
    plt.figure(figsize=(6, 6))
    plt.imshow(rp, cmap='binary')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    plt.savefig(OUTDIR, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    # print(f"Saved RP plot: {save_path}")


def create_RP(np_array, eps=0.05, steps=255, OUTDIR="out"):
    RP_images = []  # 用列表存放每个RP图
    for i in np.arange(np_array.shape[1]):
        out_path = os.path.join(OUTDIR, f"{i}RP.png")
        RP_image = numpy_array2RP(np_array[:, i], eps=eps, steps=steps, OUTDIR=out_path)
        # print(RP_image.shape)
        RP_images.append(RP_image)  # 追加到列表中
    RP_images = np.stack(RP_images)  # 转成 numpy 数组 (N, 360, 360)
    return RP_images


if __name__ == '__main__':
    np_array = np.random.randn(180, 2)
    # for i in np.arange(np_array.shape[1]):
    #     RP_image = numpy_array2RP(np_array, eps=0.05, steps=3)
    #     print(RP_image.shape)
    rp_images = create_RP(np_array)
    print(rp_images.shape)
