import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict, Any, Optional



def kb4_project(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                fx: float, fy: float, cx: float, cy: float,
                k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kannala-Brandt 投影：3D点 -> 像素坐标
    r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
    """
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(rho, Z + 1e-12)

    theta2 = theta * theta
    r = theta * (1.0 + k1*theta2 + k2*theta2**2 + k3*theta2**3 + k4*theta2**4)

    scale = np.where(rho > 1e-12, r / rho, 1.0)
    u = fx * X * scale + cx
    v = fy * Y * scale + cy
    return u, v


def kb4_unproject(u: np.ndarray, v: np.ndarray,
                  fx: float, fy: float, cx: float, cy: float,
                  k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0,
                  max_iter: int = 10, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kannala-Brandt 反投影：像素坐标 -> 单位视线方向 (Newton-Raphson)

    Args:
        u, v: pixel coordinates
        fx, fy, cx, cy: camera intrinsics
        k1, k2, k3, k4: KB4 distortion coefficients
        max_iter: maximum Newton-Raphson iterations
        tol: convergence tolerance

    Returns:
        X, Y, Z: unit ray direction in camera frame
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.sqrt(x**2 + y**2)

    # 小r近似：当r很小时，theta ≈ r（线性近似），不需要迭代
    small_r_mask = r < 1e-6

    theta = r.copy()
    for _ in range(max_iter):
        # 中间迭代clamp，防止theta发散到极大值导致overflow
        theta = np.clip(theta, 0.0, np.pi)

        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        f = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8) - r
        df = 1.0 + 3*k1*theta2 + 5*k2*theta4 + 7*k3*theta6 + 9*k4*theta8
        # Protect against small/negative derivative (ensures convergence)
        df = np.maximum(df, 1e-8)
        delta = f / df
        theta = theta - delta

        # Early stop：当最大残差足够小时提前终止
        if np.max(np.abs(delta)) < tol:
            break

    # Clamp theta to valid range [0, pi] to prevent invalid trigonometric values
    theta = np.clip(theta, 0.0, np.pi)

    # 小r时直接使用线性近似 theta = r
    theta = np.where(small_r_mask, r, theta)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    scale = np.where(r > 1e-12, sin_theta / r, 1.0)

    X = x * scale
    Y = y * scale
    Z = cos_theta
    return X, Y, Z


def rescale_intrinsics_kb4(K: Dict[str, float],
                           distortion: Dict[str, float],
                           src_resolution: Tuple[int, int],
                           dst_resolution: Tuple[int, int]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    KB4 鱼眼相机内参缩放 - 畸变参数在归一化坐标下定义，不需要缩放
    """
    src_h, src_w = src_resolution
    dst_h, dst_w = dst_resolution

    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)

    K_new = {
        "fx": K.get("fx", 1.0) * scale_x,
        "fy": K.get("fy", 1.0) * scale_y,
        "cx": K.get("cx", src_w * 0.5) * scale_x,
        "cy": K.get("cy", src_h * 0.5) * scale_y,
    }
    # KB4 畸变参数不需要缩放
    return K_new, distortion.copy()


def is_low_shm() -> bool:
    """
    检测系统共享内存 (/dev/shm) 是否较小。
    在 Docker 容器或某些 Linux 环境下，较小的 shm 会导致 DataLoader 崩溃。
    对于 macOS (用户当前环境)，通常返回 False。
    """
    if sys.platform == "darwin":
        return False  # macOS 不依赖 /dev/shm 进行 PyTorch DataLoader 通信
    
    if not os.path.exists("/dev/shm"):
        return False

    try:
        shm_stats = os.statvfs("/dev/shm")
        # 计算总大小 (GB)
        shm_size = (shm_stats.f_frsize * shm_stats.f_blocks) / (1024 ** 3)
        if shm_size < 2.0:  # 如果小于 2GB，认为是低内存环境
            return True
    except Exception:
        pass
    return False

def ensure_dir(path: str):
    """确保目录存在，不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)

class Logger:
    """简单的日志记录器，同时输出到控制台和文件，支持 context manager"""
    def __init__(self, filename: Optional[str] = None):
        self.filename = filename
        self.fh = None
        if filename:
            try:
                self.fh = open(filename, "a", buffering=1)  # 行缓冲
            except IOError as e:
                print(f"Warning: Cannot open log file {filename}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def log(self, msg: str):
        print(msg, flush=True)
        if self.fh:
            try:
                self.fh.write(str(msg) + "\n")
                self.fh.flush()
            except IOError:
                pass

    def close(self):
        if self.fh:
            try:
                self.fh.close()
            except IOError:
                pass
            self.fh = None

def read_txt_skip_first_line(path: str) -> np.ndarray:
    """读取 TUM 格式的 txt 文件，跳过第一行 header"""
    try:
        return np.loadtxt(path, dtype=np.float64, skiprows=1)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return np.array([])



class QuaternionUtils:
    """
    统一的四元数工具类 (Numpy 版本)。
    用于数据加载、评估和轨迹合成。
    格式: [x, y, z, w]
    """
    # 预定义单位四元数，避免重复创建
    _IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.shape == (0,):
            return q
        if q.shape[-1] != 4:
            raise ValueError(f"Quaternion must have last dim=4, got shape={q.shape}")
        original_shape = q.shape
        q_flat = q.reshape(-1, 4)
        norms = np.linalg.norm(q_flat, axis=1, keepdims=True)
        eps_mask = norms[:, 0] < 1e-12
        norms[eps_mask] = 1.0
        q_flat = q_flat / norms
        if np.any(eps_mask):
            q_flat[eps_mask] = QuaternionUtils._IDENTITY
        return q_flat.reshape(original_shape)

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)
        if q1.shape == (0,) or q2.shape == (0,):
            return np.array([], dtype=np.float64)
        if q1.shape[-1] != 4 or q2.shape[-1] != 4:
            raise ValueError(f"Quaternion multiply expects (...,4) inputs, got q1={q1.shape}, q2={q2.shape}")

        q1b, q2b = np.broadcast_arrays(q1, q2)
        x1, y1, z1, w1 = q1b[..., 0], q1b[..., 1], q1b[..., 2], q1b[..., 3]
        x2, y2, z2, w2 = q2b[..., 0], q2b[..., 1], q2b[..., 2], q2b[..., 3]
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return QuaternionUtils.normalize(np.stack([x, y, z, w], axis=-1))

    @staticmethod
    def inverse(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.shape == (0,):
            return q
        if q.shape[-1] != 4:
            raise ValueError(f"Quaternion must have last dim=4, got shape={q.shape}")
        result = np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], axis=-1).astype(np.float64)
        return QuaternionUtils.normalize(result)

    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        q = QuaternionUtils.normalize(q)
        if q.shape == (0,):
            return np.array([], dtype=np.float64)
        if q.shape[-1] != 4:
            raise ValueError(f"Quaternion must have last dim=4, got shape={q.shape}")

        q_flat = q.reshape(-1, 4)
        x = q_flat[:, 0]
        y = q_flat[:, 1]
        z = q_flat[:, 2]
        w = q_flat[:, 3]

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        R = np.empty((q_flat.shape[0], 3, 3), dtype=np.float64)
        R[:, 0, 0] = 1 - 2 * (yy + zz)
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)
        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = 1 - 2 * (xx + zz)
        R[:, 1, 2] = 2 * (yz - wx)
        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = 1 - 2 * (xx + yy)

        return R.reshape(q.shape[:-1] + (3, 3))

    @staticmethod
    def compose_se3(t1: np.ndarray, q1: np.ndarray, t2: np.ndarray, q2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        组合两个位姿变换 T_world_new = T_world_curr @ T_curr_next
        """
        R1 = QuaternionUtils.to_rotation_matrix(q1)
        # t_new = t1 + R1 * t2
        t = t1 + R1 @ t2
        # q_new = q1 * q2
        q = QuaternionUtils.multiply(q1, q2)
        return t, q


def align_trajectory(estimated: np.ndarray, ground_truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Umeyama 算法：计算两个轨迹之间的最佳刚体变换（对齐）。
    用于评估 ATE (Absolute Trajectory Error)。

    Args:
        estimated: [N, 3] 预测轨迹
        ground_truth: [N, 3] 真实轨迹

    Returns:
        R: [3, 3] 旋转矩阵
        t: [3] 平移向量
    """
    assert estimated.shape == ground_truth.shape

    # 边界检查：至少需要3个点
    if estimated.shape[0] < 3:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    mu_est = np.mean(estimated, axis=0)
    mu_gt = np.mean(ground_truth, axis=0)

    est_centered = estimated - mu_est
    gt_centered = ground_truth - mu_gt

    # 协方差矩阵
    H = est_centered.T @ gt_centered

    # 处理 NaN/Inf
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)

    # 检查退化情况（共线点）
    if np.linalg.matrix_rank(H) < 3:
        return np.eye(3, dtype=np.float64), mu_gt - mu_est

    try:
        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 处理反射情况 (det(R) ≈ -1)
        if np.linalg.det(R) < -0.9:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64), mu_gt - mu_est

    t = mu_gt - R @ mu_est

    return R, t


def rescale_intrinsics_pinhole(K: Dict[str, float],
                              src_resolution: Tuple[int, int],
                              dst_resolution: Tuple[int, int],
                              crop: Optional[Tuple[int, int, int, int]] = None) -> Tuple[Dict[str, float], Tuple[float, float]]:
    src_h, src_w = src_resolution
    dst_h, dst_w = dst_resolution
    fx = float(K.get("fx", 1.0))
    fy = float(K.get("fy", 1.0))
    cx = float(K.get("cx", src_w * 0.5))
    cy = float(K.get("cy", src_h * 0.5))

    if crop is not None:
        x0, y0, crop_w, crop_h = crop
        cx = cx - float(x0)
        cy = cy - float(y0)
        src_w = crop_w
        src_h = crop_h

    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)

    fx_new = fx * scale_x
    fy_new = fy * scale_y
    cx_new = cx * scale_x
    cy_new = cy * scale_y

    K_new = {
        "fx": fx_new,
        "fy": fy_new,
        "cx": cx_new,
        "cy": cy_new,
    }
    return K_new, (scale_x, scale_y)


def warp_events_flow(xw: np.ndarray, yw: np.ndarray, tw: np.ndarray,
                     omega: np.ndarray, K: Dict[str, float],
                     resolution: Tuple[int, int], t_ref: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据角速度 (omega) 对事件进行去旋转 (Derotation)。
    用于预处理阶段消除相机的纯旋转运动，使网络专注于平移。
    """
    if xw.size == 0:
        return xw, yw
        
    fx = float(K.get("fx", resolution[1]))
    fy = float(K.get("fy", resolution[0]))
    cx = float(K.get("cx", resolution[1] * 0.5))
    cy = float(K.get("cy", resolution[0] * 0.5))
    
    x = xw - cx
    y = yw - cy
    
    # 时间差
    dt = (tw - t_ref).astype(np.float32)
    
    wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])
    
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # 瞬时光流公式 (旋转部分)
    flow_u = (xy / fy) * wx - (fx + x2 / fx) * wy + (fx / fy) * y * wz
    flow_v = (fy + y2 / fy) * wx - (xy / fx) * wy - (fy / fx) * x * wz
    
    xw_corr = xw - flow_u * dt
    yw_corr = yw - flow_v * dt
    
    return xw_corr, yw_corr


# -----------------------------------------------------------------------------
# PyTorch Geometry Utilities
# -----------------------------------------------------------------------------

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    n = q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit = torch.zeros_like(q)
    unit[..., 3] = 1.0
    return torch.where(n > 1e-8, q / n, unit)

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return quat_normalize(torch.stack([x, y, z, w], dim=-1))

def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    return quat_normalize(torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1))

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)
    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)

def compose_se3(t: torch.Tensor, q: torch.Tensor, t_delta: torch.Tensor, q_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    R = quat_to_rotmat(q)
    t_new = t + torch.matmul(R, t_delta.unsqueeze(-1)).squeeze(-1)
    q_new = quat_multiply(q, q_delta)
    return t_new, q_new


def safe_geodesic_loss(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = torch.sum(q1 * q2, dim=-1).abs()
    dot = torch.clamp(dot, min=float(eps), max=1.0 - float(eps))
    return 2.0 * torch.acos(dot)


def compute_rpe_loss(prev_pred_full: torch.Tensor, pred: torch.Tensor,
                     prev_gt_full: torch.Tensor, gt: torch.Tensor,
                     weight: float) -> torch.Tensor:
    t01_p = prev_pred_full[:, 0:3].contiguous()
    q01_p = prev_pred_full[:, 3:7].contiguous()
    t12_p = pred[:, 0:3].contiguous()
    q12_p = pred[:, 3:7].contiguous()

    t01_g = prev_gt_full[:, 0:3].contiguous()
    q01_g = prev_gt_full[:, 3:7].contiguous()
    t12_g = gt[:, 0:3].contiguous()
    q12_g = gt[:, 3:7].contiguous()

    q01_p = quat_normalize(q01_p)
    q12_p = quat_normalize(q12_p)
    q01_g = quat_normalize(q01_g)
    q12_g = quat_normalize(q12_g)

    R01_p = quat_to_rotmat(q01_p)
    R01_g = quat_to_rotmat(q01_g)

    t02_p = t01_p + torch.matmul(R01_p, t12_p.unsqueeze(-1)).squeeze(-1)
    t02_g = t01_g + torch.matmul(R01_g, t12_g.unsqueeze(-1)).squeeze(-1)

    q02_p = quat_multiply(q01_p, q12_p)
    q02_g = quat_multiply(q01_g, q12_g)

    loss_t = F.smooth_l1_loss(t02_p, t02_g, beta=0.1)
    loss_r = safe_geodesic_loss(q02_p, q02_g).mean()

    return float(weight) * (loss_t + loss_r)

def warp_events_flow_torch(xw: torch.Tensor, yw: torch.Tensor, tw: torch.Tensor,
                           omega: torch.Tensor, K: Dict[str, float],
                           resolution: Tuple[int, int], t_ref: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """针孔模型的事件去旋转warp"""
    if xw.numel() == 0:
        return xw, yw
    H, W = int(resolution[0]), int(resolution[1])
    fx = float(K.get("fx", W))
    fy = float(K.get("fy", H))
    cx = float(K.get("cx", W * 0.5))
    cy = float(K.get("cy", H * 0.5))
    x = xw - cx
    y = yw - cy
    dt = (tw - float(t_ref)).float()
    wx, wy, wz = omega[0].float(), omega[1].float(), omega[2].float()
    x2 = x * x
    y2 = y * y
    xy = x * y
    flow_u = (xy / fy) * wx - (fx + x2 / fx) * wy + (fx / fy) * y * wz
    flow_v = (fy + y2 / fy) * wx - (xy / fx) * wy - (fy / fx) * x * wz
    xw_corr = xw - flow_u * dt
    yw_corr = yw - flow_v * dt
    return xw_corr, yw_corr


def kb4_unproject_torch(u: torch.Tensor, v: torch.Tensor,
                        fx: float, fy: float, cx: float, cy: float,
                        k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0,
                        max_iter: int = 10, tol: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    KB4 反投影 (PyTorch版本): 像素坐标 -> 单位视线方向
    使用 Newton-Raphson 迭代求解

    A1.3: 增加了收敛保护:
    - 中间迭代clamp防止发散
    - early stop当残差足够小时
    - 小r近似避免不必要的迭代
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = torch.sqrt(x * x + y * y)

    # A1.3: 小r近似 - 当r很小时, theta ≈ r (线性近似)
    # 此时 r(θ) ≈ θ, 不需要迭代
    small_r_mask = r < 1e-6

    # Newton-Raphson 迭代求 theta
    theta = r.clone()

    for i in range(max_iter):
        # A1.3: 中间迭代clamp防止发散到无效范围
        theta = torch.clamp(theta, 0.0, 3.14159265)

        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        f = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - r
        df = 1.0 + 3 * k1 * theta2 + 5 * k2 * theta4 + 7 * k3 * theta6 + 9 * k4 * theta8
        df = torch.clamp(df, min=1e-8)

        delta = f / df
        theta = theta - delta

        # A1.3: early stop - 当最大残差足够小时提前终止
        if torch.max(torch.abs(delta)) < tol:
            break

    # 限制 theta 范围 [0, pi]
    theta = torch.clamp(theta, 0.0, 3.14159265)

    # A1.3: 小r时直接使用线性近似 theta = r
    theta = torch.where(small_r_mask, r, theta)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    scale = torch.where(r > 1e-12, sin_theta / r, torch.ones_like(r))

    X = x * scale
    Y = y * scale
    Z = cos_theta
    return X, Y, Z


def kb4_project_torch(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor,
                      fx: float, fy: float, cx: float, cy: float,
                      k1: float = 0.0, k2: float = 0.0, k3: float = 0.0, k4: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KB4 投影 (PyTorch版本): 3D点 -> 像素坐标
    r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
    """
    rho = torch.sqrt(X * X + Y * Y)
    theta = torch.atan2(rho, Z + 1e-12)

    theta2 = theta * theta
    r = theta * (1.0 + k1 * theta2 + k2 * theta2 * theta2 +
                 k3 * theta2 * theta2 * theta2 + k4 * theta2 * theta2 * theta2 * theta2)

    scale = torch.where(rho > 1e-12, r / rho, torch.ones_like(rho))
    u = fx * X * scale + cx
    v = fy * Y * scale + cy
    return u, v


def warp_events_flow_torch_kb4(xw: torch.Tensor, yw: torch.Tensor, tw: torch.Tensor,
                                omega: torch.Tensor, K: Dict[str, float],
                                distortion: Dict[str, float],
                                resolution: Tuple[int, int], t_ref: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    KB4 鱼眼模型的事件去旋转warp

    流程:
    1. KB4 unproject: 像素 -> 单位球面上的3D点
    2. 旋转补偿: 在3D空间中应用反向旋转
    3. KB4 project: 3D点 -> 像素

    Args:
        xw, yw: 事件像素坐标
        tw: 事件时间戳
        omega: 角速度 [wx, wy, wz] (rad/s)
        K: 相机内参 {fx, fy, cx, cy}
        distortion: KB4畸变参数 {k1, k2, k3, k4}
        resolution: (H, W)
        t_ref: 参考时间

    Returns:
        xw_corr, yw_corr: 去旋转后的像素坐标
        valid_mask: 布尔mask，标记在图像边界内的有效事件 (A1.5)
    """
    if xw.numel() == 0:
        empty_mask = torch.ones(0, dtype=torch.bool, device=xw.device)
        return xw, yw, empty_mask

    H, W = int(resolution[0]), int(resolution[1])
    fx = float(K.get("fx", W))
    fy = float(K.get("fy", H))
    cx = float(K.get("cx", W * 0.5))
    cy = float(K.get("cy", H * 0.5))
    k1 = float(distortion.get("k1", 0.0))
    k2 = float(distortion.get("k2", 0.0))
    k3 = float(distortion.get("k3", 0.0))
    k4 = float(distortion.get("k4", 0.0))

    # 时间差
    dt = (tw - float(t_ref)).float()

    # 1. KB4 反投影到单位球面
    X, Y, Z = kb4_unproject_torch(xw, yw, fx, fy, cx, cy, k1, k2, k3, k4)

    # 2. 使用 SO(3) Rodrigues 公式计算旋转补偿
    # R = exp([ω]× · Δt) = I + sin(θ)·K + (1-cos(θ))·K²
    # 其中 θ = ||ω·dt||, K = [ω×]/||ω||
    wx, wy, wz = omega[0].float(), omega[1].float(), omega[2].float()

    # 旋转向量 = -omega * dt (反向旋转补偿)
    rv_x = -wx * dt
    rv_y = -wy * dt
    rv_z = -wz * dt

    # 旋转角度 θ = ||rotation_vector||
    theta_sq = rv_x * rv_x + rv_y * rv_y + rv_z * rv_z
    theta = torch.sqrt(theta_sq + 1e-12)

    # A1.1: 小角度极限保护 - theta < 1e-6 时直接使用恒等变换
    # 避免数值不稳定导致的微小旋转偏差
    small_angle_mask = theta < 1e-6

    # 归一化旋转轴 (防止除零)
    inv_theta = 1.0 / (theta + 1e-12)
    kx = rv_x * inv_theta
    ky = rv_y * inv_theta
    kz = rv_z * inv_theta

    # Rodrigues 公式: R = I + sin(θ)·K + (1-cos(θ))·K²
    # 其中 K 是反对称矩阵 [k×]
    # 直接计算 p' = R·p:
    # p' = p + sin(θ)·(k × p) + (1-cos(θ))·(k × (k × p))
    #    = p + sin(θ)·(k × p) + (1-cos(θ))·(k·(k·p) - p)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    one_minus_cos = 1.0 - cos_theta

    # k × p (叉乘)
    cross_x = ky * Z - kz * Y
    cross_y = kz * X - kx * Z
    cross_z = kx * Y - ky * X

    # k·p (点乘)
    dot = kx * X + ky * Y + kz * Z

    # Rodrigues 公式: p' = cos(θ)·p + sin(θ)·(k×p) + (1-cos(θ))·(k·p)·k
    X_rot = cos_theta * X + sin_theta * cross_x + one_minus_cos * dot * kx
    Y_rot = cos_theta * Y + sin_theta * cross_y + one_minus_cos * dot * ky
    Z_rot = cos_theta * Z + sin_theta * cross_z + one_minus_cos * dot * kz

    # 小角度时使用原始坐标 (恒等变换)
    X = torch.where(small_angle_mask, X, X_rot)
    Y = torch.where(small_angle_mask, Y, Y_rot)
    Z = torch.where(small_angle_mask, Z, Z_rot)

    # 3. KB4 投影回像素坐标
    xw_corr, yw_corr = kb4_project_torch(X, Y, Z, fx, fy, cx, cy, k1, k2, k3, k4)

    # A1.5: 返回有效mask而不是clamp，避免边界处的伪密度堆积
    # 有效事件 = 在图像边界内 且 在相机前方(Z > 0)
    valid_mask = (xw_corr >= 0) & (xw_corr < W) & (yw_corr >= 0) & (yw_corr < H) & (Z > 0)

    return xw_corr, yw_corr, valid_mask


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    From: On the Continuity of Rotation Representations in Neural Networks, CVPR 2019
    Args:
        d6: [..., 6] tensor containing two 3D vectors (a1, a2)
    Returns:
        [..., 3, 3] rotation matrices
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts 3x3 rotation matrix to 6D representation.
    Args:
        matrix: [..., 3, 3] rotation matrices
    Returns:
        [..., 6] 6D rotation representation
    """
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)

def associate_by_timestamp(t_est: np.ndarray, t_gt: np.ndarray, max_dt: float) -> Tuple[np.ndarray, np.ndarray]:
    i, j = 0, 0
    idx_est, idx_gt = [], []
    while i < len(t_est) and j < len(t_gt):
        de = t_est[i] - t_gt[j]
        if abs(de) <= max_dt:
            idx_est.append(i)
            idx_gt.append(j)
            i += 1
            j += 1
        elif de < 0:
            i += 1
        else:
            j += 1
    return np.asarray(idx_est, dtype=np.int64), np.asarray(idx_gt, dtype=np.int64)

def align_trajectory_with_timestamps(est_pos: np.ndarray, est_t: np.ndarray,
                                     gt_pos: np.ndarray, gt_t: np.ndarray,
                                     max_dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ie, ig = associate_by_timestamp(est_t, gt_t, max_dt)
    if ie.size < 3:
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        return R, t, ie, ig
    A = est_pos[ie]
    B = gt_pos[ig]
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)
    A_c = A - mu_A
    B_c = B - mu_B
    H = A_c.T @ B_c
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
    except np.linalg.LinAlgError:
        R = np.eye(3, dtype=np.float64)
    t = mu_B - R @ mu_A
    return R, t, ie, ig


def align_trajectory_with_timestamps_sim3(est_pos: np.ndarray, est_t: np.ndarray,
                                         gt_pos: np.ndarray, gt_t: np.ndarray,
                                         max_dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Performs SIM(3) alignment (Rotation, Translation, Scale) with light outlier rejection."""
    ie, ig = associate_by_timestamp(est_t, gt_t, max_dt)
    if ie.size < 3:
        return np.eye(3), np.zeros(3), 1.0, ie, ig

    A0 = est_pos[ie]
    B0 = gt_pos[ig]

    min_scale = 1e-4
    max_scale = 1e4

    def _umeyama_sim3(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        mu_A = np.mean(A, axis=0)
        mu_B = np.mean(B, axis=0)
        A_c = A - mu_A
        B_c = B - mu_B
        n = int(A.shape[0])

        H = (A_c.T @ B_c) / max(n, 1)
        U, S, Vt = np.linalg.svd(H, full_matrices=False)

        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        RA = (R @ A_c.T).T
        denom = float(np.sum(RA * RA))
        s = 1.0 if denom < 1e-12 else float(np.sum(B_c * RA) / denom)
        s = float(np.clip(s, min_scale, max_scale))

        t = mu_B - s * (R @ mu_A)
        return R, t, s

    try:
        R, t, s = _umeyama_sim3(A0, B0)

        # Light outlier rejection based on residuals, then re-fit
        if A0.shape[0] >= 10:
            B_hat = (s * (R @ A0.T).T) + t
            res = np.linalg.norm(B_hat - B0, axis=1)
            med = float(np.median(res))
            mad = float(np.median(np.abs(res - med)))
            sigma = 1.4826 * mad
            thr = med + 3.0 * sigma
            thr = max(thr, float(np.percentile(res, 80)))
            inliers = res <= thr
            if int(np.sum(inliers)) >= 3 and int(np.sum(inliers)) < int(res.size):
                R, t, s = _umeyama_sim3(A0[inliers], B0[inliers])
    except np.linalg.LinAlgError:
        R = np.eye(3)
        t = np.zeros(3)
        s = 1.0

    return R, t, float(s), ie, ig


def compute_ols_scale_stats(est_pos: np.ndarray, gt_pos: np.ndarray, stride: int = 1) -> Dict[str, float]:
    """
    Computes global OLS scale estimate and path length ratio over a trajectory.
    Args:
        est_pos: [N,3] predicted positions
        gt_pos: [N,3] ground-truth positions
        stride: window stride for forming displacement pairs
    Returns:
        {'s_ols': float, 'path_ratio': float}
    """
    n = min(len(est_pos), len(gt_pos))
    if n < 2:
        return {'s_ols': float('nan'), 'path_ratio': float('nan')}
    stride = max(int(stride), 1)
    stride = min(stride, max(n - 1, 1))
    idx0 = np.arange(0, n - stride)
    idx1 = idx0 + stride
    dp_est = est_pos[idx1] - est_pos[idx0]
    dp_gt = gt_pos[idx1] - gt_pos[idx0]
    num = float(np.sum(np.einsum('ij,ij->i', dp_est, dp_gt)))
    den = float(np.sum(np.linalg.norm(dp_est, axis=1) ** 2)) + 1e-12
    s_ols = num / den
    path_ratio = float(np.sum(np.linalg.norm(dp_est, axis=1))) / (float(np.sum(np.linalg.norm(dp_gt, axis=1))) + 1e-12)
    return {'s_ols': s_ols, 'path_ratio': path_ratio}
