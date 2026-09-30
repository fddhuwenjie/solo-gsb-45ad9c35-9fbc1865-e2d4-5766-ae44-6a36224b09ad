"""把 OKLCH 颜色安全映射到 sRGB 色域（POST /gamut-map）。

接收 CSS Color 4 的 ``oklch()``（含可选 alpha 与 ``none`` hue）。若原色
已在 sRGB 色域内（经 OKLab -> linear sRGB -> sRGB 转换后三通道均在
[0,1]，容 1e-7 浮点噪声），原样返回；否则**固定 L、h 与 alpha**，在
[0, 原 C] 上二分搜索仍处于色域内的最大色度 C*，避免逐通道硬裁剪造成的
色相 / 明度偏移。区间宽度不大于 tolerance 后停止，响应给出每轮上下界、
色度损失与最终 sRGB。

输入合法性（非法语法、非有限值、越界 L/alpha、负 C、tolerance 越界）
统一抛 GamutMapError，由端点映射为 HTTP 422。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .color_mix import oklab_to_srgb
from .models import GamutMapRequest, GamutMapResponse

# ------------------------------------------------------------- 常量

# 色域判定容差：吸收 sRGB -> OKLCH -> sRGB 往返的浮点噪声，使 sRGB
# 原色（如 rgb(0 0 255)）不会因 1e-13 量级的越界被误判为超色域
_GAMUT_EPS = 1e-7
# tolerance 的合法区间
_TOL_MIN = 1e-7
_TOL_MAX = 1e-3
# CSS Color 4：oklch 的 C 百分比按 0.4 为满量程解析
_C_PERCENT_REF = 0.4
# 二分轮数硬上限：正常输入约 20~40 轮，上限仅为抵御天文数字 C 导致的
# 浮点停滞（区间宽度小于中点处 1ulp 时），不影响正常终止条件
_MAX_ITER = 1000

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_OKLCH_RE = re.compile(rf"^oklch\(\s*(?P<body>.*?)\s*\)$", re.DOTALL)
_NUM_TOKEN_RE = re.compile(rf"^\s*({_NUMBER})(%?)\s*$")
_ANGLE_RE = re.compile(rf"^\s*({_NUMBER})(deg|rad|grad|turn)?\s*$", re.IGNORECASE)


class GamutMapError(ValueError):
    """请求不满足色域映射前提（映射为 HTTP 422）。"""


@dataclass
class OklchColor:
    """解析后的 oklch() 颜色。hue 为 None 表示缺失分量（none）。"""

    raw: str
    L: float
    C: float
    hue: float | None  # 度数；缺失为 None
    alpha: float


# ------------------------------------------------------------- 解析


def _parse_scalar(token: str, what: str) -> tuple[float, bool]:
    """解析「裸数字 / 百分比」分量，返回 (数值, 是否百分比)。"""
    m = _NUM_TOKEN_RE.match(token)
    if m is None:
        raise GamutMapError(f"{what} 分量 {token!r} 不是合法数值或百分比")
    value = float(m.group(1))
    if not math.isfinite(value):
        raise GamutMapError(f"{what} 分量必须为有限数值")
    return value, bool(m.group(2))


def _parse_lightness(token: str) -> float:
    value, is_percent = _parse_scalar(token, "L")
    L = value / 100.0 if is_percent else value
    if not 0.0 <= L <= 1.0:
        raise GamutMapError(
            f"L 必须在 [0,1]（百分比 [0%,100%]）内，收到 {token.strip()!r}"
        )
    return L


def _parse_chroma(token: str) -> float:
    value, is_percent = _parse_scalar(token, "C")
    C = value / 100.0 * _C_PERCENT_REF if is_percent else value
    if C < 0.0:
        raise GamutMapError(f"C 不能为负（收到 {token.strip()!r}）")
    return C


def _parse_hue(token: str) -> float | None:
    token = token.strip()
    if token.lower() == "none":
        return None
    m = _ANGLE_RE.match(token)
    if m is None:
        raise GamutMapError(
            f"hue {token!r} 不合法（应为 none 或数字 + deg/rad/grad/turn）"
        )
    value = float(m.group(1))
    if not math.isfinite(value):
        raise GamutMapError("hue 必须为有限数值")
    unit = (m.group(2) or "deg").lower()
    if unit == "deg":
        return value
    if unit == "rad":
        return math.degrees(value)
    if unit == "grad":
        return value * 0.9
    return value * 360.0  # turn


def _parse_alpha(token: str) -> float:
    token = token.strip()
    if token.lower() == "none":
        # 缺失 alpha 按 CSS Color 4 视为 100% 不透明
        return 1.0
    value, is_percent = _parse_scalar(token, "alpha")
    alpha = value / 100.0 if is_percent else value
    if not 0.0 <= alpha <= 1.0:
        raise GamutMapError(
            f"alpha 必须在 [0,1]（百分比 [0%,100%]）内，收到 {token!r}"
        )
    return alpha


def parse_oklch(text: str) -> OklchColor:
    """解析 CSS Color 4 现代空格语法的 oklch(L C H [/ A])。"""
    if not isinstance(text, str) or not text.strip():
        raise GamutMapError("颜色不能为空，应为 oklch(L C H / alpha)")
    raw = text.strip()

    m = _OKLCH_RE.match(raw.replace("\n", " "))
    if m is None:
        raise GamutMapError(f"颜色 {text!r} 不是合法的 oklch() 颜色")
    body = m.group("body").strip()
    if "," in body:
        raise GamutMapError("oklch() 使用现代空格语法，不允许逗号分隔分量")
    if body.count("/") > 1:
        raise GamutMapError("oklch() 至多接受一个斜杠 alpha 分量")

    if "/" in body:
        channels_text, alpha_text = body.split("/", 1)
        if not alpha_text.strip():
            raise GamutMapError("斜杠后缺少 alpha 分量")
        alpha = _parse_alpha(alpha_text)
    else:
        channels_text, alpha = body, 1.0

    tokens = channels_text.split()
    if len(tokens) != 3:
        raise GamutMapError(
            f"oklch() 需要 L、C、H 三个分量与可选 alpha，收到 {len(tokens)} 个"
        )
    L = _parse_lightness(tokens[0])
    C = _parse_chroma(tokens[1])
    hue = _parse_hue(tokens[2])
    return OklchColor(raw, L, C, hue, alpha)


# ------------------------------------------------------------- 转换与判定


def oklch_to_srgb(L: float, C: float, hue: float | None) -> tuple[float, float, float]:
    """OKLCH -> OKLab -> sRGB（返回未裁剪的 sRGB 通道）。"""
    angle = 0.0 if hue is None else hue
    rad = math.radians(angle)
    return oklab_to_srgb(L, C * math.cos(rad), C * math.sin(rad))


def _in_srgb_gamut(rgb: tuple[float, float, float]) -> bool:
    """三通道均在 [0,1]（容 _GAMUT_EPS）且有限即视为 sRGB 色域内。"""
    return all(
        math.isfinite(c) and -_GAMUT_EPS <= c <= 1.0 + _GAMUT_EPS for c in rgb
    )


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _to_byte(v: float) -> int:
    return int(math.floor(_clamp01(v) * 255.0 + 0.5))


# ------------------------------------------------------------- 输出工具


def _r9(v):
    if v is None:
        return None
    if isinstance(v, float):
        if abs(v) < 5e-13:
            v = 0.0
        return round(v, 9)
    return v


def _fmt(v: float) -> str:
    text = f"{_r9(v):.9f}".rstrip("0").rstrip(".")
    return text or "0"


def _oklch_css(L: float, C: float, hue: float | None, alpha: float) -> str:
    h_text = "none" if hue is None else _fmt(hue)
    text = f"oklch({_fmt(L)} {_fmt(C)} {h_text}"
    if alpha != 1.0:
        text += f" / {_fmt(alpha)}"
    return text + ")"


def _srgb_css(rgb: tuple[float, float, float], alpha: float) -> str:
    r, g, b = (_to_byte(c) for c in rgb)
    text = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return f"rgba({r}, {g}, {b}, {text or '0'})"


def _color_out(color: OklchColor, L: float, C: float) -> dict:
    return {
        "css": _oklch_css(L, C, color.hue, color.alpha),
        "l": _r9(L),
        "c": _r9(C),
        "hue": _r9(color.hue),  # null 即 none（C=0 时允许）
        "alpha": _r9(color.alpha),
    }


# ------------------------------------------------------------- 总入口


def gamut_map(req: GamutMapRequest) -> GamutMapResponse:
    color = parse_oklch(req.color)

    tol = req.tolerance
    if not math.isfinite(tol):
        raise GamutMapError("tolerance 必须为有限数值")
    if not _TOL_MIN <= tol <= _TOL_MAX:
        raise GamutMapError(
            f"tolerance 必须在 [{_TOL_MIN:g}, {_TOL_MAX:g}] 内，收到 {tol}"
        )

    trace: list[str] = [
        f"解析 {color.raw!r}：L={color.L:g}，C={color.C:g}，"
        + ("hue=none" if color.hue is None else f"hue={color.hue:g}°")
        + f"，alpha={color.alpha:g}",
    ]

    rgb_original = oklch_to_srgb(color.L, color.C, color.hue)
    original_out = _color_out(color, color.L, color.C)

    # 原色已在 sRGB 色域内：原样返回，不做任何裁剪 / 映射
    if _in_srgb_gamut(rgb_original):
        trace.append(
            f"原色转 sRGB 得 [{_r9(rgb_original[0])}, {_r9(rgb_original[1])}, "
            f"{_r9(rgb_original[2])}]，三通道均在 [0,1] 内，无需色域映射"
        )
        srgb = [_r9(c) for c in rgb_original]
        return GamutMapResponse(
            original=original_out,
            mapped=original_out,
            srgb_unclamped=srgb,
            srgb=srgb,
            srgb_css=_srgb_css(rgb_original, color.alpha),
            chroma_loss={"absolute": 0.0, "relative": 0.0},
            gamut_mapped=False,
            tolerance=tol,
            iterations=[],
            final_interval=None,
            trace=trace,
        )

    over = [i for i, c in enumerate(rgb_original) if c < -_GAMUT_EPS or c > 1.0 + _GAMUT_EPS]
    names = ["R", "G", "B"]
    trace.append(
        f"原色转 sRGB 得 [{_r9(rgb_original[0])}, {_r9(rgb_original[1])}, "
        f"{_r9(rgb_original[2])}]，{'、'.join(names[i] for i in over)} 通道超出 "
        f"[0,1]；固定 L={color.L:g}、"
        + ("hue=none" if color.hue is None else f"hue={color.hue:g}°")
        + f" 与 alpha={color.alpha:g}，在 C ∈ [0, {color.C:g}] 上二分搜索"
        f"（tolerance={tol:g}）"
    )

    # C=0（灰轴）必在色域内，因此越界只可能发生在 C>0；lo 始终是已知
    # 色域内的安全下界，hi 始终是越界上界
    lo, hi = 0.0, color.C
    iterations: list[dict] = []
    while hi - lo > tol and len(iterations) < _MAX_ITER:
        mid = (lo + hi) * 0.5
        rgb_mid = oklch_to_srgb(color.L, mid, color.hue)
        inside = _in_srgb_gamut(rgb_mid)
        n = len(iterations) + 1
        iterations.append({
            "iteration": n,
            "lower": _r9(lo),
            "upper": _r9(hi),
            "midpoint": _r9(mid),
            "srgb": [_r9(c) for c in rgb_mid],
            "in_gamut": inside,
        })
        if inside:
            trace.append(
                f"第 {n} 轮：区间 [{_r9(lo)}, {_r9(hi)}]，中点 C={_r9(mid)} "
                f"-> sRGB [{_r9(rgb_mid[0])}, {_r9(rgb_mid[1])}, {_r9(rgb_mid[2])}] "
                "在色域内，下界抬到中点"
            )
            lo = mid
        else:
            trace.append(
                f"第 {n} 轮：区间 [{_r9(lo)}, {_r9(hi)}]，中点 C={_r9(mid)} "
                f"-> sRGB [{_r9(rgb_mid[0])}, {_r9(rgb_mid[1])}, {_r9(rgb_mid[2])}] "
                "越界，上界压到中点"
            )
            hi = mid

    mapped_rgb = oklch_to_srgb(color.L, lo, color.hue)
    mapped_rgb_clamped = tuple(_clamp01(c) for c in mapped_rgb)
    loss = color.C - lo
    relative = loss / color.C if color.C > 0.0 else 0.0
    trace.append(
        f"区间宽度 {_r9(hi - lo)} 不大于 tolerance={tol:g}，搜索结束："
        f"C 由 {color.C:g} 降到 {_r9(lo)}（色度损失 {_r9(loss)}，"
        f"相对损失 {_r9(relative)}），映射后 sRGB "
        f"[{_r9(mapped_rgb_clamped[0])}, {_r9(mapped_rgb_clamped[1])}, "
        f"{_r9(mapped_rgb_clamped[2])}]"
    )

    return GamutMapResponse(
        original=original_out,
        mapped=_color_out(color, color.L, lo),
        srgb_unclamped=[_r9(c) for c in rgb_original],
        srgb=[_r9(c) for c in mapped_rgb_clamped],
        srgb_css=_srgb_css(mapped_rgb_clamped, color.alpha),
        chroma_loss={"absolute": _r9(loss), "relative": _r9(relative)},
        gamut_mapped=True,
        tolerance=tol,
        iterations=iterations,
        final_interval={
            "lower": _r9(lo),
            "upper": _r9(hi),
            "width": _r9(hi - lo),
        },
        trace=trace,
    )
