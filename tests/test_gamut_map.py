"""端到端测试：POST /gamut-map。"""
import math

import pytest
from fastapi.testclient import TestClient

from app.gamut_map import GamutMapError, gamut_map, oklch_to_srgb, parse_oklch
from app.main import app
from app.models import GamutMapRequest

client = TestClient(app)


def map(body):
    resp = client.post("/gamut-map", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def expect422(body):
    resp = client.post("/gamut-map", json=body)
    assert resp.status_code == 422, resp.text
    return resp.json()["detail"]


def engine(text, tol=1e-6):
    return gamut_map(GamutMapRequest(color=text, tolerance=tol))


# ---------- 原色已在 sRGB 色域内 ----------


def test_in_gamut_color_returned_as_is():
    # oklch 中一个普通的暖色应当在 sRGB 内，原样返回
    data = map({"color": "oklch(0.7 0.1 50)", "tolerance": 1e-6})
    assert data["gamut_mapped"] is False
    assert data["iterations"] == []
    assert data["final_interval"] is None
    assert data["original"] == data["mapped"]
    assert data["srgb_unclamped"] == data["srgb"]
    assert data["chroma_loss"] == {"absolute": 0.0, "relative": 0.0}
    for c in data["srgb"]:
        assert 0.0 <= c <= 1.0


def test_gray_axis_in_gamut():
    for L in (0.0, 0.5, 1.0):
        data = map({"color": f"oklch({L} 0 none)", "tolerance": 1e-6})
        assert data["gamut_mapped"] is False
        assert data["mapped"]["hue"] is None
        assert data["mapped"]["l"] == L
        assert data["mapped"]["c"] == 0.0


def test_srgb_primary_roundtrip_inside_gamut():
    # sRGB 纯蓝转 OKLCH 再转回，含 1e-6 级往返误差，应仍判为色域内
    # （而非被无谓地降 C）
    from app.color_mix import srgb_to_oklab
    L, a, b2 = srgb_to_oklab(0.0, 0.0, 1.0)
    h = math.degrees(math.atan2(b2, a)) % 360.0
    C = math.hypot(a, b2)
    data = map({"color": f"oklch({L:.12f} {C:.12f} {h:.12f})",
                "tolerance": 1e-6})
    assert data["gamut_mapped"] is False
    r, g, b = data["srgb"]
    assert abs(r) < 1e-5 and abs(g) < 1e-5 and abs(b - 1.0) < 1e-5


def test_in_gamut_preserves_alpha():
    data = map({"color": "oklch(0.7 0.1 50 / 0.4)", "tolerance": 1e-6})
    assert data["original"]["alpha"] == 0.4
    assert data["mapped"]["alpha"] == 0.4
    assert data["srgb_css"].endswith(", 0.4)")


# ---------- 超色域 -> 二分降 C ----------


def test_out_of_gamut_high_chroma_gets_mapped():
    # oklch(0.7 0.4 145) 是著名的超 sRGB 鲜绿色
    text = "oklch(0.705 0.28 145)"
    rgb_pre = oklch_to_srgb(0.705, 0.28, 145.0)
    assert max(rgb_pre) > 1.0 + 1e-7 or min(rgb_pre) < -1e-7
    data = map({"color": text, "tolerance": 1e-6})

    assert data["gamut_mapped"] is True
    o, m = data["original"], data["mapped"]
    # L、h、alpha 固定不变
    assert m["l"] == o["l"]
    assert m["hue"] == o["hue"] == 145.0
    assert m["alpha"] == o["alpha"] == 1.0
    # C 严格下降且非负
    assert 0.0 < m["c"] < o["c"]
    # 最终 sRGB 在 [0,1]，未裁剪结果确实越界（h=145 方向为负 R 越界）
    assert any(c < -1e-7 or c > 1.0 + 1e-7 for c in data["srgb_unclamped"])
    assert all(0.0 <= c <= 1.0 for c in data["srgb"])
    # 色度损失
    assert data["chroma_loss"]["absolute"] == pytest.approx(o["c"] - m["c"], abs=1e-9)
    assert 0.0 < data["chroma_loss"]["relative"] < 1.0
    assert data["chroma_loss"]["relative"] == pytest.approx(
        (o["c"] - m["c"]) / o["c"], abs=1e-9
    )


def test_mapped_result_actually_in_gamut():
    data = map({"color": "oklch(0.85 0.3 80)", "tolerance": 1e-6})
    L, C, h = (data["mapped"]["l"], data["mapped"]["c"], data["mapped"]["hue"])
    rgb = oklch_to_srgb(L, C, h)
    for c in rgb:
        assert -1e-7 <= c <= 1.0 + 1e-7


def test_binary_search_iteration_trace():
    data = map({"color": "oklch(0.705 0.28 145)", "tolerance": 1e-6})
    its = data["iterations"]
    assert len(its) >= 2
    assert its[0]["iteration"] == 1
    assert its[0]["lower"] == 0.0
    assert its[0]["upper"] == pytest.approx(0.28)
    assert its[0]["midpoint"] == pytest.approx(0.14)

    # 区间单调收窄；每轮 midpoint 等于本轮区间中点，in_gamut 决定下轮走向
    for i, step in enumerate(its):
        # 响应数值保留 9 位小数，容差相应放宽
        assert step["midpoint"] == pytest.approx(
            (step["lower"] + step["upper"]) / 2, abs=2e-9
        )
        assert len(step["srgb"]) == 3
        if i + 1 < len(its):
            nxt = its[i + 1]
            if step["in_gamut"]:
                assert nxt["lower"] == step["midpoint"]
                assert nxt["upper"] == step["upper"]
            else:
                assert nxt["upper"] == step["midpoint"]
                assert nxt["lower"] == step["lower"]
    # 末轮之后区间已收窄
    fin = data["final_interval"]
    assert fin["width"] <= 1e-6 + 1e-12
    assert fin["lower"] == data["mapped"]["c"]
    # lower 一定来自某次 in_gamut=True 的中点
    assert fin["lower"] < fin["upper"]
    # 中点判定至少出现一次越界
    assert any(not s["in_gamut"] for s in its)
    # trace 含每轮与终止描述
    assert sum("轮：区间" in t for t in data["trace"]) == len(its)
    assert any("搜索结束" in t for t in data["trace"])


def test_tolerance_controls_precision():
    coarse = map({"color": "oklch(0.705 0.28 145)", "tolerance": 1e-3})
    fine = map({"color": "oklch(0.705 0.28 145)", "tolerance": 1e-7})
    assert coarse["final_interval"]["width"] <= 1e-3 + 1e-12
    assert fine["final_interval"]["width"] <= 1e-7 + 1e-12
    # 更细容差不应产生更差的下界
    assert fine["mapped"]["c"] >= coarse["mapped"]["c"] - 1e-9
    # 容差越大轮数越少
    assert len(coarse["iterations"]) < len(fine["iterations"])


def test_hue_and_alpha_fixed_when_mapped():
    data = map({"color": "oklch(0.6 0.35 300 / 0.25)", "tolerance": 1e-6})
    assert data["gamut_mapped"] is True
    m = data["mapped"]
    assert m["l"] == 0.6
    assert m["hue"] == 300.0
    assert m["alpha"] == 0.25
    assert "/ 0.25" in m["css"]
    assert data["srgb_css"].endswith(", 0.25)")


def test_c0_with_hue_none_returned_as_is():
    data = map({"color": "oklch(0.42 0 none / 0.9)", "tolerance": 1e-6})
    assert data["gamut_mapped"] is False
    assert data["mapped"]["hue"] is None
    assert "none" in data["mapped"]["css"]
    assert data["mapped"]["css"] == "oklch(0.42 0 none / 0.9)"


# ---------- 解析合法性 ----------


@pytest.mark.parametrize(
    "text,L,C,hue,alpha",
    [
        ("oklch(0.5 0.1 120)", 0.5, 0.1, 120.0, 1.0),
        ("oklch(50% 0.1 120)", 0.5, 0.1, 120.0, 1.0),
        ("oklch(0.5 25% 120)", 0.5, 0.1, 120.0, 1.0),  # C% 按 0.4 满量程
        ("oklch(0.5 0.1 120deg)", 0.5, 0.1, 120.0, 1.0),
        ("oklch(0.5 0.1 0.5turn)", 0.5, 0.1, 180.0, 1.0),
        ("oklch(0.5 0.1 3.141592653589793rad)", 0.5, 0.1, 180.0, 1.0),
        ("oklch(0.5 0.1 200grad)", 0.5, 0.1, 180.0, 1.0),
        ("oklch(0.5 0.1 120 / 0.5)", 0.5, 0.1, 120.0, 0.5),
        ("oklch(0.5 0.1 120 / 50%)", 0.5, 0.1, 120.0, 0.5),
        ("oklch(0.5 0.1 none)", 0.5, 0.1, None, 1.0),
        ("oklch(100% 0 0 / 0)", 1.0, 0.0, 0.0, 0.0),
        ("  oklch(0.5 0.1 120)  ", 0.5, 0.1, 120.0, 1.0),
    ],
)
def test_parse_ok(text, L, C, hue, alpha):
    c = parse_oklch(text)
    assert c.L == pytest.approx(L)
    assert c.C == pytest.approx(C)
    if hue is None:
        assert c.hue is None
    else:
        assert c.hue == pytest.approx(hue)
    assert c.alpha == pytest.approx(alpha)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "red",
        "#ff0000",
        "rgb(1 2 3)",
        "oklab(0.5 0.1 0)",
        "oklch(0.5 0.1)",               # 缺 hue
        "oklch(0.5 0.1 120 0.5)",       # 第四分量缺斜杠
        "oklch(0.5, 0.1, 120)",         # 旧式逗号
        "oklch(0.5 0.1 120 / 0.5 / 0.5)",  # 多个斜杠
        "oklch(0.5 0.1 120 /)",         # 斜杠后为空
        "oklch()",
        "oklch(0.5 0.1 120",            # 缺右括号
        "oklch(0.5 x 120)",             # C 非数值
        "oklch(foo 0.1 120)",           # L 非数值
        "oklch(0.5 0.1 blue)",          # hue 非角度 / none
        "oklch(0.5 0.1 120 / x)",       # alpha 非数值
        "oklch(1.01 0.1 120)",          # L 越界
        "oklch(-0.01 0.1 120)",
        "oklch(101% 0.1 120)",
        "oklch(-1% 0.1 120)",
        "oklch(0.5 0.1 120 / 1.1)",     # alpha 越界
        "oklch(0.5 0.1 120 / -0.1)",
        "oklch(0.5 0.1 120 / 101%)",
        "oklch(0.5 -0.1 120)",          # 负 C
        "oklch(0.5 -0.000001 120)",
    ],
)
def test_invalid_syntax_422(text):
    detail = expect422({"color": text, "tolerance": 1e-6})
    assert isinstance(detail, str) and detail


def test_non_finite_values_422():
    # JSON 可以表达 1e999 -> Infinity（Python float），但序列化侧
    # 直接测引擎更直接
    for text in (
        "oklch(NaN 0.1 120)",
        "oklch(0.5 Infinity 120)",
        "oklch(0.5 0.1 Infinity)",
        "oklch(0.5 0.1 120 / NaN)",
    ):
        with pytest.raises(GamutMapError):
            parse_oklch(text)


@pytest.mark.parametrize("tol", [0, 1e-8, 9.999e-8 - 1e-12, 1.001e-3, 1.0, -1e-6])
def test_tolerance_out_of_range_422(tol):
    detail = expect422({"color": "oklch(0.5 0.1 120)", "tolerance": tol})
    assert "tolerance" in detail


@pytest.mark.parametrize("tol", [1e-7, 1e-6, 1e-3])
def test_tolerance_boundaries_accepted(tol):
    data = map({"color": "oklch(0.5 0.1 120)", "tolerance": tol})
    assert data["tolerance"] == tol


def test_missing_fields_422():
    assert client.post("/gamut-map", json={"color": "oklch(0.5 0.1 120)"}).status_code == 422
    assert client.post("/gamut-map", json={"tolerance": 1e-6}).status_code == 422


# ---------- 输出序列化 ----------


def test_css_serialization_in_gamut():
    data = map({"color": "oklch(0.7 0.1 50)", "tolerance": 1e-6})
    css = data["mapped"]["css"]
    assert css.startswith("oklch(") and css.endswith(")")
    # alpha=1 时不输出斜杠
    assert "/" not in css
    # hue 保留原值
    assert " 50" in css


def test_srgb_css_eight_bit_in_range():
    data = map({"color": "oklch(0.705 0.28 145)", "tolerance": 1e-6})
    css = data["srgb_css"]
    import re
    m = re.match(r"^rgba\((\d+), (\d+), (\d+), (.+)\)$", css)
    assert m is not None
    for ch in m.groups()[:3]:
        assert 0 <= int(ch) <= 255


def test_negative_hue_angle_preserved():
    # 越界色，负 hue 在计算时与 +360 等价；输出保留原始书写语义
    data = map({"color": "oklch(0.705 0.28 -215)", "tolerance": 1e-6})
    assert data["gamut_mapped"] is True
    assert data["mapped"]["hue"] == -215.0
    # 与 145° 结果的 sRGB 应当一致
    ref = map({"color": "oklch(0.705 0.28 145)", "tolerance": 1e-6})
    assert data["srgb"] == pytest.approx(ref["srgb"], abs=1e-9)


def test_original_unclamped_reflects_oob_channels():
    data = map({"color": "oklch(0.705 0.28 145)", "tolerance": 1e-6})
    pre = data["srgb_unclamped"]
    # 该方向 R 为负越界
    assert pre[0] < 0.0
    # 最终值在边界上（判定带 1e-7 容差，故为 0 附近的非负小量），
    # 8 位量化后必为 0
    assert 0.0 <= data["srgb"][0] <= 1e-6
    assert data["srgb_css"].startswith("rgba(0,")


def test_extremely_large_chroma_terminates():
    # 天文数字 C 也必须在轮数上限内终止且结果在色域内
    data = map({"color": "oklch(0.6 1e9 200)", "tolerance": 1e-6})
    assert data["gamut_mapped"] is True
    assert len(data["iterations"]) <= 1000
    assert all(0.0 <= c <= 1.0 for c in data["srgb"])
    assert data["final_interval"]["width"] <= 1e-6 + 1e-12
