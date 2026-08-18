import os
import csv
import cv2
import numpy as np

# Version file: code_lan1.py
# Sua loi false OK khi camera/ROI qua toi.


# =========================================================
# CAU HINH MAC DINH
# =========================================================

BASE_WIDTH = 400
BASE_HEIGHT = 400

SAVE_DEBUG_IMAGES = True

DRAW_LABEL_FOR_NG = True
DRAW_LABEL_FOR_D5 = True
DRAW_MAPPING_IMAGE = True


# =========================================================
# NGUONG MAU HSV
# OpenCV dung H trong khoang 0-180
# =========================================================

# Nen PCB mau xanh
LOWER_GREEN = np.array([35, 35, 35])
UPPER_GREEN = np.array([95, 255, 255])

# Mau den/toi chung: dung cho D, Q, U
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 90])

# Mask rieng cho dien tro SMT than den/xam
LOWER_SMT_RESISTOR = np.array([0, 0, 0])
UPPER_SMT_RESISTOR = np.array([180, 160, 170])

# Diode D1-D4 mau cam
LOWER_ORANGE = np.array([5, 50, 50])
UPPER_ORANGE = np.array([35, 255, 255])

# D5 mau trang / xam sang
LOWER_WHITE = np.array([0, 0, 80])
UPPER_WHITE = np.array([180, 180, 255])

# Dac trung trang/xam sang
LOW_SATURATION_MAX = 150
BRIGHT_VALUE_MIN = 80


# =========================================================
# KIEM TRA CHAT LUONG ANH / ANH SANG
# =========================================================
# Muc dich:
# - Khong cho anh gan nhu den hoan toan bi hieu nham la than linh kien den.
# - Anh toi vua phai se duoc tang sang nhe truoc khi nhan dien.
# - Anh qua toi, khong con thong tin thi tra ve INVALID_LIGHT thay vi OK.

NEAR_BLACK_VALUE = 10
MIN_GLOBAL_MEAN_GRAY = 14.0
MIN_GLOBAL_P90_GRAY = 30.0
MIN_GLOBAL_DYNAMIC_RANGE = 18.0
MAX_GLOBAL_NEAR_BLACK_RATIO = 0.90

LOW_LIGHT_ENHANCE_MEAN = 90.0
TARGET_LOW_LIGHT_MEAN = 110.0
LOW_LIGHT_BURST_FRAMES = 5


# =========================================================
# NGUONG MAC DINH THEO LOAI LINH KIEN
# =========================================================

DEFAULTS_BY_TYPE = {
    "resistor": {
        # Dien tro SMT:
        # - chi kiem tra core ROI o giua than linh kien
        # - tranh pad han, duong mach, bong toi o hai dau
        "max_green_ratio": 0.35,

        "min_smt_resistor_ratio": 0.30,
        "min_smt_resistor_area_ratio": 0.10,

        "min_smt_width_ratio": 0.60,
        "min_smt_height_ratio": 0.45,
        "max_smt_center_offset_ratio": 0.22,
        "min_smt_extent": 0.45,

        # 2 dac trung them cho rieng dien tro SMT
        # gray_std: do lech sang trong core ROI
        # laplacian_var: do texture / do chi tiet trong core ROI
        "min_gray_std": 8.0,
        "min_laplacian_var": 15.0,

        "min_contour_area": 4,
    },

    "orange": {
        "max_green_ratio": 0.85,
        "min_orange_ratio": 0.06,
        "min_orange_area_ratio": 0.02,
        "min_black_ratio": 0.04,
        "min_black_area_ratio": 0.015,
        "min_contour_area": 5,
    },

    "black": {
        "max_green_ratio": 0.85,
        "min_black_ratio": 0.08,
        "min_black_area_ratio": 0.025,
        "min_contour_area": 5,
    },

    "special_white": {
        "max_green_ratio": 0.90,
        "min_white_ratio": 0.04,
        "min_white_area_ratio": 0.015,
        "min_low_sat_bright_ratio": 0.04,
        "min_low_sat_bright_area_ratio": 0.015,
        "min_contour_area": 2,
    }
}


# =========================================================
# DANH SACH ROI LINH KIEN THEO THU TU COT
# bbox = (x, y, w, h)
# =========================================================

COMPONENT_ROIS = [
    # =====================================================
    # VI TRI 1 DEN 10 - HANG 1
    # =====================================================
    {"name": "R1",  "bbox": (0,   17, 27, 44)},   # 1
    {"name": "R2",  "bbox": (34,  17, 28, 44)},   # 2
    {"name": "R3",  "bbox": (71,  17, 25, 44)},   # 3
    {"name": "R4",  "bbox": (105, 17, 26, 44)},   # 4
    {"name": "R5",  "bbox": (139, 17, 27, 44)},   # 5
    {"name": "R6",  "bbox": (175, 17, 26, 44)},   # 6
    {"name": "R7",  "bbox": (210, 17, 25, 43)},   # 7
    {"name": "R8",  "bbox": (245, 17, 24, 43)},   # 8
    {"name": "R9",  "bbox": (278, 17, 26, 43)},   # 9
    {"name": "R10", "bbox": (313, 17, 26, 43)},   # 10

    # =====================================================
    # VI TRI 11 DEN 20 - HANG 2
    # =====================================================
    {"name": "R11", "bbox": (0,   64, 27, 43)},   # 11
    {"name": "R12", "bbox": (34,  64, 28, 43)},   # 12
    {"name": "R13", "bbox": (71,  64, 25, 43)},   # 13
    {"name": "R14", "bbox": (105, 64, 26, 43)},   # 14
    {"name": "R15", "bbox": (139, 64, 27, 43)},   # 15
    {"name": "R16", "bbox": (175, 64, 26, 43)},   # 16
    {"name": "R17", "bbox": (210, 64, 25, 43)},   # 17
    {"name": "R18", "bbox": (245, 63, 24, 44)},   # 18
    {"name": "R19", "bbox": (278, 63, 26, 44)},   # 19
    {"name": "R20", "bbox": (313, 62, 26, 45)},   # 20

    # =====================================================
    # VI TRI 21 DEN 30 - HANG 3
    # =====================================================
    {"name": "R21", "bbox": (4,   115, 22, 29)},  # 21
    {"name": "R22", "bbox": (41,  115, 20, 29)},  # 22
    {"name": "R23", "bbox": (76,  115, 19, 29)},  # 23
    {"name": "R24", "bbox": (110, 114, 20, 30)},  # 24
    {"name": "R25", "bbox": (144, 114, 21, 30)},  # 25
    {"name": "R26", "bbox": (180, 114, 20, 30)},  # 26
    {"name": "R27", "bbox": (214, 114, 20, 30)},  # 27
    {"name": "R28", "bbox": (247, 114, 20, 30)},  # 28
    {"name": "R29", "bbox": (283, 112, 20, 30)},  # 29
    {"name": "R30", "bbox": (317, 112, 19, 30)},  # 30

    # =====================================================
    # VI TRI 31 DEN 40 - HANG 4
    # =====================================================
    {"name": "R31", "bbox": (4,   147, 22, 33)},  # 31
    {"name": "R32", "bbox": (41,  147, 20, 33)},  # 32
    {"name": "R33", "bbox": (76,  147, 19, 33)},  # 33
    {"name": "R34", "bbox": (110, 147, 20, 33)},  # 34
    {"name": "R35", "bbox": (144, 147, 21, 33)},  # 35
    {"name": "R36", "bbox": (180, 147, 20, 33)},  # 36
    {"name": "R37", "bbox": (214, 147, 20, 33)},  # 37
    {"name": "R38", "bbox": (247, 147, 20, 33)},  # 38
    {"name": "R39", "bbox": (283, 145, 20, 35)},  # 39
    {"name": "R40", "bbox": (317, 145, 19, 34)},  # 40

    # =====================================================
    # VI TRI 41 DEN 49 - HANG 5
    # =====================================================
    {"name": "R41", "bbox": (5,   186, 26, 53)},  # 41
    {"name": "R42", "bbox": (46,  185, 27, 54)},  # 42
    {"name": "R43", "bbox": (85,  185, 28, 54)},  # 43
    {"name": "R44", "bbox": (122, 185, 27, 52)},  # 44
    {"name": "R45", "bbox": (163, 186, 23, 50)},  # 45
    {"name": "R46", "bbox": (201, 186, 26, 52)},  # 46
    {"name": "R47", "bbox": (239, 186, 24, 53)},  # 47
    {"name": "R48", "bbox": (275, 186, 24, 51)},  # 48
    {"name": "R49", "bbox": (312, 185, 26, 52)},  # 49

    # =====================================================
    # CAC LINH KIEN Q, D VA U
    # =====================================================

    # Vi tri 50 = Q3
    {"name": "Q3", "bbox": (108, 246, 30, 32)},   # 50

    # Vi tri 51 = D5
    {"name": "D5", "bbox": (165, 254, 24, 44)},   # 51

    # Vi tri 52 den 55 = D4 den D1
    {"name": "D4", "bbox": (199, 247, 28, 54)},   # 52
    {"name": "D3", "bbox": (236, 247, 28, 54)},   # 53
    {"name": "D2", "bbox": (274, 247, 27, 54)},   # 54
    {"name": "D1", "bbox": (312, 247, 27, 54)},   # 55

    # Vi tri 56 = Q4
    {"name": "Q4", "bbox": (108, 284, 30, 28)},   # 56

    # Vi tri 57 = Q5
    {"name": "Q5", "bbox": (108, 318, 30, 28)},   # 57

    # Vi tri 58 = Q1
    {"name": "Q1", "bbox": (166, 317, 29, 29)},   # 58

    # Vi tri 59 = Q6
    {"name": "Q6", "bbox": (108, 352, 30, 30)},   # 59

    # Vi tri 60 = Q2
    {"name": "Q2", "bbox": (166, 351, 29, 31)},   # 60

    # Vi tri 61 = U1
    {"name": "U1", "bbox": (237, 317, 121, 67)},  # 61
]


# =========================================================
# OVERRIDE RIENG CHO LINH KIEN DAC BIET
# =========================================================

COMPONENT_OVERRIDES = {
    "D5": {
        "custom_inner_bbox": (168, 263, 18, 26),
        "expected_color": "special_white",
        "max_green_ratio": 0.90,
        "min_white_ratio": 0.04,
        "min_white_area_ratio": 0.015,
        "min_low_sat_bright_ratio": 0.04,
        "min_low_sat_bright_area_ratio": 0.015,
        "min_contour_area": 2
    }
}


# =========================================================
# DANH GIA ANH SANG TOAN ROI 400x400
# =========================================================

def analyze_lighting(image):
    """
    Danh gia anh sang tren anh ROI goc, truoc khi tang sang.

    Anh qua toi se lam tat ca pixel roi vao mask den. Neu khong chan,
    code co the hieu nham nen toi la than linh kien va bao OK sai.
    """
    if image is None or image.size == 0:
        return {
            "is_valid": False,
            "reason": "empty_image",
            "mean_gray": 0.0,
            "median_gray": 0.0,
            "p10_gray": 0.0,
            "p90_gray": 0.0,
            "dynamic_range": 0.0,
            "near_black_ratio": 1.0,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_gray = float(np.mean(gray))
    median_gray = float(np.median(gray))
    p10_gray = float(np.percentile(gray, 10))
    p90_gray = float(np.percentile(gray, 90))
    dynamic_range = p90_gray - p10_gray
    near_black_ratio = float(np.mean(gray <= NEAR_BLACK_VALUE))

    reasons = []

    if mean_gray < MIN_GLOBAL_MEAN_GRAY:
        reasons.append("mean_too_low")

    if p90_gray < MIN_GLOBAL_P90_GRAY:
        reasons.append("p90_too_low")

    if dynamic_range < MIN_GLOBAL_DYNAMIC_RANGE:
        reasons.append("dynamic_range_too_low")

    if near_black_ratio > MAX_GLOBAL_NEAR_BLACK_RATIO:
        reasons.append("near_black_ratio_too_high")

    return {
        "is_valid": len(reasons) == 0,
        "reason": ";".join(reasons),
        "mean_gray": mean_gray,
        "median_gray": median_gray,
        "p10_gray": p10_gray,
        "p90_gray": p90_gray,
        "dynamic_range": dynamic_range,
        "near_black_ratio": near_black_ratio,
    }


# =========================================================
# TANG SANG ANH TOI VUA PHAI
# =========================================================

def enhance_low_light_image(image, lighting):
    """
    Chi tang sang khi anh van con thong tin.

    - Gamma correction: nang vung toi.
    - CLAHE tren kenh L: tang tuong phan cuc bo.
    - Bilateral filter: giam nhieu nhung giu bien.

    Anh qua toi da bi mat thong tin se khong duoc dua vao day.
    """
    mean_gray = max(float(lighting.get("mean_gray", 0.0)), 1.0)

    if mean_gray >= LOW_LIGHT_ENHANCE_MEAN:
        return image.copy(), False, 1.0

    normalized_mean = np.clip(mean_gray / 255.0, 1e-4, 0.9999)
    normalized_target = np.clip(TARGET_LOW_LIGHT_MEAN / 255.0, 1e-4, 0.9999)

    gamma = np.log(normalized_target) / np.log(normalized_mean)
    gamma = float(np.clip(gamma, 0.35, 1.0))

    lut = np.array([
        np.clip(((value / 255.0) ** gamma) * 255.0, 0, 255)
        for value in range(256)
    ], dtype=np.uint8)

    gamma_image = cv2.LUT(image, lut)

    lab = cv2.cvtColor(gamma_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    enhanced = cv2.bilateralFilter(
        enhanced,
        d=5,
        sigmaColor=35,
        sigmaSpace=35
    )

    return enhanced, True, gamma


# =========================================================
# GHI BAO CAO ANH SANG
# =========================================================

def save_lighting_report(lighting, output_path):
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "is_valid",
            "reason",
            "mean_gray",
            "median_gray",
            "p10_gray",
            "p90_gray",
            "dynamic_range",
            "near_black_ratio"
        ])
        writer.writerow([
            lighting["is_valid"],
            lighting["reason"],
            f"{lighting['mean_gray']:.6f}",
            f"{lighting['median_gray']:.6f}",
            f"{lighting['p10_gray']:.6f}",
            f"{lighting['p90_gray']:.6f}",
            f"{lighting['dynamic_range']:.6f}",
            f"{lighting['near_black_ratio']:.6f}"
        ])


def draw_invalid_light_result(image, lighting):
    result_image = image.copy()

    overlay = result_image.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (result_image.shape[1], result_image.shape[0]),
        (0, 0, 0),
        -1
    )
    result_image = cv2.addWeighted(result_image, 0.35, overlay, 0.65, 0)

    cv2.rectangle(
        result_image,
        (8, 130),
        (result_image.shape[1] - 8, 270),
        (0, 0, 180),
        -1
    )

    cv2.putText(
        result_image,
        "INVALID LIGHT - ANH QUA TOI",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result_image,
        "Them den roi chup lai, khong ket luan OK/NG.",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    metric_text = (
        f"mean={lighting['mean_gray']:.1f}  "
        f"p90={lighting['p90_gray']:.1f}  "
        f"range={lighting['dynamic_range']:.1f}"
    )

    cv2.putText(
        result_image,
        metric_text,
        (20, 242),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return result_image


# =========================================================
# SUY LUAN LOAI LINH KIEN THEO TEN
# =========================================================

def infer_expected_color(name):
    upper_name = name.upper()

    if upper_name == "D5":
        return "special_white"

    if upper_name in ["D1", "D2", "D3", "D4"]:
        return "orange"

    if upper_name.startswith("Q") or upper_name.startswith("U"):
        return "black"

    if upper_name.startswith("R"):
        return "resistor"

    return "resistor"


def build_component_config(component):
    config = component.copy()

    name = config["name"]
    expected_color = infer_expected_color(name)
    config["expected_color"] = expected_color

    defaults = DEFAULTS_BY_TYPE.get(expected_color, {})
    for key, value in defaults.items():
        config.setdefault(key, value)

    if name in COMPONENT_OVERRIDES:
        for key, value in COMPONENT_OVERRIDES[name].items():
            if key in ["name", "bbox"]:
                continue
            config[key] = value

    return config


# =========================================================
# KIEM TRA MAPPING
# =========================================================

def validate_component_mapping(components):
    print("\n===== CHECK MAPPING LINH KIEN =====")
    print(f"Tong so linh kien: {len(components)}")

    names_to_check = [
        "D1", "D2", "D3", "D4", "D5",
        "U1", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"
    ]

    seen_names = {}

    for index, component in enumerate(components, start=1):
        name = component["name"]
        bbox = component["bbox"]

        if name in seen_names:
            print(
                f"CANH BAO: Trung ten linh kien {name} "
                f"tai index {seen_names[name]} va {index}"
            )
        else:
            seen_names[name] = index

        if name in names_to_check:
            print(f"{index}.{name} -> bbox={bbox}")

    print("===================================\n")


# =========================================================
# SCALE VA CLIP BBOX
# =========================================================

def scale_bbox(bbox, image_shape):
    img_h, img_w = image_shape[:2]

    scale_x = img_w / BASE_WIDTH
    scale_y = img_h / BASE_HEIGHT

    x, y, w, h = bbox

    x = int(round(x * scale_x))
    y = int(round(y * scale_y))
    w = int(round(w * scale_x))
    h = int(round(h * scale_y))

    return x, y, w, h


def clip_bbox(x, y, w, h, image_shape):
    img_h, img_w = image_shape[:2]

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))

    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    return x, y, w, h


# =========================================================
# TAO INNER ROI VA CORE ROI
# =========================================================

def create_inner_bbox(x, y, w, h):
    if w > h * 1.3:
        shrink_x = int(w * 0.30)
        shrink_y = int(h * 0.15)

    elif h > w * 1.3:
        shrink_x = int(w * 0.15)
        shrink_y = int(h * 0.30)

    else:
        shrink_x = int(w * 0.20)
        shrink_y = int(h * 0.20)

    inner_x = x + shrink_x
    inner_y = y + shrink_y
    inner_w = max(1, w - 2 * shrink_x)
    inner_h = max(1, h - 2 * shrink_y)

    return inner_x, inner_y, inner_w, inner_h


def create_resistor_core_bbox(x, y, w, h):
    """
    Core ROI rieng cho dien tro SMT.
    Lay vung giua than linh kien, khong lay qua nho
    de tranh mot dom toi nho cung bi xem la OK.
    """

    if w >= h:
        # Dien tro nam ngang
        shrink_x = int(w * 0.30)
        shrink_y = int(h * 0.25)
    else:
        # Dien tro nam doc
        shrink_x = int(w * 0.25)
        shrink_y = int(h * 0.30)

    core_x = x + shrink_x
    core_y = y + shrink_y
    core_w = max(1, w - 2 * shrink_x)
    core_h = max(1, h - 2 * shrink_y)

    return core_x, core_y, core_w, core_h


# =========================================================
# TAO MASK TU COMPONENT
# =========================================================

def create_masks_from_component(image_shape, component):
    outer_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    inner_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    core_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    x, y, w, h = scale_bbox(component["bbox"], image_shape)
    x, y, w, h = clip_bbox(x, y, w, h, image_shape)

    if "custom_inner_bbox" in component:
        ix, iy, iw, ih = scale_bbox(component["custom_inner_bbox"], image_shape)
        ix, iy, iw, ih = clip_bbox(ix, iy, iw, ih, image_shape)
    else:
        ix, iy, iw, ih = create_inner_bbox(x, y, w, h)
        ix, iy, iw, ih = clip_bbox(ix, iy, iw, ih, image_shape)

    if component.get("expected_color") == "resistor":
        cx, cy, cw, ch = create_resistor_core_bbox(x, y, w, h)
        cx, cy, cw, ch = clip_bbox(cx, cy, cw, ch, image_shape)
    else:
        cx, cy, cw, ch = ix, iy, iw, ih

    cv2.rectangle(
        outer_mask,
        (x, y),
        (x + w - 1, y + h - 1),
        255,
        -1
    )

    cv2.rectangle(
        inner_mask,
        (ix, iy),
        (ix + iw - 1, iy + ih - 1),
        255,
        -1
    )

    cv2.rectangle(
        core_mask,
        (cx, cy),
        (cx + cw - 1, cy + ch - 1),
        255,
        -1
    )

    return (
        outer_mask,
        inner_mask,
        core_mask,
        (x, y, w, h),
        (ix, iy, iw, ih),
        (cx, cy, cw, ch)
    )


# =========================================================
# TIM CONTOUR LON NHAT
# =========================================================

def find_largest_contour_area(binary_mask, roi_area, min_contour_area):
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    max_area = 0
    best_contour = None
    valid_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            continue

        valid_count += 1

        if area > max_area:
            max_area = area
            best_contour = contour

    area_ratio = max_area / roi_area if roi_area > 0 else 0

    return max_area, area_ratio, best_contour, valid_count


def empty_feature_result(
    green_mask,
    black_mask,
    smt_resistor_mask,
    orange_mask,
    white_mask,
    low_sat_bright_mask
):
    return {
        "green_ratio": 0,
        "black_ratio": 0,
        "smt_resistor_ratio": 0,
        "orange_ratio": 0,
        "white_ratio": 0,
        "low_sat_bright_ratio": 0,

        "max_black_area": 0,
        "max_black_area_ratio": 0,
        "max_smt_resistor_area": 0,
        "max_smt_resistor_area_ratio": 0,
        "max_orange_area": 0,
        "max_orange_area_ratio": 0,
        "max_white_area": 0,
        "max_white_area_ratio": 0,
        "max_low_sat_bright_area": 0,
        "max_low_sat_bright_area_ratio": 0,

        "gray_std": 0,
        "laplacian_var": 0,

        "best_black_contour": None,
        "best_smt_resistor_contour": None,
        "best_orange_contour": None,
        "best_white_contour": None,
        "best_low_sat_bright_contour": None,

        "green_mask": green_mask,
        "black_mask": black_mask,
        "smt_resistor_mask": smt_resistor_mask,
        "orange_mask": orange_mask,
        "white_mask": white_mask,
        "low_sat_bright_mask": low_sat_bright_mask,

        "black_contour_count": 0,
        "smt_resistor_contour_count": 0,
        "orange_contour_count": 0,
        "white_contour_count": 0,
        "low_sat_bright_contour_count": 0
    }


# =========================================================
# TINH DAC TRUNG MAU TRONG MASK
# =========================================================

def calculate_color_features(image, feature_mask, min_contour_area):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    black_mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)

    smt_resistor_mask = cv2.inRange(
        hsv,
        LOWER_SMT_RESISTOR,
        UPPER_SMT_RESISTOR
    )

    orange_mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    white_mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    low_sat_mask = cv2.inRange(s_channel, 0, LOW_SATURATION_MAX)
    bright_mask = cv2.inRange(v_channel, BRIGHT_VALUE_MIN, 255)
    low_sat_bright_mask = cv2.bitwise_and(low_sat_mask, bright_mask)

    not_green_mask = cv2.bitwise_not(green_mask)

    low_sat_bright_not_green_mask = cv2.bitwise_and(
        low_sat_bright_mask,
        low_sat_bright_mask,
        mask=not_green_mask
    )

    green_mask = cv2.bitwise_and(green_mask, green_mask, mask=feature_mask)
    black_mask = cv2.bitwise_and(black_mask, black_mask, mask=feature_mask)

    smt_resistor_mask = cv2.bitwise_and(
        smt_resistor_mask,
        smt_resistor_mask,
        mask=feature_mask
    )

    orange_mask = cv2.bitwise_and(orange_mask, orange_mask, mask=feature_mask)
    white_mask = cv2.bitwise_and(white_mask, white_mask, mask=feature_mask)

    low_sat_bright_not_green_mask = cv2.bitwise_and(
        low_sat_bright_not_green_mask,
        low_sat_bright_not_green_mask,
        mask=feature_mask
    )

    kernel = np.ones((3, 3), np.uint8)

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    black_mask = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    smt_resistor_mask = cv2.morphologyEx(
        smt_resistor_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    orange_mask = cv2.morphologyEx(
        orange_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    low_sat_bright_not_green_mask = cv2.morphologyEx(
        low_sat_bright_not_green_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    roi_area = cv2.countNonZero(feature_mask)

    if roi_area == 0:
        return empty_feature_result(
            green_mask,
            black_mask,
            smt_resistor_mask,
            orange_mask,
            white_mask,
            low_sat_bright_not_green_mask
        )

    # 2 dac trung bo sung cho ROI:
    # - gray_std: do bien thien do sang
    # - laplacian_var: do texture / chi tiet
    roi_pixels = gray_blur[feature_mask > 0]

    if roi_pixels.size > 0:
        gray_std = float(np.std(roi_pixels))
    else:
        gray_std = 0.0

    masked_gray = cv2.bitwise_and(gray_blur, gray_blur, mask=feature_mask)
    laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
    lap_pixels = laplacian[feature_mask > 0]

    if lap_pixels.size > 0:
        laplacian_var = float(np.var(lap_pixels))
    else:
        laplacian_var = 0.0

    green_area = cv2.countNonZero(green_mask)
    black_area = cv2.countNonZero(black_mask)
    smt_resistor_area = cv2.countNonZero(smt_resistor_mask)
    orange_area = cv2.countNonZero(orange_mask)
    white_area = cv2.countNonZero(white_mask)
    low_sat_bright_area = cv2.countNonZero(low_sat_bright_not_green_mask)

    green_ratio = green_area / roi_area
    black_ratio = black_area / roi_area
    smt_resistor_ratio = smt_resistor_area / roi_area
    orange_ratio = orange_area / roi_area
    white_ratio = white_area / roi_area
    low_sat_bright_ratio = low_sat_bright_area / roi_area

    max_black_area, max_black_area_ratio, best_black_contour, black_count = find_largest_contour_area(
        black_mask,
        roi_area,
        min_contour_area
    )

    (
        max_smt_resistor_area,
        max_smt_resistor_area_ratio,
        best_smt_resistor_contour,
        smt_resistor_count
    ) = find_largest_contour_area(
        smt_resistor_mask,
        roi_area,
        min_contour_area
    )

    max_orange_area, max_orange_area_ratio, best_orange_contour, orange_count = find_largest_contour_area(
        orange_mask,
        roi_area,
        min_contour_area
    )

    max_white_area, max_white_area_ratio, best_white_contour, white_count = find_largest_contour_area(
        white_mask,
        roi_area,
        min_contour_area
    )

    (
        max_low_sat_bright_area,
        max_low_sat_bright_area_ratio,
        best_low_sat_bright_contour,
        low_sat_bright_count
    ) = find_largest_contour_area(
        low_sat_bright_not_green_mask,
        roi_area,
        min_contour_area
    )

    return {
        "green_ratio": green_ratio,
        "black_ratio": black_ratio,
        "smt_resistor_ratio": smt_resistor_ratio,
        "orange_ratio": orange_ratio,
        "white_ratio": white_ratio,
        "low_sat_bright_ratio": low_sat_bright_ratio,

        "max_black_area": max_black_area,
        "max_black_area_ratio": max_black_area_ratio,
        "max_smt_resistor_area": max_smt_resistor_area,
        "max_smt_resistor_area_ratio": max_smt_resistor_area_ratio,
        "max_orange_area": max_orange_area,
        "max_orange_area_ratio": max_orange_area_ratio,
        "max_white_area": max_white_area,
        "max_white_area_ratio": max_white_area_ratio,
        "max_low_sat_bright_area": max_low_sat_bright_area,
        "max_low_sat_bright_area_ratio": max_low_sat_bright_area_ratio,

        "gray_std": gray_std,
        "laplacian_var": laplacian_var,

        "best_black_contour": best_black_contour,
        "best_smt_resistor_contour": best_smt_resistor_contour,
        "best_orange_contour": best_orange_contour,
        "best_white_contour": best_white_contour,
        "best_low_sat_bright_contour": best_low_sat_bright_contour,

        "green_mask": green_mask,
        "black_mask": black_mask,
        "smt_resistor_mask": smt_resistor_mask,
        "orange_mask": orange_mask,
        "white_mask": white_mask,
        "low_sat_bright_mask": low_sat_bright_not_green_mask,

        "black_contour_count": black_count,
        "smt_resistor_contour_count": smt_resistor_count,
        "orange_contour_count": orange_count,
        "white_contour_count": white_count,
        "low_sat_bright_contour_count": low_sat_bright_count
    }


# =========================================================
# KIEM TRA HINH DANG THAN DIEN TRO SMT
# =========================================================

def is_smt_resistor_body_shape_ok(result, config):
    contour = result.get("best_smt_resistor_contour")

    if contour is None:
        return False

    fx, fy, fw, fh = result["feature_bbox"]

    if fw <= 0 or fh <= 0:
        return False

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 0 or h <= 0:
        return False

    contour_area = cv2.contourArea(contour)
    rect_area = w * h
    extent = contour_area / rect_area if rect_area > 0 else 0

    width_ratio = w / fw
    height_ratio = h / fh

    contour_cx = x + w / 2
    contour_cy = y + h / 2

    feature_cx = fx + fw / 2
    feature_cy = fy + fh / 2

    center_offset_x = abs(contour_cx - feature_cx) / fw
    center_offset_y = abs(contour_cy - feature_cy) / fh

    min_width_ratio = config.get("min_smt_width_ratio", 0.60)
    min_height_ratio = config.get("min_smt_height_ratio", 0.45)
    max_center_offset = config.get("max_smt_center_offset_ratio", 0.22)
    min_extent = config.get("min_smt_extent", 0.45)

    # Loai vung toi bi vo, mong, nhu net chu/duong mach/pad nho
    if extent < min_extent:
        return False

    center_ok = (
        center_offset_x <= max_center_offset
        and center_offset_y <= max_center_offset
    )

    if not center_ok:
        return False

    # Dien tro nam ngang
    if fw >= fh:
        if width_ratio >= min_width_ratio and height_ratio >= min_height_ratio:
            return True

    # Dien tro nam doc
    else:
        if height_ratio >= min_width_ratio and width_ratio >= min_height_ratio:
            return True

    return False


# =========================================================
# QUYET DINH OK / NG THEO LOAI LINH KIEN
# =========================================================

def calculate_resistor_soft_ng_score(result, config):
    """
    NG mem cho dien tro SMT.

    Khong ket luan NG chi vi 1 dieu kien bi fail.
    Moi dieu kien fail se cong 1 diem.
    Neu tong diem >= resistor_ng_score_threshold thi moi bao NG.

    Cac dieu kien duoc cham diem:
    1. green_ratio qua cao
    2. smt_resistor_ratio qua thap
    3. max_smt_resistor_area_ratio qua thap
    4. smt_shape_ok sai
    5. gray_std va laplacian_var deu qua thap
    """

    green_ratio = result["green_ratio"]
    smt_resistor_ratio = result["smt_resistor_ratio"]
    max_smt_resistor_area_ratio = result["max_smt_resistor_area_ratio"]
    gray_std = result.get("gray_std", 0)
    laplacian_var = result.get("laplacian_var", 0)
    shape_ok = result.get("smt_shape_ok", False)

    max_green_ratio = config.get("max_green_ratio", 0.35)
    min_smt_resistor_ratio = config.get("min_smt_resistor_ratio", 0.30)
    min_smt_resistor_area_ratio = config.get(
        "min_smt_resistor_area_ratio",
        0.10
    )
    min_gray_std = config.get("min_gray_std", 8.0)
    min_laplacian_var = config.get("min_laplacian_var", 15.0)

    score = 0
    reasons = []

    if green_ratio > max_green_ratio:
        score += 1
        reasons.append("green_high")

    if smt_resistor_ratio < min_smt_resistor_ratio:
        score += 1
        reasons.append("smt_ratio_low")

    if max_smt_resistor_area_ratio < min_smt_resistor_area_ratio:
        score += 1
        reasons.append("smt_area_low")

    if not shape_ok:
        score += 1
        reasons.append("shape_fail")

    # Anh iCam bi mo: texture chi tinh la 1 loi khi ca 2 chi so deu thap.
    # Neu chi gray_std thap hoac chi laplacian thap thi chua nen ket luan NG.
    if gray_std < min_gray_std and laplacian_var < min_laplacian_var:
        score += 1
        reasons.append("texture_low")

    return score, reasons


# =========================================================
# QUYET DINH OK / NG THEO LOAI LINH KIEN
# =========================================================

def decide_component_status(result, config):
    expected_color = result["expected_color"]

    green_ratio = result["green_ratio"]
    black_ratio = result["black_ratio"]
    smt_resistor_ratio = result["smt_resistor_ratio"]
    orange_ratio = result["orange_ratio"]
    white_ratio = result["white_ratio"]
    low_sat_bright_ratio = result["low_sat_bright_ratio"]

    max_black_area_ratio = result["max_black_area_ratio"]
    max_smt_resistor_area_ratio = result["max_smt_resistor_area_ratio"]
    max_orange_area_ratio = result["max_orange_area_ratio"]
    max_white_area_ratio = result["max_white_area_ratio"]
    max_low_sat_bright_area_ratio = result["max_low_sat_bright_area_ratio"]

    max_green_ratio = config.get("max_green_ratio", 0.85)
    green_too_much = green_ratio > max_green_ratio

    # Mac dinh cho linh kien khong phai R
    result["resistor_ng_score"] = ""
    result["resistor_ng_reasons"] = ""

    has_any_object_sign = (
        black_ratio >= 0.04
        or orange_ratio >= 0.04
        or white_ratio >= 0.04
        or low_sat_bright_ratio >= 0.04
        or max_black_area_ratio >= 0.015
        or max_orange_area_ratio >= 0.015
        or max_white_area_ratio >= 0.015
        or max_low_sat_bright_area_ratio >= 0.015
    )

    # -----------------------------------------------------
    # R: DIEN TRO SMT - NG MEM
    # -----------------------------------------------------
    if expected_color == "resistor":
        ng_score, ng_reasons = calculate_resistor_soft_ng_score(result, config)

        # Mac dinh: fail tu 3 dieu kien tro len moi NG.
        # Co the override rieng tung R:
        # "R48": {"resistor_ng_score_threshold": 4}
        ng_score_threshold = config.get("resistor_ng_score_threshold", 3)

        result["resistor_ng_score"] = ng_score
        result["resistor_ng_reasons"] = ";".join(ng_reasons)

        if ng_score >= ng_score_threshold:
            return "NG"

        return "OK"

    # -----------------------------------------------------
    # D1-D4: DIODE MAU CAM
    # -----------------------------------------------------
    if expected_color == "orange":
        min_orange_ratio = config.get("min_orange_ratio", 0.06)
        min_orange_area_ratio = config.get("min_orange_area_ratio", 0.02)
        min_black_ratio = config.get("min_black_ratio", 0.04)
        min_black_area_ratio = config.get("min_black_area_ratio", 0.015)

        has_orange = (
            orange_ratio >= min_orange_ratio
            or max_orange_area_ratio >= min_orange_area_ratio
        )

        has_black = (
            black_ratio >= min_black_ratio
            or max_black_area_ratio >= min_black_area_ratio
        )

        if green_too_much and not has_orange and not has_black:
            return "NG"

        if has_orange or has_black:
            return "OK"

        return "NG"

    # -----------------------------------------------------
    # D5: THAN TRANG / XAM SANG
    # -----------------------------------------------------
    if expected_color == "special_white":
        min_white_ratio = config.get("min_white_ratio", 0.04)
        min_white_area_ratio = config.get("min_white_area_ratio", 0.015)
        min_low_sat_bright_ratio = config.get("min_low_sat_bright_ratio", 0.04)
        min_low_sat_bright_area_ratio = config.get(
            "min_low_sat_bright_area_ratio",
            0.015
        )

        has_white = (
            white_ratio >= min_white_ratio
            or max_white_area_ratio >= min_white_area_ratio
        )

        has_low_sat_bright = (
            low_sat_bright_ratio >= min_low_sat_bright_ratio
            or max_low_sat_bright_area_ratio >= min_low_sat_bright_area_ratio
        )

        if green_ratio <= max_green_ratio and (has_white or has_low_sat_bright):
            return "OK"

        return "NG"

    # -----------------------------------------------------
    # Q / U: LINH KIEN THAN DEN
    # -----------------------------------------------------
    if expected_color == "black":
        min_black_ratio = config.get("min_black_ratio", 0.08)
        min_black_area_ratio = config.get("min_black_area_ratio", 0.025)

        has_black = (
            black_ratio >= min_black_ratio
            or max_black_area_ratio >= min_black_area_ratio
        )

        if not has_black:
            return "NG"

        if green_ratio <= max_green_ratio and has_black:
            return "OK"

        if green_too_much and not has_black:
            return "NG"

        if has_black:
            return "OK"

        return "NG"

    # -----------------------------------------------------
    # MAC DINH
    # -----------------------------------------------------
    if green_ratio <= max_green_ratio and has_any_object_sign:
        return "OK"

    return "NG"

# =========================================================
# KIEM TRA MOT LINH KIEN
# =========================================================

def check_one_component(image, component):
    config = build_component_config(component)

    index = config.get("index")
    name = config["name"]
    expected_color = config["expected_color"]

    min_contour_area = config.get("min_contour_area", 8)

    (
        outer_mask,
        inner_mask,
        core_mask,
        outer_bbox,
        inner_bbox,
        core_bbox
    ) = create_masks_from_component(
        image.shape,
        config
    )

    if expected_color == "resistor":
        feature_mask = core_mask
        feature_bbox = core_bbox
    else:
        feature_mask = inner_mask
        feature_bbox = inner_bbox

    features = calculate_color_features(
        image,
        feature_mask,
        min_contour_area
    )

    result = {
        "index": index,
        "name": name,
        "expected_color": expected_color,

        "green_ratio": features["green_ratio"],
        "black_ratio": features["black_ratio"],
        "smt_resistor_ratio": features["smt_resistor_ratio"],
        "orange_ratio": features["orange_ratio"],
        "white_ratio": features["white_ratio"],
        "low_sat_bright_ratio": features["low_sat_bright_ratio"],

        "gray_std": features["gray_std"],
        "laplacian_var": features["laplacian_var"],

        "max_black_area": features["max_black_area"],
        "max_black_area_ratio": features["max_black_area_ratio"],
        "max_smt_resistor_area": features["max_smt_resistor_area"],
        "max_smt_resistor_area_ratio": features["max_smt_resistor_area_ratio"],
        "max_orange_area": features["max_orange_area"],
        "max_orange_area_ratio": features["max_orange_area_ratio"],
        "max_white_area": features["max_white_area"],
        "max_white_area_ratio": features["max_white_area_ratio"],
        "max_low_sat_bright_area": features["max_low_sat_bright_area"],
        "max_low_sat_bright_area_ratio": features["max_low_sat_bright_area_ratio"],

        "best_black_contour": features["best_black_contour"],
        "best_smt_resistor_contour": features["best_smt_resistor_contour"],
        "best_orange_contour": features["best_orange_contour"],
        "best_white_contour": features["best_white_contour"],
        "best_low_sat_bright_contour": features["best_low_sat_bright_contour"],

        "outer_bbox": outer_bbox,
        "inner_bbox": inner_bbox,
        "core_bbox": core_bbox,
        "feature_bbox": feature_bbox,

        "outer_mask": outer_mask,
        "inner_mask": inner_mask,
        "core_mask": core_mask,
        "feature_mask": feature_mask,

        "green_mask": features["green_mask"],
        "black_mask": features["black_mask"],
        "smt_resistor_mask": features["smt_resistor_mask"],
        "orange_mask": features["orange_mask"],
        "white_mask": features["white_mask"],
        "low_sat_bright_mask": features["low_sat_bright_mask"]
    }

    if expected_color == "resistor":
        result["smt_shape_ok"] = is_smt_resistor_body_shape_ok(result, config)
    else:
        result["smt_shape_ok"] = ""

    result["status"] = decide_component_status(result, config)

    return result


# =========================================================
# CHON CONTOUR DE VE
# =========================================================

def choose_contour_for_drawing(result, expected_color):
    if expected_color == "resistor":
        if result["best_smt_resistor_contour"] is not None:
            return result["best_smt_resistor_contour"]
        return result["best_black_contour"]

    if expected_color == "orange":
        if result["best_orange_contour"] is not None:
            return result["best_orange_contour"]
        return result["best_black_contour"]

    if expected_color == "special_white":
        if result["best_white_contour"] is not None:
            return result["best_white_contour"]
        return result["best_low_sat_bright_contour"]

    if expected_color == "black":
        return result["best_black_contour"]

    if result["best_black_contour"] is not None:
        return result["best_black_contour"]

    return result["best_low_sat_bright_contour"]


# =========================================================
# VE KET QUA LEN ANH
# =========================================================

def draw_component_result(result_image, result):
    index = result["index"]
    name = result["name"]
    status = result["status"]
    expected_color = result["expected_color"]

    x, y, w, h = result["outer_bbox"]
    ix, iy, iw, ih = result["inner_bbox"]
    cx, cy, cw, ch = result["core_bbox"]

    if status == "OK":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    # Outer ROI: xanh neu OK, do neu NG
    cv2.rectangle(
        result_image,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    # Inner ROI: vang
    cv2.rectangle(
        result_image,
        (ix, iy),
        (ix + iw, iy + ih),
        (0, 255, 255),
        1
    )

    # Core ROI cua dien tro SMT: tim
    if expected_color == "resistor":
        cv2.rectangle(
            result_image,
            (cx, cy),
            (cx + cw, cy + ch),
            (255, 0, 255),
            1
        )

    contour = choose_contour_for_drawing(result, expected_color)

    if contour is not None:
        cv2.drawContours(
            result_image,
            [contour],
            -1,
            color,
            1
        )

    should_draw_label = False

    if DRAW_LABEL_FOR_NG and status == "NG":
        should_draw_label = True

    if DRAW_LABEL_FOR_D5 and name == "D5":
        should_draw_label = True

    if should_draw_label:
        label = f"{index}.{name}:{status}"

        cv2.putText(
            result_image,
            label,
            (x, max(y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA
        )


# =========================================================
# VE TOM TAT LEN ANH
# =========================================================

def draw_missing_summary(result_image, missing_components):
    if len(missing_components) == 0:
        text = "OK - Khong phat hien linh kien thieu"
        color = (0, 255, 0)
    else:
        text = f"NG - Nghi thieu {len(missing_components)} linh kien"
        color = (0, 0, 255)

    cv2.rectangle(
        result_image,
        (5, 5),
        (390, 32),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        result_image,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )


# =========================================================
# VE ANH MAPPING DEBUG
# =========================================================

def draw_mapping_image(image, components, output_path="debug_component_mapping.jpg"):
    mapping_image = image.copy()

    for index, component in enumerate(components, start=1):
        x, y, w, h = scale_bbox(component["bbox"], image.shape)
        x, y, w, h = clip_bbox(x, y, w, h, image.shape)

        cv2.rectangle(
            mapping_image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            1
        )

        label = f"{index}.{component['name']}"

        cv2.putText(
            mapping_image,
            label,
            (x, max(y - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(output_path, mapping_image)


# =========================================================
# LUU DEBUG IMAGE
# =========================================================

def save_debug_images(result):
    index = result["index"]
    name = result["name"]

    prefix = f"debug_components/{index:02d}_{name}"

    cv2.imwrite(f"{prefix}_green_mask.jpg", result["green_mask"])
    cv2.imwrite(f"{prefix}_black_mask.jpg", result["black_mask"])
    cv2.imwrite(f"{prefix}_smt_resistor_mask.jpg", result["smt_resistor_mask"])
    cv2.imwrite(f"{prefix}_orange_mask.jpg", result["orange_mask"])
    cv2.imwrite(f"{prefix}_white_mask.jpg", result["white_mask"])
    cv2.imwrite(f"{prefix}_low_sat_bright_mask.jpg", result["low_sat_bright_mask"])
    cv2.imwrite(f"{prefix}_inner_mask.jpg", result["inner_mask"])
    cv2.imwrite(f"{prefix}_core_mask.jpg", result["core_mask"])
    cv2.imwrite(f"{prefix}_feature_mask.jpg", result["feature_mask"])


# =========================================================
# LUU CSV TAT CA LINH KIEN
# =========================================================

def save_results_csv(all_results, csv_output_path):
    with open(csv_output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "index",
            "name",
            "expected_color",
            "status",

            "green_ratio",
            "black_ratio",
            "smt_resistor_ratio",
            "orange_ratio",
            "white_ratio",
            "low_sat_bright_ratio",
            "gray_std",
            "laplacian_var",

            "max_black_area",
            "max_black_area_ratio",
            "max_smt_resistor_area",
            "max_smt_resistor_area_ratio",
            "smt_shape_ok",

            "max_orange_area",
            "max_orange_area_ratio",
            "max_white_area",
            "max_white_area_ratio",
            "max_low_sat_bright_area",
            "max_low_sat_bright_area_ratio",
            "resistor_ng_score",
            "resistor_ng_reasons",

            "outer_bbox",
            "inner_bbox",
            "core_bbox",
            "feature_bbox"
        ])

        for item in all_results:
            writer.writerow([
                item["index"],
                item["name"],
                item["expected_color"],
                item["status"],

                f"{item['green_ratio']:.6f}",
                f"{item['black_ratio']:.6f}",
                f"{item['smt_resistor_ratio']:.6f}",
                f"{item['orange_ratio']:.6f}",
                f"{item['white_ratio']:.6f}",
                f"{item['low_sat_bright_ratio']:.6f}",
                f"{item['gray_std']:.6f}",
                f"{item['laplacian_var']:.6f}",

                f"{item['max_black_area']:.2f}",
                f"{item['max_black_area_ratio']:.6f}",
                f"{item['max_smt_resistor_area']:.2f}",
                f"{item['max_smt_resistor_area_ratio']:.6f}",
                item["smt_shape_ok"],

                f"{item['max_orange_area']:.2f}",
                f"{item['max_orange_area_ratio']:.6f}",
                f"{item['max_white_area']:.2f}",
                f"{item['max_white_area_ratio']:.6f}",
                f"{item['max_low_sat_bright_area']:.2f}",
                f"{item['max_low_sat_bright_area_ratio']:.6f}",
                item.get("resistor_ng_score", ""),
                item.get("resistor_ng_reasons", ""),

                item["outer_bbox"],
                item["inner_bbox"],
                item["core_bbox"],
                item["feature_bbox"]
            ])


# =========================================================
# LUU DANH SACH LINH KIEN THIEU TXT
# =========================================================

def save_missing_components_txt(missing_components, output_path="missing_components.txt"):
    with open(output_path, mode="w", encoding="utf-8") as file:
        file.write("===== DANH SACH LINH KIEN NGHI THIEU =====\n\n")

        if len(missing_components) == 0:
            file.write("Khong co linh kien nao bi nghi thieu.\n")
            return

        for item in missing_components:
            file.write(
                f"{item['index']}.{item['name']} | "
                f"type={item['expected_color']} | "
                f"status={item['status']} | "
                f"green={item['green_ratio']:.6f} | "
                f"black={item['black_ratio']:.6f} | "
                f"smt_resistor={item['smt_resistor_ratio']:.6f} | "
                f"smt_area={item['max_smt_resistor_area_ratio']:.6f} | "
                f"smt_shape_ok={item['smt_shape_ok']} | "
                f"orange={item['orange_ratio']:.6f} | "
                f"white={item['white_ratio']:.6f} | "
                f"low_sat_bright={item['low_sat_bright_ratio']:.6f} | "
                f"gray_std={item['gray_std']:.6f} | "
                f"laplacian_var={item['laplacian_var']:.6f} | "
                f"ng_score={item.get('resistor_ng_score', '')} | "
                f"ng_reasons={item.get('resistor_ng_reasons', '')} | "
                f"outer_bbox={item['outer_bbox']} | "
                f"inner_bbox={item['inner_bbox']} | "
                f"core_bbox={item['core_bbox']} | "
                f"feature_bbox={item['feature_bbox']}\n"
            )


# =========================================================
# LUU DANH SACH LINH KIEN THIEU CSV
# =========================================================

def save_missing_components_csv(missing_components, output_path="missing_components.csv"):
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "index",
            "name",
            "expected_color",
            "status",

            "green_ratio",
            "black_ratio",
            "smt_resistor_ratio",
            "max_smt_resistor_area_ratio",
            "smt_shape_ok",

            "orange_ratio",
            "white_ratio",
            "low_sat_bright_ratio",
            "gray_std",
            "laplacian_var",
            "resistor_ng_score",
            "resistor_ng_reasons",

            "outer_bbox",
            "inner_bbox",
            "core_bbox",
            "feature_bbox"
        ])

        for item in missing_components:
            writer.writerow([
                item["index"],
                item["name"],
                item["expected_color"],
                item["status"],

                f"{item['green_ratio']:.6f}",
                f"{item['black_ratio']:.6f}",
                f"{item['smt_resistor_ratio']:.6f}",
                f"{item['max_smt_resistor_area_ratio']:.6f}",
                item["smt_shape_ok"],

                f"{item['orange_ratio']:.6f}",
                f"{item['white_ratio']:.6f}",
                f"{item['low_sat_bright_ratio']:.6f}",
                f"{item['gray_std']:.6f}",
                f"{item['laplacian_var']:.6f}",
                item.get("resistor_ng_score", ""),
                item.get("resistor_ng_reasons", ""),

                item["outer_bbox"],
                item["inner_bbox"],
                item["core_bbox"],
                item["feature_bbox"]
            ])


# =========================================================
# KIEM TRA TOAN BO LINH KIEN
# =========================================================

def check_all_components(
    image_path,
    components,
    output_path="result_component_check.jpg",
    csv_output_path="result_component_check.csv",
    missing_txt_path="missing_components.txt",
    missing_csv_path="missing_components.csv"
):
    validate_component_mapping(components)

    print("Current folder:", os.getcwd())
    print("Image path:", image_path)
    print("Image exists:", os.path.exists(image_path))

    raw_image = cv2.imread(image_path)

    if raw_image is None:
        raise ValueError(f"Khong doc duoc anh PCB: {image_path}")

    lighting = analyze_lighting(raw_image)

    lighting_report_path = (
        os.path.splitext(csv_output_path)[0]
        + "_lighting.csv"
    )
    save_lighting_report(lighting, lighting_report_path)

    print("\n===== KIEM TRA ANH SANG =====")
    print(
        f"mean={lighting['mean_gray']:.3f}, "
        f"median={lighting['median_gray']:.3f}, "
        f"p90={lighting['p90_gray']:.3f}, "
        f"range={lighting['dynamic_range']:.3f}, "
        f"near_black={lighting['near_black_ratio']:.3f}"
    )

    # Anh gan nhu den hoan toan: khong du thong tin de nhan dien.
    # Phai dung tai day de tranh moi vung deu bi xem la mau den cua linh kien.
    if not lighting["is_valid"]:
        result_image = draw_invalid_light_result(raw_image, lighting)
        cv2.imwrite(output_path, result_image)

        save_results_csv([], csv_output_path)
        save_missing_components_txt([], missing_txt_path)
        save_missing_components_csv([], missing_csv_path)

        return {
            "final_status": (
                "INVALID_LIGHT - Anh qua toi, khong du thong tin de "
                "ket luan linh kien. Hay them anh sang va chup lai."
            ),
            "missing_components": [],
            "components": [],
            "lighting": lighting,
            "light_valid": False,
            "low_light_enhanced": False,
            "gamma": 1.0,
            "lighting_report_path": lighting_report_path,
        }

    image, low_light_enhanced, gamma = enhance_low_light_image(
        raw_image,
        lighting
    )

    if low_light_enhanced:
        cv2.imwrite("debug_low_light_enhanced.jpg", image)
        print(f"Da tang sang anh toi. Gamma={gamma:.3f}")
    else:
        print("Anh sang du, khong can tang sang.")

    if DRAW_MAPPING_IMAGE:
        draw_mapping_image(
            image,
            components,
            output_path="debug_component_mapping.jpg"
        )

    result_image = image.copy()

    all_results = []
    missing_components = []

    if SAVE_DEBUG_IMAGES:
        os.makedirs("debug_components", exist_ok=True)

    for index, component in enumerate(components, start=1):
        component_with_index = component.copy()
        component_with_index["index"] = index

        result = check_one_component(image, component_with_index)
        all_results.append(result)

        if result["status"] != "OK":
            missing_components.append(result)

        draw_component_result(result_image, result)

        if SAVE_DEBUG_IMAGES:
            save_debug_images(result)

    draw_missing_summary(result_image, missing_components)

    light_text = (
        f"Light mean={lighting['mean_gray']:.1f} "
        f"p90={lighting['p90_gray']:.1f} "
        f"enhanced={'YES' if low_light_enhanced else 'NO'}"
    )

    cv2.rectangle(
        result_image,
        (3, result_image.shape[0] - 22),
        (result_image.shape[1] - 3, result_image.shape[0] - 3),
        (0, 0, 0),
        -1
    )
    cv2.putText(
        result_image,
        light_text,
        (7, result_image.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.imwrite(output_path, result_image)

    save_results_csv(all_results, csv_output_path)
    save_missing_components_txt(missing_components, missing_txt_path)
    save_missing_components_csv(missing_components, missing_csv_path)

    if len(missing_components) == 0:
        final_status = "OK - Tat ca linh kien deu co dau hieu xuat hien"
    else:
        missing_names = [
            f"{item['index']}.{item['name']}"
            for item in missing_components
        ]

        final_status = "NG - Nghi thieu: " + ", ".join(missing_names)

    return {
        "final_status": final_status,
        "missing_components": missing_components,
        "components": all_results,
        "lighting": lighting,
        "light_valid": True,
        "low_light_enhanced": low_light_enhanced,
        "gamma": gamma,
        "lighting_report_path": lighting_report_path,
    }


# =========================================================
# IN KET QUA
# =========================================================

def print_results(result):
    print("\n===== KET QUA TOAN MACH =====")
    print(result["final_status"])

    lighting = result.get("lighting", {})

    if lighting:
        print("\n===== THONG SO ANH SANG =====")
        print(
            f"valid={lighting.get('is_valid')} | "
            f"reason={lighting.get('reason', '')} | "
            f"mean={lighting.get('mean_gray', 0):.3f} | "
            f"median={lighting.get('median_gray', 0):.3f} | "
            f"p90={lighting.get('p90_gray', 0):.3f} | "
            f"range={lighting.get('dynamic_range', 0):.3f} | "
            f"near_black={lighting.get('near_black_ratio', 0):.3f}"
        )

    if not result.get("light_valid", True):
        print("\nKhong kiem tra linh kien vi anh qua toi.")
        print("Hay them den, giam khoang cach hoac tang exposure roi chup lai.")
        return

    print(
        "Low-light enhancement:",
        "ON" if result.get("low_light_enhanced") else "OFF",
        "| gamma=",
        f"{result.get('gamma', 1.0):.3f}"
    )

    print("\n===== DANH SACH LINH KIEN NGHI THIEU =====")

    if len(result["missing_components"]) == 0:
        print("Khong co linh kien nao bi nghi thieu.")
    else:
        for item in result["missing_components"]:
            print(
                f"- {item['index']}.{item['name']} | "
                f"type={item['expected_color']} | "
                f"green={item['green_ratio']:.3f}, "
                f"black={item['black_ratio']:.3f}, "
                f"smt_resistor={item['smt_resistor_ratio']:.3f}, "
                f"smt_area={item['max_smt_resistor_area_ratio']:.3f}, "
                f"smt_shape_ok={item['smt_shape_ok']}, "
                f"orange={item['orange_ratio']:.3f}, "
                f"white={item['white_ratio']:.3f}, "
                f"low_sat_bright={item['low_sat_bright_ratio']:.3f}, "
                f"gray_std={item['gray_std']:.3f}, "
                f"laplacian_var={item['laplacian_var']:.3f}, "
                f"outer_bbox={item['outer_bbox']}, "
                f"inner_bbox={item['inner_bbox']}, "
                f"core_bbox={item['core_bbox']}, "
                f"feature_bbox={item['feature_bbox']}"
            )

    print("\n===== KET QUA TUNG LINH KIEN =====")

    for item in result["components"]:
        print(
            f"{item['index']}.{item['name']}: {item['status']} | "
            f"type={item['expected_color']} | "
            f"green={item['green_ratio']:.3f}, "
            f"black={item['black_ratio']:.3f}, "
            f"smt_resistor={item['smt_resistor_ratio']:.3f}, "
            f"smt_area={item['max_smt_resistor_area_ratio']:.3f}, "
            f"smt_shape_ok={item['smt_shape_ok']}, "
            f"orange={item['orange_ratio']:.3f}, "
            f"white={item['white_ratio']:.3f}, "
            f"low_sat_bright={item['low_sat_bright_ratio']:.3f}, "
            f"gray_std={item['gray_std']:.3f}, "
            f"laplacian_var={item['laplacian_var']:.3f}, "
            f"black_area={item['max_black_area_ratio']:.3f}, "
            f"orange_area={item['max_orange_area_ratio']:.3f}, "
            f"white_area={item['max_white_area_ratio']:.3f}, "
            f"low_sat_bright_area={item['max_low_sat_bright_area_ratio']:.3f}, "
            f"outer_bbox={item['outer_bbox']}, "
            f"inner_bbox={item['inner_bbox']}, "
            f"core_bbox={item['core_bbox']}, "
            f"feature_bbox={item['feature_bbox']}"
        )

    print("\n===== DEBUG RIENG D5 =====")

    for item in result["components"]:
        if item["name"] == "D5":
            print(
                f"{item['index']}.{item['name']}: {item['status']} | "
                f"type={item['expected_color']} | "
                f"green={item['green_ratio']:.3f}, "
                f"white={item['white_ratio']:.3f}, "
                f"low_sat_bright={item['low_sat_bright_ratio']:.3f}, "
                f"gray_std={item['gray_std']:.3f}, "
                f"laplacian_var={item['laplacian_var']:.3f}, "
                f"white_area={item['max_white_area_ratio']:.3f}, "
                f"low_sat_bright_area={item['max_low_sat_bright_area_ratio']:.3f}, "
                f"outer_bbox={item['outer_bbox']}, "
                f"inner_bbox={item['inner_bbox']}"
            )


# =========================================================
# CHAY CHUONG TRINH
# =========================================================


# =========================================================
# CHAY REALTIME CAMERA - VE KHUNG ROI 400x400 O GIUA
# NHAN C HOAC SPACE DE CHUP ROI VA KIEM TRA
# =========================================================

def get_center_roi_400(frame):
    """
    Lay vung ROI 400x400 nam giua frame camera.
    Anh dua vao ham check_all_components phai dung kich thuoc 400x400
    de khop voi he toa do COMPONENT_ROIS.
    """
    frame_h, frame_w = frame.shape[:2]
    roi_size = 400

    if frame_w < roi_size or frame_h < roi_size:
        raise ValueError(
            f"Frame camera qua nho: {frame_w}x{frame_h}. "
            f"Can toi thieu {roi_size}x{roi_size}."
        )

    x1 = (frame_w - roi_size) // 2
    y1 = (frame_h - roi_size) // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    roi = frame[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)


def draw_camera_guide(preview, roi_box, lighting=None):
    x1, y1, x2, y2 = roi_box

    if lighting is not None and lighting.get("is_valid", False):
        guide_color = (0, 255, 0)
        light_status = "LIGHT OK"
    else:
        guide_color = (0, 0, 255)
        light_status = "TOO DARK - THEM DEN"

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        guide_color,
        2
    )

    cv2.putText(
        preview,
        "ROI 400x400 - Dat PCB vao khung nay",
        (x1, max(y1 - 12, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        guide_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        preview,
        "Nhan C/SPACE de chup ROI - Q de thoat",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    if lighting is not None:
        light_text = (
            f"{light_status} | mean={lighting['mean_gray']:.1f} "
            f"p90={lighting['p90_gray']:.1f} "
            f"range={lighting['dynamic_range']:.1f}"
        )

        cv2.rectangle(
            preview,
            (10, 42),
            (620, 70),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            preview,
            light_text,
            (15, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            guide_color,
            2,
            cv2.LINE_AA
        )

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    cv2.line(preview, (center_x, y1), (center_x, y2), guide_color, 1)
    cv2.line(preview, (x1, center_y), (x2, center_y), guide_color, 1)

    return preview


def capture_center_roi_burst(cap, frame_count=LOW_LIGHT_BURST_FRAMES):
    """
    Lay trung vi nhieu frame de giam nhieu camera trong dieu kien toi.
    Camera va PCB can duoc giu co dinh trong luc chup.
    """
    roi_frames = []

    for _ in range(max(1, frame_count)):
        ret, frame = cap.read()

        if not ret:
            continue

        try:
            roi, _ = get_center_roi_400(frame)
            roi_frames.append(roi)
        except ValueError:
            continue

    if len(roi_frames) == 0:
        return None

    if len(roi_frames) == 1:
        return roi_frames[0]

    stack = np.stack(roi_frames, axis=0)
    median_roi = np.median(stack, axis=0).astype(np.uint8)

    return median_roi


def run_realtime_camera():
    # Ban dang dung webcam index 1 trong code cu.
    # Neu khong mo duoc, doi thanh 0 hoac 2.
    CAMERA_INDEX = 1

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Khong mo duoc camera.")
        print("Thu doi CAMERA_INDEX = 0 hoac CAMERA_INDEX = 2.")
        return

    print("Dang mo camera...")
    print("Dat PCB vao khung ROI 400x400 o giua man hinh.")
    print("Khung xanh: anh sang du de kiem tra.")
    print("Khung do: anh qua toi, chuong trinh se khong bao OK sai.")
    print("Nhan C hoac SPACE de chup ROI va kiem tra.")
    print("Nhan Q de thoat.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Khong doc duoc frame tu camera.")
            break

        preview = frame.copy()
        roi = None
        current_lighting = None

        try:
            roi, roi_box = get_center_roi_400(frame)
            current_lighting = analyze_lighting(roi)
            preview = draw_camera_guide(
                preview,
                roi_box,
                current_lighting
            )
        except ValueError as error:
            cv2.putText(
                preview,
                str(error),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Camera Preview", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        if key == ord("c") or key == ord("C") or key == 32:
            if roi is None:
                print("Khong the chup vi chua tao duoc ROI 400x400.")
                continue

            captured_roi = capture_center_roi_burst(
                cap,
                frame_count=LOW_LIGHT_BURST_FRAMES
            )

            if captured_roi is None:
                captured_roi = roi.copy()

            capture_path = "captured.jpg"
            cv2.imwrite(capture_path, captured_roi)

            captured_lighting = analyze_lighting(captured_roi)

            print("\nDa chup ROI 400x400:", capture_path)
            print(
                f"Anh sang: mean={captured_lighting['mean_gray']:.3f}, "
                f"p90={captured_lighting['p90_gray']:.3f}, "
                f"range={captured_lighting['dynamic_range']:.3f}, "
                f"valid={captured_lighting['is_valid']}"
            )
            print("Dang kiem tra linh kien...")

            result = check_all_components(
                image_path=capture_path,
                components=COMPONENT_ROIS,
                output_path="result_component_check.jpg",
                csv_output_path="result_component_check.csv",
                missing_txt_path="missing_components.txt",
                missing_csv_path="missing_components.csv"
            )

            print_results(result)

            result_image = cv2.imread("result_component_check.jpg")

            if result_image is not None:
                cv2.imshow("Ket qua kiem tra", result_image)

            print("\nDa luu anh ket qua: result_component_check.jpg")
            print("Da luu anh ROI da chup: captured.jpg")
            print("Da luu CSV: result_component_check.csv")
            print("Da luu bao cao anh sang: result_component_check_lighting.csv")
            print("Da luu danh sach thieu: missing_components.txt")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_camera()
