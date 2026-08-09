"""PWA 아이콘 생성 스크립트 — 1회성 유틸. 실행 후 icons/*.png 를 생성한다.
   ♩(콰터노트) 모양을 직접 그려서 폰트 글리프 의존성을 없앰."""
import math
from PIL import Image, ImageDraw
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'icons')
os.makedirs(OUT_DIR, exist_ok=True)

PRIMARY = (0, 118, 206)      # --primary
SKY     = (79, 195, 247)     # --sky (gradient end)
WHITE   = (255, 255, 255)


def gradient_bg(size, corner_radius_ratio=0.0):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    px = img.load()
    for y in range(size):
        t = y / (size - 1)
        r = int(PRIMARY[0] + (SKY[0] - PRIMARY[0]) * t)
        g = int(PRIMARY[1] + (SKY[1] - PRIMARY[1]) * t)
        b = int(PRIMARY[2] + (SKY[2] - PRIMARY[2]) * t)
        for x in range(size):
            px[x, y] = (r, g, b, 255)
    if corner_radius_ratio > 0:
        mask = Image.new('L', (size, size), 0)
        d = ImageDraw.Draw(mask)
        rad = int(size * corner_radius_ratio)
        d.rounded_rectangle([0, 0, size - 1, size - 1], radius=rad, fill=255)
        img.putalpha(mask)
    return img


def draw_quarter_note(draw, cx, cy, scale):
    """중심(cx,cy) 기준 콰터노트(♩) 실루엣을 그린다. scale=음표 전체 높이 기준."""
    head_w = scale * 0.34
    head_h = scale * 0.26
    stem_w = scale * 0.09
    stem_h = scale * 0.78

    head_cx = cx - scale * 0.12
    head_cy = cy + scale * 0.30

    # 부리(머리) — 살짝 기울어진 타원
    bbox = [head_cx - head_w / 2, head_cy - head_h / 2, head_cx + head_w / 2, head_cy + head_h / 2]
    draw.ellipse(bbox, fill=WHITE)

    # 기둥(stem)
    stem_x = head_cx + head_w / 2 - stem_w * 0.55
    stem_top = head_cy - stem_h
    draw.rectangle([stem_x, stem_top, stem_x + stem_w, head_cy], fill=WHITE)


def make_icon(size, maskable=False):
    corner_ratio = 0.0 if maskable else 0.22
    img = gradient_bg(size, corner_ratio)
    draw = ImageDraw.Draw(img)
    # maskable 아이콘은 OS가 원형/둥근사각형으로 잘라내므로 안전 영역(가운데 80%)에만 그린다.
    note_scale = size * (0.34 if maskable else 0.42)
    cx, cy = size * 0.52, size * 0.5
    draw_quarter_note(draw, cx, cy, note_scale)
    return img


sizes_any = [192, 512]
for s in sizes_any:
    make_icon(s, maskable=False).save(os.path.join(OUT_DIR, f'icon-{s}.png'))

make_icon(512, maskable=True).save(os.path.join(OUT_DIR, 'icon-maskable-512.png'))

# apple-touch-icon: iOS는 자체적으로 둥근 모서리를 씌우므로 사각형(각진) 배경으로 생성
apple = gradient_bg(180, corner_radius_ratio=0.0)
d = ImageDraw.Draw(apple)
draw_quarter_note(d, 180 * 0.52, 180 * 0.5, 180 * 0.42)
apple.convert('RGB').save(os.path.join(OUT_DIR, 'apple-touch-icon.png'))

# favicon
make_icon(64, maskable=False).save(os.path.join(OUT_DIR, 'favicon-64.png'))

print('done:', os.listdir(OUT_DIR))
