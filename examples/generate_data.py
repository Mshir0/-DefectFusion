from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent / "data"

def save(path, defect=None, index=0):
    image = Image.new("RGB", (256, 256), (180, 185, 190))
    draw = ImageDraw.Draw(image)
    for y in range(0, 256, 16):
        draw.line((0, y, 256, y), fill=(150, 155, 160), width=1)
    if defect == "scratch":
        draw.line((40, 40 + index * 8, 210, 180), fill=(35, 35, 35), width=5)
    elif defect == "spot":
        draw.ellipse((90, 90, 150, 150), fill=(30, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

for i in range(3): save(ROOT / "normal" / f"normal_{i}.png", index=i)
save(ROOT / "test.png", "scratch")
print(f"wrote example data to {ROOT}")
