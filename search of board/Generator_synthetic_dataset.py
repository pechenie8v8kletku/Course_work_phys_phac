import os
import random
import numpy as np
import json
from PIL import Image, ImageDraw
import colorsys
import pycocotools.mask as maskr
from multiprocessing import Pool

IMG_SIZE = (512, 512)
PRIMITIVE_RANGE = (10, 69)
NOISE_DELTA = 15
SAVE_DIR = "generated_dataset"
OUTPUT_CONTOURS = "training_contours"
OUTPUT_PNGS = "training_img"
START_INDEX = 2000
END_INDEX = 3000

def find_boundaries(mask_img):
    mask_np = np.array(mask_img)
    boundaries = np.zeros_like(mask_np, dtype=np.uint8)
    for y in range(1, mask_np.shape[0] - 1):
        for x in range(1, mask_np.shape[1] - 1):
            square_area = mask_np[y-1:y+2, x-1:x+2]
            if np.any(square_area != mask_np[y, x]):
                boundaries[y, x] = 255
    return boundaries

def random_color():
    return tuple(random.randint(0, 255) for _ in range(3)) + (255,)

def unique_mask_color(used_colors):
    while True:
        color = tuple(random.randint(0, 255) for _ in range(3))
        if color not in used_colors:
            used_colors.add(color)
            return color

def add_primitives(draw, num_primitives, mask_draw, used_colors):
    for _ in range(num_primitives):
        shape_type = random.choice(["rectangle", "ellipse", "line"])
        x1, y1 = random.randint(0, IMG_SIZE[0]), random.randint(0, IMG_SIZE[1])
        x2, y2 = random.randint(0, IMG_SIZE[0]), random.randint(0, IMG_SIZE[1])
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        color = random_color()
        mask_color = unique_mask_color(used_colors)
        if shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=color)
            mask_draw.rectangle([x1, y1, x2, y2], fill=mask_color)
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color)
            mask_draw.ellipse([x1, y1, x2, y2], fill=mask_color)
        elif shape_type == "line":
            line_width = random.randint(1, 5)
            draw.line([x1, y1, x2, y2], fill=color, width=line_width)
            mask_draw.line([x1, y1, x2, y2], fill=mask_color, width=line_width)

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def apply_hsv_noise(image):
    image = image.convert("RGB")
    np_image = np.array(image, dtype=np.float32) / 255.0
    hsv_image = np.zeros_like(np_image)
    for i in range(np_image.shape[0]):
        for j in range(np_image.shape[1]):
            hsv_image[i, j] = colorsys.rgb_to_hsv(*np_image[i, j, :3])
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            h, s, v = hsv_image[i, j]
            h += random.uniform(-NOISE_DELTA / 360.0, NOISE_DELTA / 360.0)
            s += random.uniform(-NOISE_DELTA / 100.0, NOISE_DELTA / 100.0)
            v += random.uniform(-NOISE_DELTA / 100.0, NOISE_DELTA / 100.0)
            hsv_image[i, j] = [h % 1.0, clamp(s, 0.0, 1.0), clamp(v, 0.0, 1.0)]
    noisy_image = (np.array([[colorsys.hsv_to_rgb(*hsv) for hsv in row] for row in hsv_image]) * 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def generate_image(i):
    img = Image.new("RGBA", IMG_SIZE, (0, 0, 0, 255))
    mask = Image.new("RGB", IMG_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    mask_draw = ImageDraw.Draw(mask)
    used_colors = set()
    num_primitives = random.randint(*PRIMITIVE_RANGE)
    add_primitives(draw, num_primitives, mask_draw, used_colors)
    bounds = find_boundaries(mask)
    '''cv2.imshow('Masks on Image', bounds)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    rle = maskr.encode(np.asfortranarray(bounds))
    annotation = {"segmentation": {'size': rle[0]['size'], 'counts': rle[0]['counts'].decode('utf-8')}}
    os.makedirs(OUTPUT_CONTOURS, exist_ok=True)
    os.makedirs(OUTPUT_PNGS, exist_ok=True)
    with open(os.path.join(OUTPUT_CONTOURS, f"mask_{i:03d}_annotations_skelet.json"), 'w') as f:
        json.dump(annotation, f, indent=4)
    apply_hsv_noise(img).save(os.path.join(OUTPUT_PNGS, f"image_{i:03d}.png"))
    print(f"Сгенерировано {i}")

if __name__ == "__main__":
    with Pool() as pool:
        pool.map(generate_image, range(START_INDEX, END_INDEX))