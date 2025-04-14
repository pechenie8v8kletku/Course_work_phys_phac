import os
import cv2
import random
import numpy as np
import json
from PIL import Image, ImageDraw
import colorsys
import pycocotools.mask as maskr
from multiprocessing import Pool

STRENGTH=(0.7,5)
IMG_SIZE = (512, 512)
PRIMITIVE_RANGE = (10,25)
NOISE_DELTA = 5
SAVE_DIR = "generated_dataset"
OUTPUT_CONTOURS = "training_masks"
OUTPUT_PNGS = "training_img"
START_INDEX = 0
END_INDEX = 1000
NUM_VECTORS=32
# цветные вектора H(0-179) S(0-255) V(0-255)
color_list={(0,20,230):1,#bel
            (0,20,190):2,#sv ser
            (0,20,140):3,
            (0,20,90):4,
            (0,20,20):5
            }
# яркостные вектора надо где то еще в 2-3 раза больше
bright_list={(0,220,220):6,
             (20,220,220):9,
             (40,220,220):12,
             (60,220,220):15,
             (80,220,220):18,
             (100,220,220):21,
             (120,220,220):24,
             (140,220,220):27,
             (160,220,220):30,
             (180,220,220):6,
             (0,90,90):8,
             (0,150,150):7,
             (20,150,150):10,
             (20,90,90):11,
             (40,150,150):12,
             (40,90,90):13,
             (60,150,150):16,
             (60,90,90):17,
             (80,150,150):19,
             (80,90,90):20,
             (100,150,150):22,
             (100,90,90):23,
             (120,150,150):25,
             (120,90,90):26,
             (140,150,150):28,
             (140,90,90):29,
             (160,150,150):31,
             (160,90,90):32,
             (180,150,150):7,(180,90,90):8
             }
color_list1={(0,20,230):1,
            (0,20,190):2,
            (0,20,140):3,
            (0,20,90):4,
            (0,20,20):5,
            (0, 220, 220): 6,
            (20, 220, 220): 9,
            (40, 220, 220): 12,
            (60, 220, 220): 15,
            (80, 220, 220): 18,
            (100, 220, 220): 21,
            (120, 220, 220): 24,
            (140, 220, 220): 27,
            (160, 220, 220): 30,
            (180, 220, 220): 6,
            (0, 90, 90): 8,
            (0, 150, 150): 7,
            (20, 150, 150): 10,
            (20, 90, 90): 11,
            (40, 150, 150): 12,
            (40, 90, 90): 13,
            (60, 150, 150): 16,
            (60, 90, 90): 17,
            (80, 150, 150): 19,
            (80, 90, 90): 20,
            (100, 150, 150): 22,
            (100, 90, 90): 23,
            (120, 150, 150): 25,
            (120, 90, 90): 26,
            (140, 150, 150): 28,
            (140, 90, 90): 29,
            (160, 150, 150): 31,
            (160, 90, 90): 32,
            (180, 150, 150): 7, (180, 90, 90): 8
             }
class_to_color = {v: k for k, v in color_list1.items()}

def apply_gradient_layer(img,color, angle_range=(0, 90),x_max=512,x_min=0,y_max=512,y_min=0):
    img=np.array(img)
    angle = random.uniform(angle_range[0], angle_range[1])
    angle_rad = np.radians(angle)
    grad_x = np.cos(angle_rad)
    grad_y = np.sin(angle_rad)
    len_x=x_max-x_min+30
    len_y=y_max-y_min+30
    strength_s=np.minimum(random.uniform(STRENGTH[0],STRENGTH[1]),255/(1+color[1])-1)/1.42/2
    strength_v=np.minimum(random.uniform(STRENGTH[0],STRENGTH[1]),255/(1+color[2])-1)/1.42/2
    for x in range(IMG_SIZE[0]):
        for y in range(IMG_SIZE[1]):
            h,s,v=img[x,y]
            s =s * (1 + strength_s * (((y-y_min)/len_y)*grad_y+((x-x_min)/len_x)*grad_x))
            v =v * (1 + strength_v * (((y-y_min)/len_y)*grad_y+((x-x_min)/len_x)*grad_x))
            img[x,y] = h,s,v
    return img
def random_color():
    return (random.randint(0,255),random.randint(0, 255),random.randint(0, 255))

def find_nearest_vec(vec,list1):
    min_dist=float("inf")
    nearest_class=None
    for v,cls in list1.items():
        dis=np.linalg.norm(np.array(v)-np.array(vec))
        if dis<min_dist:
            min_dist=dis
            nearest_class=cls
    return nearest_class

def select_vector(vec):
    if vec[1]>50 and vec[2]>40:
        vec1=list(vec)
        vec1[0]=(vec1[0]*179/255)
        vec1=tuple(vec1)
        color_class = find_nearest_vec(vec1,bright_list)
    elif vec[2]<40:
        color_class=5
    else:

        color_class=find_nearest_vec(vec,color_list)

    return color_class

def generate(i):
    mask_arr = np.full(IMG_SIZE,5, dtype=np.uint8)
    mask = Image.fromarray(mask_arr)
    draw_mask=ImageDraw.Draw(mask)
    num_primitives = random.randint(*PRIMITIVE_RANGE)
    img=Image.fromarray((add_primitives(num_primitives, draw_mask)).astype('uint8'),mode="HSV")
    noisy=add_gauss(img)
    mask_matr=np.array(mask)
    '''
    hsv_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    for class_idx, hsv in class_to_color .items():
        h,s,v=hsv
        hsv=h*255/180,s,v
        hsv_mask[mask_matr == class_idx] = hsv
    imgg = Image.fromarray(hsv_mask, mode="HSV").convert("RGB")
    imgg.save(f"bebrazdec.png")
    '''


    annotation=to_json(mask_matr)
    os.makedirs(OUTPUT_CONTOURS, exist_ok=True)
    os.makedirs(OUTPUT_PNGS, exist_ok=True)
    with open(os.path.join(OUTPUT_CONTOURS, f"mask_{i:03d}_annotations.json"), 'w') as f:
        json.dump(annotation, f, indent=4)
    noisy.save(os.path.join(OUTPUT_PNGS, f"image_{i:03d}.png"))


def add_primitives(num_primitives,mask):
    loch=Image.new("HSV", IMG_SIZE, (0, 0, 0))
    loch=np.array(loch)
    for _ in range(num_primitives):
        shape_type = random.choice(["rectangle", "ellipse", "star","polygon"])
        x1, y1 = random.randint(0, IMG_SIZE[0]), random.randint(0, IMG_SIZE[1])
        x2, y2 = random.randint(0, IMG_SIZE[0]), random.randint(0, IMG_SIZE[1])
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        color = random_color()
        local= Image.new("HSV", IMG_SIZE, (0, 0, 0))
        local_draw=ImageDraw.Draw(local)
        y_max=IMG_SIZE[0]
        x_max=IMG_SIZE[0]
        x_min=0
        y_min=0
        if shape_type == "rectangle":


            local_draw.rectangle([x1, y1, x2, y2], fill=color)
            mask.rectangle([x1, y1, x2, y2], fill=select_vector(color))

        elif shape_type == "ellipse":
            local_draw.ellipse([x1, y1, x2, y2], fill=color)
            mask.ellipse([x1, y1, x2, y2], fill=select_vector(color))


        elif shape_type == "polygon":
            num_angles=random.randint(3,16)
            rad=random.randint(30,100)
            cx=(x1+x2)/2
            cy=(y1+y2)/2
            points=[]
            x_max = np.minimum(cx + rad*1.42, IMG_SIZE[0])
            x_min = np.maximum(cx - rad*1.42, 0)
            y_max = np.minimum(cy + rad*1.42, IMG_SIZE[1])
            y_min = np.maximum(cy - rad*1.42, 0)
            for i in range(num_angles):
                angle=2*i*np.pi/num_angles
                x = int(cx + rad * np.cos(angle))
                y = int(cy + rad * np.sin(angle))
                points.append((x,y))
            local_draw.polygon(points, fill=color)
            mask.polygon(points, fill=select_vector(color))



        elif shape_type == "star":
            num_rays = random.randint(4, 7)
            cx, cy = random.randint(50, 450), random.randint(50, 450)
            inner_radius = random.randint(10, 30)
            outer_radius = random.randint(40, 70)
            points = []
            x_max=np.minimum(cx+outer_radius*1.42,IMG_SIZE[0])
            x_min=np.maximum(cx-outer_radius*1.42,1)
            y_max=np.minimum(cy+outer_radius*1.42,IMG_SIZE[1])
            y_min=np.maximum(cy-outer_radius*1.42,1)
            for i in range(num_rays * 2):
                angle = i * (3.14159 / num_rays)
                r = outer_radius if i % 2 == 0 else inner_radius
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                points.append((x, y))
            local_draw.polygon(points, fill=color)
            mask.polygon(points,fill=select_vector(color))
        v=random.randint(0,1)
        if v==1:
            local=apply_gradient_layer(local,color,x_max=x_max,x_min=x_min,y_max=y_max,y_min=y_min)
        hueta=(np.all(np.array(local) == [0, 0, 0], axis=-1)).astype(np.uint8)
        hueta=np.stack([hueta] * 3, axis=-1)
        loch=loch*hueta+np.array(local).astype(np.uint8)
    return loch

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))
def add_gauss(image):
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

def to_json(array):
    annotations = []
    for i in range(1,NUM_VECTORS+1):
        mask=np.full(IMG_SIZE,i,dtype=np.uint8)
        intersection=np.where(mask == array, mask, 0)
        rle=maskr.encode(np.asfortranarray(intersection))
        annotation = {
            "segmentation": {
                'class': i,
                'size': rle['size'],
                'counts': rle['counts'].decode('utf-8')
            },
        }
        annotations.append(annotation)
    return annotations



if __name__ == "__main__":
    with Pool() as pool:
        pool.map(generate, range(START_INDEX, END_INDEX))






