import os
import random
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

def build_font_dict(target_root):
    font_dict = defaultdict(dict)
    font_list = os.listdir(target_root)
    for font in font_list:
        font_path = os.path.join(target_root, font)
        if not os.path.isdir(font_path):
            continue
        for fname in os.listdir(font_path):
            if '+' not in fname or not fname.endswith('.jpg'):
                continue
            try:
                _, char = fname[:-4].split('+')
                font_dict[font][char] = os.path.join(font_path, fname)
            except:
                continue
    return font_dict

def sample_pairs(font_dict, content_root, output_dir, n_samples=100):
    content_img_dir = os.path.join(output_dir, 'content_img')
    GT_img_dir = os.path.join(output_dir, 'GT_img')
    style_img_dir = os.path.join(output_dir, 'style_img')

    os.makedirs(content_img_dir, exist_ok=True)
    os.makedirs(GT_img_dir, exist_ok=True)
    os.makedirs(style_img_dir, exist_ok=True)

    fonts = list(font_dict.keys())
    saved = 0

    for _ in tqdm(range(n_samples), desc="Sampling"):
        fontA, fontB = random.sample(fonts, 2)  # fontB is target font

        common_chars = list(set(font_dict[fontA].keys()) & set(font_dict[fontB].keys()))
        if len(common_chars) < 1 or len(font_dict[fontB]) < 2:
            continue

        char = random.choice(common_chars)
        style_choices = list(set(font_dict[fontB].keys()) - {char})
        if not style_choices:
            continue
        style_char = random.choice(style_choices)

        # Pick content image from a different font (not fontB)
        content_fonts = [f for f in fonts if f != fontB and char in font_dict[f]]
        if not content_fonts:
            continue
        content_font = random.choice(content_fonts)
        content_img_path = font_dict[content_font][char]

        try:
            content_img = Image.open(content_img_path)
            GT_img = Image.open(font_dict[fontB][char])
            style_img = Image.open(font_dict[fontB][style_char])

            idx = saved + 1
            content_img.save(os.path.join(content_img_dir, f"{idx}.jpg"))
            GT_img.save(os.path.join(GT_img_dir, f"{idx}.jpg"))
            style_img.save(os.path.join(style_img_dir, f"{idx}.jpg"))

            saved += 1
        except Exception as e:
            print(f"Error processing {char}: {e}")
            continue

        if saved >= n_samples:
            break

    print(f"Finished sampling {saved} samples.")


if __name__ == "__main__":
    content_root = "/scratch/yl10337/FontDiffuser/data_examples/train/ContentImage"
    target_root = "/scratch/yl10337/FontDiffuser/data_examples/train/TargetImage"
    output_dir = "/scratch/yl10337/FontDiffuser/data_examples/train/sampled_output"

    n_samples = 3600  # number of samples you want to generate
    font_dict = build_font_dict(target_root)
    sample_pairs(font_dict, content_root, output_dir, n_samples=n_samples)