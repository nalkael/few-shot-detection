import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,10],
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    #data_path = 'datasets/cocosplit/datasplit/trainvalno5k.json'
    data_path = 'datasets/merged_ortho_coco/train/_annotations.coco.json'
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
        if a['iscrowd'] == 1:
            continue
        anno[a['category_id']].append(a)

    # for i in range(args.seeds[0], args.seeds[1]):
    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        # random.seed(0)
        for c in ID2CLASS.keys():
            img_ids = {}
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []
            # for shots in [1, 5, 10, 20]:
            for shots in [10, 20, 30]: # 10-shot to 50-shot
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots: # maybe change the constrain condition to avoid dead loop
                            break
                    if len(sample_shots) == shots: # maybe the same as above @ Huaixin
                        break
                new_data = {
                    #'info': data['info'],
                    #'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(data_path, ID2CLASS[c], shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    save_dir = os.path.join('datasets', 'cocosplit', 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    ID2CLASS = {

        1: "Gasschieberdeckel",
        2: "Kanalschachtdeckel",
        3: "Sinkkaesten",
        4: "Unterflurhydrant",
        5: "Versorgungsschacht",
        6: "Wasserschieberdeckel",

        7: "aircraft",
        8: "oiltank",
        9: "overpass",
        10: "playground",
        
        # 13: "other",
        # 14: "propeller",
        # 15: "pushbacktruck",
        # 16: "stairtruck",
        # 17: "trainer",
        # 18: "truck",
        # 19: "van",
        
        # 15: "bench",
        # 16: "bird",
        # 17: "cat",
        # 18: "dog",
        # 19: "horse",
        # 20: "sheep",
        # 21: "cow",
        # 22: "elephant",
        # 23: "bear",
        # 24: "zebra",
        # 25: "giraffe",
        # 27: "backpack",
        # 28: "umbrella",
        # 31: "handbag",
        # 32: "tie",
        # 33: "suitcase",
        # 34: "frisbee",
        # 35: "skis",
        # 36: "snowboard",
        # 37: "sports ball",
        # 38: "kite",
        # 39: "baseball bat",
        # 40: "baseball glove",
        # 41: "skateboard",
        # 42: "surfboard",
        # 43: "tennis racket",
        # 44: "bottle",
        # 46: "wine glass",
        # 47: "cup",
        # 48: "fork",
        # 49: "knife",
        # 50: "spoon",
        # 51: "bowl",
        # 52: "banana",
        # 53: "apple",
        # 54: "sandwich",
        # 55: "orange",
        # 56: "broccoli",
        # 57: "carrot",
        # 58: "hot dog",
        # 59: "pizza",
        # 60: "donut",
        # 61: "cake",
        # 62: "chair",
        # 63: "couch",
        # 64: "potted plant",
        # 65: "bed",
        # 67: "dining table",
        # 70: "toilet",
        # 72: "tv",
        # 73: "laptop",
        # 74: "mouse",
        # 75: "remote",
        # 76: "keyboard",
        # 77: "cell phone",
        # 78: "microwave",
        # 79: "oven",
        # 80: "toaster",
        # 81: "sink",
        # 82: "refrigerator",
        # 84: "book",
        # 85: "clock",
        # 86: "vase",
        # 87: "scissors",
        # 88: "teddy bear",
        # 89: "hair drier",
        # 90: "toothbrush",
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
