from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import argparse
from training.dataset import ImageFolderDataset

imagenet64_label_to_class = {
    "vegetables": {938: 'cauliflower', 937: 'broccoli', 945: 'bell pepper', 943: 'cucumber', 939: 'zucchini'},
    "dogs": {263: 'corgi', 258: 'samoyed', 250: 'husky', 207: 'golden retriever', 151: 'chihuahua'},
    "cats": {292: 'tiger', 281: 'tabby cat', 291: 'lion', 290: 'jaguar', 285: 'egyptian cat'}, 
    "bears": {297: 'sloth bear', 296: 'ice bear', 295: 'black bear', 294: 'brown bear', 388: 'panda'},
    "fishs": {2: 'white shark', 6: 'sting ray', 391: 'salmon', 1: 'gold fish', 148: 'orca'},
    "automobiles": {407: 'ambulance', 569: 'garbage truck', 609: 'jeep', 751: 'race car', 656: 'minivan'},
    "boats": {628: 'ship', 510: 'container ship', 724: 'pirate ship', 833: 'submarine', 814: 'speedboat'},
    "wild herbivores": {339: 'horse', 340: 'zebra', 350: 'goat', 345: 'ox', 354: 'camel'},
}

def create_imagenet_editing_subgroups(dataset_path, sample_save_dir):
    class_to_directory = {}
    for directory, class_dict in imagenet64_label_to_class.items():
        for class_label, _ in class_dict.items():
            class_to_directory[class_label] = directory

    dataset_obj = ImageFolderDataset(path=dataset_path, use_labels=True, xflip=False, cache=True)

    if not os.path.exists(sample_save_dir):
        os.makedirs(sample_save_dir)

    for i in tqdm(range(len(dataset_obj))):
        img = dataset_obj[i]
        label = np.argmax(img[1])

        if label in class_to_directory.keys():
            array = np.transpose(img[0], (1, 2, 0))
            image = Image.fromarray(array)
            class_name = imagenet64_label_to_class[class_to_directory[label]][label]
            group_dir = os.path.join(sample_save_dir, class_to_directory[label], str(label))
            os.makedirs(group_dir, exist_ok=True)
            count = len(os.listdir(group_dir))
            image.save(f"{group_dir}/{class_name}_{count}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create subgroups of ImageNet images for editing task.")
    parser.add_argument("--dataset_path", required=True, help="Path to the ImageNet64 dataset.")
    parser.add_argument("--save_dir", required=True, help="Path to save the subgroup samples.")

    args = parser.parse_args()
    create_imagenet_editing_subgroups(args.dataset_path, args.save_dir)

