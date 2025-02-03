import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse

def organize_imagenet_dataset(annotations_dir, images_dir):

    # Loop through XML files in the annotations directory
    for xml_file in tqdm(os.listdir(annotations_dir)):
        if xml_file.endswith('.xml'):
            # Parse the XML file
            xml_path = os.path.join(annotations_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get the object class name (first <name> tag)
            object_class = root.find('.//object/name').text

            # Create a class directory 
            class_dir = os.path.join(images_dir, object_class)
            os.makedirs(class_dir, exist_ok=True)

            # Find the corresponding JPEG file (same name as XML file but with .JPEG extension)
            image_file = xml_file.replace('.xml', '.JPEG')
            image_path = os.path.join(images_dir, image_file)

            # Move the JPEG file to the class directory
            if os.path.exists(image_path):
                shutil.move(image_path, class_dir)
            else:
                print(f"Image {image_file} not found!")

    print("Files moved to class bins successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize ImageNet validation set into class directories based on annotations.")
    parser.add_argument("--annote_dir", required=True, help="Path to the annotations directory.")
    parser.add_argument("--images_dir", required=True, help="Path to the images directory.")
    args = parser.parse_args()

    organize_imagenet_dataset(args.annote_dir, args.images_dir)
