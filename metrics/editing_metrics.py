import numpy as np
import os
import torch
import dnnlib
import lpips

from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch_utils import distributed as dist

#----------------------------------------------------------------------------


cifar10_label_to_class = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
imagenet64_label_to_class = {"vegetables": {938: 'cauliflower', 937: 'broccoli', 945: 'bell pepper', 943: 'cucumber', 939: 'zucchini'},
                             "dogs": {263: 'corgi', 258: 'samoyed', 250: 'husky', 207: 'golden retriever', 151: 'chihuahua'},
                             "cats": {292: 'tiger', 281: 'tabby cat', 291: 'lion', 290: 'jaguar', 285: 'egyptian cat'}, 
                             "bears": {297: 'sloth bear', 296: 'ice bear', 295: 'black bear', 294: 'brown bear', 388: 'panda'},
                             "fishs": {2: 'white shark', 6: 'sting ray', 391: 'salmon', 1: 'gold fish', 148: 'orca'},
                             "automobiles": {407: 'ambulance', 569: 'garbage truck', 609: 'jeep', 751: 'race car', 656: 'minivan'},
                             "boats": {628: 'ship', 510: 'container ship', 724: 'pirate ship', 833: 'submarine', 814: 'speedboat'},
                             "wild herbivores": {339: 'horse', 340: 'zebra', 350: 'goat', 345: 'ox', 354: 'camel'},
                             }
dataset_labels = {"cifar10":cifar10_label_to_class, "im64":imagenet64_label_to_class}
template = "a photo of {}"


def render(img, save_path):
    img = (img + 1) / 2.0    
    save_image(img, save_path, nrow=4) # saving as a grid, 4 images per row


def compute_editing_metrics_presampled_2k(opts, sample_size=16):
    device = opts.device
    w = opts.w_scale
    save_root_dir = opts.im64_visualize_dir
    im64_subg_dir = opts.im64_subg_dir

    transform = transforms.Compose([
        transforms.ToTensor(), # convert image to (C, H, W) format with values between 0 and 1
    ])

    lpips_model = lpips.LPIPS(net='alex').to(device)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def get_text_embeddings(texts):
        inputs = tokenizer(texts, return_tensors = "pt").to(device)
        text_embeddings = model.get_text_features(**inputs)
        return text_embeddings / text_embeddings.norm(dim=-1,keepdim=True)
    
    def get_image_embeddings(images):
        image_inputs = processor(text=None, images=images, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)
        image_embedding = model.get_image_features(image_inputs)
        return image_embedding / image_embedding.norm(dim=-1,keepdim=True)

    def get_similarity_score(text_emb, image_emb):
        return text_emb @ image_emb.T

    metrics_dict = {}
    num_classes = 1000
    sorted_groups = sorted(os.listdir(im64_subg_dir))

    with torch.no_grad():
        for j in range(len(sorted_groups)):
            if j % opts.num_gpus != dist.get_rank():
                continue
            
            group = sorted_groups[j]
            print(f"[rank{dist.get_rank()}] reached here ready to edit group {group}")

            labels = os.listdir(os.path.join(im64_subg_dir,group))
            one_hot_matrix = torch.zeros((len(labels), num_classes), dtype=torch.int).to(device)
            for i in range(len(labels)):
                one_hot_matrix[i,int(labels[i])] = 1

            metrics_dict[group] = {"lpips": {}, "clip": {}}   

            for i in range(len(labels)):
                src_class_name = imagenet64_label_to_class[group][int(labels[i])]
                src_batch_fnames = [f"{src_class_name}_{i}.png" for i in range(sample_size)]
                src_class_labels = one_hot_matrix[i].unsqueeze(0).repeat(sample_size, 1)

                src_batch_images = []
                for filename in src_batch_fnames:
                    image_path = os.path.join(im64_subg_dir, group, labels[i], filename)
                    image = Image.open(image_path).convert('RGB')  # ensure the image is in RGB mode
                    image_tensor = transform(image)  # apply transformation (will normalize to [0, 1])
                    src_batch_images.append(image_tensor)
                src_batch_images = torch.stack(src_batch_images).to(device) * 2 - 1

                # Save image
                save_dir = os.path.join(save_root_dir,group,src_class_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                render(src_batch_images.cpu(), os.path.join(save_dir, f"src_{src_class_name}.png"))

                for j in range(len(labels)):
                    if i==j: 
                        # Skip reconstruction.
                        continue
                    else:
                        tar_class_name = imagenet64_label_to_class[group][int(labels[j])]
                        tar_class_labels = one_hot_matrix[j].unsqueeze(0).repeat(sample_size, 1)
                        edit_images, _ = opts.editing_fn(opts.G, src_batch_images, src_labels=src_class_labels, target_labels=tar_class_labels, **opts.G_kwargs)

                        # Save image.
                        render(edit_images.cpu(), os.path.join(save_dir,f"w={str(int(w))}_src_{src_class_name}_tar_{tar_class_name}.png") )

                        lpips_value = lpips_model(src_batch_images, edit_images).detach().flatten().cpu().numpy()

                        target_prompt = template.format(tar_class_name)
                        text_emb = get_text_embeddings(target_prompt)
                        norm_edit_images = (edit_images + 1.0) / 2.0 # [-1,1] -> [0,1]
                        image_emb = get_image_embeddings(norm_edit_images)
                        clip_score = get_similarity_score(text_emb, image_emb).detach().flatten().cpu().numpy()
                        
                        metrics_dict[group]["lpips"][f'{src_class_name}_to_{tar_class_name}'] = lpips_value.mean()
                        metrics_dict[group]["clip"][f'{src_class_name}_to_{tar_class_name}'] = clip_score.mean()
            
            metrics_dict[group]["mean_lpips"] = np.mean(list(metrics_dict[group]["lpips"].values()))
            metrics_dict[group]["mean_clip"] = np.mean(list(metrics_dict[group]["clip"].values()))
        
        # Flatten dictionary.
        metrics_full_dict = {}
        for group, metric_dict in metrics_dict.items():
            metrics_full_dict[f"w_{str(int(w))}_group_{group}_mean_lpips"] = metrics_dict[group]["mean_lpips"]
            metrics_full_dict[f"w_{str(int(w))}_group_{group}_mean_clip"] = metrics_dict[group]["mean_clip"]

            for subgroup, metrics in metrics_dict[group]["lpips"].items():
                metrics_full_dict[f"w_{str(int(w))}_subgroup_{subgroup}_lpips"] =  metrics_dict[group]["lpips"][subgroup]
                metrics_full_dict[f"w_{str(int(w))}_subgroup_{subgroup}_clip"] =  metrics_dict[group]["clip"][subgroup]
        
    return metrics_full_dict


def compute_editing_metrics(opts, dataset_name="cifar10"):
    dataset = dnnlib.util.construct_class_by_name(**opts.test_dataset_kwargs)
    assert dataset.has_labels

    label_to_class = dataset_labels[dataset_name]

    device = opts.device
    num_items = len(dataset)
    num_classes = len(label_to_class)
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=128, **data_loader_kwargs)

    lpips_model = lpips.LPIPS(net='alex').to(device)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    def get_text_embeddings(texts):
        inputs = tokenizer(texts, return_tensors = "pt").to(device)
        text_embeddings = model.get_text_features(**inputs)
        return text_embeddings / text_embeddings.norm(dim=-1,keepdim=True)
    
    def get_image_embeddings(images):
        image_inputs = processor(text=None, images=images, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)
        image_embedding = model.get_image_features(image_inputs)
        return image_embedding / image_embedding.norm(dim=-1,keepdim=True)

    def get_similarity_score(text_emb, image_emb):
        return text_emb @ image_emb.T

    metrics_dict = {metric: np.zeros((num_classes, num_classes)) for metric in ["clip", "lpips"]}
    metrics_full_dict = {f'{metric}_full': 0 for metric in ["clip", "lpips"]}
    
    count = np.zeros((num_classes, num_classes))

    one_hot_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int).to(device)
    for i in range(num_classes):
        one_hot_matrix[i,i] = 1

    with torch.no_grad():
        for images, labels in dataloader:
            batch_size = images.shape[0]

            images = images.to(device).to(torch.float32) / 127.5 - 1
            input_labels = labels.to(device)
            input_labels_idx = torch.argmax(input_labels, dim=1)

            for i in range(num_classes):
                target_labels = one_hot_matrix[i].repeat(batch_size, 1).to(device)
                edit_images, _ = opts.editing_fn(opts.G, images, src_labels=input_labels, target_labels=target_labels, **opts.G_kwargs)

                lpips_value = lpips_model(images, edit_images).detach().flatten().cpu().numpy()

                target_prompt = template.format(label_to_class[torch.argmax(one_hot_matrix[i]).item()])
                text_emb = get_text_embeddings(target_prompt)
                norm_edit_images = (edit_images + 1.0) / 2.0 # [-1,1] -> [0,1]
                image_emb = get_image_embeddings(norm_edit_images)
                clip_score = get_similarity_score(text_emb, image_emb).detach().flatten().cpu().numpy()

                for j in range(batch_size):
                    metrics_dict["lpips"][input_labels_idx[j],i] += lpips_value[j]
                    metrics_dict["clip"][input_labels_idx[j],i] += clip_score[j]
                    count[input_labels_idx[j],i] += 1
        
    for metric in ["clip", "lpips"]:
        metrics_dict[metric] /= count

        total_sum = np.sum(metrics_dict[metric]) - np.sum(np.diag(metrics_dict[metric]))
        num_non_diagonal = num_classes * num_classes - num_classes
        metrics_full_dict[f"{metric}_full"] = total_sum / num_non_diagonal
    
    for metric in ["clip", "lpips"]:
        for i in range(num_classes):
            for j in range(num_classes):
                metrics_full_dict[f"{metric}_src{i}_tar{j}"] = metrics_dict[metric][i, j]
    
    return metrics_full_dict