from medmnist import INFO
import medmnist
from os import makedirs, path, getcwd, chdir

import argparse
import subprocess

from medmnistc.corruptions.registry import CORRUPTIONS_DS, DATASET_RGB
from PIL import Image

import numpy as np
from tqdm import trange
from torchvision.transforms import ToPILImage

import os

class AugMedMNIST(object):
    def __init__(self, 
                 train_corruptions : dict,
                 RGB: bool):
        
        assert len(train_corruptions) > 0, f"You need to define some corruptions firsts."
        
        self.train_corruptions = train_corruptions
        self.train_corruptions_keys = list(self.train_corruptions.keys())

        if RGB:
            self.mode = "RGB"
        else:
            self.mode = "L"
    
    def allow_identity(self, allow_identity: bool):
        if allow_identity:
            if "identity" not in self.train_corruptions_keys: self.train_corruptions_keys.append('identity')
        else:
            if "identity" in self.train_corruptions_keys: self.train_corruptions_keys.remove('identity')

    def __call__(self, img):
        corr = np.random.choice(self.train_corruptions_keys)

        img = img.convert('RGB')

        if corr == 'identity': 
            return img, corr
        
        return Image.fromarray(self.train_corruptions[corr].apply(img, augmentation=True)).convert(mode=self.mode), corr

def dataset_preparator(data_flag: str, size: int, prepare: bool = True, corrupt: bool = False, corruption: str = None,
                  split_to_use:str = None):
    
    info = INFO[data_flag]

    print(info["python_class"], "with classes: ", info["label"], "\n")

    if not prepare:
        return len(info["label"])

    np.random.seed(42)
    if corrupt:
        
        if corruption is not None:
            my_c = {corruption: CORRUPTIONS_DS[data_flag][corruption]}
            data_name = data_flag + "_corrupted_" + corruption
        else:
            my_c = CORRUPTIONS_DS[data_flag]
            data_name = data_flag + "_corrupted"
        
        trans = AugMedMNIST(train_corruptions=my_c, RGB=DATASET_RGB[data_flag])
        
    else:
        trans = None
        data_name = data_flag
    
    data_name = data_name + "_" + str(size)
    
    data_class = getattr(medmnist, info['python_class'])

    categories = [x for x in range(len(info["label"]))]
    for split in ["train", "val", "test"]:
        
        if split_to_use is not None and split != split_to_use:
            continue
        
        if corrupt:
            if split == "test":
                trans.allow_identity(False)
            else:
                trans.allow_identity(True)
        
        dataset = data_class(split=split, download=True, size=size, transform = trans)

        for category in categories:
            try:
                makedirs(f"data/{data_name}/{split}/{category}", exist_ok=False)
            except:
                print(f"\nDataset {data_name} already prepared. If you want to re-prepare it, please remove the folder "
                      f"data/{data_name}")

                np.random.seed(None)
                return len(info["label"])

        for idx in trange(len(dataset), desc=f"Saving {split} dataset", mininterval=1):
            
            if corrupt:
                (img, t), label = dataset[idx]
                t = "__" + t
            else:
                (img, label) = dataset[idx]
                t=""
    
            img.save(f"data/{data_name}/{split}/{label[0]}/{split}{idx}_{label[0]}{t}.png")

    # write categories descriptions
    with open(f"data/{data_name}/categories.txt", "w") as f:
        f.write(str(info["label"]))

    np.random.seed(None)
    return len(categories)


def download_MedViT():
    
    if not os.path.exists('MedViT/pretrained/medvit_large.pth'):
        print("Downloading MedViT model...")
    
        model_urls = {
            "medvit_large": "https://dl.dropboxusercontent.com/scl/fi/ardpdupcmfwf58yaw91hp/MedViT_large_im1k.pth?rlkey=gftk4ngacr3k98nht3wyefhuz&st=w36m4hsm&dl=0",
        }

        makedirs("pretrained", exist_ok=True)

        import urllib.request

        urllib.request.urlretrieve(model_urls["medvit_large"], f"pretrained/medvit_large.pth")
    else:
        print("MedViT model already downloaded.")


def fine_tune_model(model:str, data_name: str, n_classes: int, batch_size: int, gpus:int, epochs:int, corruption:bool):
    
    if corruption:
        data_name = data_name + "_corrupted_224"
    else:
        data_name = data_name + "_224"

    folder_data = "data"
    folder_output = f"{model}/output"
    
    if model == "MedViT":
        download_MedViT()
        model_name = "MedViT_large"
        resume = "MedViT/pretrained/medvit_large.pth"
    else:
        model_name = model
    
    # Make sure the script is executable
    subprocess.call(["chmod", "+x", './../models/finetuning/train.sh'])
    
    commands = ['./../models/finetuning/train.sh', f'{gpus}',
                '--batch-size', f'{batch_size}',
                '--epochs', f'{epochs}',
                '--model', f'{model_name}',
                '--input-size', '224',
                '--data-set', 'image_folder',
                '--finetune',
                '--data-path', f'{folder_data}/{data_name}/train/',
                '--output-dir', f'{folder_output}/{data_name}/',
                '--pin-mem', '--num_workers', f'{gpus*2}',
                '--eval_data_path', f'{folder_data}/{data_name}/val',
                '--nb_classes', f'{n_classes}', ]
    
    if model == "MedViT":
        commands.extend(['--resume', resume])
    
    subprocess.call(commands)

def main(args):
    
    new_cwd =path.join(getcwd(), f"models_training")
    print("working in: ",new_cwd)
    os.makedirs(new_cwd, exist_ok=True)
    chdir(new_cwd)
    
    n_classes = dataset_preparator(args.data_name, 224, args.prepare_data, args.corrupt, args.corruption_type, args.split_to_use)

    if args.fine_tune:
        fine_tune_model(args.model, args.data_name, n_classes, args.batch_size, args.gpus, args.epochs, args.corrupt)
    
if __name__ == "__main__":
    
    datasets = ['bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist', 
                'octmnist', 'organamnist', 'organcmnist', 'organsmnist', 
                'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist']
    
    models = ['MedViT', 'vgg16', 'densenet121','convnext_base','deit_base_patch16_224','swin_base_patch4_window7_224','swin_tiny_patch4_window7_224','coat_tiny','efficientvit_b0','gcvit_xxtiny','vit_base_patch16_224.mae','vit_base_patch16_224']

    desc = """
    This script has two main functionalities:
        1) Prepare model:
            - download dataset and prepare it (corrupted or not)
            - download pretrained model
        2) Fine tune model:
            - fine tune the model with the selected dataset
    """

    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)

    #General arguments
    parser.add_argument("--data_name", type=str, required=True, help="Dataset to be prepared/used for fine tuning", choices=datasets)
    parser.add_argument("--model", type=str, help="Model to be prepared/fine tuned.", choices=models)
    
    #Dataset arguments
    parser.add_argument("--corrupt", action=argparse.BooleanOptionalAction, help="Prepare a corrupted dataset. Default: no", default=False)
    parser.add_argument("--split_to_use", type=str, help="Split to use for the dataset preparation. Default: prepare all splits", choices=["train", "val", "test"], default=None)
    parser.add_argument("--corruption_type", type=str, help="Corruption type to use for the dataset preparation. Default: choose randomly", 
                        default=None)
    parser.add_argument("--prepare_data", action=argparse.BooleanOptionalAction, default=False,
                        help="prepare the dataset. Default: no")
    
    #Fine tuning arguments
    parser.add_argument("--batch_size", help="Batch size for fine tuning. Default 8.", type=int, default=8)
    parser.add_argument("--epochs", help="Epochs for fine tuning. Default: 100", type=int, default=100)
    parser.add_argument("--gpus", help="Number of  GPU to use. Default: 1", type=int, default=1)
    parser.add_argument("--fine_tune", action=argparse.BooleanOptionalAction,
                        help="Fine tune the model with the selected data")
    
    #Other arguments
    parser.add_argument("--silent", action=argparse.BooleanOptionalAction, help="Silent mode", default=False)
    args = parser.parse_args()

    if not args.silent:
        message = f"""
        
        Recalling the selected options:
        
        - Model: {args.model} -> The model to be fine tuned
        - Dataset: {args.data_name} -> The dataset to be used
        
        - Fine tune: {args.fine_tune} -> If True, the model will be fine tuned
        - Prepare data: {args.prepare_data} -> If True, the dataset will be prepared
        
        - Dataset options:
            - Corrupt: {args.corrupt} -> If True, the dataset will be corrupted
            - Split to use: {args.split_to_use} -> The split to use for the dataset preparation. Default: All
            - Corruption type: {args.corruption_type} -> The corruption type to use for the dataset preparation. Default: choose randomly
        
        - Fine tuning options:
            - Batch size: {args.batch_size}
            - Epochs: {args.epochs} 
            - GPUs: {args.gpus}
        
        Are you sure you want to continue? [y/n]
        """
        
        #response = input(message)
        
        #while response.lower() not in ['y', 'n']:
            #response = input("Please enter 'y' or 'n': ")
        
        #if response.lower() == 'n':
            #print("Exiting...")
            #exit(0)
    
    if args.corruption_type is not None and (args.corruption_type not in CORRUPTIONS_DS[args.data_name].keys() and args.corruption_type!="all"):
        print(f"Corruption type {args.corruption_type} not available for the dataset {args.data_name}.")
        exit(1)
    
    if args.corruption_type is not None and args.corruption_type=="all":
        for c_t in CORRUPTIONS_DS[args.data_name].keys():
            args.corruption_type = c_t
            main(args)
    else:
        main(args)
