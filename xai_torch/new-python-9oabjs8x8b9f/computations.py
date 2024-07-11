import pickle
import torch
import pandas as pd
from tqdm import tqdm
import zipfile

def compute(test_dataset_file, model_processor_file, model_architecture_type, target_layer, saving_dir):
        """
        lib:https://github.com/jacobgil/pytorch-grad-cam/tree/master?tab=readme-ov-file
        targets are set to None: If targets is None, the highest scoring category
        Need to wrap up the huggingface model to torch model.

        model_architecture_type will be responsible for choosing the reshaping of the output tensors.
        https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html

        target layers: list of target layers 

        -----------------------------------------------Theory:-----------------------------------------------------------

        xai algorithms requires input of the layers which has output (Batch_size, Channels, w, h)

        Batch_size: number of images.
        Channels  : Features (for cnn number of kernels).
        w, h      : size of the features (for cnn size of the kernels.)

        for the transformer based architectures we need to construct the reshape_transform function.

        what is reshape_transform function ?
        To apply the xai on transformer architecture we need layers which has output (batchsize, patchsize*patchsize+1 or patchsize*patchsize, features.)
        because it does not produce 2D feature maps as  cnn.
        Once we get the output in this shape we need to transform it to match it (batch_size, features, patchsize, patchsize)

        How to build reshape_transform function ?
        For ViT : if the output shape is (1, 197, 192): 197 represents the number of tokens and the first one is for [cls] .
                we ignore the first token so (1, 196, 192) -> (1, 14, 14, 192) ->( 1, 192, 14, 14 )[just like cnn]

        For SwiT : there is no [cls] token in SwiT, so (1, 196, 192) -> (1, 14, 14, 192) ->( 1, 192, 14, 14 )[just like cnn]

        For the Users:
        select the suitable layer, that match the output shape as discribe above.

        How to check which layers to select?

        If you are using pre-train models and not sure about the summary.
        Use 

        def print_model_summary(model, input_size, batch_size=-1, device="cuda"):
            from torchinfo import torchinfo
            summary_info = torchinfo.summary(model, input_size=(batch_size, *input_size), device=device, verbose=2)
            print(summary_info)

        or try to print the model.

        Some of the common choice for the target layers.

        #my_model.resnet.encoder.stages[-1].layers[-1]
        #my_model.vit.encoder.layer[-4].output
        #my_model.swinv2.layernorm
        """

        #, "ScoreCAM", "AblationCAM" this 2 are slow so, just opt out for now.
        import numpy as np
        from PIL import Image
        import os
        from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # Ensure the saving directory exists

        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)

        # Unzip the test dataset file
        with zipfile.ZipFile(test_dataset_file, 'r') as zip_ref:
            zip_ref.extractall(saving_dir)

        # Find the Excel file in the extracted folder
        # Function to find the Excel file recursively
        def find_excel_file(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.xlsx'):
                        return os.path.join(root, file)
            return None
        
        # Find the Excel file in the extracted folder
        excel_file = find_excel_file(saving_dir)
        
        if excel_file is None:
            raise FileNotFoundError("Excel file not found in the unzipped directory.")
        
        # Load the dataset from the Excel file
        dataframe = pd.read_excel(excel_file)
        
        # Load the model and processor from the pickle file
        with open(model_processor_file, 'rb') as f:
            loaded_dict = pickle.load(f)
        
        model = loaded_dict['model']
        processor = loaded_dict['processor']
        model.eval()
        

        def reshape_transform_vit_huggingface(x):
            activations = x[:, 1:, :]
            x = np.sqrt(activations.shape[1])
            activations = activations.view(activations.shape[0], int(x), int(x), activations.shape[2])
            activations = activations.transpose(2, 3).transpose(1, 2)
            return activations

        def reshape_transform_SwiT_huggingface(x):
            activations = x
            x = np.sqrt(activations.shape[1])
            activations = activations.view(activations.shape[0], int(x), int(x), activations.shape[2])
            activations = activations.transpose(2, 3).transpose(1, 2)
            return activations

        if model_architecture_type == "cnn":
            transform = None
            xai_algo = ["GradCAM", "HiResCAM", "GradCAMPlusPlus", "XGradCAM", "EigenCAM"]
            [os.makedirs(f"{saving_dir}/xai/{algo}", exist_ok=True) for algo in xai_algo]

        if model_architecture_type == "ViT":
            transform = reshape_transform_vit_huggingface
            xai_algo = ["GradCAM", "HiResCAM", "GradCAMPlusPlus", "XGradCAM", "EigenCAM"]
            [os.makedirs(f"{saving_dir}/xai/{algo}", exist_ok=True) for algo in xai_algo]

        if model_architecture_type == "SwiT":
            transform = reshape_transform_SwiT_huggingface
            xai_algo = ["GradCAM", "HiResCAM", "GradCAMPlusPlus", "XGradCAM", "EigenCAM"]
            [os.makedirs(f"{saving_dir}/xai/{algo}", exist_ok=True) for algo in xai_algo]

        
        target_layers = [eval(target_layer)]
        GradCAM_, HiResCAM_, GradCAMPlusPlus_, XGradCAM_, EigenCAM_ = [], [], [], [], []
        for algo in xai_algo:
            for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
                file_path = os.path.join(os.path.dirname(excel_file), row['filename'])
                if not os.path.exists(file_path):
                    print(f"Image file {file_path} not found.")
                    continue
                image = Image.open(file_path).convert("RGB")
                img_name = row['filename']
                input_tensor = processor(image)
                input_tensor = input_tensor.unsqueeze(0)
                cam = eval(algo)(model=model, target_layers=target_layers, reshape_transform=transform)
                targets = None

                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]

                w, h = grayscale_cam.shape[0], grayscale_cam.shape[1]
                image = image.resize((w,h))
                visualization = show_cam_on_image(np.array(image)/255, grayscale_cam, use_rgb=True)

                img = Image.fromarray(visualization)
                img.save(f"{saving_dir}/xai/{algo}/{img_name}")

                eval(algo + "_").append(f"{saving_dir}/xai/{algo}/{img_name}")
        return {"GradCAM":GradCAM_, "HiResCAM":HiResCAM_, "GradCAMPlusPlus":GradCAMPlusPlus_, "XGradCAM":XGradCAM_, "EigenCAM":EigenCAM_}


def test():
    """Test the compute function."""

    print("Running test")
