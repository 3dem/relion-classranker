import importlib
import os
import argparse
import sys

try:
    import torch
except ImportError:
    print("PYTHON ERROR: The required python module 'torch' was not found.")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("PYTHON ERROR: The required python module 'numpy' was not found.")
    exit(1)

if sys.version_info < (3, 0):
    # This script requires Python 3. A Syntax error here means you are running it in Python 2.
    raise Exception('This script supports Python 3 or above.')


def setup_model(name: str) -> str:
    bundle_name_to_link = {
        "v1.0": "https://zenodo.org/record/7733060/files/original.zip",
    }
    dest = os.path.join(
        torch.hub.get_dir(), "checkpoints", "relion_class_ranker", name
    )
    if os.path.isfile(os.path.join(dest, "download_complete.txt")):
        return dest

    print(f"Installing model bundle ({name})...")
    import zipfile

    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    torch.hub.download_url_to_file(bundle_name_to_link[name], dest + ".zip")

    with zipfile.ZipFile(dest + ".zip", "r") as zip_object:
        zip_object.extractall(path=os.path.split(dest)[0])

    os.remove(dest + ".zip")

    with open(os.path.join(dest, "download_complete.txt"), "w") as f:
        f.write("Successfully downloaded model bundle")

    print(f"Model bundle ({name}) successfully installed.")

    state_dict_path = os.path.join(dest, "checkpoint.pt")

    return state_dict_path


def init_model(
        state_dict_path: str,
        device: str = "cpu"
) -> torch.nn.Module:
    # Load checkpoint file
    checkpoint = torch.load(state_dict_path, map_location="cpu")

    # Create a loader for the module using the string contents
    model_module_loader = importlib.machinery.SourceFileLoader("module_name", "<string>")

    # Set the file contents as the data for the loader
    model_module_loader.set_data("<string>", checkpoint['model_definition'].encode("utf-8"))

    # Load the module
    model_module = model_module_loader.load_module()

    # Load the model
    model = model_module.Model().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_name', type=str, default="v1.0")
    parser.add_argument('project_dir', type=str)
    args = parser.parse_args()

    project_dir = args.project_dir

    model_state_dict_path = setup_model(args.model_name)
    model = init_model(model_state_dict_path)

    feature_fn = os.path.join(project_dir, "features.npy")
    images_fn = os.path.join(project_dir, "images.npy")

    features = np.load(feature_fn)
    images = np.load(images_fn)

    count = features.shape[0]

    features_tensor = torch.Tensor(features)
    images_tensor = torch.unsqueeze(torch.Tensor(images), 1)
    score = model(images_tensor, features_tensor).detach().cpu().numpy()

    for i in range(count):
        print(score[i, 0], end=" ")


if __name__ == "__main__":
    main()
