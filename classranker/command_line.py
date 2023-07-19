import os
import argparse
import sys
import types

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
    model_list = {
        "v1.0": [
            "ftp://ftp.mrc-lmb.cam.ac.uk/pub/dari/classranker_v1.0.ckpt.gz",
            "68a9855c16d7bab64b7e73e1e1442c7bf898f227ffd9a19c48ddfd2cf0646d73"
        ]
    }

    dest_dir = os.path.join(torch.hub.get_dir(), "checkpoints", "relion_class_ranker")
    model_path = os.path.join(dest_dir, f"{name}.ckpt")
    model_path_gz = model_path + ".gz"
    complete_check_path = os.path.join(dest_dir, f"{name}_download_complete.txt")

    if os.path.isfile(complete_check_path):
        return model_path

    print(f"Installing Classranker model ({name})...")

    os.makedirs(dest_dir, exist_ok=True)

    import gzip, shutil
    torch.hub.download_url_to_file(model_list[name][0], model_path_gz, hash_prefix=model_list[name][1])
    with gzip.open(model_path_gz, 'rb') as f_in:
        with open(model_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(model_path_gz)

    with open(complete_check_path, "w") as f:
        f.write("Successfully downloaded model")

    print(f"Model ({name}) successfully installed in {dest_dir}.")

    return model_path


def init_model(
        state_dict_path: str,
        device: str = "cpu"
) -> torch.nn.Module:
    # Load checkpoint file
    checkpoint = torch.load(state_dict_path, map_location="cpu")

    # Dynamically include model as a module
    # Make sure download file integrity is checked for this, otherwise major security risk
    model_module = types.ModuleType("classranker_model")
    exec(checkpoint['model_definition'], model_module.__dict__)
    sys.modules["classranker_model"] = model_module

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
