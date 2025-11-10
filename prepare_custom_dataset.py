import os, shutil, random
from tqdm import tqdm

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset(real_path, fake_path, output_path, train_ratio=0.75, val_ratio=0.15):
    make_dir(output_path)
    for folder in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            make_dir(os.path.join(output_path, folder, cls))

    def copy_split(src_dir, cls):
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]
        }
        for split, subset in splits.items():
            for f in tqdm(subset, desc=f"{cls} → {split}"):
                src = os.path.join(src_dir, f)
                dst = os.path.join(output_path, split, cls, f)
                shutil.copy2(src, dst)

    copy_split(real_path, "real")
    copy_split(fake_path, "fake")
    print(f"\nSplit complete → saved under {output_path}")

if __name__ == "__main__":
    split_dataset("./custom_dataset/real", "./custom_dataset/fake", "./split_dataset")
