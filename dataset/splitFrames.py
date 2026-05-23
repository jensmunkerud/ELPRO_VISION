import os
import shutil


SOURCE_DIR = "newDataset/captured_frames"
OUTPUT_DIR = "newDataset/split_frames"
N_FOLDERS = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_frames(source_dir, output_dir, n_folders):
	files = [
		name for name in sorted(os.listdir(source_dir))
		if os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS
	]

	if not files:
		print(f"No image files found in {source_dir}")
		return

	for index, name in enumerate(files):
		folder_number = (index % n_folders) + 1
		target_dir = os.path.join(output_dir, f"folder_{folder_number}")
		os.makedirs(target_dir, exist_ok=True)
		shutil.copy2(os.path.join(source_dir, name), os.path.join(target_dir, name))

	print(f"Split {len(files)} frames into {n_folders} folders under {output_dir}")


if __name__ == "__main__":
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	split_frames(SOURCE_DIR, OUTPUT_DIR, N_FOLDERS)