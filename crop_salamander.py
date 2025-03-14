import cropped_predictions as cp

from tqdm import tqdm
from pathlib import Path

# TODO read from a file?
skeleton = [
    ("nose", "neck"),
    ("lhead", "neck"),
    ("rhead", "neck"),
    ("neck", "spine1"),
    ("spine2", "spine1"),
    ("spine2", "spine3"),
    ("tailbase", "spine3"),
    ("tailbase", "tail1"),
    ("tail2", "tail1"),
    ("tail2", "tail3"),
    ("tailtip", "tail3"),
    ("spine1", "larm"),
    ("spine1", "rarm"),
    ("lelbow", "larm"),
    ("relbow", "rarm"),
    ("tailbase", "rknee"),
    ("tailbase", "lknee"),
    ("rfoot", "rknee"),
    ("lfoot", "lknee"),
]

data_dir = Path("/home/ks3582/salamander")
outputs_dir = Path("/home/ks3582/synced/outputs/salamander/cropzoom")

ind = cp.Dataset()
ind.data_dir = Path(data_dir)
ind.labels_file = data_dir / "CollectedData.csv"
ind.single_preds_file = (
    outputs_dir
    / "detector_0/image_preds/CollectedData.csv/predictions.csv"
)
ind.pose_preds_file = (
    outputs_dir
    / "pose_supervised_0/image_preds/cropped_CollectedData.csv/remapped_predictions.csv"
)
ind.bbox_file = outputs_dir / "detector_0/image_preds/CollectedData.csv/bbox.csv"
ind.read_files()
ind.skeleton = skeleton

ood = cp.Dataset()
ood.data_dir = Path(data_dir)
ood.labels_file = data_dir / "CollectedData_new.csv"
ood.single_preds_file = (
    outputs_dir
    / "detector_0/image_preds/CollectedData_new.csv/predictions.csv"
)
ood.pose_preds_file = (
    outputs_dir
    / "pose_supervised_0/image_preds/cropped_CollectedData_new.csv/remapped_predictions.csv"
)
ood.bbox_file = (
    outputs_dir / "detector_0/image_preds/CollectedData_new.csv/bbox.csv"
)
ood.read_files()
ood.skeleton = skeleton

output_dir = Path("/home/ks3582/pose_vs_ctx_gallery_salamander/")

# Previews
for img_path in tqdm(ind.single_preds_df.index):
    red, green = ind.generate_annotated_image(img_path)

    img_path = Path(img_path)

    red_path = output_dir / "ind" / img_path.with_stem(img_path.stem + "_red")
    red_path.parent.mkdir(parents=True, exist_ok=True)
    red.save(red_path)

    green_path = output_dir / "ind" / img_path.with_stem(img_path.stem + "_green")
    green_path.parent.mkdir(parents=True, exist_ok=True)
    green.save(green_path)


for img_path in tqdm(ood.single_preds_df.index):
    red, green = ood.generate_annotated_image(img_path)

    img_path = Path(img_path)

    red_path = output_dir / "ood" / img_path.with_stem(img_path.stem + "_red")
    red_path.parent.mkdir(parents=True, exist_ok=True)
    red.save(red_path)

    green_path = output_dir / "ood" / img_path.with_stem(img_path.stem + "_green")
    green_path.parent.mkdir(parents=True, exist_ok=True)
    green.save(green_path)


cp.create_image_gallery_html(output_dir)
