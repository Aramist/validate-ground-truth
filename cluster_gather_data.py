from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

VIDEO_SIZE = (640, 512)

dataset_names = [
    "sologerbil-4m-e1",
    "edison-4m-e1",
    "gerbilearbud-4m-e1",
    "speaker-4m-e1",
    "hexapod-8m-e2",
    "mouseearbud-24m-e3",
]

# Path to processed dataset on cluster
dataset_files = [
    Path("/mnt/home/atanelus/ceph/neurips_datasets/audio") / f"{name}_audio.h5"
    for name in dataset_names
]

# Path to metadata on cluster (provides link between dataset index and video frame #, path)
metadata_files = [
    Path("/mnt/home/atanelus/ceph/datasets/solo_gerbil_metadata.csv"),
    Path("/mnt/home/atanelus/ceph/datasets/edison_metadata.csv"),
    Path("/mnt/home/atanelus/ceph/datasets/gerbilearbud_metadata.csv"),
    Path("/mnt/home/atanelus/ceph/datasets/speaker_metadata.csv"),
    None,
    Path(
        "/mnt/home/atanelus/ceph/princeton_data/partial_datasets/2024_03_08_16_28_20.csv"
    ),
]

corner_points_file = [
    Path("/mnt/home/atanelus/dataset_corner_points/antiphonal_corner_points.npy"),
    Path("/mnt/home/atanelus/dataset_corner_points/robot_corner_points.npy"),
    Path("/mnt/home/atanelus/dataset_corner_points/earbud_corner_points.npy"),
    Path("/mnt/home/atanelus/dataset_corner_points/speaker_corner_points.npy"),
    None,
    None,
]


arena_dims = [
    np.array([572, 360]),
    np.array([572, 360]),
    np.array([572, 360]),
    np.array([558.9, 355.6]),
    None,
    np.array([615, 615]),
]

to_list = lambda x: "[" + " ".join(map(str, x)) + "]"


def make_affine(corner_points: np.ndarray, arena_dims: np.ndarray) -> np.ndarray:
    """Generates the homography matrix to convert points in the arena coordinate frame
    to the video pixel coordinate frame.
    """

    dest = corner_points
    aw, ah = arena_dims / 2
    source = np.array(
        [
            [-aw, ah],
            [aw, ah],
            [aw, -ah],
            [-aw, -ah],
        ]
    )

    H, _ = cv2.findHomography(source, dest, method=0)  # OLS
    return H


def convert_points(points, H):
    # Given a stream and a point (in pixels), converts to inches within the global coordinate frame
    # Pixel coordinates should be presented in (x, y) order, with the origin in the top-left corner of the frame
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # System is M * [x_px y_px 1] ~ [x_r y_r 1]
    ones = np.ones((*points.shape[:-1], 1))
    points = np.concatenate([points, ones], axis=-1)
    prod = np.einsum("ij,...j->...i", H, points)[..., :-1]  # remove ones row
    return prod


def gather_images_and_gtruth(
    dataset_idx: int, output_dir: Path, num_samples: int = 50
) -> None:
    dset_name = dataset_names[dataset_idx]
    output_path = output_dir / dset_name
    if output_path.exists():
        print(f"Dataset {dset_name} already exists in {output_dir}. Skipping.")
        return
    output_path.mkdir(parents=True, exist_ok=False)
    (output_path / "images").mkdir()

    metadata_path = metadata_files[dataset_idx]
    corner_points_path = corner_points_file[dataset_idx]
    dataset_path = dataset_files[dataset_idx]

    if metadata_path is None:  # or corner_points_path is None:
        print(f"Dataset {dset_name} does not have metadata or corner points. Skipping.")
        return

    # columns: _, video_path, frame_idx
    metadata = pd.read_csv(metadata_path)
    corner_points = np.load(corner_points_path) if corner_points_path else None
    adims = arena_dims[dataset_idx]
    H = make_affine(corner_points, adims) if corner_points is not None else None

    new_df = pd.DataFrame(
        columns=[
            "idx_in_dataset",
            "ground_truth_x",
            "ground_truth_y",
            "manual_x",
            "manual_y",
        ]
    )

    idx_in_dataset = []
    ground_truth_x = []
    ground_truth_y = []
    # Sample random frames from metadata
    rand_rows = metadata.sample(n=num_samples, replace=False)
    for idx, row in rand_rows.iterrows():
        video_path = row["video_path"]
        frame_idx = row["frame_idx"]
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = video.read()
        if not ret:
            print(f"Failed to read frame {frame_idx} from {video_path}. Skipping.")
            continue

        with h5py.File(dataset_path, "r") as f:
            ground_truth = f["locations"][idx]

        # Convert points to global coordinate frame
        points = (
            convert_points(ground_truth, H).squeeze() if H is not None else ground_truth
        )
        # Save image and ground truth
        image_path = output_path / "images" / f"{idx}.png"
        cv2.imwrite(str(image_path), frame)

        idx_in_dataset.append(idx)

        if len(points.shape) == 1:
            points = points[None, :]
        ground_truth_x.append(points[:, 0])
        ground_truth_y.append(points[:, 1])

    new_df = pd.DataFrame(
        {
            "idx_in_dataset": idx_in_dataset,
            "ground_truth_x": list(map(lambda a: to_list(a), ground_truth_x)),
            "ground_truth_y": list(map(lambda a: to_list(a), ground_truth_y)),
            "manual_x": [list() for _ in range(len(idx_in_dataset))],
            "manual_y": [list() for _ in range(len(idx_in_dataset))],
        }
    )
    new_df.to_csv(output_path / "annotations.csv", index=False)


if __name__ == "__main__":
    output_dir = Path(".")

    for i in range(len(dataset_names)):
        gather_images_and_gtruth(i, output_dir)
