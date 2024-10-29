from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

HEADER_HEIGHT = 300

datasets = [
    # "dyadgerbil-4m-e1",
    # 'dyadmouse-24m-e3',
    "sologerbil-4m-e1",
    "edison-4m-e1",
    # 'solomouse-24m-e3',
    "gerbilearbud-4m-e1",
    "speaker-4m-e1",
    "hexapod-8m-e2",
    "mouseearbud-24m-e3",
]


def string_to_numpy(string: str) -> np.ndarray:
    if isinstance(string, float) and np.isnan(string):
        return np.array([])
    if isinstance(string, np.ndarray):
        return string
    string = string.replace("[", "").replace("]", "").strip()
    return np.fromstring(string, sep=" ").astype(int)


def to_list_str(arr: np.ndarray) -> list:
    if len(arr) == 0:
        return "[]"
    return "[" + " ".join(map(str, arr)) + "]"


def verify_data_presence(dataset_idx: int) -> None:
    dset_data_path = Path(datasets[dataset_idx])
    image_dir = dset_data_path / "images"
    save_file = dset_data_path / "annotations.csv"

    if list(image_dir.glob("*.png")) and save_file.exists():
        return
    raise FileNotFoundError(
        f"Dataset '{datasets[dataset_idx]}' is missing data. Please find it."
    )


def generate_gui_header(progress: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Generates a header image for the GUI. Resolution is 1920 x HEADER_HEIGHT.
    Includes user instructions and a legend for the annotations.
    """
    im_width, im_height = 1920, HEADER_HEIGHT
    im = np.full((im_height, im_width, 3), 255, dtype=np.uint8)  # white background
    # Add instructions to top left
    instructions = [
        "Controls:",
        "- Left click: mark ground truth location",
        "- Right click: remove mark",
        "- H: previous image",
        "- L: next image",
        "- Q: quit (annotations are automatically saved)",
    ]
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    (_, text_height), _ = cv2.getTextSize(
        instructions[0], font_face, font_scale, thickness
    )

    for i, instruction in enumerate(instructions):
        padding = 10

        cv2.putText(
            im,
            instruction,
            (padding, padding + text_height + i * (text_height + padding)),
            font_face,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    # Legend on top right
    legend = ["Legend:", "Original GT", "Hand annotated GT"]
    # find width of longest line
    (legend_width, _), _ = cv2.getTextSize(legend[2], font_face, font_scale, thickness)
    cv2.putText(
        im,
        legend[0],
        (im_width - legend_width - text_height - padding, padding + text_height),
        font_face,
        font_scale,
        (0, 0, 0),
        thickness,
    )
    # cv2.putText(
    #     im,
    #     legend[1],
    #     (
    #         im_width - legend_width - text_height - padding,
    #         2 * padding + 2 * text_height,
    #     ),
    #     font_face,
    #     font_scale,
    #     (0, 0, 255),  # red
    #     thickness,
    # )
    # cv2.circle(
    #     im,
    #     (
    #         im_width - padding - text_height // 2,
    #         2 * padding + text_height + text_height // 2,
    #     ),
    #     text_height // 2 - 2,
    #     (0, 0, 255),
    #     -1,
    # )
    cv2.putText(
        im,
        legend[2],
        (
            im_width - legend_width - text_height - padding,
            3 * padding + 3 * text_height,
        ),
        font_face,
        font_scale,
        (0, 180, 0),  # green
        thickness,
    )
    cv2.circle(
        im,
        (
            im_width - padding - text_height // 2,
            3 * padding + 2 * text_height + text_height // 2,
        ),
        text_height // 2 - 2,
        (0, 180, 0),
        -1,
    )

    if progress is not None:
        # Add current frame at bottom center
        progress_text = f"{progress[0]+1} / {progress[1]}"
        (progress_width, _), _ = cv2.getTextSize(
            progress_text, font_face, font_scale, thickness
        )
        textpos_x = im_width // 2 - progress_width // 2
        textpos_y = im_height - padding
        cv2.putText(
            im,
            progress_text,
            (textpos_x, textpos_y),
            font_face,
            font_scale,
            (0, 0, 0),
            thickness,
        )

    return im


def pick_dataset() -> int:
    print("Choose a dataset:")
    for i, dataset in enumerate(datasets):
        print(f"({i + 1}) {dataset}")

    while True:
        try:
            choice = int(input("Enter the number of the dataset: "))
            if choice < 1 or choice > len(datasets):
                raise ValueError
            break
        except ValueError:
            print(
                f"Invalid choice. Please enter a number between 1 and {len(datasets)}."
            )

    return choice - 1


class GUI:
    def __init__(self, dataset_idx: int):
        self.dataset_idx = dataset_idx
        self.cur_data_idx = 0
        self.win_name = f"Dataset: {datasets[self.dataset_idx]}"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        self.should_save = False
        self.displayed_frame = None
        self.load_data()
        self.load_frame()
        # self.control_loop()

    def load_data(self):
        data_path = Path(datasets[self.dataset_idx]) / "annotations.csv"
        df = pd.read_csv(
            data_path,
            dtype={
                "manual_x": str,
                "manual_y": str,
                "ground_truth_x": str,
                "ground_truth_y": str,
            },
        )
        # convert strings in manual_x, manual_y, ground_truth_x, ground_truth_y to numpy arrays
        df["manual_x"] = df["manual_x"].apply(string_to_numpy)
        df["manual_y"] = df["manual_y"].apply(string_to_numpy)
        df["ground_truth_x"] = df["ground_truth_x"].apply(string_to_numpy)
        df["ground_truth_y"] = df["ground_truth_y"].apply(string_to_numpy)
        self.df = df.astype(
            {
                "manual_x": "object",
                "manual_y": "object",
                "ground_truth_x": "object",
                "ground_truth_y": "object",
            }
        )

    def save_data(self):
        if self.should_save:
            data_path = Path(datasets[self.dataset_idx]) / "annotations.csv"
            # dont want to apply to_list to working df
            temp_copy = self.df.copy()
            temp_copy["manual_x"] = temp_copy["manual_x"].apply(to_list_str)
            temp_copy["manual_y"] = temp_copy["manual_y"].apply(to_list_str)
            temp_copy.to_csv(data_path, index=False)
            self.should_save = False

    def load_frame(self):
        idx_in_dataset = self.df.iloc[self.cur_data_idx]["idx_in_dataset"]
        frame_path = (
            Path(datasets[self.dataset_idx]) / "images" / f"{idx_in_dataset}.png"
        )
        frame_without_header = cv2.imread(str(frame_path))

        _, im_width = frame_without_header.shape[:2]
        header_height = HEADER_HEIGHT
        self.resized_header_height = int(header_height * (im_width / 1920))
        header = generate_gui_header((self.cur_data_idx, len(self.df)))
        self.header = cv2.resize(header, (im_width, self.resized_header_height))

        self.displayed_frame = np.concatenate(
            (self.header, frame_without_header), axis=0
        )

    def frame_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            row = self.df.iloc[self.cur_data_idx]
            manual_x = row["manual_x"]
            manual_y = row["manual_y"]

            manual_x = np.append(manual_x, x)
            # correct for header height
            manual_y = np.append(manual_y, y - self.resized_header_height)
            self.df.at[self.cur_data_idx, "manual_x"] = manual_x.tolist()
            self.df.at[self.cur_data_idx, "manual_y"] = manual_y.tolist()
            self.should_save = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            row = self.df.iloc[self.cur_data_idx]
            manual_x = row["manual_x"]
            manual_y = row["manual_y"]

            if len(manual_x) == 0:
                return

            pts = np.stack((manual_x, manual_y), axis=-1)

            mouse_pt = np.array([x, y - self.resized_header_height])

            closest_idx = np.argmin(np.linalg.norm(pts - mouse_pt[None, :], axis=1))
            manual_x = np.delete(manual_x, closest_idx)
            manual_y = np.delete(manual_y, closest_idx)

            self.df.at[self.cur_data_idx, "manual_x"] = manual_x.tolist()
            self.df.at[self.cur_data_idx, "manual_y"] = manual_y.tolist()
            self.should_save = True

    def markup_frame(self):
        frame_copy = self.displayed_frame.copy()
        row = self.df.iloc[self.cur_data_idx]
        manual_x = row["manual_x"]
        manual_y = row["manual_y"]
        for x, y in zip(manual_x, manual_y):
            cx, cy = int(x), int(y + self.resized_header_height)
            cv2.circle(frame_copy, (cx, cy), 5, (0, 180, 0), -1)

        ### Uncomment to display ground truth points
        # gt_x = row["ground_truth_x"]
        # gt_y = row["ground_truth_y"]
        # for x, y in zip(gt_x, gt_y):
        #     cx, cy = int(x), int(y + self.resized_header_height)
        #     cv2.circle(frame_copy, (cx, cy), 5, (0, 0, 255), -1)

        return frame_copy

    def control_loop(self):
        cv2.setMouseCallback(self.win_name, self.frame_callback)
        while True:
            frame = self.markup_frame()
            cv2.imshow(self.win_name, frame)
            key = cv2.waitKey(33)  # max out at 30 fps

            if key == ord("q"):
                self.save_data()
                break
            elif key == ord("h"):
                self.cur_data_idx = max(0, self.cur_data_idx - 1)
                self.load_frame()
            elif key == ord("l"):
                self.cur_data_idx = min(len(self.df) - 1, self.cur_data_idx + 1)
                self.load_frame()
            self.save_data()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset_idx = pick_dataset()
    verify_data_presence(dataset_idx)
    gui = GUI(dataset_idx)
    gui.control_loop()
