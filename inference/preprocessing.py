# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import dlib
import numpy as np
import torch
from omegaconf import DictConfig


class BboxTuple(NamedTuple):
    """
    A type representing a bounding box.
    """

    x1: int
    y1: int
    x2: int
    y2: int


Tube = List[Tuple[int, BboxTuple]]


def load_video(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """
    Load a video into a list of frames.

    Args:
    video_path (Path): The path to the video file to load.

    Returns:
        Tuple[List[np.ndarray], float]: A list of video frames and the framerate
    """

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames, fps


def write_video(frames: np.ndarray, output_path: str, fps: float) -> None:
    """
    Write a video frame array to disk

    Args:
    frames (np.ndarray): Video frame array of shape `(T, H, W, C)`
    output_path (str): Outputh path where video gets written
    fps (float): Output framerate

    Returns:
        None
    """

    frames = frames[..., ::-1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frames.shape[2], frames.shape[1]))
    for i in range(frames.shape[0]):
        out.write(frames[i])
    out.release()


def calculate_optical_flow(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
    """
    Calculate the optical flow between consecutive frames in a sequence of frames.

    Args:
        frames (Sequence[np.ndarray]): Sequence of frames.

    Returns:
        List[np.ndarray]: List of magnitudes representing the optical flow.
    """
    previous_gray = None
    optical_flow_magnitudes = []

    for current_frame in frames:
        scale_ratio = determine_scale_ratio(current_frame)

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        resized_dimensions = (
            int(current_frame.shape[1] * scale_ratio),
            int(current_frame.shape[0] * scale_ratio),
        )
        current_gray = cv2.resize(current_gray, resized_dimensions)

        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            normalized_magnitude = normalize_magnitude(magnitude)
            resized_magnitude = cv2.resize(
                normalized_magnitude, (current_frame.shape[1], current_frame.shape[0])
            )
        else:
            resized_magnitude = np.zeros(
                (current_frame.shape[0], current_frame.shape[1]), dtype=np.uint8
            )

        previous_gray = current_gray
        optical_flow_magnitudes.append(resized_magnitude)

    return optical_flow_magnitudes


def determine_scale_ratio(frame: np.ndarray) -> float:
    """
    Determine the scale ratio based on the frame's width.

    Args:
        frame (np.ndarray): The current frame.

    Returns:
        float: The scale ratio.
    """
    if frame.shape[1] > 640 and frame.shape[1] > 960:
        return 1 / 4
    elif frame.shape[1] >= 960 and frame.shape[1] < 1280:
        return 1 / 6
    elif frame.shape[1] >= 1280:
        return 1 / 8
    else:
        return 1 / 2


def normalize_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """
    Normalize the magnitude of the optical flow.

    Args:
        magnitude (np.ndarray): The magnitude of the optical flow.

    Returns:
        np.ndarray: The normalized magnitude.
    """
    return (
        255.0 * (magnitude - magnitude.min()) / max(float(magnitude.max() - magnitude.min()), 1)
    ).astype(np.uint8)


def find_target_bbox(
    bbox_arr: Sequence[Sequence[BboxTuple]],
    opts: Sequence[np.ndarray],
    iou_thr: float = 0.5,
    len_ratio_thr: float = 0.5,
) -> Tuple[Optional[BboxTuple], List[Tube],]:
    """
    Function to find the target bounding box and tubes.

    Args:
        bbox_arr (Sequence[Sequence[BboxTuple]]): Sequence of bounding boxes.
        opts (Sequence[np.ndarray]): Sequence of optical flow arrays.
        iou_thr (float, optional): Intersection over Union threshold. Defaults to 0.5.
        len_ratio_thr (float, optional): Length ratio threshold. Defaults to 0.5.

    Returns:
        Tuple[Optional[BboxTuple], List[Tube]: Target bounding box and tubes.
    """
    tubes = []
    total_bboxes = sum(len(x) for x in bbox_arr)

    while total_bboxes > 0:
        anchor = next((i, bbox_arr[i].pop()) for i, bboxes in enumerate(bbox_arr) if bboxes)
        tube = [anchor]
        for i, bboxes in enumerate(bbox_arr):
            if anchor[0] == i or not bboxes:
                continue
            ious = np.array([get_iou(anchor[1], bbox) for bbox in bboxes])
            max_iou_index = ious.argmax()
            if ious[max_iou_index] > iou_thr:
                target_bbox = bboxes.pop(max_iou_index)
                tube.append([i, target_bbox])
        tubes.append(tube)
        total_bboxes -= len(tube)

    mean_vals_and_tubes = [
        (calculate_mean_val(tube, opts), tube)
        for tube in tubes
        if len(tube) / len(opts) > len_ratio_thr
    ]
    _, best_tube = max(mean_vals_and_tubes) if mean_vals_and_tubes else (-1, None)

    target_bbox = (
        tuple(np.array([bbox[1] for bbox in best_tube]).mean(axis=0)) if best_tube else None
    )
    return target_bbox, tubes


@lru_cache(maxsize=None)
def get_iou(bbox_a: BboxTuple, bbox_b: BboxTuple) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    The IoU is defined as the area of the overlap between the two bounding boxes divided by the area of their union.

    Args:
        bbox_a (BboxTuple): The first bounding box, represented as a tuple of (x1, y1, x2, y2).
        bbox_b (BboxTuple): The second bounding box, represented as a tuple of (x1, y1, x2, y2).

    Returns:
        float: The IoU between the two bounding boxes.
    """
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(bbox_a_area + bbox_b_area - inter_area)

    return iou


def calculate_mean_val(tube: Tube, opts: List[np.ndarray]) -> float:
    """
    Function to calculate the mean value of a tube.

    Args:
        tube (Tube): Tube to calculate mean value for.
        opts (List[np.ndarray]): List of options.

    Returns:
        float: Mean value of the tube.
    """
    mean_val = sum(
        opts[frame_index][max(y0, 0) : y1, max(x0, 0) : x1].mean()
        for frame_index, (x0, y0, x1, y1) in ((x[0], tuple(map(int, x[1]))) for x in tube)
    )
    return mean_val / len(tube)


def calculate_bbox_from_tubes(tubes: Sequence[Tube]) -> Optional[BboxTuple]:
    """
    Calculate bounding box from tubes.

    Args:
        tubes (Sequence[Tube]): Sequence of tubes. A tube is a sequence of (frame index, bounding box) tuples.

    Returns:
        Optional[BboxTuple]: Bounding box, or None if not found.
    """
    total_sizes = [
        sum((bbox[3] - bbox[0]) * (bbox[2] - bbox[1]) for _, bbox in tube) for tube in tubes
    ]
    if max(total_sizes) > 0:
        idx = np.array(total_sizes).argmax()
        return tuple(np.array([x for _, x in tubes[idx]]).mean(axis=0))
    return None


def crop_resize(imgs: Sequence[np.ndarray], bbox: BboxTuple, target_size: int) -> np.ndarray:
    """
    This function crops and resizes frames based on the provided bounding box and target size.

    Args:
        imgs (Sequence[np.ndarray]): Sequence of frames to be processed.
        bbox (BboxTuple): Bounding box coordinates (x0, y0, x1, y1).
        target_size (int): The target size for the output images.

    Returns:
        np.ndarray: Stacked array of processed frames.
    """
    x0, y0, x1, y1 = bbox

    exp = abs((x1 - x0) - (y1 - y0)) / 2
    if x1 - x0 < y1 - y0:
        x0, x1 = x0 - exp, x1 + exp
    else:
        y0, y1 = y0 - exp, y1 + exp
    x0, x1, y0, y1 = map(int, (x0, x1, y0, y1))

    # Calculate expansion values for each side
    left_expand = max(-x0, 0)
    up_expand = max(-y0, 0)
    right_expand = max(x1 - imgs[0].shape[1] + 1, 0)
    down_expand = max(y1 - imgs[0].shape[0] + 1, 0)

    # Pad, crop, and resize each frame
    rois = np.stack(
        [
            cv2.resize(
                cv2.copyMakeBorder(
                    img,
                    up_expand,
                    down_expand,
                    left_expand,
                    right_expand,
                    cv2.BORDER_CONSTANT,
                    (0, 0, 0),
                )[y0 + up_expand : y1 + up_expand, x0 + left_expand : x1 + left_expand],
                (target_size, target_size),
            )
            for img in imgs
        ]
    )

    return rois


def temporal_sampling(
    frames: torch.Tensor,
    start_idx: int,
    end_idx: int,
    num_samples: int,
    return_index: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx + 0.5, end_idx + 0.5, num_samples, device=frames.device)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)

    if return_index:
        return new_frames, index
    return new_frames


def tensor_normalize(
    tensor: torch.Tensor,
    mean: Union[torch.Tensor, Tuple[float, float, float]],
    std: Union[torch.Tensor, Tuple[float, float, float]],
) -> torch.Tensor:
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, tuple):
        mean = torch.tensor(mean, device=tensor.device)
    if isinstance(std, tuple):
        std = torch.tensor(std, device=tensor.device)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def uniform_crop(
    images: torch.Tensor, size: int, spatial_idx: int, scale_size: Optional[int] = None
) -> torch.Tensor:
    """
    Perform uniform spatial sampling on the images.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped


def get_num_padding_frames(
    idx: torch.Tensor,
    num_frames: int,
    sampling_rate: int,
    fps: float,
    target_fps: float,
) -> int:
    """
    Get the number of padding frames based on the provided parameters

    Args:
    idx (torch.Tensor): A tensor containing indices.
    num_frames (int): The total number of frames.
    sampling_rate (int): The rate at which frames are sampled.
    fps (float): The original frames per second.
    target_fps (float): The target frames per second.

    Returns:
    int: The number of padding frames.
    """

    num_unique = len(torch.unique(idx))

    # Frames duplicated via interpolation should not count as padding
    if target_fps > (fps * sampling_rate):
        num_non_padding = math.floor(num_unique * target_fps / (fps * sampling_rate))
    else:
        num_non_padding = num_unique
    return num_frames - num_non_padding


class Preprocessor:
    def __init__(self, config: DictConfig, device: torch.device):
        self.config = config
        self.device = device

        if config.hog_detector:
            self.detector = dlib.get_frontal_face_detector()
        else:
            self.detector = dlib.cnn_face_detection_model_v1(config.detector_path)

    def detect_frame(self, frame: np.ndarray) -> List[BboxTuple]:
        """
        Detect faces in a frame using either a HOG detector or a CNN-based detector.

        Args:
            frame (np.ndarray): The input frame to be processed.

        Returns:
            List[BboxTuple]: A list of bounding box tuples, each tuple containing the
            coordinates (left, top, right, bottom) of a detected object.
        """
        if self.config.detection_downsample:
            scale_ratio = determine_scale_ratio(frame)
            frame = cv2.resize(frame, (0, 0), fx=scale_ratio, fy=scale_ratio)
        else:
            scale_ratio = 1

        if self.config.hog_detector:
            return [
                (
                    rect.left() // scale_ratio,
                    rect.top() // scale_ratio,
                    rect.right() // scale_ratio,
                    rect.bottom() // scale_ratio,
                )
                for rect in self.detector(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 1)
            ]
        else:
            return [
                (
                    d.rect.left() // scale_ratio,
                    d.rect.top() // scale_ratio,
                    d.rect.right() // scale_ratio,
                    d.rect.bottom() // scale_ratio,
                )
                for d in self.detector(frame, 1)
            ]

    def detect_faces(self, frames: Sequence[np.ndarray]) -> Tuple[List[List[BboxTuple]], int]:
        """
        Detect faces in a sequence of images.
        This function applies a face detector to each image in the input sequence. The detected faces are returned as
        bounding boxes. The function also returns the maximum number of faces detected in a single image.

        Args:
            frames (Sequence[np.ndarray]): A sequence of frames in which to detect faces.

        Returns:
            Tuple[List[List[BboxTuple]], int]: A tuple where the first element is a list of lists of bounding boxes
            for the detected faces, and the second element is the maximum number of faces detected in a single image.
        """
        bboxes = [self.detect_frame(frame) for frame in frames]
        max_num_faces = max([len(x) for x in bboxes])

        return bboxes, max_num_faces

    def _expand_bboxes(self, bboxes: Sequence[Sequence[BboxTuple]]) -> None:
        """
        Expand bounding boxes based on the expansion configuration.
        This function iterates over each bounding box in the provided sequence and expands it according to the
        expansion configuration defined in `self.config`. The expansion is performed in all four directions: left,
        up, right, and down.

        Args:
            bboxes (Sequence[Sequence[BboxTuple]]): A sequence of sequences of bounding boxes. Each inner sequence
            represents a set of bounding boxes for a particular frame.

        Returns:
            None
        """
        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):
                x0, y0, x1, y1 = bboxes[i][j]
                w, h = x1 - x0 + 1, y1 - y0 + 1
                x0, y0, x1, y1 = (
                    x0 - w * self.config.left_exp,
                    y0 - h * self.config.up_exp,
                    x1 + w * self.config.right_exp,
                    y1 + h * self.config.down_exp,
                )
                bboxes[i][j] = (x0, y0, x1, y1)

    def _bbox_from_bboxes_mean(self, bboxes: Sequence[Sequence[BboxTuple]]) -> BboxTuple:
        """
        Calculate the mean bounding box from a sequence of bounding boxes.
        This function takes a sequence of bounding boxes, filters out those that do not have exactly one element,
        and then calculates the mean bounding box from the remaining ones.

        Args:
            bboxes (Sequence[Sequence[BboxTuple]]): A sequence of sequences of bounding boxes. Each inner sequence
            represents a set of bounding boxes for a particular frame.

        Returns:
            BboxTuple: The mean bounding box calculated from the input bounding boxes.
        """
        return tuple(np.array([x for x in bboxes if len(x) == 1]).mean(axis=0)[0])

    def _try_bbox_from_optical_flow(
        self,
        frames: Sequence[np.ndarray],
        bboxes: Sequence[Sequence[BboxTuple]],
    ) -> Optional[BboxTuple]:
        """
        Try to find a bounding box from optical flow.
        This function calculates the optical flow from the given frames, then tries to find a target bounding box
        based on the calculated optical flow and the given bounding boxes. If no bounding box is found, it tries to
        calculate a bounding box from the tubes.

        Args:
            frames (Sequence[np.ndarray]): A sequence of frames from a video.
            bboxes (Sequence[Sequence[BboxTuple]]): A sequence of bounding boxes for each frame.

        Returns:
            Optional[BboxTuple]: The found bounding box, or None if no bounding box could be found.
        """

        opts = calculate_optical_flow(frames)

        opts = opts[:: self.config.detection_sampling_rate]

        bbox, tubes = find_target_bbox(
            bboxes, opts, self.config.iou_threshold, self.config.num_ratio_threshold
        )

        if bbox is None and tubes:
            bbox = calculate_bbox_from_tubes(tubes)

        return bbox

    def _bbox_from_center_crop(self, frame: np.ndarray) -> BboxTuple:
        """
        Obtain a bounding box by center cropping a video frame.
        This function calculates the center of the frame and then creates a bounding box around the center.
        The size of the bounding box is half of the smaller dimension of the frame.

        Args:
            frame (np.ndarray): Frame array

        Returns:
            BboxTuple: Bounding box for center crop.
        """
        W, H = frame.shape[1], frame.shape[0]
        cx, cy, size = W // 2, H // 2, min(W, H) // 2
        return (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)

    def _sample_frames_for_feature_extraction(
        self, frames: torch.Tensor, fps: float
    ) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Samples clips with a fixed number of frames in a sliding window for the full video.
        The clips are then stacked in the batch dimension. If the video is shorter than `num_frames`,
        it will be padded. The method returns the number of padding frames.
        This method does not support repeat augmentation.
        """

        frames_list = []
        num_padding_frames_list = []

        stride = self.config.feature_extraction_stride
        sampling_rate = self.config.sampling_rate
        num_frames = self.config.num_frames

        clip_sz = sampling_rate * num_frames / self.config.target_fps * fps

        for i in range(0, frames.shape[0], stride * sampling_rate):
            start_idx, end_idx = i, i + clip_sz - 1
            new_frames, idx = temporal_sampling(
                frames, start_idx, end_idx, num_frames, return_index=True
            )

            num_padding_frames_list.append(
                get_num_padding_frames(idx, num_frames, sampling_rate, fps, self.config.target_fps)
            )

            frames_list.append(new_frames)

            if end_idx >= frames.shape[0]:
                break

        new_frames = torch.stack(frames_list, dim=0)

        new_frames = tensor_normalize(new_frames, tuple(self.config.mean), tuple(self.config.std))
        frames = new_frames.permute(0, 4, 1, 2, 3)  # b t h w c -> b c t h w

        num_padding_frames = torch.tensor(
            num_padding_frames_list, dtype=torch.long, device=frames.device
        )

        return {"frames": frames, "padding": num_padding_frames}

    def __call__(self, video_path: Path) -> Dict[str, torch.Tensor]:
        """
        Process a video and return regions of interest (ROIs).
        This function loads a video, detects faces in the video frames, expands the bounding boxes of the detected faces,
        and then tries to find a bounding box either by calculating the mean of the bounding boxes or by using optical flow.
        If no bounding box is found, it creates one by center cropping the first frame. Finally, it crops and resizes the
        frames according to the found bounding box and returns the resulting regions of interest.

        Args:
            video_path (Path): The path to the video file to process.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing a frames tensor and a padding tensor
        """

        assert video_path.is_file(), f"{video_path} does not exist."

        t0 = time.time()

        frames, fps = load_video(video_path)

        t1 = time.time()

        bboxes, max_num_faces = self.detect_faces(frames[:: self.config.detection_sampling_rate])

        t2 = time.time()

        self._expand_bboxes(bboxes)

        if max_num_faces == 1:
            bbox = self._bbox_from_bboxes_mean(bboxes)
        else:
            bbox = self._try_bbox_from_optical_flow(frames, bboxes)

        if bbox is None:
            bbox = self._bbox_from_center_crop(frames[0])

        t3 = time.time()

        rois = crop_resize(frames, bbox, self.config.target_size)

        t4 = time.time()

        if self.config.debug:
            write_video(rois, f"{video_path.stem}_roi_crop.mp4", fps)

        out = self._sample_frames_for_feature_extraction(
            torch.from_numpy(rois).to(self.device), fps=fps
        )

        t5 = time.time()

        if self.config.verbose:
            print(f"2. Preprocessing: {t5 - t0:.3f}s")
            print(f" - Loading: {t1 - t0:.3f}s")
            print(f" - Face detection: {t2 - t1:.3f}s")
            print(f" - Finding bboxes: {t3 - t2:.3f}s")
            print(f" - Crop and resize: {t4 - t3:.3f}s")
            print(f" - Sampling: {t5 - t4:.3f} s")

        return out
