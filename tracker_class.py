from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple
import torch, cv2
import os, requests
import testing.utils as hf
# Sam imports
from tracker_mobilesam import BaseTracker 
import open3d as o3d
# FCCLIP imports
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from fcclip import add_maskformer2_config, add_fcclip_config
from predictor import VisualizationDemo

DEBUG_VISUALIZATION = False

class CameraVideoData:
    """
    A class representing a single viewpoint in a video sequence,
    containing a list of frames and camera intrinsics and extrinsics.

    Attributes:
        frames: A list of frames from the video sequence.
        depths: A list of depth images from the video sequence.
        intrinsics (numpy.ndarray): The camera intrinsics matrix.
        extrinsics (numpy.ndarray): The camera extrinsics matrix.
    """
    def __init__(self, frames: list, depths: list, camera_intrinsics: np.ndarray, camera_extrinsics: np.ndarray = np.eye(3), depth_mm_to_m: bool = True):
        assert len(frames) == len(depths)
        self.frames = np.array(frames)
        self.depths = np.array(depths) / 1000 if depth_mm_to_m else depths
        self.intrinsics = camera_intrinsics
        self.extrinsics = camera_extrinsics
    
    def __len__(self):
        return len(self.frames)
    
    # def add_frame(self, frame, depth):
    #     self.frames = np.concatenate((self.frames, np.array([frame])))
    #     self.depths = np.concatenate((self.depths, np.array([depth])))

    


class Tracker:
    """
    A class that tracks an object in a video sequence given a list of viewpoints.
    """
    def __init__(self,
                 fcclip_config_file: str,
                 fcclip_weight_file: str,
                 sam_xmem_checkpoint: str,
                 fcclip_text_prompt = "ball",
                 ) -> None:
        fcclip_config = self._setup_cfg(fcclip_config_file, fcclip_weight_file)
        self.fcclip = VisualizationDemo(fcclip_config, txt_prompt=fcclip_text_prompt)
        self.tracker = BaseTracker(sam_xmem_checkpoint, device="cuda:0")
    

    def track(self, viewpoints: List[CameraVideoData]) -> List[float]:
        """
        This method returns a list of 3D positions of an object in the world frame, given a list of viewpoints.
        The object that is tracked is the object that fcclip was configured to track given the text prompt
        """

        # Assert that the number of frames is equal for all of your viewpoints.
        assert all(len(i) == len(viewpoints[0]) for i in viewpoints)

        #Variables to keep track of during iteration
        object_poses = []
        mask_is_template = True

        current_viewpoint, current_mask = self._get_largest_valid_template_mask([frame.frames[0] for frame in viewpoints])
        object_poses.append(self._get_object_pose(viewpoints, current_viewpoint, current_mask, current_frame=0, template_mask=mask_is_template))
        for i in range(1, len(viewpoints[0])):
            # plt.imshow(current_mask)
            # Create a new numpy array by blending the mask with the image using alpha blending
            if DEBUG_VISUALIZATION:
                self._save_visualization(current_mask, viewpoints[current_viewpoint].frames[i-1], viewpoints[current_viewpoint].depths[i-1], f"viewpoint{i}", mask_is_template)
            try:
                # Todo: We can comment out some of the code in the tracker that is not related to the mask
                if(mask_is_template):
                    self.tracker.clear_memory()
                    current_mask, _, _= self.tracker.track(viewpoints[current_viewpoint].frames[i], current_mask)
                    mask_is_template = False
                else:
                    current_mask, _, _= self.tracker.track(viewpoints[current_viewpoint].frames[i])
                if(np.count_nonzero(current_mask) == 0):
                    raise IndexError
                object_pose = self._get_object_pose(viewpoints, current_viewpoint, current_mask, current_frame=i, template_mask=mask_is_template)
                if np.isnan(object_pose).any():
                    raise IndexError
            # TODO: change this to not index error, but the error it raises 
            except IndexError:
                current_viewpoint, current_mask = self._get_largest_valid_template_mask([frame.frames[i] for frame in viewpoints])
                mask_is_template = True
                object_pose = self._get_object_pose(viewpoints, current_viewpoint, current_mask, current_frame=i, template_mask=mask_is_template)
            object_poses.append(object_pose)
        return object_poses
    
    def _save_visualization(self, mask, rgb, depth, path, is_template):
        plt.cla(); plt.clf()
        if not is_template:
            mask = mask.reshape(rgb.shape[:2])
        masked_image = .3 * rgb /255 + .7 * np.stack([mask, mask, mask], axis=-1)
        plt.imshow(masked_image)
        plt.savefig(path)

    def _get_object_pose(self, viewpoints: List[CameraVideoData], current_viewpoint: int, 
                         current_mask: torch.Tensor, current_frame: int, template_mask: bool) -> np.ndarray:
        """
        Calculates the 3D position of an object in the world frame given a list of viewpoints, the current viewpoint index,
        the current mask of the object in the current viewpoint, and the current frame index.

        Args:
            viewpoints (List[CameraVideoData]): A list of CameraVideoData objects representing the viewpoints.
            current_viewpoint (int): The index of the current viewpoint in the list of viewpoints.
            current_mask (torch.Tensor): The mask of the object in the current viewpoint.
            current_frame (int): The index of the current frame in the video sequence.

        Returns:
            np.ndarray: The 3D position of the object in the world frame.
        """
        depth = viewpoints[current_viewpoint].depths[current_frame]

        # Get the masked points of the object, and then filter out the zero points
        points = get_img_points(viewpoints[current_viewpoint].intrinsics, depth)
        if not template_mask:   
            current_mask = current_mask.astype(bool)
        points = points[current_mask.flatten()]

        idxs = np.argwhere(np.sum(np.abs(points), axis=1) > 0.0001).flatten()
        points = points[idxs]

        # Get the average point of the object to be the center
        object_center = np.mean(points, axis=0)
        if np.isnan(object_center).any():
            # Why this happens, the robot blocks the object except for a sliver. The sliver is accuratly tracked,
            # but has no depth information. This leads to have a Nan. We should if we encounter a NAN switch cameras. 
            # Get first mask should also calculate all of the masks, and then instead of picking the first, pick the largest.
            print('nand')
        # transform the object center to the world frame
        object_center = viewpoints[current_viewpoint].extrinsics @ np.append(object_center, 1)

        if DEBUG_VISUALIZATION:
            pcd = hf.get_pcd(viewpoints[current_viewpoint].intrinsics, viewpoints[current_viewpoint].frames[current_frame], viewpoints[current_viewpoint].depths[current_frame])
            hf.add_point(pcd, [1, 0, 0], object_center[:3])
            o3d.visualization.draw_geometries([pcd])
        return object_center[:3]
    

    def _get_largest_valid_template_mask(self, rgbs:List[np.ndarray]) -> Tuple[int, torch.Tensor]:
        """
        Returns the index and mask of the valid mask that is the largest in the given list of RGB images.

        Args:
            rgbs (List[np.ndarray]): A list of RGB images.

        Returns:
            Tuple[int, torch.Tensor]: A tuple containing the index of the first valid template and its corresponding mask.
        """
        masks = []
        for i, rgb in enumerate(rgbs):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            try:
                _, _, template_mask = self.fcclip.run_on_image(bgr)
                masks.append((i, template_mask[0]))
            except TypeError:
                continue
        if len(masks) > 0:
            # pick the mask with the most true values in it
            return max(masks, key=lambda x: np.count_nonzero(x[1]))
        
        # raise IndexError("No valid template masks were found in the given list of RGB images.")
        os.makedirs('./debug', exist_ok=True)
        for i, rgb in enumerate(rgbs):
            plt.cla(); plt.clf()
            plt.imshow(rgb)
            plt.savefig(f'./debug/rgb{i}.png')
        raise IndexError(f"No valid template masks were found in the given list of RGB images. Saved to ./debug folder. Trying to detect {self.fcclip.txt_prompt}")


    def _setup_cfg(self, fcclip_cfg_file, fcclip_weight_file):
        """
        Sets up the config dict for the FCCLIP model.
        Args:
            fcclip_cfg_file (str): The path to the FCCLIP configuration file.
            fcclip_weight_file (str): The path to the FCCLIP weight file.

        Returns:
            detectron2.config.config.CfgNode: The configuration for the FCCLIP model.
        """
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_fcclip_config(cfg)
        cfg.merge_from_file(fcclip_cfg_file)
        cfg.merge_from_list(["MODEL.WEIGHTS", fcclip_weight_file])
        cfg.freeze()
        return cfg


def get_img_points(ints, depth_image) -> np.ndarray:
    """
    Computes the 3D coordinates of each pixel in an image, given its depth map and camera intrinsics.

    Args:
        ints (numpy.ndarray): The camera intrinsics matrix, of shape (3, 3).
        depth_image (numpy.ndarray): The depth map of the image, of shape (height, width).

    Returns:
        numpy.ndarray: An array of points in camera frame
    """
    height, width = depth_image.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)

    px = (px - ints[0, 2]) * (depth_image / ints[0, 0])
    py = (py - ints[1, 2]) * (depth_image / ints[1, 1])
    
    points = np.float32([px, py, depth_image]).transpose(1, 2, 0)

    pts = np.array(points).reshape(-1, 3)
    return pts
