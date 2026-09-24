"""Configurable fixed-shape real-robot DP3 dataset."""
from diffusion_policy_3d.dataset.generic_zarr_dataset import GenericZarrDataset


class GenericDP3Dataset(GenericZarrDataset):
    expected_point_cloud_shape = (1024, 3)
    expected_state_shape = (54,)
    expected_action_shape = (54,)

    def __init__(self, *, expected_num_points=1024, expected_dim=54, **kwargs):
        self.expected_point_cloud_shape = (int(expected_num_points), 3)
        self.expected_state_shape = (int(expected_dim),)
        self.expected_action_shape = (int(expected_dim),)
        super().__init__(**kwargs)
