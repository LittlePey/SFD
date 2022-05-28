import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

        if self.point_encoding_config.get('used_feature_list_pseudo', False):
            self.used_feature_list_pseudo = self.point_encoding_config.used_feature_list_pseudo
            self.src_feature_list_pseudo = self.point_encoding_config.src_feature_list_pseudo

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    @property
    def num_point_features_pseudo(self):
        return getattr(self, self.point_encoding_config.encoding_type_pseudo)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        data_dict['use_lead_xyz'] = use_lead_xyz

        if self.point_encoding_config.get('used_feature_list_pseudo', False):
            data_dict['points_pseudo'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type_pseudo)(
                data_dict['points_pseudo']
            )
            data_dict['use_lead_xyz_pseudo'] = use_lead_xyz

        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True

    def absolute_coordinates_encoding_pseudo(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list_pseudo)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list_pseudo:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list_pseudo.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True