import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, in_channel=9, out_channels=32):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, out_channels, 1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1,2)

        return x

class CPConvs(nn.Module):
    def __init__(self):
        super(CPConvs, self).__init__()
        self.pointnet1_fea = PointNet(  6,12)
        self.pointnet1_wgt = PointNet(  6,12)
        self.pointnet1_fus = PointNet(108,12)

        self.pointnet2_fea = PointNet( 12,24)
        self.pointnet2_wgt = PointNet(  6,24)
        self.pointnet2_fus = PointNet(216,24)

        self.pointnet3_fea = PointNet( 24,48)
        self.pointnet3_wgt = PointNet(  6,48)
        self.pointnet3_fus = PointNet(432,48)

    def forward(self, points_features, points_neighbor):
        if points_features.shape[0] == 0:
            return points_features

        N, F = points_features.shape
        N, M = points_neighbor.shape
        point_empty = (points_neighbor == 0).nonzero()
        points_neighbor[point_empty[:,0], point_empty[:,1]] = point_empty[:,0]

        pointnet_in_xiyiziuiviri = torch.index_select(points_features[:,[0,1,2,6,7,8]],0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet_in_x0y0z0u0v0r0 = points_features[:,[0,1,2,6,7,8]].unsqueeze(dim=1).repeat([1,M,1])
        pointnet_in_xyzuvr       = pointnet_in_xiyiziuiviri - pointnet_in_x0y0z0u0v0r0
        points_features[:, 3:6] /= 255.0
        
        pointnet1_in_fea        = points_features[:,:6].view(N,1,-1)
        pointnet1_out_fea       = self.pointnet1_fea(pointnet1_in_fea).view(N,-1)
        pointnet1_out_fea       = torch.index_select(pointnet1_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet1_out_wgt       = self.pointnet1_wgt(pointnet_in_xyzuvr)
        pointnet1_feas          = pointnet1_out_fea * pointnet1_out_wgt
        pointnet1_feas          = self.pointnet1_fus(pointnet1_feas.reshape(N,1,-1)).view(N,-1)   

        pointnet2_in_fea        = pointnet1_feas.view(N,1,-1)
        pointnet2_out_fea       = self.pointnet2_fea(pointnet2_in_fea).view(N,-1)
        pointnet2_out_fea       = torch.index_select(pointnet2_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet2_out_wgt       = self.pointnet2_wgt(pointnet_in_xyzuvr)
        pointnet2_feas           = pointnet2_out_fea * pointnet2_out_wgt
        pointnet2_feas          = self.pointnet2_fus(pointnet2_feas.reshape(N,1,-1)).view(N,-1)

        pointnet3_in_fea        = pointnet2_feas.view(N,1,-1)
        pointnet3_out_fea       = self.pointnet3_fea(pointnet3_in_fea).view(N,-1)
        pointnet3_out_fea       = torch.index_select(pointnet3_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet3_out_wgt       = self.pointnet3_wgt(pointnet_in_xyzuvr)
        pointnet3_feas           = pointnet3_out_fea * pointnet3_out_wgt
        pointnet3_feas          = self.pointnet3_fus(pointnet3_feas.reshape(N,1,-1)).view(N,-1)
 
        pointnet_feas     = torch.cat([pointnet3_feas, pointnet2_feas, pointnet1_feas, points_features[:,:6]], dim=-1)
        return pointnet_feas

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.pseudo_in, self.valid_in = channels
        middle = self.valid_in // 4
        self.fc1 = nn.Linear(self.pseudo_in, middle)
        self.fc2 = nn.Linear(self.valid_in, middle)
        self.fc3 = nn.Linear(2*middle, 2)
        self.conv1 = nn.Sequential(nn.Conv1d(self.pseudo_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(self.valid_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())

    def forward(self, pseudo_feas, valid_feas):
        batch = pseudo_feas.size(0)

        pseudo_feas_f = pseudo_feas.transpose(1,2).contiguous().view(-1, self.pseudo_in)
        valid_feas_f = valid_feas.transpose(1,2).contiguous().view(-1, self.valid_in)

        pseudo_feas_f_ = self.fc1(pseudo_feas_f)
        valid_feas_f_ = self.fc2(valid_feas_f)
        pseudo_valid_feas_f = torch.cat([pseudo_feas_f_, valid_feas_f_],dim=-1)
        weight = torch.sigmoid(self.fc3(pseudo_valid_feas_f))

        pseudo_weight = weight[:,0].squeeze()
        pseudo_weight = pseudo_weight.view(batch, 1, -1)

        valid_weight = weight[:,1].squeeze()
        valid_weight = valid_weight.view(batch, 1, -1)

        pseudo_features_att = self.conv1(pseudo_feas)  * pseudo_weight
        valid_features_att     =  self.conv2(valid_feas)      *  valid_weight

        return pseudo_features_att, valid_features_att

class GAF(nn.Module):
    def __init__(self, pseudo_in, valid_in, outplanes):
        super(GAF, self).__init__()
        self.attention = Attention(channels = [pseudo_in, valid_in])
        self.conv1 = torch.nn.Conv1d(valid_in + valid_in, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, pseudo_features, valid_features):
        pseudo_features_att, valid_features_att=  self.attention(pseudo_features, valid_features)
        fusion_features = torch.cat([valid_features_att, pseudo_features_att], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


