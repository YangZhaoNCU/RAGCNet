import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
# from logger import get_missing_parameters_message, get_unexpected_parameters_message
from utils import Adj_matrix_gen
from knn_cuda import KNN
from pointnet_util import PointNetFeaturePropagation
from dataloader import get_data, extract_triangles_by_class_2


def furthest_point_sample(points_face, npoint):
    device = points_face.device
    xyz = points_face[:, :, 9:12]
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids #sampled pointcloud index, [B, npoint]


def fps(data, number):
    fps_idx = furthest_point_sample(data, number)
    fps_idx = fps_idx.to(torch.int)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data
"""
***************************************************************************************
"""
class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        xyz = xyz.contiguous()
        center = fps(xyz, self.num_group)  # B G 3
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 12).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center  # [B,G,M,12] [B,G,12]


class GraphAttention(nn.Module):
    def __init__(self, feature_dim, out_dim, K):
        super(GraphAttention, self).__init__()
        # self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.k = K
    def forward(self, neighborhood, x, feature):

        B, G, C = x.shape
        x = x.contiguous()
        neighbor_feature = neighborhood
        centre = x.view(B, G, 1, C).expand(B, G, self.k, C)
        delta_f = torch.cat([centre-neighbor_feature, neighbor_feature], dim=3).permute(0, 3, 2, 1)
        e = self.conv(delta_f)
        e = e.permute(0, 3, 2, 1)
        attention = F.softmax(e, dim=2)
        graph_feature = torch.sum(torch.mul(attention, feature.permute(0, 2, 3, 1)), dim=2) .permute(0, 2, 1)
        return graph_feature


class AFF(nn.Module):

    def __init__(self, channels=256, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # xa = x + y
        # print(xa.shape)
        xl = self.local_att(x)
        xg = self.global_att(y)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + y * (1 - wei)
        return xo
"""
***************************************************************************************
"""
class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        # self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 1),
            # nn.BatchNorm1d(128)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 1),
            # nn.BatchNorm1d(512)
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 12)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, 512)


class get_model(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()
###########################################################################
        self.cls_dim = cls_dim
        # self.num_heads = 6
        self.group_size = 32
        self.num_group = 2000
        self.encoder_dims = 512
################### transformer_feature ####################################
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Conv1d(12, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        self.propagation_0 = PointNetFeaturePropagation(in_channel=512 + 12,
                                                        mlp=[512, 512])
        self.propagation_1 = PointNetFeaturePropagation(in_channel=448 + 12,
                                                        mlp=[448, 1024])

################### cor_feature ####################################
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                     self.bn1_c,
                                     nn.LeakyReLU(negative_slope=0.2))
        #
        self.conv2_c = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                     self.bn2_c,
                                     nn.LeakyReLU(negative_slope=0.2))
        #
        self.conv3_c = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                     self.bn3_c,
                                     nn.LeakyReLU(negative_slope=0.2))
        #
        self.conv4_c = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                     self.bn4_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.attention_layer1_c = GraphAttention(feature_dim=12, out_dim=64, K=self.group_size)
        self.attention_layer2_c = GraphAttention(feature_dim=12, out_dim=128, K=self.group_size)
        self.attention_layer3_c = GraphAttention(feature_dim=12, out_dim=256, K=self.group_size)

#################### nor_feature ####################################
        self.bn1_n = nn.BatchNorm1d(64)
        self.bn2_n = nn.BatchNorm1d(128)
        self.bn3_n = nn.BatchNorm1d(256)
        self.bn4_n = nn.BatchNorm1d(512)
        self.conv1_1_n = nn.Sequential(nn.Conv1d(12, 64, kernel_size=1, bias=False),
                                       self.bn1_n,
                                       # nn.LayerNorm([64, 16000]),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.conv2_1_n = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                       self.bn2_n,
                                       # nn.LayerNorm([128, 16000]),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.conv3_1_n = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),
                                       self.bn3_n,
                                       # nn.LayerNorm([256, 16000]),
                                       nn.LeakyReLU(negative_slope=0.2))
        self.conv4_n = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                       self.bn4_n,
                                       # nn.LayerNorm([256, 16000]),
                                       nn.LeakyReLU(negative_slope=0.2))

########################################################
        # self.convs1 = nn.Conv1d(1984, 1024, 1)
        # self.dp1 = nn.Dropout(0.2)
        '''feature-wise attention'''
        self.aff = AFF(1024, 4)
        self.fa = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                nn.BatchNorm1d(1024),
                                nn.LeakyReLU(0.2))
        self.convs2 = nn.Conv1d(1024, 512, 1)
        self.dp2 = nn.Dropout(0.2)
        self.convs3 = nn.Conv1d(512, 256, 1)
        self.dp3 = nn.Dropout(0.2)
        self.convs4 = nn.Conv1d(256, 128, 1)
        self.dp4 = nn.Dropout(0.2)
        # self.bns1 = nn.BatchNorm1d(1024)
        self.bns2 = nn.BatchNorm1d(512)
        self.bns3 = nn.BatchNorm1d(256)
        self.bns4 = nn.BatchNorm1d(128)
        self.pred = nn.Conv1d(128, self.cls_dim, 1)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, point_faces, index_face):
        pts = point_faces[:, :, :12]
        x = point_faces.transpose(2, 1)
        # coor = x[:, :12, :]
        nor = x[:, 12:, :]
        adj = Adj_matrix_gen(index_face)
        adj_2 = torch.matmul(adj, adj)

        nor1_1 = self.conv1_1_n(nor)  # [1, 64, 16000]
        nor1_1 = (adj @ nor1_1.transpose(-1, -2)).transpose(-1, -2)
        nor2_1 = self.conv2_1_n(nor1_1)  # [1, 128, 16000]
        nor2_1 = (adj @ nor2_1.transpose(-1, -2)).transpose(-1, -2)
        nor3_1 = self.conv3_1_n(nor2_1)  # [1, 256, 16000]
        nor3_1 = (adj @ nor3_1.transpose(-1, -2)).transpose(-1, -2)
        nor = torch.cat((nor1_1, nor2_1, nor3_1), dim=1)
        nor = (adj_2 @ nor.transpose(-1, -2)).transpose(-1, -2)
        nor = self.conv4_n(nor)

        B, N, C = pts.shape
        neighborhood, center = self.group_divider(pts)
        center_feature = center.view(B, self.num_group, 1, C).repeat(1, 1, self.group_size, 1)
        coor_1 = self.conv1_c((neighborhood + center_feature).permute(0, 3, 1, 2))
        coor_feature_1 = self.attention_layer1_c(neighborhood, center, coor_1)
        coor_2 = self.conv2_c(coor_1)
        coor_feature_2 = self.attention_layer2_c(neighborhood, center, coor_2)
        coor_3 = self.conv3_c(coor_2) # B G M 12
        coor_feature_3 = self.attention_layer3_c(neighborhood, center, coor_3)
        cor_feature = torch.cat((coor_feature_1, coor_feature_2, coor_feature_3), dim=1)

        group_input_tokens = self.encoder(neighborhood)  # B G N
        x = group_input_tokens.transpose(-1, -2)  # B 96 G

        pos = self.pos_embed(center.transpose(-1, -2))
        x_max = torch.max(pos, 2)[0]
        x_avg = torch.mean(pos, 2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 576*2
        # x_global_feature = (adj @ x_global_feature.transpose(-1, -2)).transpose(-1, -2)

        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x) # 1024
        # f_level_0 = (adj @ f_level_0.transpose(-1, -2)).transpose(-1, -2)

        f_level_1 = self.propagation_1(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), cor_feature)
        f_level_1 = (adj @ f_level_1.transpose(-1, -2)).transpose(-1, -2)

        x_global = torch.cat((f_level_0, x_global_feature), 1)
        x_global = (adj @ x_global.transpose(-1, -2)).transpose(-1, -2)

        # coor = torch.cat((f_level_0, f_level_1, x_global_feature), 1)
        coor = self.aff(f_level_1, x_global) #全部特征

        coor = (adj_2 @ coor.transpose(-1, -2)).transpose(-1, -2)
        coor = self.conv4_c(coor)

        avgSum_coor = coor.sum(1) / 512
        avgSum_nor = nor.sum(1) / 512
        avgSum = avgSum_coor + avgSum_nor
        weight_coor = (avgSum_coor / avgSum).reshape(1, 1, 16000)
        weight_nor = (avgSum_nor / avgSum).reshape(1, 1, 16000)
        x = torch.cat((coor * weight_coor, nor * weight_nor), dim=1)

        weight = self.fa(x)
        x = weight * x

        x = self.relu(self.bns2(self.convs2(x)))
        x = self.dp2(x)
        x = self.relu(self.bns3(self.convs3(x)))
        x = self.dp3(x)
        x = self.relu(self.bns4(self.convs4(x)))
        x = self.dp4(x)
        pred = self.pred(x)
        pred = F.log_softmax(pred, dim=1)
        pred = pred.permute(0, 2, 1)

        pred_choice = pred.contiguous().view(-1, 33)
        pred_choice = pred_choice.data.max(1)[1].unsqueeze(1)

        triangles_by_class = []
        for class_index in range(33):
            triangles_class = extract_triangles_by_class_2(point_faces[:,:,:12].squeeze(0), pred_choice, class_index)
            triangles_by_class.append(triangles_class)

        pred_center = []
        for pred_tensor in triangles_by_class:
            pred_tensor = pred_tensor.cuda()
            t = pred_tensor[:, 9:12]
            is_all_zero = torch.all(t == 0)
            centers = torch.mean(t, dim=0) if not is_all_zero else torch.zeros(3).cuda()
            pred_center.append(centers)
        pred_center = torch.stack(pred_center, dim=0)
        return pred, pred_center    # , rebuild_points, gt_points



if __name__ == "__main__":
    model = get_model(cls_dim=33)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    print(get_parameter_number(model))




