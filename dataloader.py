import torch
import torch.nn as nn
from plyfile import PlyData
import numpy as np
from torch.utils.data import DataLoader,Dataset,random_split
import os
import random
import pandas as pd
from scipy.spatial import distance_matrix
from scipy import sparse as sp
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from misc import fps

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


labels = ((255, 255, 255), (255, 0, 0), (255, 125, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
          (0, 0, 255), (255, 0, 255), (0, 125, 255))

def face(xyz):
    face_normal = []
    for f_idx in range(xyz.shape[0]):
        x = xyz[f_idx, :3]
        y = xyz[f_idx, 3:6]
        z = xyz[f_idx, 6:9]
        xy = x - y
        xz = x - z
        face = [xy[1] * xz[2]-xz[1] * xy[2], xy[2] * xz[0]-xz[2] * xy[0], xy[0] * xz[1]-xz[0] * xy[1]]
        face_normal.append(face)

    return np.asarray(face_normal)


def Adj_matrix_gen(face):
    N = face.shape[1]
    face0 = face.repeat(1, N).view(N*N, 3)
    face1 = face.repeat(N, 1)
    b = (face0 == face1)
    b = b[:, 0] + b[:, 1] + b[:, 2]
    a = b.view(N, N)
    adj = torch.where(a == True, 1., 0.)

    return adj


def extract_triangles_by_class(points_face, label_face, class_index):
    class_indices = np.where(label_face == class_index)[0]
    # print(class_indices)
    class_triangles = points_face[class_indices]
    # print(class_triangles.shape)
    return class_triangles

def extract_triangles_by_class_2(points_face, pred_choice, class_index):
    class_indices = torch.where(pred_choice == class_index)[0]
    class_triangles = points_face[class_indices]
    # print(class_triangles.shape)
    return class_triangles

def get_data(path=""):
    labels = ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255],
              [0, 0, 255], [255, 0, 255], [30, 144, 255], [0, 255, 127], [127, 255, 0], [255, 246, 143],
              [60, 179, 113], [255, 106, 106], [131, 111, 255], [255, 211, 155], [255, 99, 71], [155, 48, 255],
              [255, 48, 48], [0, 191, 255], [255, 165, 0], [202, 255, 112],
              [200, 255, 255], [255, 228, 255], [255, 155, 255], [255, 69, 0], [139, 0, 0], [144, 238, 144],
              [0, 139, 139], [0, 0, 139], [139, 0, 139], [255, 105, 180], [230, 230, 250], [255, 228, 181], [255, 255, 255]
              )
    row_data = PlyData.read(path)  # read ply file
    points = np.array(pd.DataFrame(row_data.elements[0].data))
    # print(points)
    faces = np.array(pd.DataFrame(row_data.elements[1].data))
    # print("faces:", faces)
    n_face = faces.shape[0]  # number of faces
    xyz = points[:, :3]  # coordinate of vertex shape=[N, 3]
    normal = points[:, 3:]  # normal of vertex shape=[N, 3]
    label_face = np.zeros([n_face, 1]).astype('int32')
    label_face_onehot = np.zeros([n_face, 33]).astype(('int32'))
    """ index of faces shape=[N, 3] """
    index_face = np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3)
    # print("faces[0]:",(faces[:, 0]))
    """ RGB of faces shape=[N, 3] """
    RGB_face = faces[:, 1:4]
    """ coordinate of 3 vertexes  shape=[N, 9] """
    xyz_face = np.concatenate((xyz[index_face[:, 0], :], xyz[index_face[:, 1], :],xyz[index_face[:, 2], :]), axis=1)
    # print(xyz_face)
    """  normal of 3 vertexes  shape=[N, 9] """
    normal_vertex = np.concatenate((normal[index_face[:, 0], :], normal[index_face[:, 1], :],normal[index_face[:, 2], :]), axis=1)

    normal_face = face(xyz_face)
    x1, y1, z1 = xyz_face[:, 0], xyz_face[:, 1], xyz_face[:, 2]
    x2, y2, z2 = xyz_face[:, 3], xyz_face[:, 4], xyz_face[:, 5]
    x3, y3, z3 = xyz_face[:, 6], xyz_face[:, 7], xyz_face[:, 8]
    x_centre = (x1 + x2 + x3) / 3
    y_centre = (y1 + y2 + y3) / 3
    z_centre = (z1 + z2 + z3) / 3
    centre_face = np.concatenate((x_centre.reshape(n_face,1),y_centre.reshape(n_face,1),z_centre.reshape(n_face,1)), axis=1)
    """ get points of each face, concat all of above, shape=[N, 24]"""
    points_face = np.concatenate((xyz_face, centre_face, normal_vertex, normal_face), axis=1).astype('float32')
    """ get label of each face """
    for i, label in enumerate(labels):
        label_face[(RGB_face == label).all(axis=1)] = i
        label_face_onehot[(RGB_face == label).all(axis=1), i] = 1
    return index_face, points_face, label_face, label_face_onehot, points, RGB_face


def get_data_2(path=""):
    labels = ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255],
              [0, 0, 255], [255, 0, 255], [30, 144, 255], [0, 255, 127], [127, 255, 0], [255, 246, 143],
              [60, 179, 113], [255, 106, 106], [131, 111, 255], [255, 211, 155], [255, 99, 71], [155, 48, 255],
              [255, 48, 48], [0, 191, 255], [255, 165, 0], [202, 255, 112],
              [200, 255, 255], [255, 228, 255], [255, 155, 255], [255, 69, 0], [139, 0, 0], [144, 238, 144],
              [0, 139, 139], [0, 0, 139], [139, 0, 139], [255, 105, 180], [230, 230, 250], [255, 228, 181], [255, 255, 255]
              )
    row_data = PlyData.read(path)  # read ply file
    points = np.array(pd.DataFrame(row_data.elements[0].data))
    faces = np.array(pd.DataFrame(row_data.elements[1].data))
    n_face = faces.shape[0]  # number of faces
    xyz = points[:, :3]  # coordinate of vertex shape=[N, 3]
    normal = points[:, 3:]  # normal of vertex shape=[N, 3]
    label_face = np.zeros([n_face, 1]).astype('int32')
    label_face_onehot = np.zeros([n_face, 33]).astype(('int32'))
    """ index of faces shape=[N, 3] """
    index_face = np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3)
    print(index_face)
    """ RGB of faces shape=[N, 3] """
    RGB_face = faces[:, 1:4]
    """ coordinate of 3 vertexes  shape=[N, 9] """
    xyz_face = np.concatenate((xyz[index_face[:, 0], :], xyz[index_face[:, 1], :],xyz[index_face[:, 2], :]), axis=1)
    """  normal of 3 vertexes  shape=[N, 9] """
    normal_vertex = np.concatenate((normal[index_face[:, 0], :], normal[index_face[:, 1], :],normal[index_face[:, 2], :]), axis=1)

    normal_face = face(xyz_face)
    x1, y1, z1 = xyz_face[:, 0], xyz_face[:, 1], xyz_face[:, 2]
    x2, y2, z2 = xyz_face[:, 3], xyz_face[:, 4], xyz_face[:, 5]
    x3, y3, z3 = xyz_face[:, 6], xyz_face[:, 7], xyz_face[:, 8]
    x_centre = (x1 + x2 + x3) / 3
    y_centre = (y1 + y2 + y3) / 3
    z_centre = (z1 + z2 + z3) / 3
    centre_face = np.concatenate((x_centre.reshape(n_face,1),y_centre.reshape(n_face,1),z_centre.reshape(n_face,1)), axis=1)
    """ get points of each face, concat all of above, shape=[N, 24]"""
    points_face = np.concatenate((xyz_face, centre_face, normal_vertex, normal_face), axis=1).astype('float32')
    """ get label of each face """
    for i, label in enumerate(labels):
        label_face[(RGB_face == label).all(axis=1)] = i
        label_face_onehot[(RGB_face == label).all(axis=1), i] = 1
    """get triangles of each class: class num = 33"""
    triangles_by_class = []
    for class_index in range(33):
        triangles_class = extract_triangles_by_class(points_face, label_face, class_index)
        triangles_by_class.append(triangles_class)
    return index_face, points_face, label_face, label_face_onehot, points, RGB_face, triangles_by_class

def generate_plyfile(index_face, point_face, label_face, path= " "):
    """
    Input:
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
        label_face: label of face [N, 1]
        path: path to save new generated ply file
    Return:
    """
    unique_index = np.unique(index_face.flatten())  # get unique points index
    flag = np.zeros([unique_index.max()+1, 2]).astype('uint64')
    order = 0
    with open(path, "a") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write("element vertex " + str(unique_index.shape[0]) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("element face " + str(index_face.shape[0]) + "\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")
        for i, index in enumerate(index_face):
            for j, data in enumerate(index):
                if flag[data, 0] == 0:  # if this point has not been wrote
                    xyz = point_face[i, 3*j:3*(j+1)]  # Get coordinate
                    xyz_nor = point_face[i, 3*(j+3):3*(j+4)]
                    f.write(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + " " + str(xyz_nor[0]) + " "
                            + str(xyz_nor[1]) + " " + str(xyz_nor[2]) + "\n")
                    flag[data, 0] = 1  # this point has been wrote
                    flag[data, 1] = order  # give point a new index
                    order = order + 1  # index add 1 for next point

        labels_change_color = [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255],
                               [0, 0, 255], [255, 0, 255], [30, 144, 255], [0, 255, 127], [127, 255, 0],
                               [255, 246, 143],
                               [60, 179, 113], [255, 106, 106], [131, 111, 255], [255, 211, 155], [255, 99, 71],
                               [155, 48, 255],
                               [255, 48, 48], [0, 191, 255], [255, 165, 0], [202, 255, 112],
                               [200, 255, 255], [255, 228, 255], [255, 155, 255], [255, 69, 0], [139, 0, 0],
                               [144, 238, 144],
                               [0, 139, 139], [0, 0, 139], [139, 0, 139], [255, 105, 180], [230, 230, 250],
                               [255, 228, 181], [255, 255, 255]
                               ]

        for i, data in enumerate(index_face):  # write new point index for every face
            RGB = labels_change_color[label_face[i, 0]]  # Get RGB value according to face label
            f.write(str(3) + " " + str(int(flag[data[0], 1])) + " " + str(int(flag[data[1], 1])) + " "
                    + str(int(flag[data[2], 1])) + " " + str(RGB[0]) + " " + str(RGB[1]) + " "
                    + str(RGB[2]) + " " + str(255) + "\n")
        f.close()


def farthest_point_sample(points_face, npoint):
    """
    Input:
        points_face: point_face data, [B, N, 12]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = points_face.device
    xyz = points_face[:,:,9:12]
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class plydataset(Dataset):

    def __init__(self, path="data/train", mode='train', model='normal'):
        self.mode = mode
        self.model = model
        self.root_path = path
        self.file_list = os.listdir(path)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_face, label_face, label_face_onehot, points, RGB_face, cls = get_data_2(path=read_path)
        RGB_face = torch.from_numpy(RGB_face.astype(float))
        raw_points_face = points_face.copy()

        x_bias = random.uniform(-6, 6)
        y_bias = random.uniform(-8, 8)
        z_bias = random.uniform(-5, 5)
        theta = random.uniform(-np.pi*0.1, np.pi*0.1)
        Z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        if self.mode == 'train':
            points_face[:, 0] = points_face[:, 0] + x_bias
            points_face[:, 3] = points_face[:, 3] + x_bias
            points_face[:, 6] = points_face[:, 6] + x_bias
            points_face[:, 9] = points_face[:, 9] + x_bias
            points_face[:, 12] = points_face[:, 12] + x_bias
            points_face[:, 15] = points_face[:, 15] + x_bias
            points_face[:, 18] = points_face[:, 18] + x_bias
            points_face[:, 21] = points_face[:, 21] + x_bias

            points_face[:, 1] = points_face[:, 1] + y_bias
            points_face[:, 4] = points_face[:, 4] + y_bias
            points_face[:, 7] = points_face[:, 7] + y_bias
            points_face[:, 10] = points_face[:, 10] + y_bias
            points_face[:, 13] = points_face[:, 13] + y_bias
            points_face[:, 16] = points_face[:, 16] + y_bias
            points_face[:, 19] = points_face[:, 19] + y_bias
            points_face[:, 22] = points_face[:, 22] + y_bias

            points_face[:, 2] = points_face[:, 2] + z_bias
            points_face[:, 5] = points_face[:, 5] + z_bias
            points_face[:, 8] = points_face[:, 8] + z_bias
            points_face[:, 11] = points_face[:, 11] + z_bias
            points_face[:, 14] = points_face[:, 14] + z_bias
            points_face[:, 17] = points_face[:, 17] + z_bias
            points_face[:, 20] = points_face[:, 20] + z_bias
            points_face[:, 23] = points_face[:, 23] + z_bias

            points_face[:, :3] = points_face[:, :3].dot(Z.transpose())
            points_face[:, 3:6] = points_face[:, 3:6].dot(Z.transpose())
            points_face[:, 6:9] = points_face[:, 6:9].dot(Z.transpose())
            points_face[:, 9:12] = points_face[:, 9:12].dot(Z.transpose())
            points_face[:, 12:15] = points_face[:, 12:15].dot(Z.transpose())
            points_face[:, 15:18] = points_face[:, 15:18].dot(Z.transpose())
            points_face[:, 18:21] = points_face[:, 18:21].dot(Z.transpose())
            points_face[:, 21:24] = points_face[:, 21:24].dot(Z.transpose())

        # move all mesh to origin
        centre = points_face[:, 9:12].mean(axis=0)
        points_face[:, 0:3] -= centre
        points_face[:, 3:6] -= centre
        points_face[:, 6:9] -= centre
        points_face[:, 9:12] = (points_face[:, 0:3] + points_face[:, 3:6] + points_face[:, 6:9]) / 3
        points[:, :3] -= centre
        max = points.max()
        points_face[:, :12] = points_face[:, :12] / max

        # normalized data
        maxs = points[:, :3].max(axis=0)
        mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = points[:, :3].std(axis=0)
        nmeans = points[:, 3:].mean(axis=0)
        nstds = points[:, 3:].std(axis=0)
        nmeans_f = points_face[:, 21:].mean(axis=0)
        nstds_f = points_face[:, 21:].std(axis=0)
        for i in range(3):
            #normalize coordinate
            points_face[:, i] = (points_face[:, i] - means[i]) / stds[i]  # point 1
            points_face[:, i + 3] = (points_face[:, i + 3] - means[i]) / stds[i]  # point 2
            points_face[:, i + 6] = (points_face[:, i + 6] - means[i]) / stds[i]  # point 3
            points_face[:, i + 9] = (points_face[:, i + 9] - mins[i]) / (maxs[i] - mins[i])  # centre
            #normalize normal vector
            points_face[:, i + 12] = (points_face[:, i + 12] - nmeans[i]) / nstds[i]  # normal1
            points_face[:, i + 15] = (points_face[:, i + 15] - nmeans[i]) / nstds[i]  # normal2
            points_face[:, i + 18] = (points_face[:, i + 18] - nmeans[i]) / nstds[i]  # normal3
            points_face[:, i + 21] = (points_face[:, i + 21] - nmeans_f[i]) / nstds_f[i]  # face normal
        return index_face, points_face, label_face, label_face_onehot, self.file_list[item], raw_points_face, RGB_face, cls


class Group(nn.Module):  # FPS + KNN  FPS用于选取center KNN将选取每个center对应的块中的点
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 24   xyz: B：batch size   N: 点的个数
            ---------------------------
            output: B G M 24   M：group size（每个group的点的个数）
            center : B G 24
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size , self.num_group, self.group_size, 24).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center



if __name__ == "__main__":
    print(" ")
    index_face, points_face, label_face, label_face_onehot, points, RGB_face, _ = get_data_2('data/testL/067_L.ply')





