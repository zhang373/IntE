import numpy as np
from Build_Test_Dataset import generate_and_modify_dataset
import math
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, MeanShift
import random
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from inte_general_class import utils
from scipy.stats import mode

class Clusterer:
    def __init__(self, distance_dataset, num_clusters):
        self.distance_dataset = distance_dataset
        self.num_clusters = num_clusters
        self.points = self.extract_unique_points()

    def extract_unique_points(self):
        points = set()
        for ((x1, y1), (x2, y2), distance) in self.distance_dataset:
            points.add((x1, y1))
            points.add((x2, y2))
        return np.array(list(points))

    def kmeans_clustering(self):
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(self.points)
        return kmeans.labels_

    def hierarchical_clustering(self):
        hierarchical = AgglomerativeClustering(n_clusters=self.num_clusters)
        return hierarchical.fit_predict(self.points)

    def dbscan_clustering(self):
        # DBSCAN does not take num_clusters as an argument, so we use a different approach
        # You may need to adjust eps and min_samples based on your dataset
        eps = 0.5  # Example value, adjust based on your dataset
        min_samples = 10  # Example value, adjust based on your dataset
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(self.points)

    def spectral_clustering(self):
        spectral = SpectralClustering(n_clusters=self.num_clusters, affinity='nearest_neighbors')
        return spectral.fit_predict(self.points)

    def mean_shift_clustering(self, initial_bandwidth=0.5, bandwidth_increment=0.1):
        # MeanShift does not take num_clusters as an argument, so we use a bandwidth
        # and adjust it until the number of clusters equals num_clusters.
        bandwidth = initial_bandwidth
        while True:
            mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            mean_shift.fit(self.points)
            labels = mean_shift.labels_
            unique_labels = np.unique(labels)
            num_clusters_found = len(unique_labels) - (
                1 if -1 in unique_labels else 0)  # Exclude noise label if present
            if num_clusters_found == self.num_clusters:
                break
            elif num_clusters_found < self.num_clusters:
                # If we find fewer clusters than expected, increase the bandwidth.
                bandwidth += bandwidth_increment
            else:
                # If we find more clusters than expected, we continue to increase bandwidth
                # in the hope that some clusters will merge.
                bandwidth += bandwidth_increment
        return labels

    def fuzzy_cmeans_clustering(self, m=2):
        # 计算距离矩阵
        distances = pairwise_distances(self.points, metric='euclidean')
        # 标准化距离矩阵
        distances = StandardScaler().fit_transform(distances)
        # 执行模糊C-均值聚类
        fcm = KMeans(n_clusters=self.num_clusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        fcm.fit(distances)
        fcm.fit(distances)
        # 返回聚类结果
        return fcm.labels_

    def remap_labels(self, labels):
        reflection = {}
        for item in labels:
            if item not in reflection.keys():
                reflection[item] = len(reflection.keys())
        for index in range(len(labels)):
            labels[index] = reflection[labels[index]]
        return labels

    def merge_cluster_labels(self):
        # 获取各个聚类方法的结果并重新映射标签
        kmeans_labels = self.remap_labels(self.kmeans_clustering())
        hierarchical_labels = self.remap_labels(self.hierarchical_clustering())
        #dbscan_labels = self.remap_labels(self.dbscan_clustering())
        spectral_labels = self.remap_labels(self.spectral_clustering())
        mean_shift_labels = self.remap_labels(self.mean_shift_clustering())
        fuzzy_cmeans_labels = self.remap_labels(self.fuzzy_cmeans_clustering())
        label_list = ["kmeans_labels", "hierarchical_labels", "spectral_labels", "mean_shift_labels", "fuzzy_cmeans_labels"]
        # 将所有标签合并到一个矩阵中
        all_labels = np.vstack([
            kmeans_labels,
            hierarchical_labels,
            # dbscan_labels,
            spectral_labels,
            mean_shift_labels,
            fuzzy_cmeans_labels
        ]).T

        # 将点与标签合并
        merged_data = np.hstack([self.points, all_labels])
        self.merge_dataset = merged_data
        # 返回合并后的数据
        return merged_data, label_list

    def call_forest_result(self):
        # 检查self.merge_dataset是否存在
        if not hasattr(self, 'merge_dataset'):
            raise ValueError("merge_dataset does not exist. Please run merge_cluster_labels first.")

        # 获取除了点坐标之外的所有标签列
        labels_columns = self.merge_dataset[:, -self.merge_dataset.shape[1] + 2:]

        # 使用mode函数找到每一行中出现频率最高的标签
        most_common_labels = mode(labels_columns, axis=1).mode.flatten()

        # 在self.merge_dataset最后添加一个新列，存放出现频率最高的标签
        self.merge_dataset = np.hstack([self.merge_dataset, most_common_labels.reshape(-1, 1)])

        # 计算每个聚类算法与最终投票结果之间一致的比例
        agreement_rates = []
        for i in range(labels_columns.shape[1]):
            agreement = np.sum(labels_columns[:, i] == most_common_labels) / len(most_common_labels)
            agreement_rates.append(agreement)

        # 返回更新后的merge_dataset和每个聚类算法的一致率
        return self.merge_dataset, agreement_rates


# Test Cluster Demo

# 生成随机数据集用于测试
def generate_random_dataset(num_points, num_clusters):
    random.seed(0)  # 为了可重复性设置随机种子
    centers = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(num_clusters)]
    dataset = []
    for _ in range(num_points):
        center = random.choice(centers)
        x, y = random.gauss(center[0], 1), random.gauss(center[1], 1)
        # 为了简单起见，这里我们不计算真实的距离，而是生成一个随机的距离
        distance = random.uniform(0, 20)
        dataset.append((center, (x, y), distance))
    return dataset


if __name__ == '__main__':
    # build sysc dataset
    num_classes = 3
    class_centers = [[0, 0], [20, 20], [40, 40]]
    num_points_per_class = 100
    converted_dataset = generate_and_modify_dataset(num_classes=num_classes,
                                                    class_centers=class_centers,
                                                    num_points_per_class=num_points_per_class,
                                                    randomness_strength=0.1,
                                                    error_rate=0.3,
                                                    error_strength=2)
    print(f"Total len of sysc dataset: {len(converted_dataset)}, the type: {type(converted_dataset)}, the formate: {converted_dataset[0]}")


    utils = utils()
    distances = utils.calculate_distances(converted_dataset)
    if len(distances) != math.comb(num_points_per_class*num_classes, 2):
        raise ValueError("There are some error incured in cal distance")
    print(f"Total len of sysc dataset: {len(distances)}, the type: {type(distances)}, the formate: {distances[0]}")



    # 测试代码
    num_points = 100  # 数据集中点的数量
    num_clusters = 3  # 期望的聚类数量
    # # 生成随机数据集
    # distance_dataset = generate_random_dataset(num_points, num_clusters)
    # print(f"Total len of sysc dataset in test demo: {len(distance_dataset)}, the type: {type(distance_dataset)}, the formate: {distance_dataset[0]}")

    # 创建Clusterer实例
    clusterer = Clusterer(distances, num_clusters)

    # 使用所有聚类方法
    kmeans_labels = clusterer.kmeans_clustering()
    hierarchical_labels = clusterer.hierarchical_clustering()
    dbscan_labels = clusterer.dbscan_clustering()
    spectral_labels = clusterer.spectral_clustering()
    mean_shift_labels = clusterer.mean_shift_clustering()
    clusterer_fuzzy_cmeans_clustering_label = clusterer.fuzzy_cmeans_clustering()
    # clusterer_k_medoids_clustering_label = clusterer.k_medoids_clustering()

    # 打印聚类结果
    print("K-Means Labels:", kmeans_labels)
    print("Hierarchical Labels:", hierarchical_labels)
    print("DBSCAN Labels:", dbscan_labels)
    print("Spectral Labels:", spectral_labels)
    print("Mean Shift Labels:", mean_shift_labels)
    print("clusterer_fuzzy_cmeans_clustering_label",clusterer_fuzzy_cmeans_clustering_label)
    # print("clusterer_k_medoids_clustering_label",clusterer_k_medoids_clustering_label)
    data_before_forest, label_list = clusterer.merge_cluster_labels()
    sudo_data, voting_percentage = clusterer.call_forest_result()

    print(voting_percentage)
    print("We finished")
