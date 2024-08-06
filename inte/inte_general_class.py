import numpy as np
from itertools import permutations
import pandas as pd

class utils:
    def __init__(self):
        pass

    def calculate_distances(self, converted_dataset):
        distances = []
        num_points = len(converted_dataset)

        # 对每一对点计算距离
        for i in range(num_points):
            for j in range(i + 1, num_points):  # 从i+1开始，避免重复计算和自身比较
                point1 = converted_dataset[i][0]
                point2 = converted_dataset[j][0]
                # 使用欧几里得距离公式计算两点之间的距离
                distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                distances.append((point1, point2, distance))
        return distances

    def extract_sudo_labels(self, merged_data):
        # 假设merged_data的格式是N x (2 + K)，其中N是数据点的数量，2是(x, y)坐标，K是标签的数量
        # 提取x和y坐标
        x_coords = merged_data[:, 0]
        y_coords = merged_data[:, 1]
        # 提取sudolabels（这里我们假设sudolabel是最后一个列）
        sudo_labels = merged_data[:, -1]

        # 构建列表[((x, y), sudolabel), ...]
        sudo_label_dataset = [((x, y), label) for x, y, label in zip(x_coords, y_coords, sudo_labels)]

        return sudo_label_dataset

    def merge_datasets(self, original_dataset, pseudo_label_dataset):
        pseudo_label_dataset = self.extract_sudo_labels(pseudo_label_dataset)
        # 将原始数据集转换为字典，以便于根据(x, y)查找
        original_dict = {k: v for (k, v) in original_dataset}

        # 合并数据集
        merged_dataset = []
        for (xy, sudolabel) in pseudo_label_dataset:
            # 如果在原始数据集中找到了对应的(x, y)，则合并
            if xy in original_dict:
                label = original_dict[xy]
                merged_dataset.append((xy, (label, sudolabel)))
            else:
                # 如果没有找到对应的(x, y)，则只添加伪标签
                merged_dataset.append((xy, (None, sudolabel)))
        # self.merged_dataset = merged_dataset
        return merged_dataset

    def maximize_intersection(self, real_labels, pseudo_labels):
        # 获取标签的唯一值
        unique_real_labels = np.unique(real_labels)
        unique_pseudo_labels = np.unique(pseudo_labels)

        # 计算所有可能的标签映射
        best_mapping = None
        max_intersection = -1
        for perm in permutations(unique_real_labels):
            # 创建一个映射字典
            mapping = dict(zip(unique_pseudo_labels, perm))
            # 应用映射
            mapped_pseudo_labels = np.vectorize(mapping.get)(pseudo_labels)
            # 计算交集
            intersection = np.sum(mapped_pseudo_labels == real_labels)
            # 更新最佳映射
            if intersection > max_intersection:
                max_intersection = intersection
                best_mapping = mapping

        # 应用最佳映射
        adjusted_pseudo_labels = np.vectorize(best_mapping.get)(pseudo_labels)
        return adjusted_pseudo_labels

    def adjust_sudo_labels(self, data):
        # 将数据转换为 DataFrame
        df = pd.DataFrame(data, columns=['xy', 'labels'])

        # 分解 'xy' 和 'labels' 列
        df[['x', 'y']] = pd.DataFrame(df['xy'].tolist(), index=df.index)
        df[['real_label', 'pseudo_label']] = pd.DataFrame(df['labels'].tolist(), index=df.index)

        # 删除原始的 'xy' 和 'labels' 列
        df.drop(['xy', 'labels'], axis=1, inplace=True)

        # 提取真实标签和伪标签
        real_labels = df['real_label'].values
        pseudo_labels = df['pseudo_label'].values

        # 调用 maximize_intersection 方法来调整伪标签
        adjusted_pseudo_labels = self.maximize_intersection(real_labels, pseudo_labels)

        # 将调整后的伪标签更新到 dataframe 中
        df['pseudo_label'] = adjusted_pseudo_labels

        # 将 DataFrame 转换回原始格式
        result = [(x, y, real_label, pseudo_label) for x, y, real_label, pseudo_label in
                  zip(df['x'], df['y'], df['real_label'], df['pseudo_label'])]
        return result

    def calculate_match_ratio_updated(self, data):
        """
        Calculate the proportion of (real_label, sudo_label) pairs that are matched in the given data.

        Args:
        data (list of tuples): A list where each element is a tuple containing (x, y, real_label, sudo_label).

        Returns:
        float: The proportion of matched (real_label, sudo_label) pairs.
        """
        match_count = sum(1 for _, _, real, sudo in data if real == sudo)
        total = len(data)
        return match_count / total if total != 0 else 0


class GNMR_calculator:
    def __init__(self, data, norm_factor=2):
        self.data_GNMR = data
        self.norm_factor = norm_factor

    def phi(self, sudo_label, real_label):
        return sum(1 for s, r in zip(sudo_label, real_label) if s != r)

    def calculate_GNMR(self):
        # 将数据按照real_label和sudo_label分组
        real_labels_dict = {}
        sudo_labels_dict = {}
        for x, y, real_label, sudo_label in self.data_GNMR:
            real_labels_dict.setdefault(real_label, []).append(real_label)
            sudo_labels_dict.setdefault(real_label, []).append(sudo_label)

        # 获取类别总数P
        P = len(real_labels_dict)
        total_mismatch = 0

        # 遍历每个类别计算不匹配率
        for p in real_labels_dict.keys():
            L_p_p = real_labels_dict[p]
            L_s_p = sudo_labels_dict[p]
            A_mp = len(L_p_p)  # 当前类别的数据总数

            # 计算数字不匹配率
            num_mismatch = abs(len(L_p_p) - len(L_s_p)) / (len(L_p_p))

            # 计算标签不匹配率
            label_mismatch = self.phi(L_s_p, L_p_p) / A_mp

            # 将两种不匹配率的平方求和
            total_mismatch += (num_mismatch + label_mismatch) ** 2

        # 计算GNMR
        G = ((total_mismatch / P) ** 0.5) / (P ** 0.5)
        # print(G)
        out_GNMR_pos = max((1 - G * self.norm_factor), 0)
        out_GNMR_neg = min(G*self.norm_factor, 1)
        return out_GNMR_pos,  # from 0-1, 1 stands for Good

class DDR_calculator:
    def __init__(self, data, distance_table, norm_factor=2):
        self.distance_table = distance_table
        self.distance_dict = {tuple(sorted((entry[0], entry[1]))): entry[2] for entry in distance_table}
        self.data_DDR = data
        self.norm_factor = norm_factor

    def lookup_distance(self, point1, point2):
        if point1 == point2:
            return float(0)
        key = tuple(sorted((point1, point2)))
        # 直接从字典中获取距离
        return self.distance_dict.get(key, None)

    def euclidean_distance(self, point1, point2):
        # print("point1 & point2 in line 168: ", point1, point2)
        # point1 & point2 in line 168:  (9.20208106894033, 11.513134708164444) (9.78392178336755, 10.813515674961184)
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def gamma_function(self, p, q, i, j, f, pseudo_labels):
        if f == 0:
            return pseudo_labels[i] == p and pseudo_labels[j] == p
        else:
            return pseudo_labels[i] == p and pseudo_labels[j] == q and p != q

    def calculate_DDR(self):
        data = self.data_DDR
        coordinates = [(x, y) for x, y, _, _ in data]
        pseudo_labels = [pseudo_label for _, _, _, pseudo_label in data]
        unique_pseudo_labels = list(set(pseudo_labels))
        P = len(unique_pseudo_labels)

        numerator = 0
        denominator = 0
        num_pairs_diff_classes = 0
        num_pairs_same_class = {p: 0 for p in unique_pseudo_labels}

        for p in unique_pseudo_labels:
            for q in unique_pseudo_labels:
                if p != q:
                    for i in range(len(data)):
                        for j in range(len(data)):
                            d_ij = self.lookup_distance(coordinates[i], coordinates[j])
                            if self.gamma_function(p, q, i, j, 1, pseudo_labels):
                                numerator += d_ij
                                num_pairs_diff_classes += 1
                            elif self.gamma_function(p, q, i, j, 0, pseudo_labels):
                                denominator += d_ij
                                num_pairs_same_class[p] += 1

        # Average the distances
        average_numerator = numerator / num_pairs_diff_classes if num_pairs_diff_classes > 0 else 0
        average_denominator = sum(
            denominator / num_pairs_same_class[p] for p in unique_pseudo_labels if num_pairs_same_class[p] > 0)

        # Calculate DDR
        DDR = average_numerator / average_denominator if average_denominator > 0 else float('inf')
        out_DDR_pos = min(DDR / self.norm_factor, 1)
        org_DDR = DDR
        return out_DDR_pos

class DC_calculator:
    def __init__(self, data, distance_table, norm_factor=1):
        self.distance_table = distance_table
        self.distance_dict = {tuple(sorted((entry[0], entry[1]))): entry[2] for entry in distance_table}
        self.data_DR = data
        self.groups = self.group_by_sudo_label(self.data_DR)
        self.norm_factor = norm_factor

    def group_by_sudo_label(self, data):
        groups = {}
        for x, y, real_label, sudo_label in data:
            if sudo_label not in groups:
                groups[sudo_label] = []
            groups[sudo_label].append((x, y))
        return groups

    def lookup_distance(self, point1, point2):
        if point1 == point2:
            return float(0)
        key = tuple(sorted((point1, point2)))
        # 直接从字典中获取距离
        return self.distance_dict.get(key, None)

    def euclidean_distance(self, point1, point2):
        # print("point1 & point2 in line 229: ", point1, point2)
        # point1 & point2 in line 229:  (9.318560237172958, 10.75399013166582) (9.710463130877365, 10.977509763694387)
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_average_distance(self, group):
        num_points = len(group)
        if num_points <= 1:
            return 0
        total_distance = 0
        for j in range(num_points):
            for k in range(num_points):
                if j != k:
                    total_distance += self.lookup_distance(group[j], group[k])
        return total_distance / (num_points * (num_points - 1))

    def calculate_dc_temp(self):
        P = len(self.groups)
        sum_of_averages = sum(self.calculate_average_distance(group) for group in self.groups.values())
        return sum_of_averages / P

    def calculate_DC(self):
        return min(self.calculate_dc_temp() / self.norm_factor, 1)

# Class definition for OODRCalculator
class OODRCalculator:
    def __init__(self, data, norm_factor=6):
        self.data_OODR = data
        self.norm_factor = norm_factor

    def calculate_OODR(self):
        P = len(set([real_label for _, _, real_label, _ in self.data_OODR]))
        sum_OODR = 0
        for p in range(P):
            A_mcs = set()
            A_mp = set()
            for x, y, real_label, sudo_label in self.data_OODR:
                if real_label == p:
                    A_mcs.add((x, y))
                if sudo_label == p:
                    A_mp.add((x, y))
            if len(A_mcs) > 0:
                sum_OODR += (len(A_mcs) - len(A_mcs.intersection(A_mp))) / len(A_mcs)
        temp = P / sum_OODR if sum_OODR!=0 else 5   # All correct
        OODR_pos = min(temp/self.norm_factor, 1)
        return OODR_pos


class inte_core:
    def __init__(self, data_with_2_labels, distance_table, norm_factor_GNMR=2, norm_factor_DDR=4, norm_factor_DC=2, norm_factor_OODR=6):
        self.distance_table = distance_table
        self.distance_dict = {tuple(sorted((entry[0], entry[1]))): entry[2] for entry in distance_table}
        self.data = data_with_2_labels
        print("data_with_2_labels: ", data_with_2_labels[0])
        self.norm_factor_GNMR, self.norm_factor_DDR, self.norm_factor_DC, self.norm_factor_OODR =\
            norm_factor_GNMR, norm_factor_DDR, norm_factor_DC, norm_factor_OODR

    def inte_calculation(self):
        gnmr = GNMR_calculator(data=self.data, norm_factor=self.norm_factor_GNMR)
        ddr = DDR_calculator(data=self.data, norm_factor=self.norm_factor_DDR, distance_table=self.distance_table)
        dc = DC_calculator(data=self.data, norm_factor=self.norm_factor_DC, distance_table=self.distance_table)
        oodr = OODRCalculator(data=self.data, norm_factor=self.norm_factor_OODR)

        result = {"gnmr": float(gnmr.calculate_GNMR()[0]), "ddr": float(ddr.calculate_DDR()),
                  "dc": float(dc.calculate_DC()), "oodr": float(oodr.calculate_OODR())}

        return result

    def filter_inconsistent_data_to_dataframe(self):
        """
        Filter out data where real label and sudo label are inconsistent and return a DataFrame.

        Args:
        data (list of tuples): List where each element is a tuple containing ((sentence1, sentence2), (real_label, sudo_label)).

        Returns:
        pandas.DataFrame: DataFrame containing only the elements where real_label and sudo_label are different,
                          with columns 'sentence1', 'sentence2', 'priori labeling', 'sudo labeling'.
        """
        inconsistent_data = [((x, y), (rl, sl)) for (x, y, rl, sl) in self.data if rl != sl]
        if len(inconsistent_data) == 0:
            inconsistent_data = [((None, None), (None, None))]
        # Convert to DataFrame
        df = pd.DataFrame(inconsistent_data, columns=['sentence pair', 'labels'])
        df[['sentence1', 'sentence2']] = pd.DataFrame(df['sentence pair'].tolist(), index=df.index)
        df[['priori labeling', 'sudo labeling']] = pd.DataFrame(df['labels'].tolist(), index=df.index)
        df.drop(columns=['sentence pair', 'labels'], inplace=True)

        return df

    # 典型性top k个点的函数
    def find_typical_points_per_sudo_label(self, k):
        data = self.data
        distance_dict = self.distance_dict
        sudo_label_groups = {}
        for x, y, real_label, sudo_label in data:
            if sudo_label not in sudo_label_groups:
                sudo_label_groups[sudo_label] = []
            sudo_label_groups[sudo_label].append(((x, y), sudo_label))

        typical_points = []
        for sudo_label, points in sudo_label_groups.items():
            avg_distances = []
            for point, label in points:
                total_distance = 0
                count = 0
                for other_point, _ in points:
                    if point != other_point:
                        key = tuple(sorted((point, other_point)))
                        total_distance += distance_dict.get(key, float('inf'))
                        count += 1
                avg_distance = total_distance / count if count > 0 else float('inf')
                avg_distances.append((point, sudo_label, avg_distance))

            avg_distances.sort(key=lambda x: x[2])
            typical_points.extend(avg_distances[:k])

        # 创建DataFrame
        df_typical = pd.DataFrame(typical_points, columns=['Point', 'sudoLabel', 'Average Distance'])
        df_typical['Table'] = 'Typical'

        return df_typical

    # 奇特性top k个点的函数，每个sudo_label返回top k个样本
    def find_unique_points_per_sudo_label(self, k):
        data = self.data
        distance_dict = self.distance_dict
        sudo_label_groups = {}
        for x, y, real_label, sudo_label in data:
            if sudo_label not in sudo_label_groups:
                sudo_label_groups[sudo_label] = []
            sudo_label_groups[sudo_label].append(((x, y), sudo_label))

        unique_points = []
        for sudo_label, points in sudo_label_groups.items():
            avg_distances = []
            for point, label in points:
                total_distance = 0
                for other_point, _ in points:
                    if point != other_point:
                        key = tuple(sorted((point, other_point)))
                        total_distance += distance_dict.get(key, float('inf'))
                avg_distance = total_distance / (len(points) - 1) if len(points) > 1 else float('inf')
                avg_distances.append((point, label, avg_distance))

            # 按平均距离降序排序，并取top k个点
            avg_distances.sort(key=lambda x: x[2], reverse=True)
            unique_points.extend(avg_distances[:k])

        # 创建DataFrame
        df_unique = pd.DataFrame(unique_points, columns=['Point', 'sudoLabel', 'Average Distance'])
        df_unique['Table'] = 'Unique'

        return df_unique

