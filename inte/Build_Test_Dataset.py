import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm, t, uniform, expon, binom
import unittest
import copy

class DatasetGenerator:
    def __init__(self, num_classes, distribution='normal', params=None):
        self.num_classes = num_classes
        self.distribution = distribution
        self.params = params if params else {'loc': 0, 'scale': 1, 'df': 5, 'randomness': 0.5, 'error_shift': 3}
        self.datasets = []
        self.class_centers = np.array([[0, 0]] * num_classes)  # 初始化每个类的中心点
        self.num_points_per_class = 100  # Default number of points per class
        self.randomness_strength = 0.1  # Default randomness strength
        self.error_strength = 0.1  # Default error strength

    def set_distribution(self, distribution, params):
        self.distribution = distribution
        self.params = params

    def set_class_centers(self, centers):
        self.class_centers = np.array(centers)

    def generate_points(self, num_points, class_label):
        params = self.params.copy()
        params['loc'] = self.class_centers[class_label]

        if self.distribution == 'normal':
            points = np.random.normal(params['loc'], params['scale'], (num_points, 2))
        elif self.distribution == 't':
            points = np.random.standard_t(params['df'], (num_points, 2)) + params['loc']
        elif self.distribution == 'uniform':
            points = np.random.uniform(params['loc'] - params['scale'], params['loc'] + params['scale'],
                                       (num_points, 2))
        elif self.distribution == 'exponential':
            points = np.random.exponential(params['scale'], (num_points, 2)) + params['loc']
        elif self.distribution == 'binomial':
            points = np.random.binomial(params['n'], params['p'], (num_points, 2)) + params['loc']
        else:
            raise ValueError("Unsupported distribution type")

        # 这里我们不添加随机性，只生成点集
        points = np.hstack((points, np.full((num_points, 1), class_label)))
        return points

    def add_randomness(self, points, randomness):
        # 只对前两列（坐标）添加随机性，标签列（第三列）保持不变
        points[:, :2] += np.random.uniform(-randomness, randomness, (points.shape[0], 2))
        return points

    def generate_datasets(self, num_points_per_class=None):
        if num_points_per_class is not None:
            self.num_points_per_class = num_points_per_class
        for class_label in range(self.num_classes):
            points = self.generate_points(self.num_points_per_class, class_label)
            self.datasets.append(points)
        self.datasets = np.vstack(self.datasets)

    def introduce_errors(self, error_rate, error_strength, out_source_data=None):
        if out_source_data is None:
            data = self.datasets
        else:
            data = out_source_data
        if error_strength is not None:
            self.error_strength = error_strength
        num_errors = int(error_rate * data.shape[0])
        error_indices = np.random.choice(data.shape[0], num_errors, replace=False)
        error_points = data[error_indices]

        for i in error_indices:
            while True:
                shift = np.random.normal(0, self.error_strength * self.params['error_shift'], 2)
                new_point = data[i, :2] + shift
                if not np.any(np.all(data == np.append(new_point, data[i, 2]), axis=1)):
                    data[i, :2] = new_point
                    break
        if out_source_data is None:
            self.datasets = data
        return data

    # Original plot function
    # def plot_datasets(self, original_datasets=None, title='Dataset', save_path=None):
    #     plt.figure(figsize=(10, 10))
    #     if original_datasets is not None:
    #         plt.scatter(original_datasets[:, 0], original_datasets[:, 1], c='blue', label='Original Points')
    #         plt.scatter(self.datasets[:, 0], self.datasets[:, 1], c='red', label='Modified Points')
    #     else:
    #         plt.scatter(self.datasets[:, 0], self.datasets[:, 1], c='green', label='Generated Points')
    #     plt.title(title)  # 添加图表标题
    #     plt.legend()
    #     if save_path:
    #         plt.savefig(save_path)  # 保存图片到指定路径
    #         print(f"Plot saved to {save_path}")
    #     plt.show()

    def plot_datasets(self, original_datasets=None, updated_datasets=None, title='Dataset', save_path=None):
        plt.figure(figsize=(10, 10))
        if original_datasets is not None:
            cmap = cm.get_cmap('viridis', self.num_classes)
            for class_label in range(self.num_classes):
                # 使用 NumPy 将列表转换为数组
                original_datasets_array = np.array(original_datasets)
                # 找到原始数据集中没有的点的索引
                different_points = self.find_non_overlapping_points(original_datasets, updated_datasets)
                print("Current percentage: ", len(different_points)/len(updated_datasets))
                if len(different_points) == 0:
                    pass
                else:
                    # 将 different_points 转换为 NumPy 数组
                    different_points_array = np.array(different_points)
                    # 绘制不同的点
                    plt.scatter(different_points_array[:, 0], different_points_array[:, 1],
                                c=cmap(class_label), label=f'Modified Points (Class {class_label})', marker='x')
                # 绘制原始数据集中的点
                plt.scatter(original_datasets_array[class_label * self.num_points_per_class:(class_label + 1) * self.num_points_per_class, 0], original_datasets_array[class_label * self.num_points_per_class:(class_label + 1) * self.num_points_per_class, 1], c=cmap(class_label), label=f'Original Points (Class {class_label})', marker='o')
        else:
            cmap = cm.get_cmap('viridis', self.num_classes)
            for class_label in range(self.num_classes):
                points = np.array(self.datasets[class_label * self.num_points_per_class:(class_label + 1) * self.num_points_per_class])
                plt.scatter(points[:, 0], points[:, 1], c=cmap(class_label), label=f'Generated Points (Class {class_label})', marker='o')
        plt.title(title)  # 添加图表标题
        plt.legend()
        if save_path:
            plt.savefig(save_path)  # 保存图片到指定路径
            print(f"Plot saved to {save_path}")
        plt.show()

    def find_non_overlapping_points(self, list1, list2):
        # 将list1中的点转换为集合，便于快速查找
        points_set = {(point[0], point[1]) for point in list1}
        # 过滤list2，只保留不在list1中的点
        non_overlapping_points = [point for point in list2 if (point[0], point[1]) not in points_set]
        return non_overlapping_points

    def convert_dataset_format(self, out_source_data=None):
        if out_source_data is None:
            # Convert dataset format from [[x1, y1, class_label1]] to [[(x1, y1), class_label1]]
            converted_datasets = [((x, y), label) for x, y, label in self.datasets]
            return converted_datasets
        else:
            converted_datasets = [((x, y), label) for x, y, label in out_source_data]
            return converted_datasets

    def get_dataset(self):
        return self.datasets



def generate_and_modify_dataset(num_classes, class_centers, num_points_per_class=3, randomness_strength=0.1, error_rate=1, error_strength=0.1, draw_not=False):
    if num_classes != len(class_centers):
        raise ValueError(f"You should keep num_classes {num_classes} and len(class_centers) {len(class_centers)} the same length!")
    # 初始化数据集生成器
    generator = DatasetGenerator(num_classes=num_classes)
    generator.set_class_centers(class_centers)

    # 1. 生成一个基础数据集
    generator.generate_datasets(num_points_per_class=num_points_per_class)
    base_dataset = generator.get_dataset()

    # 2. 画出这个数据集
    if draw_not:
        generator.plot_datasets(title="Original Dataset")

    # 3. 向这个数据集增添小范围随机性
    generator.add_randomness(base_dataset, randomness_strength)
    modified_dataset_with_randomness = copy.deepcopy(generator.datasets)


    # 4. 画出修改完之后的数据集
    if draw_not:
        generator.plot_datasets(title="Dataset after adding Randomness")

    # 5. 增加error样本
    error_dataset = generator.introduce_errors(error_rate, error_strength)
    if draw_not:
        generator.plot_datasets(title="Dataset after adding error")

    # 6. 画出修改完之后的数据集
    if draw_not:
        generator.plot_datasets(original_datasets=modified_dataset_with_randomness, updated_datasets=error_dataset, title="Dataset after adding error")

    # 7. 修改数据集格式并返回数据
    converted_dataset = generator.convert_dataset_format()

    return converted_dataset

class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = DatasetGenerator(num_classes=3)
        self.generator.set_class_centers([[0, 0], [5, 5], [10, 10]])

    def test_generate_datasets(self):
        self.generator.generate_datasets(num_points_per_class=50)
        dataset = self.generator.get_dataset()
        self.assertEqual(dataset.shape[0], 150)  # 3 classes * 50 points each
        self.assertEqual(dataset.shape[1], 3)  # x, y, class_label

    def test_generate_points(self):
        points = self.generator.generate_points(10, 0)
        self.assertEqual(points.shape[0], 10)
        self.assertEqual(points.shape[1], 3)
        self.assertTrue(np.all(points[:, 2] == 0))  # Check class label

    def test_introduce_errors(self):
        self.generator.generate_datasets(num_points_per_class=50)
        self.generator.introduce_errors(error_rate=0.1, error_strength=1)
        dataset = self.generator.get_dataset()


    def test_convert_dataset_format(self):
        self.generator.generate_datasets(num_points_per_class=50)
        converted_dataset = self.generator.convert_dataset_format()
        self.assertEqual(len(converted_dataset), 150)  # 3 classes * 50 points each
        self.assertEqual(type(converted_dataset[0]), tuple)
        self.assertEqual(len(converted_dataset[0]), 2)

    def test_plot_datasets(self):
        # This test will only check if the plot function runs without errors
        self.generator.generate_datasets(num_points_per_class=50)
        self.generator.plot_datasets()
        # Since this is a visual test, no assertion is made here

    def test_unsupported_distribution(self):
        with self.assertRaises(ValueError):
            self.generator.set_distribution('unsupported', {})
            self.generator.generate_datasets(num_points_per_class=50)

# Run the tests
if __name__ == '__main__':
    # unittest.main(argv=[''], exit=False)
    # 使用示例
    num_classes = 3
    class_centers = [[0, 0], [5, 5], [10, 10]]
    converted_dataset = generate_and_modify_dataset(num_classes=num_classes,
                                                    class_centers=class_centers,
                                                    num_points_per_class=100,
                                                    randomness_strength=0.1,
                                                    error_rate=0.3,
                                                    error_strength=2)