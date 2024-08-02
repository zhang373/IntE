import math
from cluster import Clusterer
from inte_general_class import utils, inte_core
from Build_Test_Dataset import DatasetGenerator
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import json

# num_classes, class_centers, num_points_per_class=3, randomness_strength=0.1, error_rate=1, error_strength=0.1, draw_not=False
def generate_datasets_for_all_distributions(num_classes, distributions, params=None, num_points_per_class=100,
                                            randomness_strength=0.1, error_rate=1, error_strength=0.1, draw_not=False,
                                            initial_center_spread=20, center_change_base=2, scale_change_base = 2,
                                            num_samples=5, centers=None, rand_mode=False):
    datasets_dict = {}

    for distribution in distributions:
        if draw_not:
            print(f"\n\nWe are currently processing the distribution: {distribution}")
        # 存储当前分布的四组数据，每组数据包含num_samples个采样
        datasets = [[] for _ in range(4)]
        # params = {'loc': 0, 'scale': 2, 'df': 2, 'randomness': 0.5, 'error_shift': 3}  # It is given here
        generator = DatasetGenerator(num_classes, distribution, params)
        if centers !=None:
            if len(centers) != num_classes:
                raise ValueError("you should keep centers the same number with num_class!")
            initial_centers = centers
        else:
            initial_centers = np.array([np.random.rand(2) * initial_center_spread for _ in range(num_classes)])
        initial_scale = params['scale'] if 'scale' in params else 1
        randomness_init = params['randomness']
        error_shift_init = params['error_shift']
        initial_df = params['df'] if 'df' in params else 5  # 确保t分布的自由度参数被设置

        # 第一组：中心距离逐渐靠近
        if draw_not:
            print("\nWe are processing the first list! Center get closer while shift stays the same\n")
        for i in range(num_samples):
            if rand_mode:
                center_spread = initial_center_spread / (center_change_base ** i)
                generator.class_centers = np.array([np.random.rand(2) * center_spread for _ in range(num_classes)])
                    #[initial_centers[0] + np.random.rand(2) * center_spread for _ in range(num_classes)])
            else:
                generator.class_centers = initial_centers/(center_change_base ** i)
            generator.set_distribution(distribution, {'loc': 0, 'scale': initial_scale, 'df': initial_df, 'randomness': randomness_init, 'error_shift': error_shift_init})
            generator.datasets = []

            cur_parm_print = {'loc': 0, 'scale': initial_scale, 'df': initial_df}
            if draw_not:
                print(f"Currently dealing with distribution: {distribution}, center is :\n {generator.class_centers}, \nand the para is: \n{cur_parm_print}")

            generator.generate_datasets(num_points_per_class)
            modified_dataset_with_randomness = copy.deepcopy(generator.add_randomness(copy.deepcopy(generator.datasets), randomness_strength))
            error_dataset = generator.introduce_errors(error_rate, error_strength)
            converted_dataset = generator.convert_dataset_format()
            datasets[0].append(converted_dataset)

        # 第二组：中心距离保持不变，离散程度逐渐扩大
        if draw_not:
            print("\nWe are processing the second list! Center stays same while shift grow up\n")
        generator.set_class_centers(initial_centers)
        for i in range(num_samples):
            scale = initial_scale * (scale_change_base ** i)
            df = initial_df * (scale_change_base ** i)
            generator.set_distribution(distribution, {'loc': 0, 'scale': scale, 'df': df, 'randomness': randomness_init, 'error_shift': error_shift_init})
            generator.datasets = []
            if rand_mode:
                generator.class_centers = np.array([np.random.rand(2) * initial_center_spread for _ in range(num_classes)])
            else:
                generator.class_centers = initial_centers

            cur_parm_print = {'loc': 0, 'scale': scale, 'df': df}
            if draw_not:
                print(f"Currently dealing with distribution: {distribution}, center is :\n {generator.class_centers}, \nand the para is: \n{cur_parm_print}")

            generator.generate_datasets(num_points_per_class)
            modified_dataset_with_randomness = copy.deepcopy(generator.add_randomness(copy.deepcopy(generator.datasets), randomness_strength))
            error_dataset = generator.introduce_errors(error_rate, error_strength)
            converted_dataset = generator.convert_dataset_format()
            datasets[1].append(converted_dataset)

        # 第三组：中心距离逐渐减小，离散程度逐渐变大
        if draw_not:
            print("\nWe are processing the third list! Center get closer while shift grows up\n")
        for i in range(num_samples):
            if rand_mode:
                center_spread = initial_center_spread / (center_change_base ** i)
                generator.class_centers = np.array([np.random.rand(2) * center_spread for _ in range(num_classes)])
                    #[initial_centers[0] + np.random.rand(2) * center_spread for _ in range(num_classes)])
            else:
                generator.class_centers = initial_centers/(center_change_base ** i)
            scale = initial_scale * (scale_change_base ** i)
            df = initial_df * (scale_change_base ** i)
            generator.set_distribution(distribution, {'loc': 0, 'scale': scale, 'df': df, 'randomness': randomness_init, 'error_shift': error_shift_init})
            generator.datasets = []

            cur_parm_print = {'loc': 0, 'scale': scale, 'df': df}
            if draw_not:
                print(f"Currently dealing with distribution: {distribution}, center is :\n {generator.class_centers}, \nand the para is: \n{cur_parm_print}")

            generator.generate_datasets(num_points_per_class)
            modified_dataset_with_randomness = copy.deepcopy(generator.add_randomness(copy.deepcopy(generator.datasets), randomness_strength))
            error_dataset = generator.introduce_errors(error_rate, error_strength)
            converted_dataset = generator.convert_dataset_format()
            datasets[2].append(converted_dataset)

        # 第四组：中心距离逐渐减小，离散程度逐渐变小
        if draw_not:
            print("\nWe are processing the last list! Center get closer while shift going down\n")
        for i in range(num_samples):
            if rand_mode:
                center_spread = initial_center_spread / (center_change_base ** i)
                generator.class_centers = np.array([np.random.rand(2) * center_spread for _ in range(num_classes)])
                    #[initial_centers[0] + np.random.rand(2) * center_spread for _ in range(num_classes)])
            else:
                generator.class_centers = initial_centers/(center_change_base ** i)
            scale = initial_scale / (scale_change_base ** i)
            df = initial_df / (scale_change_base ** i)
            generator.set_distribution(distribution, {'loc': 0, 'scale': scale, 'df': df, 'randomness': randomness_init, 'error_shift': error_shift_init})
            generator.datasets = []

            cur_parm_print = {'loc': 0, 'scale': scale, 'df': df}
            if draw_not:
                print(f"Currently dealing with distribution: {distribution}, center is :\n {generator.class_centers}, \nand the para is: \n{cur_parm_print}")

            generator.generate_datasets(num_points_per_class)
            modified_dataset_with_randomness = copy.deepcopy(generator.add_randomness(copy.deepcopy(generator.datasets), randomness_strength))
            error_dataset = generator.introduce_errors(error_rate, error_strength)
            converted_dataset = generator.convert_dataset_format()
            datasets[3].append(converted_dataset)

        # 将当前分布的数据集添加到字典中
        datasets_dict[distribution] = datasets

    return datasets_dict

def inte_eval(converted_dataset):
    # init inte_general class object
    Utils = utils()
    distances = Utils.calculate_distances(converted_dataset)
    if len(distances) != math.comb(num_points_per_class * num_classes, 2):
        raise ValueError("There are some error incured in cal distance")
    print(f"Total len of sysc dataset: {len(distances)}, the type: {type(distances)}, the formate: {distances[0]}")

    # 创建Clusterer实例
    clusterer = Clusterer(distances, num_classes)
    data_before_forest, label_list = clusterer.merge_cluster_labels()
    sudo_data, voting_percentage = clusterer.call_forest_result()

    # init inte and process the data with inte class
    data = Utils.merge_datasets(original_dataset=converted_dataset, pseudo_label_dataset=sudo_data)
    data = Utils.adjust_sudo_labels(data=data)  # new data: [(x,y),(real_label,sudo_label)]
    print("inte.calculate_match_ratio(data)", Utils.calculate_match_ratio_updated(data))

    IntE = inte_core(data_with_2_labels=data, distance_table=distances)
    # {'gnmr': 1.0, 'ddr': 0.23540444013948753, 'dc': 1.0, 'oodr': 1.0}
    out = IntE.inte_calculation()
    print("The IntE score for this task: ", out)
    return out


class DatasetEvaluator:
    def __init__(self, eval_func=None):
        self.result_dir = 'result'
        self.colors = ['blue', 'green', 'red', 'purple', 'orange']
        self.linestyles = ['-', '--', '-.', ':']
        if eval_func == None:
            self.inte_eval = self.inte_eval_random
        else:
            self.inte_eval = eval_func

    def inte_eval_random(self, dataset):
        # 示例评分函数，实际应用中应替换为真实的评分逻辑
        return {'gnmr': np.random.rand(), 'ddr': np.random.rand(), 'dc': np.random.rand(), 'oodr': np.random.rand()}

    def generate_datasets_for_all_distributions(self):
        # 示例数据生成函数，实际应用中应替换为真实的数据生成逻辑
        # 返回值应该是分布名称到数据集列表的映射
        # 每个分布包含四个组，每个组包含多个converted_dataset
        return {
            'normal': [
                [{'data': 1}, {'data': 2}, {'data': 3}, {'data': 4}, {'data': 5}],
                [{'data': 6}, {'data': 7}, {'data': 8}, {'data': 9}, {'data': 10}],
                [{'data': 11}, {'data': 12}, {'data': 13}, {'data': 14}, {'data': 15}],
                [{'data': 16}, {'data': 17}, {'data': 18}, {'data': 19}, {'data': 20}]
            ],
            'unique': [
                [{'data': 1}, {'data': 2}, {'data': 3}, {'data': 4}, {'data': 5}],
                [{'data': 6}, {'data': 7}, {'data': 8}, {'data': 9}, {'data': 10}],
                [{'data': 11}, {'data': 12}, {'data': 13}, {'data': 14}, {'data': 15}],
                [{'data': 16}, {'data': 17}, {'data': 18}, {'data': 19}, {'data': 20}]
            ],
            'exp': [
                [{'data': 1}, {'data': 2}, {'data': 3}, {'data': 4}, {'data': 5}],
                [{'data': 6}, {'data': 7}, {'data': 8}, {'data': 9}, {'data': 10}],
                [{'data': 11}, {'data': 12}, {'data': 13}, {'data': 14}, {'data': 15}],
                [{'data': 16}, {'data': 17}, {'data': 18}, {'data': 19}, {'data': 20}]
            ],
        }

    def evaluate_and_plot(self, datasets_dict=None):
        # 创建或清空result文件夹
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        else:
            for file in os.listdir(self.result_dir):
                os.remove(os.path.join(self.result_dir, file))

        # 生成数据集并计算评分
        if datasets_dict==None:
            datasets_dict = self.generate_datasets_for_all_distributions()
        scores_dict = {}

        # 计算每个数据集的评分
        for distribution, datasets in datasets_dict.items():
            scores_dict[distribution] = []
            for group in datasets:
                group_scores = [self.inte_eval(dataset) for dataset in group]
                scores_dict[distribution].append(group_scores)

        print(scores_dict)

        # 绘制图表并保存结果
        for j in range(4):  # 对于每个组
            plt.figure(figsize=(10, 6))
            for i, (distribution, scores) in enumerate(scores_dict.items()):
                gnmr_scores = [score['gnmr'] for score in scores[j]]
                ddr_scores = [score['ddr'] for score in scores[j]]
                dc_scores = [score['dc'] for score in scores[j]]
                oodr_scores = [score['oodr'] for score in scores[j]]

                plt.plot(gnmr_scores, label=f'{distribution} gnmr', color=self.colors[i], linestyle=self.linestyles[0])
                plt.plot(ddr_scores, label=f'{distribution} ddr', color=self.colors[i], linestyle=self.linestyles[1])
                plt.plot(dc_scores, label=f'{distribution} dc', color=self.colors[i], linestyle=self.linestyles[2])
                plt.plot(oodr_scores, label=f'{distribution} oodr', color=self.colors[i], linestyle=self.linestyles[3])

            plt.title(f'Group: {j + 1}')
            plt.xlabel('Sample Index')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.result_dir, f'group{j + 1}_scores.png'))
            plt.close()

        print("graph and data are saved tp folder 'result/' in current working path")







if __name__ == "__main__":
    # basic dataset setting
    num_classes = 3         # how many class in total
    params = {'loc': 0, 'scale': 4, 'df': 4, 'randomness': 0.5, 'error_shift': 15}
    # initial para, loc(mid point, useless in this situation), scale,df(variance), 'randomness': 0.5 (useless at all), 'error_shift': 3(max error strength)
    distributions = ['normal', 't', 'uniform', 'exponential']
    num_points_per_class = 100

    # adjust dataset setting
    randomness_strength = 0.1       # (random shift)
    draw_not = False

    # running setting
    initial_center_spread = 30  # 初始中心点之间的最大距离
    center_change_base = 1.8  # 中心点变化的底数
    scale_change_base = 1   # the change factor of variance
    num_samples = 5  # 每组数据的采样次数

    # Loop para
    error_rate = 0.3          # percentage to show error
    error_strength = 0.1    # real shift = self.error_strength * self.params['error_shift']


    # num_classes, distributions, params=None, num_points_per_class=100,
    #                                             randomness_strength=0.1, error_rate=1, error_strength=0.1, draw_not=False,
    #                                             initial_center_spread=20, center_change_base=2, num_samples=5,
    datasets_dict = generate_datasets_for_all_distributions(num_classes=num_classes, distributions=distributions, params=params, num_points_per_class=num_points_per_class,
                                                            randomness_strength=randomness_strength, error_rate=error_rate, error_strength=error_strength, draw_not=False,
                                                            initial_center_spread=initial_center_spread, center_change_base=center_change_base,
                                                            scale_change_base=scale_change_base, num_samples=num_samples)
    print(datasets_dict.keys)
    print(len(datasets_dict[list(datasets_dict.keys())[0]]))
    print(len(datasets_dict[list(datasets_dict.keys())[0]][0]))


    evaluator = DatasetEvaluator(eval_func=inte_eval)
    evaluator.evaluate_and_plot(datasets_dict=datasets_dict)