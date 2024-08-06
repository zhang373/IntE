import math
import time

from cluster import Clusterer
from inte_general_class import utils, inte_core
from Build_Test_Dataset import DatasetGenerator
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            initial_centers = np.array(centers)
        else:
            initial_centers = np.array([np.random.rand(2) * initial_center_spread for _ in range(num_classes)])
        initial_scale = params['scale'] if 'scale' in params else 1
        randomness_init = params['randomness']
        error_shift_init = params['error_shift']
        initial_df = params['df'] if 'df' in params else 5  # 确保t分布的自由度参数被设置
        initial_df_decrease = initial_df * (center_change_base ** num_samples)
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
            df = (initial_df_decrease) / (scale_change_base ** i)
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
            df = (initial_df_decrease) / (scale_change_base ** i)
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
                generator.class_centers = initial_centers/(1*(center_change_base ** i))
            # TODO: Attention, this {int} is to adjust the initial point, if this is the same, the initial seeting should be too small for decreasing
            # scale = (initial_scale*(scale_change_base ** (num_samples))) / (scale_change_base ** i)
            scale = (initial_scale*8) / (scale_change_base ** (i))
            df = (initial_df) * (scale_change_base ** i)
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

    # norm_factor_GNMR=2, norm_factor_DDR=4, norm_factor_DC=2, norm_factor_OODR=6
    IntE = inte_core(data_with_2_labels=data, distance_table=distances, norm_factor_GNMR=1.5, norm_factor_DDR=2, norm_factor_DC=2, norm_factor_OODR=4)
    # {'gnmr': 1.0, 'ddr': 0.23540444013948753, 'dc': 1.0, 'oodr': 1.0}
    out = IntE.inte_calculation()
    print("The IntE score for this task: ", out)
    return out

# the same except the hyper parameters
def inte_eval_error_rate(converted_dataset):
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

    # norm_factor_GNMR=2, norm_factor_DDR=4, norm_factor_DC=2, norm_factor_OODR=6
    IntE = inte_core(data_with_2_labels=data, distance_table=distances, norm_factor_GNMR=2.5, norm_factor_DDR=2, norm_factor_DC=10, norm_factor_OODR=5)
    # {'gnmr': 1.0, 'ddr': 0.23540444013948753, 'dc': 1.0, 'oodr': 1.0}
    out = IntE.inte_calculation()
    print("The IntE score for this task: ", out)
    return out

def inte_eval_error_numsample(converted_dataset):
    # init inte_general class object
    Utils = utils()
    print(len(converted_dataset))
    distances = Utils.calculate_distances(converted_dataset)
    # this is different cause the num_points_per_class varys
    # if len(distances) != math.comb(num_points_per_class * num_classes, 2):
    #     raise ValueError("There are some error incured in cal distance and len(distances) != math.comb(num_points_per_class * num_classes, 2)")
    print(f"Total len of sysc dataset: {len(distances)}, the type: {type(distances)}, the formate: {distances[0]}")

    # 创建Clusterer实例
    clusterer = Clusterer(distances, num_classes)
    data_before_forest, label_list = clusterer.merge_cluster_labels()
    sudo_data, voting_percentage = clusterer.call_forest_result()

    # init inte and process the data with inte class
    data = Utils.merge_datasets(original_dataset=converted_dataset, pseudo_label_dataset=sudo_data)
    data = Utils.adjust_sudo_labels(data=data)  # new data: [(x,y),(real_label,sudo_label)]
    print("inte.calculate_match_ratio(data)", Utils.calculate_match_ratio_updated(data))

    # norm_factor_GNMR=2, norm_factor_DDR=4, norm_factor_DC=2, norm_factor_OODR=6
    IntE = inte_core(data_with_2_labels=data, distance_table=distances, norm_factor_GNMR=2.5, norm_factor_DDR=2, norm_factor_DC=10, norm_factor_OODR=5)
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

    def evaluate_and_plot(self, datasets_dict=None, error_name=None):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        else:
            for file in os.listdir(self.result_dir):
                os.remove(os.path.join(self.result_dir, file))

        if datasets_dict is None:
            datasets_dict = self.generate_datasets_for_all_distributions()
        scores_dict = {}

        for distribution, datasets in datasets_dict.items():
            scores_dict[distribution] = []
            for group in datasets:
                group_scores = [self.inte_eval(dataset) for dataset in group]
                scores_dict[distribution].append(group_scores)

        print(scores_dict)

        image_name_list = ['center close vs shift same', 'center same vs shift increase',
                           'center close vs shift increase', 'center close vs shift close']

        # Define the mappings for distributions and metrics
        distribution_handles = [
            plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=distribution) for
            color, distribution in zip(self.colors, datasets_dict.keys())]
        metric_handles = [plt.Line2D([], [], color='black', marker='None', linestyle=style, linewidth=2, label=metric)
                          for style, metric in zip(self.linestyles, ['gnmr', 'ddr', 'dc', 'oodr'])]

        for j in range(4):
            plt.figure(figsize=(10, 6))
            for i, (distribution, scores) in enumerate(scores_dict.items()):
                gnmr_scores = [score['gnmr'] for score in scores[j]]
                ddr_scores = [score['ddr'] for score in scores[j]]
                dc_scores = [score['dc'] for score in scores[j]]
                oodr_scores = [score['oodr'] for score in scores[j]]

                plt.plot(gnmr_scores, label='gnmr', color=self.colors[i], linestyle=self.linestyles[0])
                plt.plot(ddr_scores, label='ddr', color=self.colors[i], linestyle=self.linestyles[1])
                plt.plot(dc_scores, label='dc', color=self.colors[i], linestyle=self.linestyles[2])
                plt.plot(oodr_scores, label='oodr', color=self.colors[i], linestyle=self.linestyles[3])

            plt.title(f'Group: {j + 1}: {image_name_list[j]}')
            plt.xlabel('Sample Index')
            plt.ylabel('Score')

            # Create two legends
            distribution_legend = plt.legend(handles=distribution_handles, title='Distribution', loc='upper left')
            metric_legend = plt.legend(handles=metric_handles, title='Metric', loc='upper right')

            plt.gca().add_artist(distribution_legend)
            plt.gca().add_artist(metric_legend)

            plt.grid(False)
            plt.savefig(os.path.join(self.result_dir, f'group{j + 1}_scores.png'))
            plt.close()

        print("Graphs and data are saved to folder 'result/' in current working directory.")

    def evaluate_and_plot_vs_error_rate(self, datasets_dict, result_dir='result'):
        """评估并绘制不同分布和指标的分数。"""
        # 创建或清空结果目录
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        else:
            for file in os.listdir(result_dir):
                os.remove(os.path.join(result_dir, file))

        # 定义颜色和线型
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        linestyles = ['-', '--', '-.', ':']

        # 定义分布和指标的映射
        distribution_handles = [
            plt.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=distribution)
            for color, distribution in zip(colors, datasets_dict.keys())
        ]
        metric_handles = [
            plt.Line2D([], [], color='black', marker='None', linestyle=style, linewidth=2, label=metric)
            for style, metric in zip(linestyles, ['gnmr', 'ddr', 'dc', 'oodr'])
        ]

        # 创建图和轴
        fig, ax = plt.subplots(figsize=(10, 6))

        # 对每个分布和指标进行评估和绘图
        for i, (distribution, datasets) in enumerate(datasets_dict.items()):
            for j, metric in enumerate(['gnmr', 'ddr', 'dc', 'oodr']):
                scores = [self.inte_eval(dataset)[metric] for dataset in datasets]
                ax.plot(scores, label=f'{distribution} - {metric}', color=colors[i], linestyle=linestyles[j])

        # 设置标题和标签
        ax.set_title('Score Evolution for Different Distributions and Metrics')
        ax.set_xlabel('Error Rate')
        ax.set_ylabel('Score')

        # 创建两个图例并放置在右上角
        distribution_legend = ax.legend(handles=distribution_handles, title='Distribution', loc='upper right',
                                        bbox_to_anchor=(0.88, 1))
        metric_legend = ax.legend(handles=metric_handles, title='Metric', loc='upper right', bbox_to_anchor=(1, 1))

        ax.add_artist(distribution_legend)
        ax.add_artist(metric_legend)

        ax.grid(False)
        plt.savefig(os.path.join(result_dir, 'score_evolution_200_50_1.png'))
        plt.close()

        print("Graph is saved to folder 'result/' in current working directory.")

    def cal_scores(self, dataset_list):
        # 初始化存储分数的字典
        scores = {'gnmr': [], 'ddr': [], 'dc': [], 'oodr': []}

        # 遍历每个数据集，获取评分并存储
        for dataset in dataset_list:
            current_scores = self.inte_eval(dataset)
            for key in scores:
                scores[key].append(current_scores[key])

        # 计算均值和标准差
        mean_std = {}
        for key in scores:
            mean_std[key] = {
                'mean': np.mean(scores[key]),
                'std': np.std(scores[key])
            }

        return mean_std

    def calculate_mean_std(self, error_dict):
        """
        Calculate the mean and standard deviation for different error rates and number of samples.

        Args:
        - error_dict (dict): A dictionary containing the datasets with error rates and number of samples.

        Returns:
        - A dictionary with mean and standard deviation for each combination of error rate and number of samples.
        """
        # Initialize a dictionary to store the results
        results = {}
        # Iterate over each distribution in the error_dict
        tqdm_bar_index = 0
        for distribution, datasets in error_dict.items():
            # Initialize a list for this distribution in the results dictionary
            results[distribution] = {}
            # Iterate over each error rate
            for error_rate_data in datasets:
                error_rate = error_rate_data[0]
                results[distribution][str(error_rate)] = {}
                sample_data = error_rate_data[1:]
                # For each number of samples, calculate mean and std
                for num_samples_data in sample_data:
                    tqdm_bar_index += 1
                    print(f"\n\nWe are processing {tqdm_bar_index} out of total {total_amount}!, we have spend {time.time()-time_start} seconds")
                    num_samples = num_samples_data[0]
                    dataset_lists = num_samples_data[1:]  # Assuming scores are stored in a list of lists
                    mean_std_scores = self.cal_scores(dataset_lists)
                    results[distribution][str(error_rate)][str(num_samples)] = mean_std_scores
        return results

    def process_result(self, result, sub_score_or_not, sub_distribution_or_not):
        # 确保result文件夹存在且为空
        if not os.path.exists('result'):
            os.makedirs('result')
        else:
            for file in os.listdir('result'):
                os.remove(os.path.join('result', file))

        # 遍历result字典并处理数据
        for distribution, error_rates in result.items():
            for error_rate_str, num_samples in error_rates.items():
                error_rate = float(error_rate_str)
                for num_sample_str, mean_std_scores in num_samples.items():
                    num_sample = int(num_sample_str)

                    # 为每个score创建一个DataFrame
                    for score, stats in mean_std_scores.items():
                        data = {
                            'distribution': distribution,
                            'error_rate': error_rate,
                            'num_sample': num_sample,
                            'score': score,
                            'mean': stats['mean'],
                            'std': stats['std']
                        }

                        # 创建DataFrame
                        df = pd.DataFrame(data, index=[0])

                        # 根据 sub_score_or_not 和 sub_distribution_or_not 处理数据
                        if sub_score_or_not and sub_distribution_or_not:
                            # 单独保存每个分数类型的每个分布的结果
                            filename = f'result/{score}_{distribution}.csv'
                            df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
                        elif sub_score_or_not:
                            # 合并分布但单独保存每个分数类型的结果
                            df = df.drop(columns=['distribution'])
                            filename = f'result/{score}_all_distributions.csv'
                            df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
                        elif sub_distribution_or_not:
                            # 合并分数类型但单独保存每个分布的结果
                            df = df.drop(columns=['score'])
                            filename = f'result/all_scores_{distribution}.csv'
                            df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
                        else:
                            # 合并所有分数类型和分布
                            df = df.drop(columns=['distribution', 'score'])
                            filename = f'result/all_scores_all_distributions.csv'
                            df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

    def process_result_old(self, result, sub_score_or_not, sub_distribution_or_not):
        # 确保result文件夹存在且为空
        if not os.path.exists('result'):
            os.makedirs('result')
        else:
            for file in os.listdir('result'):
                os.remove(os.path.join('result', file))

        # 初始化一个空的DataFrame来收集所有数据
        all_data = pd.DataFrame()
        # 遍历result字典并处理数据
        for distribution, error_rates in result.items():
            for error_rate, num_samples in error_rates.items():
                for num_sample, mean_std_scores in num_samples.items():
                    # 获取当前数据点的分数、均值和标准差
                    scores = list(mean_std_scores.keys())
                    means = [mean_std_scores[score]['mean'] for score in scores]
                    stds = [mean_std_scores[score]['std'] for score in scores]

                    # 创建一个临时的DataFrame用于当前的数据点
                    temp_df = pd.DataFrame({
                        'distribution': [distribution] * len(scores),
                        'error_rate': [error_rate] * len(scores),
                        'num_sample': [num_sample] * len(scores),
                        'score': scores,
                        'mean': means,
                        'std': stds
                    })

                    # 将临时DataFrame添加到all_data中
                    all_data = pd.concat([all_data, temp_df], ignore_index=True)

        # 根据 sub_score_or_not 和 sub_distribution_or_not 处理all_data DataFrame
        if sub_score_or_not:
            # 遍历每个分数类型
            for score in ['gnmr', 'ddr', 'dc', 'oodr']:
                score_df = all_data[all_data['score'] == score]
                if sub_distribution_or_not:
                    # 遍历每个分布
                    for distrib in score_df['distribution'].unique():
                        distrib_df = score_df[score_df['distribution'] == distrib]
                        distrib_df = distrib_df.drop(columns=['distribution', 'score'])
                        distrib_df = distrib_df.set_index(['error_rate', 'num_sample'])
                        distrib_df.to_csv(f'result/{score}_{distrib}.csv')
                else:
                    # 如果不需要按分布分开，则合并所有分布
                    score_df = score_df.drop(columns=['distribution', 'score'])
                    score_df = score_df.groupby(['error_rate', 'num_sample']).mean()
                    score_df.to_csv(f'result/{score}.csv')
        else:
            if sub_distribution_or_not:
                # 遍历每个分布
                for distrib in all_data['distribution'].unique():
                    distrib_df = all_data[all_data['distribution'] == distrib]
                    distrib_df = distrib_df.drop(columns=['distribution', 'score'])
                    distrib_df = distrib_df.set_index(['error_rate', 'num_sample'])
                    distrib_df.to_csv(f'result/all_scores_{distrib}.csv')
            else:
                # 合并所有分数和分布
                all_data = all_data.drop(columns=['distribution', 'score'])
                all_data = all_data.groupby(['error_rate', 'num_sample']).mean()
                all_data.to_csv(f'result/all_scores.csv')

    def eval_sample_error(self, error_dataset_dict, sub_score_or_not=True, sub_distribution_or_not=True):
        result = self.calculate_mean_std(error_dataset_dict)
        self.process_result(result, sub_score_or_not, sub_distribution_or_not)

def generate_datasets_under_errorrate_for_all_distributions(num_classes, distributions, params=None, num_points_per_class=100,
                                            randomness_strength=0.1, error_rate=1, error_strength=0.1, draw_not=False,
                                            initial_center_spread=20, center_change_base=2, scale_change_base = 2,
                                            num_samples=5, centers=None, rand_mode=False):
    error_dict = {}

    for distribution in distributions:
        if draw_not:
            print(f"\n\nWe are currently processing the distribution: {distribution}")

        # params = {'loc': 0, 'scale': 2, 'df': 2, 'randomness': 0.5, 'error_shift': 3}  # It is given here
        generator = DatasetGenerator(num_classes, distribution, params)
        # centers
        if centers !=None:
            if len(centers) != num_classes:
                raise ValueError("you should keep centers the same number with num_class!")
            initial_centers = np.array(centers)
        else:
            initial_centers = np.array([np.random.rand(2) * initial_center_spread for _ in range(num_classes)])

        # scales
        initial_scale = params['scale'] if 'scale' in params else 1
        randomness_init = params['randomness']
        error_shift_init = params['error_shift']
        initial_df = params['df'] if 'df' in params else 5  # 确保t分布的自由度参数被设置
        initial_df_decrease = initial_df * (center_change_base ** num_samples)
        # data list with error rate increase
        if draw_not:
            print("\n We are processing the first list! Center get closer while shift stays the same\n")
        if rand_mode:
            center_spread = initial_center_spread / (center_change_base ** i)
            generator.class_centers = np.array([np.random.rand(2) * center_spread for _ in range(num_classes)])
            # [initial_centers[0] + np.random.rand(2) * center_spread for _ in range(num_classes)])
        else:
            pass

        generator.class_centers = initial_centers
        generator.set_distribution(distribution,
                                   {'loc': 0, 'scale': initial_scale, 'df': initial_df, 'randomness': randomness_init,
                                    'error_shift': error_shift_init})
        generator.datasets = []

        cur_parm_print = {'loc': 0, 'scale': initial_scale, 'df': initial_df}
        if draw_not:
            print(
                f"Currently dealing with distribution: {distribution}, center is :\n {generator.class_centers}, \nand the para is: \n{cur_parm_print}")

        generator.generate_datasets(num_points_per_class)

        step = 1.0 / num_samples
        error_rate = 0.0
        # store data
        datasets = []
        for i in range(num_samples+1):
            modified_dataset_with_randomness = copy.deepcopy(generator.add_randomness(copy.deepcopy(generator.datasets), randomness_strength))
            # print(modified_dataset_with_randomness)
            error_dataset = generator.introduce_errors(error_rate, error_strength, out_source_data=modified_dataset_with_randomness)
            converted_dataset = generator.convert_dataset_format(error_dataset)
            datasets.append(converted_dataset)
            error_rate += step
        error_dict[distribution] = datasets

    return error_dict


def generate_datasets_under_errorrate_for_all_distributions_varience_detection(num_classes, distributions, params=None, num_points_per_class=100,num_point_slices=10,
                                            randomness_strength=0.1, error_rate=1, error_strength=0.1, draw_not=False,
                                            initial_center_spread=20, center_change_base=2, scale_change_base = 2,
                                            num_samples=5, num_repeated_test=10, centers=None, rand_mode=False):
    error_dict = {}
    for distribution in distributions:
        if draw_not:
            print(f"\n\nWe are currently processing the distribution: {distribution}")
        # params = {'loc': 0, 'scale': 2, 'df': 2, 'randomness': 0.5, 'error_shift': 3}  # It is given here
        generator = DatasetGenerator(num_classes, distribution, params)
        # centers
        if centers !=None:
            if len(centers) != num_classes:
                raise ValueError("you should keep centers the same number with num_class!")
            initial_centers = np.array(centers)
        else:
            initial_centers = np.array([np.random.rand(2) * initial_center_spread for _ in range(num_classes)])

        # scales
        initial_scale = params['scale'] if 'scale' in params else 1
        randomness_init = params['randomness']
        error_shift_init = params['error_shift']
        initial_df = params['df'] if 'df' in params else 5  # 确保t分布的自由度参数被设置
        initial_df_decrease = initial_df * (center_change_base ** num_samples)
        # data list with error rate increase
        if draw_not:
            print("\n We are processing the first list! Center get closer while shift stays the same\n")
        if rand_mode:
            center_spread = initial_center_spread / (center_change_base ** i)
            generator.class_centers = np.array([np.random.rand(2) * center_spread for _ in range(num_classes)])
            # [initial_centers[0] + np.random.rand(2) * center_spread for _ in range(num_classes)])
        else:
            pass

        generator.class_centers = initial_centers
        generator.set_distribution(distribution,
                                   {'loc': 0, 'scale': initial_scale, 'df': initial_df, 'randomness': randomness_init,
                                    'error_shift': error_shift_init})


        cur_parm_print = {'loc': 0, 'scale': initial_scale, 'df': initial_df}
        if draw_not:
            print(
                f"Currently dealing with distribution: {distribution}, center is :\n {generator.class_centers}, \nand the para is: \n{cur_parm_print}")

        step = 1.0 / num_samples
        samples_step = num_points_per_class/num_point_slices
        error_rate = 0.0
        cur_samples = 0
        # store data
        datasets = []
        for i in range(num_samples+1):
            repeted_datas = [error_rate]
            cur_samples = 0
            for j in range(num_point_slices):
                cur_samples += samples_step
                cur_samples = round(cur_samples)
                print(cur_samples)
                different_damples_data = [cur_samples]
                if draw_not:
                    print(f"Currently we are simulate {cur_samples} points in each dataset "
                          f"(totally {num_repeated_test} datasets) under error rate {error_rate}")
                for k in range(num_repeated_test):
                    generator.datasets = []
                    generator.generate_datasets(cur_samples)
                    modified_dataset_with_randomness = copy.deepcopy(generator.add_randomness(copy.deepcopy(generator.datasets), randomness_strength))
                    error_dataset = generator.introduce_errors(error_rate, error_strength, out_source_data=modified_dataset_with_randomness)
                    converted_dataset = generator.convert_dataset_format(error_dataset)
                    different_damples_data.append(converted_dataset)
                repeted_datas.append(different_damples_data)
            datasets.append(repeted_datas)
            error_rate += step
        error_dict[distribution] = datasets
    return error_dict


if __name__ == "__main__":
    ExperimentType = 2
    if ExperimentType == 1:
        # basic dataset setting
        num_classes = 3         # how many class in total
        params = {'loc': 0, 'scale': 0.5, 'df': 1, 'randomness': 0.5, 'error_shift': 20}
        # initial para, loc(mid point, useless in this situation), scale,df(variance), 'randomness': 0.5 (useless at all), 'error_shift': 3(max error strength)
        distributions = ['normal', 'uniform', 'exponential']
        num_points_per_class = 200

        # adjust dataset setting
        randomness_strength = 0.1       # (random shift)
        draw_not = False

        # running setting
        initial_center_spread = 30  # 初始中心点之间的最大距离, useless
        center_change_base = 1.9  # 中心点变化的底数
        scale_change_base = 1.9   # the change factor of variance
        num_samples = 25  # 每组数据的采样次数
        class_centers = [[0, 0], [20, 20], [40, 40]]

        # Loop para
        error_rate = 0         # percentage to show error
        error_strength = 1    # real shift = self.error_strength * self.params['error_shift']


        # num_classes, distributions, params=None, num_points_per_class=100,
        #                                             randomness_strength=0.1, error_rate=1, error_strength=0.1, draw_not=False,
        #                                             initial_center_spread=20, center_change_base=2, num_samples=5,
        datasets_dict = generate_datasets_for_all_distributions(num_classes=num_classes, distributions=distributions, params=params, num_points_per_class=num_points_per_class,
                                                                randomness_strength=randomness_strength, error_rate=error_rate, error_strength=error_strength, draw_not=True,
                                                                initial_center_spread=initial_center_spread, center_change_base=center_change_base,
                                                                scale_change_base=scale_change_base, num_samples=num_samples, centers=class_centers)
        print(datasets_dict.keys)
        print(len(datasets_dict[list(datasets_dict.keys())[0]]))
        print(len(datasets_dict[list(datasets_dict.keys())[0]][0]))


        evaluator = DatasetEvaluator(eval_func=inte_eval)
        evaluator.evaluate_and_plot(datasets_dict=datasets_dict)

    if ExperimentType == 2:
        # TODO: Run this again when you are free
        # basic dataset setting
        num_classes = 3  # how many class in total
        params = {'loc': 0, 'scale': 0.5, 'df': 1, 'randomness': 0.5, 'error_shift': 13}
        # initial para, loc(mid point, useless in this situation), scale,df(variance), 'randomness': 0.5 (useless at all), 'error_shift': 3(max error strength)
        distributions = ['normal', 'uniform', 'exponential']
        num_points_per_class = 200

        # adjust dataset setting
        randomness_strength = 0.1  # (random shift)
        draw_not = False

        # running setting
        initial_center_spread = 30  # 初始中心点之间的最大距离, useless
        center_change_base = 1.9  # 中心点变化的底数
        scale_change_base = 1.9  # the change factor of variance
        num_samples = 50  # 每组数据的采样次数
        class_centers = [[0, 0], [10, 10], [20, 20]]

        # Loop para
        error_rate = 0  # percentage to show error
        error_strength = 1  # real shift = self.error_strength * self.params['error_shift']

        datasets_dict = generate_datasets_under_errorrate_for_all_distributions(num_classes=num_classes, distributions=distributions, params=params, num_points_per_class=num_points_per_class,
                                                                randomness_strength=randomness_strength, error_rate=error_rate, error_strength=error_strength, draw_not=draw_not,
                                                                initial_center_spread=initial_center_spread, center_change_base=center_change_base,
                                                                scale_change_base=scale_change_base, num_samples=num_samples, centers=class_centers)

        print(datasets_dict.keys())
        eval_error_rate = DatasetEvaluator(eval_func=inte_eval_error_rate)
        eval_error_rate.evaluate_and_plot_vs_error_rate(datasets_dict)

    if ExperimentType == 3:
        # basic dataset setting
        num_classes = 3  # how many class in total
        params = {'loc': 0, 'scale': 0.5, 'df': 1, 'randomness': 0.5, 'error_shift': 13}
        # initial para, loc(mid point, useless in this situation), scale,df(variance), 'randomness': 0.5 (useless at all), 'error_shift': 3(max error strength)
        distributions = ['normal', 'uniform', 'exponential']
        num_points_per_class = 200

        # adjust dataset setting
        randomness_strength = 0.1  # (random shift)
        draw_not = True

        # running setting
        initial_center_spread = 30      # 初始中心点之间的最大距离, useless
        center_change_base = 1.9        # 中心点变化的底数
        scale_change_base = 1.9         # the change factor of variance
        num_samples = 8                 # Error rate 的变化次数
        num_repeated_test = 10           # 每次error下做多少次实验
        num_point_slices = 10           # 多少次达到最大num数量
        time_start = time.time()
        class_centers = [[0, 0], [10, 10], [20, 20]]
        total_amount = num_point_slices * num_samples
        # Loop para
        error_rate = 0  # percentage to show error
        error_strength = 1  # real shift = self.error_strength * self.params['error_shift']

        datasets_dict_repeat = generate_datasets_under_errorrate_for_all_distributions_varience_detection(num_point_slices=num_point_slices,num_classes=num_classes, distributions=distributions, params=params, num_points_per_class=num_points_per_class,
                                                                randomness_strength=randomness_strength, error_rate=error_rate, error_strength=error_strength, draw_not=draw_not,
                                                                initial_center_spread=initial_center_spread, center_change_base=center_change_base,
                                                                scale_change_base=scale_change_base,num_repeated_test=num_repeated_test, num_samples=num_samples, centers=class_centers)

        print(datasets_dict_repeat.keys())
        eval_error_numsample = DatasetEvaluator(eval_func=inte_eval_error_numsample)
        eval_error_numsample.eval_sample_error(datasets_dict_repeat, sub_score_or_not=True, sub_distribution_or_not=True)