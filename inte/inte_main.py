import math
import time
from cluster import Clusterer
from inte_general_class import utils, inte_core
from Build_Test_Dataset import generate_and_modify_dataset

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

    # init inte_general class object
    utils = utils()
    distances = utils.calculate_distances(converted_dataset)
    if len(distances) != math.comb(num_points_per_class*num_classes, 2):
        raise ValueError("There are some error incured in cal distance")
    print(f"Total len of sysc dataset: {len(distances)}, the type: {type(distances)}, the formate: {distances[0]}")

    # 创建Clusterer实例
    clusterer = Clusterer(distances, num_classes)
    data_before_forest, label_list = clusterer.merge_cluster_labels()
    sudo_data, voting_percentage = clusterer.call_forest_result()

    # init inte and process the data with inte class
    data = utils.merge_datasets(original_dataset=converted_dataset, pseudo_label_dataset=sudo_data)
    data = utils.adjust_sudo_labels(data)        # new data: [(x,y),(real_label,sudo_label)]
    print("inte.calculate_match_ratio(data)", utils.calculate_match_ratio_updated(data))

    start_time = time.time()
    IntE = inte_core(data_with_2_labels=data, distance_table=distances)
    print(IntE.inte_calculation())
    print(IntE.filter_inconsistent_data_to_dataframe())
    end_time = time.time()
    print("Total time is: ", end_time-start_time)

    # 使用测试数据调用函数
    k_typical = 2
    k_unique = 2
    typical_points = IntE.find_typical_points_per_sudo_label(k=3)
    unique_points = IntE.find_unique_points_per_sudo_label(k=3)


    # 打印输出DataFrame
    print("Current typical data:\n", typical_points)
    print("Current unique data:\n", unique_points)