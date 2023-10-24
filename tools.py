import os
import csv
import shutil

# 指定csv文件和目标文件夹
csv_file = 'C:/Users/Administrator/Desktop/CONTRIQUE-main/dataset/test.csv'
source_dir = 'D:/Lab Works/IQA Project/DataSet/data'
target_dir = 'C:/Users/Administrator/Desktop/CONTRIQUE-main/dataset/TestDataset'

# 创建目标文件夹（如果它不存在）
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 打开CSV文件并解析第一列
with open(csv_file, 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        if row:  # 确保行不为空
            file_name = row[0]  # 获取第一列的文件名

            # 构建源文件路径和目标文件路径
            source_file_path = os.path.join(source_dir, file_name)
            target_file_path = os.path.join(target_dir, file_name)

            # 检查源文件是否存在，并复制到目标文件夹
            if os.path.exists(source_file_path):
                shutil.copy(source_file_path, target_file_path)
                print(f"复制文件: {file_name} 到 {target_dir}")
            else:
                print(f"文件 {file_name} 不存在于 {source_dir}")

print("复制完成")
