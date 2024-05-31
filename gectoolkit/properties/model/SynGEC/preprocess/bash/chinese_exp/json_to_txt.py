import json
import sys

dataset = "conll14_translation"
dataset_path = f"/home/xiaoman/project/gectoolkit2/data/{dataset}"
output_path = f"/home/xiaoman/project/gectoolkit2/proprecess/syngec/data/{dataset}"

# todo:trainset预处理
# 打开源文件和目标文件
with open(f'{dataset_path}/trainset.json', 'r', encoding='utf-8') as json_file, \
        open(f'{output_path}/trainset/src.txt', 'w', encoding='utf-8') as src_file, \
        open(f'{output_path}/trainset/tgt.txt', 'w', encoding='utf-8') as tgt_file:
    data = json.load(json_file)

    for entry in data:
        source_text = entry['source_text']  # 错误的句子
        target_text = entry['target_text']  # 正确的句子

        # 写入source_text到src.txt，并以换行分割
        src_file.write(source_text + '\n')
        # 写入target_text到tgt.txt，并以换行分割
        tgt_file.write(target_text + '\n')

    print(f'{output_path} trainset finish !!')

# # todo:validset预处理
# # 打开源文件和目标文件
with open(f'{dataset_path}/validset.json', 'r', encoding='utf-8') as json_file,\
      open(f'{output_path}/validset/src.txt', 'w', encoding='utf-8') as src_file,\
      open(f'{output_path}/validset/tgt.txt', 'w', encoding='utf-8') as tgt_file:
    data = json.load(json_file)

    for entry in data:
        source_text = entry['source_text']  # 错误的句子
        target_text = entry['target_text']  # 正确的句子

        # 写入source_text到src.txt，并以换行分割
        src_file.write(source_text + '\n')
        # 写入target_text到tgt.txt，并以换行分割
        tgt_file.write(target_text + '\n')

    print(f'{output_path} validset finish !!')

# todo:testset预处理
# 打开源文件和目标文件
with open(f'{dataset_path}/testset.json', 'r', encoding='utf-8') as json_file,\
        open(f'{output_path}/testset/src.txt', 'w', encoding='utf-8') as src_file, \
        open(f'{output_path}/testset/tgt.txt', 'w', encoding='utf-8') as tgt_file:
    data = json.load(json_file)

    for entry in data:
        source_text = entry['source_text']  # 错误的句子
        target_text = entry['target_text']  # 正确的句子

        # 写入source_text到src.txt，并以换行分割
        src_file.write(source_text + '\n')
        # 写入target_text到tgt.txt，并以换行分割
        tgt_file.write(target_text + '\n')

    print(f'{output_path} testset finish !!')


# # 检查是否存在空行，存在空行放回空行id
# def find_empty_lines(file_path):
#     empty_lines = []
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         for i, line in enumerate(lines, start=1):
#             if line.strip() == '':
#                 empty_lines.append(i)
#     if empty_lines:
#         print(f"{file_path}存在空行，行号为:", empty_lines)
#     else:
#         print(f"{file_path}文件中不存在空行。")
#     return empty_lines
#
# trainset_file_path = f'../../data/{dataset}/trainset/src.txt'  # 请替换为你的文件路径
# trainset_empty_lines = find_empty_lines(trainset_file_path)
#
# validset_file_path = f'../../data/{dataset}/trainset/src.txt'  # 请替换为你的文件路径
# validset_empty_lines = find_empty_lines(validset_file_path)
#
# testset_file_path = f'../../data/{dataset}/testset/src.txt'  # 请替换为你的文件路径
# testset_empty_lines = find_empty_lines(testset_file_path)
