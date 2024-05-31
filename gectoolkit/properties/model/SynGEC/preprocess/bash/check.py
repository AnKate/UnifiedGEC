
################################################################################
### todo: 检查是是否有连续空格
################################################################################
# 打开文档
def check_two_space(file_path):
    with open(file_path, 'r') as file:
        # 逐行读取文档内容
        pos = False
        for line_number, line in enumerate(file, start=1):
            # 检查是否存在连续的两个空格
            if '  ' in line:
                pos = True
                print(f"第 {line_number} 行存在连续的两个空格。")
                print(line)
        if pos==False:
            print(f'{file_path}不存在连续的两个空格')


################################################################################
### todo: 重新清洗cowsl2h2的数据(两个空格)
################################################################################
#
import json

def clean_two_space_json(file_path):
    # 打开JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    pos = False

    # 遍历每个对象
    for obj in data:
        id = obj['id']
        source_text = obj['source_text']
        target_text = obj['target_text']

        # 检查source_text是否包含连续的两个空格
        if '  ' in source_text:
            pos = True
            print(f"id:{id},source_text中存在连续的两个空格：{source_text}")
            # 替换连续的两个空格为一个空格
            source_text = source_text.replace('  ', ' ')
            obj['source_text'] = source_text

        # 检查target_text是否包含连续的两个空格
        if '  ' in target_text:
            pos = True
            print(f"id:{id},target_text中存在连续的两个空格：{target_text}")
            # 替换连续的两个空格为一个空格
            target_text = target_text.replace('  ', ' ')
            obj['target_text'] = target_text

    if pos == False:
        print(f'{file_path}中不存在两个空格的数据')
    else:
        # 将修改后的数据写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"已经将连续的两个空格替换为一个空格，并写回文件：{file_path}")

dataset='falko'
file_path1 = f'./data/{dataset}/trainset.json'
file_path2 = f'./data/{dataset}/validset.json'
file_path3 = f'./data/{dataset}/testset.json'
# clean_two_space(file_path1)
# clean_two_space(file_path2)
# clean_two_space(file_path3)


################################################################################
### todo: falko最后一个标点空格
################################################################################
import re
def clean_and_replace(file_path, out_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(out_path, 'w', encoding='utf-8') as output_file:
        for line in lines:
            # 清除 ' (>_<)' 这个字符
            line = line.replace('~~~~(>_<)~~~~', '')
            line = line.replace('(>_<)', '')
            line= line.replace('', '')
            # 查找所有的 '...'，'.....'，'......'，并将其替换为 '...'
            line = re.sub(r'\.{3,}', '...', line)

            output_file.write(line)

    print(f'已经清除 {file_path} 中的特殊字符，并替换了多余的点号，并将结果保存到 {out_path}。')

def add_space_before_last_punctuation_in_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(input_file, 'w', encoding='utf-8') as output:
        for line in lines:
            # 在最后一个标点符号前添加空格
            line_with_spaces = add_space_before_last_punctuation(line)
            # line_with_spaces = line_with_spaces.replace('.. .', ' ...')
            # 将处理后的行写入输出文件
            output.write(line_with_spaces)
        print(f'{input_file} 在最后一个标点符号前添加空格 successful')

def add_space_before_last_punctuation(text):
    # 定义匹配最后一个标点符号的正则表达式模式
    pattern = r'([^\s\w])$'
    # 使用正则表达式查找文本中的匹配项
    matches = re.finditer(pattern, text)
    # 对每个匹配项，在其前面添加一个空格
    for match in matches:
        start_index = match.start(1)
        end_index = match.end(1)
        text = text[:start_index] + ' ' + text[start_index:end_index] + text[end_index:]
    return text


def replace_dots(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()

    # 替换 ".. ." 为 "..."
    content = content.replace('.. .', ' ...')
    content = content.replace('...', ' ...')
    # 将替换后的内容写回原始文件
    with open(file_path, 'w') as file:
        file.write(content)



import re
def clean_two_space_txt(file_path):
    # 打开JSON文件
    with open(file_path, 'r') as file:
        # 读取文件内容
        lines = file.readlines()
    pos = False
    # 遍历每个对象
    for i, line in enumerate(lines):
        # 检查source_text是否包含连续的两个空格
        if '  ' in line:
            pos = True
            print(f"存在连续的两个空格：{line}")
            # 替换连续的两个空格为一个空格
            lines[i] = line.replace('  ', ' ')

    # 将处理后的内容写回原始文件
    with open(file_path, 'w') as file:
        file.writelines(lines)
    if pos:
        print(f'{file_path} 中清理连续的两个以上空格完成。')
    else:
        print(f'{file_path} 中不存在连续的两个空格。')


def clean_two_space_chinese(file_path):
    # 打开JSON文件
    with open(file_path, 'r') as file:
        # 读取文件内容
        lines = file.readlines()
    pos = False
    # 遍历每个对象
    for i, line in enumerate(lines):
        # 检查source_text是否包含连续的两个空格
        if '  ' in line:
            pos = True
            print(f"存在连续的两个空格：{line}")
            # 替换连续的两个空格为一个空格
            lines[i] = line.replace('  ', '')

    # 将处理后的内容写回原始文件
    with open(file_path, 'w') as file:
        file.writelines(lines)
    if pos:
        print(f'{file_path} 中清理连续的两个以上空格完成。')
    else:
        print(f'{file_path} 中不存在连续的两个空格。')

def check_lines_starting_with(pattern, file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 检查是否存在以指定模式开头的行
    pos = False
    for line_number, line in enumerate(lines, start=1):
        if line.startswith(pattern):
            pos = True
            print(f"在{file_path}第 {line_number} 行发现以 '{pattern}' 开头的行：{line.strip()}")
    if pos == False:
        print(f'{file_path}没有开头为空格')

def replace_lines_with_no_leading_space(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 替换开头是空格的行
    for i, line in enumerate(lines):
        if line.startswith(' '):
            lines[i] = line.lstrip()

    # 将修改后的内容写回原始文件
    with open(file_path, 'w') as file:
        file.writelines(lines)
    print(f'replace_lines_with_no_leading_space {file_path} successful!')

def check_fullwidth_space(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 检查是否存在全角空格
    if '　' in content:
        print(f"{file_path} 中存在全角空格。")
    else:
        print(f"{file_path} 中不存在全角空格。")

def replace_fullwidth_space(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # 替换全角空格为普通空格
    replaced_content = content.replace('　', '')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(replaced_content)


def find_lines_with_fullwidth_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    pos = False
    # 遍历每一行，检查是否包含全角字符
    for line_number, line in enumerate(lines, start=1):
        fullwidth_characters = [char for char in line if ord(char) >= 0xFF01 and ord(char) <= 0xFF5E]
        if fullwidth_characters:
            pos =True
            print(f"第 {line_number} 行包含全角字符：{line.strip()}")
    if pos == False:
        print(f'{file_path}中不存在全角字符')
    else:
        print(f'{file_path}中存在全角字符')


def replace_fullwidth_characters(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式替换全角字符为半角字符
    halfwidth_content = re.sub(r'[！-～]', lambda x: chr(ord(x.group(0)) - 0xfee0), content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(halfwidth_content)
    print(f'{file_path} 全角-半角替换成功')





###############################
# todo:评估数据集
###############################

dataset='conll14_translation'

out_path1 = f'../data/{dataset}/testset/src.txt'
out_path2 = f'../data/{dataset}/testset/tgt.txt'
out_path3 = f'../data/{dataset}/validset/src.txt'
out_path4 = f'../data/{dataset}/validset/tgt.txt'
out_path5 = f'../data/{dataset}/trainset/src.txt'
out_path6 = f'../data/{dataset}/trainset/tgt.txt'

pattern = ' '


# （1）去除脏数据：
# clean_and_replace(file_path,out_path)
# # （2）最后标点前加空格
# add_space_before_last_punctuation_in_file(out_path)
# # （3）...前加空格
# replace_dots(out_path)
# # （4）检查开头是否有空格
# check_lines_starting_with(pattern,out_path)
# # （5）清除开头的空格
# replace_lines_with_no_leading_space(out_path)
# # （6）检查是否有全角空格
# check_fullwidth_space(out_path)
# # （7）去除全角空格
# replace_fullwidth_space(out_path)
# # （8）检查是否有全角字符
# find_lines_with_fullwidth_characters(out_path)
# #（9）全角换半角
# replace_fullwidth_characters(out_path)
# # （10）是否两个连续空格
# check_two_space(out_path)
# #（11）两个连续空格变一个
# clean_two_space_txt(out_path)

#（12）最终检查
# print('\n------- testset/src.txt --------')
# check_lines_starting_with(pattern,out_path1)
# check_fullwidth_space(out_path1)
# # find_lines_with_fullwidth_characters(out_path1)
# check_two_space(out_path1)
#
# print('\n-------- testset/tgt.txt -------')
# check_lines_starting_with(pattern,out_path2)
# check_fullwidth_space(out_path2)
# # find_lines_with_fullwidth_characters(out_path2)
# check_two_space(out_path2)
#
# print('\n---------- validset/src.txt -----')
# # replace_lines_with_no_leading_space(out_path3)
# # clean_two_space_chinese(out_path3)
# check_lines_starting_with(pattern,out_path3)
# check_fullwidth_space(out_path3)
# # find_lines_with_fullwidth_characters(out_path3)
# check_two_space(out_path3)
#
# print('\n--------- validset/tgt.txt ------')
# # replace_lines_with_no_leading_space(out_path4)
# # clean_two_space_chinese(out_path4)
# check_lines_starting_with(pattern,out_path4)
# check_fullwidth_space(out_path4)
# # find_lines_with_fullwidth_characters(out_path4)
# check_two_space(out_path4)
#
# print('\n--------- trainset/src.txt ------')
# # replace_lines_with_no_leading_space(out_path5)
# # clean_two_space_chinese(out_path5)
# check_lines_starting_with(pattern,out_path5)
# check_fullwidth_space(out_path5)
# # find_lines_with_fullwidth_characters(out_path5)
# check_two_space(out_path5)
# #
# print('\n---------- trainset/tgt.txt -----')
# # replace_lines_with_no_leading_space(out_path6)
# # clean_two_space_chinese(out_path6)
# check_lines_starting_with(pattern,out_path6)
# check_fullwidth_space(out_path6)
# # find_lines_with_fullwidth_characters(out_path6)
# check_two_space(out_path6)
