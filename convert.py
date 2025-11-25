import os
import subprocess

# 设置包含 .json 文件的文件夹路径
json_folder_path = ' '

# 获取文件夹中所有的 .json 文件
json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

# 遍历每个 .json 文件，使用 labelme_json_to_dataset 转换
for json_file in json_files:
    json_file_path = os.path.join(json_folder_path, json_file)

    # 提取文件名（去除路径和扩展名）
    file_name = os.path.splitext(json_file)[0]

    # 为每个 .json 文件创建一个同名的文件夹（加上 "_json" 后缀）
    output_folder = os.path.join(json_folder_path, file_name + '_json')
    os.makedirs(output_folder, exist_ok=True)  # 如果文件夹已存在则不会报错

    # 使用 subprocess 调用 labelme_json_to_dataset 命令，生成数据集
    command = f'labelme_json_to_dataset {json_file_path} -o {output_folder}'

    # 执行命令
    try:
        subprocess.run(command, shell=True, check=True)
        print(f'转换成功: {json_file} 到 {output_folder}')
    except subprocess.CalledProcessError as e:
        print(f'转换失败: {json_file}. 错误信息: {e}')
    # os.system("labelme_json_to_dataset %s" %(path + file))