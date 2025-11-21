import os
import shutil

def classify_wav_files(root_folder, output_folder_name):
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建一个大文件夹来存放所有分类后的文件夹
    main_output_folder = os.path.join(script_dir, output_folder_name)
    os.makedirs(main_output_folder, exist_ok=True)

    # 创建一个字典来存储按倒数第二位分类的文件
    wav_files_dict = {}

    # 遍历根文件夹中的所有文件和子文件夹
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".wav"):
                # 获取文件名倒数第二位的值
                second_last_char = file[-6]
                # 如果该值还没有对应的文件夹，创建一个
                if second_last_char not in wav_files_dict:
                    wav_files_dict[second_last_char] = []
                wav_files_dict[second_last_char].append(os.path.join(subdir, file))

    # 为每个分类创建一个新的文件夹，并将文件移动过去
    for char, files in wav_files_dict.items():
        # 创建以倒数第二位值命名的文件夹
        char_folder = os.path.join(main_output_folder, char)
        os.makedirs(char_folder, exist_ok=True)
        for file_path in files:
            # 获取文件名
            file_name = os.path.basename(file_path)
            # 复制文件到新的文件夹
            shutil.copy(file_path, os.path.join(char_folder, file_name))
            print(f"Copied {file_name} to {char_folder}")

    print(f"Processing complete. All classified folders are saved in {main_output_folder}")

# 替换为你的根文件夹路径
root_folder_path = r"E:\数据库\EMODB数据库\wav"
# 替换为你想要创建的大文件夹的名字
output_folder_name = "classified_wav_files"
classify_wav_files(root_folder_path, output_folder_name)