import os

# 原始标注文件夹
annotation_folder = "SIMD/test_anno"  # 替换成你的原始标注文件夹路径
# 新的标注文件夹
new_annotation_folder = "SIMD/new_test_anno"  # 替换成你的新文件夹路径

# 创建新的标注文件夹（如果不存在）
os.makedirs(new_annotation_folder, exist_ok=True)

# 遍历所有 txt 文件
for file_name in os.listdir(annotation_folder):
    if file_name.endswith(".txt"):
        src_path = os.path.join(annotation_folder, file_name)
        dst_path = os.path.join(new_annotation_folder, file_name)
        
        # 读取并修改类别 ID
        with open(src_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:  # 确保行不为空
                parts[0] = str(int(parts[0]) + 6)  # 类别 ID +6
                new_lines.append(" ".join(parts))
        
        # 保存修改后的内容到新文件夹
        with open(dst_path, "w") as f:
            f.write("\n".join(new_lines))

print(f"所有修改后的 txt 文件已保存到 '{new_annotation_folder}'！")