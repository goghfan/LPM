import h5py
import sys

def print_h5_tree(h5_object, prefix=''):
    """
    以紧凑的树状结构打印 HDF5 对象的层级。

    参数:
    h5_object (h5py.File or h5py.Group): 要探查的 HDF5 文件或组对象。
    prefix (str): 用于构建树状线条的前缀。
    """
    # 获取当前层级下所有条目的列表，以便判断哪个是最后一个
    keys = list(h5_object.keys())
    for i, key in enumerate(keys):
        item = h5_object[key]
        
        # 判断是否为当前层级的最后一个条目
        is_last = (i == len(keys) - 1)
        
        # 根据是否为最后一个条目，选择不同的连接符
        connector = '└── ' if is_last else '├── '
        
        if isinstance(item, h5py.Group):
            print(f"{prefix}{connector}📁 {key}")
            # 为下一层递归准备新的前缀
            new_prefix = prefix + ('    ' if is_last else '│   ')
            print_h5_tree(item, new_prefix)
            
        elif isinstance(item, h5py.Dataset):
            # 将数据集信息打印在一行
            print(f"{prefix}{connector}📄 {key} (Shape: {item.shape}, Dtype: {item.dtype})")

def main():
    """
    主函数
    """
    # --- 文件路径 ---
    file_path =r'F:\desktop\2D-3D\LPM\LPM\drr_pose_dataset_with_landmarks.h5'
    
    # --- 请在这里指定您想读取的组的名称 ---
    group_name_to_read = 'specimen_1' # 示例组，您可以改成 'proj-params' 等

    try:
        with h5py.File(file_path, 'r') as hf:
            print(f"--- 成功打开 HDF5 文件: {file_path} ---")

            if group_name_to_read in hf:
                print(f"\n'{group_name_to_read}' 组的结构树:\n")
                target_group = hf[group_name_to_read]
                
                # 打印初始的组名作为树的根
                print(f"📁 {group_name_to_read}")
                # 【核心改动】调用新的树状打印函数
                print_h5_tree(target_group, prefix=' ')
                
                print(f"\n--- 探查完成 ---")
            else:
                print(f"\n错误: 未找到名为 '{group_name_to_read}' 的组。", file=sys.stderr)
                available_groups = list(hf.keys())
                print("文件顶层可用的组包括:", file=sys.stderr)
                for group_name in available_groups:
                    print(f" - {group_name}", file=sys.stderr)

    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径 '{file_path}'", file=sys.stderr)
    except Exception as e:
        print(f"读取文件时发生了一个错误: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()