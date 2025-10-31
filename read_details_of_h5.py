import h5py
import sys
import os # 导入 os 模块用于处理文件路径

def print_h5_tree(h5_object, file_handle, prefix=''):
    """
    以紧凑的树状结构将 HDF5 对象的层级写入到文件。

    参数:
    h5_object (h5py.File or h5py.Group): 要探查的 HDF5 文件或组对象。
    file_handle (file): 用于写入的已打开的文件句柄。
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
            # 【修改】将 print 重定向到文件句柄
            print(f"{prefix}{connector}📁 {key}", file=file_handle)
            # 为下一层递归准备新的前缀
            new_prefix = prefix + ('    ' if is_last else '│   ')
            # 【修改】递归调用时传入文件句柄
            print_h5_tree(item, file_handle, new_prefix)
            
        elif isinstance(item, h5py.Dataset):
            # 【修改】将 print 重定向到文件句柄
            # 将数据集信息打印在一行
            print(f"{prefix}{connector}📄 {key} (Shape: {item.shape}, Dtype: {item.dtype})", file=file_handle)

def main():
    """
    主函数
    """
    # --- 文件路径 ---
    file_path = r'F:\desktop\2D-3D\LPM\LPM\data\ipcai_2020_full_res_data\ipcai_2020_full_res_data.h5'
    
    # 【新增】自动生成输出的 txt 文件路径
    # 例如：'F:\...\drr_pose_dataset_with_landmarks.h5' -> 'F:\...\drr_pose_dataset_with_landmarks_structure.txt'
    base_name = os.path.splitext(file_path)[0]
    output_txt_path = f"{base_name}_structure.txt"

    # --- 请在这里指定您想读取的组的名称 ---
    group_name_to_read = '17-1882' # 示例组，您可以改成 'proj-params' 等

    try:
        with h5py.File(file_path, 'r') as hf:
            # 状态信息打印到控制台
            print(f"--- 成功打开 HDF5 文件: {file_path} ---")

            if group_name_to_read in hf:
                target_group = hf[group_name_to_read]
                
                # 【修改】使用 'with open' 来写入目标 txt 文件
                print(f"\n正在将 '{group_name_to_read}' 组的结构树写入文件...")
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    # 打印初始的组名作为树的根（写入文件）
                    print(f"📁 {group_name_to_read}", file=f)
                    # 【核心改动】调用新的树状打印函数，并传入文件句柄 'f'
                    print_h5_tree(target_group, f, prefix=' ')
                
                # 完成后在控制台打印确认信息
                print(f"\n--- 结构树已成功写入到: {output_txt_path} ---")
                
            else:
                # 错误信息打印到控制台
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