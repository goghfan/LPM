#!/bin/bash

# --- 配置 ---

# 1. 设置Python脚本的路径
PYTHON_SCRIPT="register.py"

# 2. 设置输出结果的根目录
OUTPUT_DIR="result"

# 3. 设置固定的参数
# (使用 \ 来换行，使其更易读)
BASE_CMD="python $PYTHON_SCRIPT \
--scene_encoder_path ./checkpoints-1030/best_scene_encoder.pth \
--image_encoder_path ./checkpoints-1030/best_image_encoder.pth \
--live_xray_path live_xray_for_registration.png \
--h5_dataset_path F:/desktop/2D-3D/LPM/LPM/drr_pose_dataset_with_landmarks.h5 \
--specimen_id 1 \
--feature_dim 1536 \
--image_height 256 \
--image_width 256 \
--feature_height 8 \
--feature_width 8 \
--n_samples_per_ray 128 \
--num_iterations 200 \
--lr_pose 1e-3 \
--generate_visualization"

# --- 脚本执行 ---

# 创建输出目录 (如果它不存在)
mkdir -p $OUTPUT_DIR

echo "开始参数扫描，结果将保存到 $OUTPUT_DIR 文件夹..."

# 循环遍历 focal_length (10, 110, 210, ..., 910)
# 注意：seq 的最后一个参数是上限，但循环不一定会包含它，
# seq 10 100 1000 会生成 10, 110, ..., 910。
# 如果您想包含 1000，请将 1000 改为 1010。这里我先按您的描述(从10到1000)来设置。
for focal in $(seq 10 100 1000); do
  # 循环遍历 near_plane (10, 110, 210, ..., 910)
  for near in $(seq 10 100 1000); do
    # 循环遍历 far_plane (100, 200, 300, ..., 1000)
    for far in $(seq 100 100 1000); do

      # --- 逻辑检查 ---
      # 保证 near_plane 必须小于 far_plane
      if [ $near -ge $far ]; then
        echo "跳过无效组合: focal=$focal, near=$near, far=$far (near >= far)"
        continue # 跳过当前循环，进入下一个
      fi

      echo "--- 正在运行: focal=$focal, near=$near, far=$far ---"

      # 1. 根据参数组合创建动态的文件名
      FILENAME_TAG="focal${focal}-near${near}-far${far}"
      OUTPUT_PNG_PATH="$OUTPUT_DIR/final_registered_drr-$FILENAME_TAG.png"
      OUTPUT_NPY_PATH="$OUTPUT_DIR/final_registered_pose-$FILENAME_TAG.npy"

      # 2. 组合完整的命令
      FULL_CMD="$BASE_CMD \
      --focal_length $focal \
      --near_plane $near \
      --far_plane $far \
      --output_pose_path $OUTPUT_NPY_PATH \
      --output_drr_path $OUTPUT_PNG_PATH"

      # 3. 打印并执行命令
      echo $FULL_CMD
      $FULL_CMD

      echo "--- 完成: $FILENAME_TAG ---"
      echo "" # 添加空行以便阅读

    done
  done
done

echo "所有参数组合已执行完毕。"