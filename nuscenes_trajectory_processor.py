from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np
import json
import os
# ===== 配置路径 =====
DATAROOT = r"F:\v1.0-mini"
OUTPUT_PATH = r"F:\v1.0-mini\all_trajectories.json"  # 明确输出路径
nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)

def get_trajectory(nusc, instance_token):
    """
    获取某个 instance 在当前场景中的完整轨迹（过去+未来）
    返回: list of (x, y, yaw, timestamp)
    """
    traj = []

    # 找到该 instance 出现的所有样本
    instance = nusc.get('instance', instance_token)
    current_ann_token = instance['first_annotation_token']

    while current_ann_token != '':
        ann = nusc.get('sample_annotation', current_ann_token)

        # 直接使用当前标注的信息
        translation = ann['translation']  # [x, y, z]
        rotation = ann['rotation']        # [w, x, y, z]
        yaw = Quaternion(rotation).yaw_pitch_roll[0]  # 只取偏航角
        timestamp = nusc.get('sample', ann['sample_token'])['timestamp']
        traj.append((translation[0], translation[1], yaw, timestamp))

        if current_ann_token == instance['last_annotation_token']:
            break
        current_ann_token = ann['next']

    return traj

# ===== 主程序：遍历所有场景，提取所有轨迹 =====
scene_count = 0
agent_count = 0

# 存储所有轨迹数据的列表
all_trajectories = []

for scene in nusc.scene:  # 遍历所有场景
    scene_count += 1
    print(f"\n=== 场景 {scene_count}: {scene['name']} ===")

    # 获取该场景的第一个 sample
    sample_token = scene['first_sample_token']
    sample = nusc.get('sample', sample_token)

    # 遍历该场景中所有标注的 agent
    for ann_token in sample['anns']:  # 遍历所有 agent
        ann = nusc.get('sample_annotation', ann_token)
        instance_token = ann['instance_token']

        # 提取完整轨迹
        traj = get_trajectory(nusc, instance_token)

        if len(traj) < 2:
            continue

        agent_count += 1

        '''# 打印进度信息
        print(f"Agent {agent_count} (类别: {ann['category_name']}) - 轨迹长度: {len(traj)}")'''

        # 存储轨迹数据
        trajectory_info = {
            'agent_id': instance_token,
            'category': ann['category_name'],
            'full_trajectory': traj,  # 完整轨迹
            'scene_name': scene['name'],
            'scene_token': scene['token']
        }

        all_trajectories.append(trajectory_info)

        # 如果需要历史/未来分割
        mid = len(traj) // 2
        hist = traj[:mid]
        futu = traj[mid:]

        # 存储历史和未来轨迹
        trajectory_info['history'] = hist
        trajectory_info['future'] = futu

print(f"\n 全部处理完成！共处理 {scene_count} 个场景，{agent_count} 个 agent")
print(f"总共提取 {len(all_trajectories)} 条完整轨迹")

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 保存轨迹数据到文件
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(all_trajectories, f, indent=2, ensure_ascii=False)

print(f"轨迹数据已保存到: {OUTPUT_PATH}")

# ======================================================================================

# 第 1 步：从 nuScenes mini 提取所有训练样本（JSONL 格式）
# ======================================================================================

def get_full_trajectory(nusc, instance_token):
    """获取完整轨迹（所有时间步）"""
    traj = []
    instance = nusc.get('instance', instance_token)
    ann_token = instance['first_annotation_token']
    while ann_token != '':
        ann = nusc.get('sample_annotation', ann_token)
        x, y = ann['translation'][:2]
        yaw = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        ts = nusc.get('sample', ann['sample_token'])['timestamp']
        traj.append({'x': x, 'y': y, 'yaw': yaw, 'ts': ts})
        ann_token = ann['next']
    return traj

def find_current_index(traj, current_ts):
    """定位当前sample在轨迹中的索引（"""
    for i, p in enumerate(traj):
        if abs(p['ts'] - current_ts) < 1e5:  # 100ms容差
            return i
    return len(traj) // 2  # 降级方案（不应触发）

def format_traj(traj_list, num_points=6):
    pts = [(round(p['x'], 2), round(p['y'], 2)) for p in traj_list[:num_points]]
    return ", ".join([f"({x},{y})" for x, y in pts])

# === 主逻辑：收集所有样本（带缓存+精准切分）===
samples = []
trajectory_cache = {}  # 缓存避免重复计算
total_scenes = len(nusc.scene)
print(f" 开始处理 {total_scenes} 个场景...")

for scene_idx, scene in enumerate(nusc.scene):
    start_count = len(samples)
    print(f"\n=== 处理场景 {scene_idx+1}/{total_scenes}: {scene['name']} ===")

    sample_token = scene['first_sample_token']
    frame_count = 0

    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        current_ts = sample['timestamp']  # 获取当前帧时间戳
        frame_count += 1

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            category = ann['category_name']
            if not (category.startswith('vehicle.') or category.startswith('human.')):
                continue

            # 使用缓存避免重复提取
            inst_tok = ann['instance_token']
            if inst_tok not in trajectory_cache:
                trajectory_cache[inst_tok] = get_full_trajectory(nusc, inst_tok)
            traj = trajectory_cache[inst_tok]

            if len(traj) < 10:
                continue

            # 以当前帧为中心切分
            curr_idx = find_current_index(traj, current_ts)
            hist = traj[max(0, curr_idx-4):curr_idx]
            futu = traj[curr_idx:curr_idx+6]

            if len(hist) < 4 or len(futu) < 6:
                continue

            # 生成方向描述（使用当前帧朝向）
            curr_yaw = hist[-1]['yaw'] if hist else traj[curr_idx]['yaw']
            direction = "东" if np.cos(curr_yaw) > 0.5 else \
                       "西" if np.cos(curr_yaw) < -0.5 else \
                       "北" if np.sin(curr_yaw) > 0.5 else "南"

            input_text = f"一辆{category.split('.')[-1]}在过去2秒的位置为：{format_traj(hist)}，朝向{direction}，在直行车道上。请预测它未来3秒的位置。"
            output_text = format_traj(futu)

            samples.append({
                "input": input_text,
                "output": output_text,
                "agent_id": inst_tok,
                "category": category,
                "scene_name": scene['name'],
                "sample_token": sample['token'],
                "current_idx": curr_idx  # 便于调试
            })

        if frame_count % 50 == 0:
            print(f"  已处理 {frame_count} 帧 | 累计样本: {len(samples)}")
        sample_token = sample['next']

    new_samples = len(samples) - start_count
    print(f" 场景 {scene_idx+1} 完成 | 处理 {frame_count} 帧 | 新增样本: {new_samples}")

# === 保存 ===
os.makedirs("data", exist_ok=True)
output_file = "data/nuscenes_all_trajectories.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\n 处理完成！总样本数: {len(samples)} | 保存至: {output_file}")
if os.path.exists(output_file):
    size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f" 文件大小: {size_mb:.2f} MB | 样本示例:")
    # 打印1个样本示例验证
    with open(output_file, "r", encoding="utf-8") as f:
        example = json.loads(f.readline())
        print(f"  Input: {example['input'][:80]}...")
        print(f"  Output: {example['output']}")