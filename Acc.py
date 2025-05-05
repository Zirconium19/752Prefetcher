import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 配置区 ---
csv_file_path = 'Peformance_100m.csv'  # 你的CSV文件路径

# !!! 请再次仔细检查并确保这些列名与你的CSV文件中的列名完全一致 !!!
x_axis_col = 'Trace'          # 包含X轴类别 (Trace) 的列名
grouping_col = 'Exp'          # 包含分组类别 (Prefetcher/Exp) 的列名
value_col = 'LLC_prefetch_acc'  # 包含Y轴数值 (LLC_prefetch_acc) 的列名 - 已改为预取准确率

# --- 图表样式配置 ---
chart_title = 'Figure 6: LLC Accuracy results for benchmarks'            # 图表标题
y_axis_label = 'LLC Prefetch Accuracy'  # Y轴标签 - 已更新
x_axis_label = 'Benchmarks'   # X轴标签
y_axis_min = 0.0              # Y轴最小值
y_axis_max = 0.5              # Y轴最大值 - 调整为0.5

# --- 控制柱子间距的关键参数 ---
bar_width = 0.15              # 每个柱子的固定宽度
bar_spacing = 0.0             # 同一组内柱子之间的间距 (0表示无间距)
group_spacing = 0.8           # 不同trace组之间的间距

# --- 图表尺寸和布局参数 ---
figure_width = 12             # 图表宽度 (英寸)
figure_height = 8             # 图表高度 (英寸)
rotate_x_labels = 45           # X轴标签旋转角度

# --- 颜色映射设置 ---
# 创建与目标图像匹配的自定义颜色列表
custom_colors = [
    '#1f74b0',  # 深蓝 - No prefetching
    '#3d9f42',  # 绿色 - SPP
    '#bbc7ca',  # 浅灰 - Bingo
    '#7fc5e5',  # 浅蓝 - Mlop
    '#7f61b1',  # 紫色 - Pythia
    '#ff7923',  # 橙色 - MPMLP(10M)
    '#66ccff',  # 浅蓝色 - MPMLP.v2(10M)
    '#ce9462',  # 棕色 - MPMLP(100M)
    '#de321f',  # 红色 - MPMLP.v4(10M)
    '#8b5147',  # 深棕 - MPMLP.v5.nseq(10M)
    '#d876b9',  # 粉红 - MPMLP.v5.nseq(100M)
    '#768281',  # 灰色 - MPMLP.v5.seq(10M)
    '#b7b93c'   # 橄榄绿 - MPMLP.v5.seq(100M)
]

# 定义显示标签映射
label_mapping = {
    'nopref': 'No prefetching',
    'spp': 'SPP',
    'bingo': 'Bingo',
    'mlop': 'Mlop',
    'pythia': 'Pythia',
    'MPMLP(10M)': 'MPMLP(10M)',
    'MPMLP(100M)': 'MPMLP(100M)',
    'Classified MPMLP(10M)': 'MPMLP.v2(10M)',
    'Classified.v4 MPMLP(10M)': 'MPMLP.v4(10M)',
    'Classified.v5 MPMLP(10M-nseq)': 'MPMLP.v5.nseq(10M)',
    'Classified.v5 MPMLP(100M-nseq)': 'MPMLP.v5.nseq(100M)',
    'Classified.v5 MPMLP(10M-seq)': 'MPMLP.v5.seq(10M)',
    'Classified.v5 MPMLP(100M-seq)': 'MPMLP.v5.seq(100M)'
}

# 定义颜色映射
color_mapping = {
    'nopref': custom_colors[0],            # 深蓝
    'spp': custom_colors[1],               # 绿色 
    'bingo': custom_colors[2],             # 浅灰
    'mlop': custom_colors[3],              # 浅蓝
    'pythia': custom_colors[4],            # 紫色
    'MPMLP(10M)': custom_colors[5],        # 橙色
    'Classified MPMLP(10M)': custom_colors[6],  # 浅蓝色
    'MPMLP(100M)': custom_colors[7],       # 棕色
    'Classified.v4 MPMLP(10M)': custom_colors[8],  # 红色
    'Classified.v5 MPMLP(10M-nseq)': custom_colors[9],  # 深棕
    'Classified.v5 MPMLP(100M-nseq)': custom_colors[10], # 粉红  
    'Classified.v5 MPMLP(10M-seq)': custom_colors[11],  # 灰色
    'Classified.v5 MPMLP(100M-seq)': custom_colors[12]  # 橄榄绿
}

# 定义按照图例显示顺序的分组标签（这会影响柱子的顺序）
ordered_labels = [
    'MPMLP(10M)',        # 橙色
    'MPMLP(100M)',       # 棕色       
    'MPMLP.v2(10M)',     # 浅蓝色
    'MPMLP.v4(10M)',     # 红色
    'MPMLP.v5.nseq(10M)',  # 深棕
    'MPMLP.v5.nseq(100M)', # 粉红
    'MPMLP.v5.seq(10M)',   # 灰色
    'MPMLP.v5.seq(100M)'   # 橄榄绿
]

# 创建反向映射：从标签到原始类别
reverse_label_mapping = {}
for cat, label in label_mapping.items():
    reverse_label_mapping[label] = cat

# 1. 加载数据
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"错误：找不到文件 '{csv_file_path}'。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"读取CSV文件时出错: {e}")
    exit()

# 检查所需列是否存在
required_cols = [x_axis_col, grouping_col, value_col]
if not all(col in df.columns for col in required_cols):
    print(f"\n错误：CSV文件中缺少必要的列。")
    print(f"代码需要的列名: {required_cols}")
    print(f"CSV文件实际包含的列名: {df.columns.tolist()}")
    print(f"请仔细检查CSV文件或修改代码中的配置变量。")
    exit()

# 确保数值列是数字类型
try:
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    if df[value_col].isnull().all():
         print(f"\n错误：列 '{value_col}' 中所有值都无法转换为数字。请检查数据。")
         exit()
    elif df[value_col].isnull().any():
        print(f"\n注意：列 '{value_col}' 中部分值无法转换为数字，这些值将在绘图中被跳过。")
        
    # 将百分比值转换为0-1范围的小数
    # 检查值是否为百分比格式（大于1）
    if df[value_col].max() > 1:
        print(f"\n注意：检测到'{value_col}'列中的值可能是百分比格式，正在将其转换为0-1范围...")
        df[value_col] = df[value_col] / 100.0

except Exception as e:
     print(f"\n错误：在尝试转换列 '{value_col}' 为数字时出错: {e}")
     exit()

# 获取原始顺序的分组类别 (Exp/Prefetcher)
original_group_order = df[grouping_col].unique()

# 2. 数据整理：使用 pivot_table 进行透视
try:
    pivot_df = pd.pivot_table(df,
                              values=value_col,
                              index=x_axis_col,
                              columns=grouping_col,
                              aggfunc='mean') # 默认使用平均值处理重复项

    # 按原始顺序重新索引列
    pivot_df = pivot_df.reindex(columns=original_group_order)

except Exception as e:
    print(f"\n数据透视 (pivot_table) 时出错: {e}")
    exit()

# 获取X轴类别 (Traces) 和 按原始顺序排列的分组类别 (Exps)
x_categories = pivot_df.index.astype(str).tolist()
group_categories = pivot_df.columns.tolist()
num_x_categories = len(x_categories)
num_group_categories = len(group_categories)

if num_x_categories == 0 or num_group_categories == 0:
    print("\n错误：数据透视后没有有效的X轴类别或分组类别，无法绘图。")
    exit()

# 添加调试信息
print("\n=== 调试信息 ===")
print("所有分组类别:", group_categories)
print("分组类别数量:", len(group_categories))
print("预定义颜色数量:", len(custom_colors))

# 打印实际的分组类别，帮助调试
print("\n实际的分组类别:")
for i, category in enumerate(group_categories):
    print(f"{i}: {category}")

# 4. 创建图表
plt.rcParams.update({'font.size': 11})  # 全局字体大小
fig, ax = plt.subplots(figsize=(figure_width, figure_height))

# 创建按顺序排列的类别列表（用于绘制柱子）
ordered_categories = []
for label in ordered_labels:
    # 查找对应的原始类别
    found = False
    for cat, mapped_label in label_mapping.items():
        if mapped_label == label and cat in group_categories:
            ordered_categories.append(cat)
            found = True
            break
    if not found:
        print(f"警告: 找不到标签为 '{label}' 的分组类别")

# 使用新的绘图方法，确保柱子紧密排列
x_positions = []  # 保存每个trace的中心位置
for i, x_cat in enumerate(x_categories):
    # 获取当前trace的所有非NaN值
    trace_data = pivot_df.loc[x_cat]
    
    # 创建临时字典来存储每个类别的值，便于按顺序访问
    category_data = {}
    for category in group_categories:
        value = trace_data[category]
        if not pd.isna(value):  # 只保留非NaN值
            category_data[category] = value
    
    # 显示哪些类别有数据
    print(f"\nTrace {x_cat} 有数据的类别:")
    for cat in category_data:
        print(f"  {cat}: {category_data[cat]}")
    
    # 按照ordered_categories的顺序处理数据
    valid_categories = []
    valid_values = []
    valid_colors = []
    valid_labels = []
    
    for category in ordered_categories:
        if category in category_data:
            value = category_data[category]
            valid_categories.append(category)
            valid_values.append(value)
            color = color_mapping.get(category, custom_colors[5 + len(valid_categories) % 8])  # 从第5个颜色开始，因为前5个是其他prefetchers
            valid_colors.append(color)
            label = label_mapping.get(category, category)
            valid_labels.append(label)
    
    # 计算当前trace的位置
    x_pos = i * (len(ordered_categories) * (bar_width + bar_spacing) + group_spacing)
    x_positions.append(x_pos + (len(valid_categories) * bar_width) / 2)  # 保存中心位置用于设置X轴刻度
    
    # 绘制当前trace的所有有效柱子
    for j, (value, color, label, category) in enumerate(zip(valid_values, valid_colors, valid_labels, valid_categories)):
        bar_pos = x_pos + j * (bar_width + bar_spacing)
        # 只在第一个trace时添加图例标签，避免重复
        if i == 0:
            rect = ax.bar(bar_pos, value, bar_width, color=color, label=label, align='edge', edgecolor=color, linewidth=0)
        else:
            rect = ax.bar(bar_pos, value, bar_width, color=color, align='edge', edgecolor=color, linewidth=0)
        
        # # 添加柱子顶部百分比标签
        # percentage = value * 100  # 转换回百分比
        # ax.text(bar_pos + bar_width/2, value + 0.01, f'{percentage:.2f}%', 
        #         ha='center', va='bottom', fontsize=8, rotation=45, color='transparent')
                
        print(f"在位置 {bar_pos} 绘制 {x_cat} 的 {category} 柱子，值 = {value}")

# 设置X轴刻度和标签位置
ax.set_xticks(x_positions)
ax.set_xticklabels(x_categories, rotation=rotate_x_labels, ha='right', fontsize=12)

# 5. 添加图表元素和样式
ax.set_ylabel(y_axis_label, fontsize=14, fontweight='bold')
ax.set_xlabel(x_axis_label, fontsize=14, fontweight='bold')
ax.set_title(chart_title, fontsize=16, fontweight='bold', loc='left')

# 设置Y轴刻度字体
ax.tick_params(axis='y', labelsize=12)

# 添加网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True) # 让网格线在柱子后面

# 设置Y轴范围
if y_axis_min is not None and y_axis_max is not None:
    ax.set_ylim(bottom=y_axis_min, top=y_axis_max)
else:
    # 自动计算适当的Y轴范围
    all_values = pivot_df.values.flatten()
    valid_values = all_values[~np.isnan(all_values)]
    if len(valid_values) > 0:
        current_ymin, current_ymax = min(valid_values), max(valid_values)
        adjusted_ymin = 0 if current_ymin >= 0 else current_ymin - (current_ymax - current_ymin) * 0.05
        adjusted_ymax = current_ymax + (current_ymax - current_ymin) * 0.08
        ax.set_ylim(bottom=adjusted_ymin, top=adjusted_ymax)

# 创建一个有序的图例
handles = []
labels = []

# 为每个顺序类别创建一个图例条目
for i, category in enumerate(ordered_categories):
    label = label_mapping.get(category, category)
    color = color_mapping.get(category, custom_colors[5 + i])  # 从第5个颜色开始
    
    # 创建一个小矩形作为该类别的图例条目
    handle = plt.Rectangle((0, 0), 1, 1, color=color, edgecolor='black', linewidth=0.5)
    handles.append(handle)
    labels.append(label)
    print(f"添加图例: {category} -> {label}, 颜色: {color}")

# 优化布局并将图例放在右侧
fig.tight_layout(rect=[0, 0, 0.85, 1])  # 调整右边界给图例留空间
legend = ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

plt.savefig('Figure2.pdf', bbox_inches='tight', format='pdf')
# 6. 显示图表
plt.show()

print("\n图表已更新。")