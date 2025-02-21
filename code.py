'''
目录
模块一、沿b轴方向晶格常数热力图绘制	2
模块二、双取代结构热力图	3
模块三、gap与导电性质热力图	4
模块四、D带中心热力图	6
模块五、能带和dos图绘制	7
'''





##模块一、沿b轴方向晶格常数热力图绘制
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_b_heatmap(file_path, sheet_name, output_path):
    """
    读取给定Excel文件路径和工作表名称的数据，并绘制d带中心的热力图，然后保存到指定路径
    并返回保存热力图的文件路径
    :param file_path: Excel文件的路径
    :param sheet_name: 工作表名称
    :param output_path: 保存热力图的文件路径
    :return: 保存热力图的文件路径
    """
    # 使用pandas读取指定工作表
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

    # 创建热力图，设置数值的字体大小和保留小数位数
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5,
                xticklabels=df.columns, yticklabels=df.index,
                annot_kws={'size': 8})  # 设置热力图中数值的字体大小

    # 添加标题
    plt.title('Heatmap of d-band Center', fontsize=16, fontweight='bold')

    # 设置x轴和y轴标签的旋转角度和字体大小
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # 保存热力图到指定路径
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

    # 显示热力图
    plt.close()  # 关闭图像，避免在内存中占用资源

    # 返回保存热力图的文件路径
    return output_path

# 主程序，调用函数并传入用户输入的文件路径、工作表名称和输出图像路径
if __name__ == "__main__":
    # 让用户输入Excel文件路径和工作表名称
    file_path = r' 1.xlsx'
    sheet_name = '1'
    output_path = r'E:\桌面\heatmap.png'  # 指定保存热力图的路径

    # 调用函数并获取保存的图像路径
    saved_image_path = plot_b _heatmap(file_path, sheet_name, output_path)
    print(f"Heatmap saved to: {saved_image_path}")





##模块二、双取代结构热力图
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def plot_heatmap_from_excel(file_path, sheet_name):
    """
    从指定的Excel文件和工作表中读取数据并绘制热力图
    :param file_path: Excel文件的路径
    :param sheet_name: Excel文件中的工作表名称
    """
    try:
        # 从Excel文件读取数据
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # 设置索引
        index = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
                 "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", 
                 "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]
        data.index = index

        # 绘制热力图
        sns.set(font_scale=0.8)
        plt.figure(figsize=(10, 8))
        plot = sns.heatmap(data, cmap="YlGnBu", vmax=2, vmin=-2, annot=True, fmt=".1f")
        
        # 添加标题和标签
        plt.xlabel("Metal Replacement Element", size=15)
        plt.ylabel("Metal Replacement Element", size=15)
        plt.title("Ef Calculation of Bimetallic Atomic Substitution in DDA-Cu Bimetallic Systems", size=20)
        
        # 设置轴标签的旋转角度
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        
        # 显示图像
        plt.show()

    except Exception as e:
        print(f"读取Excel文件或绘图时出错：{e}")


if __name__ == "__main__":
    # 指定Excel文件路径和工作表
    file_path = "形成能PLUS.xlsx"
    sheet_name = "Sheet2"
    
    # 调用绘图函数
plot_heatmap_from_excel(file_path, sheet_name)









##模块三、gap与导电性质热力图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap

def plot_bandgap_heatmap(file_path):
    """
    读取给定Excel文件路径中的数据并绘制Bandgap的热力图，同时进行特殊标记
    :param file_path: Excel文件的路径
    """
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=0)  # 读取第一个sheet数据
    sheet3 = pd.read_excel(file_path, sheet_name=1, usecols="A:AC", nrows=26, header=None)  # 读取第三个sheet的A1到AC26范围内数据，并包含第一行

    # 定义刻度标签
    labels = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
              'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
              'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

    # 确保数据的第一行及以后的数据作为数据部分
    df = df.iloc[0:, 1:]

    # 将数据类型转换为数值型，强制转换所有数据
    df = df.apply(pd.to_numeric, errors='coerce')  # 将无法转换的数据设为 NaN

    # 定义颜色映射，使用渐深颜色和特殊的红色标记
    colors = sns.color_palette("Blues", as_cmap=True)  # 使用渐深蓝色
    cmap = colors(np.linspace(0, 1, 256))
    cmap[0] = np.array([0.5, 0.8, 0, 1])  # 将 0 值设为红色
    cmap = ListedColormap(cmap)

    # 统计不同颜色区块的数量
    color_counts = {'Metal': 0, 'Semi-metal': 0, 'Semi-conductor': 0}
    for value in df.values.flatten():
        if pd.notna(value):  # 忽略 NaN 值
            if value == 0:
                color_counts['Metal'] += 1
            elif value < 0.1:
                color_counts['Semi-metal'] += 1
            elif value >= 0.1:
                color_counts['Semi-conductor'] += 1

    # 找出含有“half”的位置
    half_mask = sheet3.astype(str).applymap(lambda x: 'half' in x)

    # 绘制热力图
    plt.figure(figsize=(12, 10))  # 调整图表宽度以容纳左侧的图例
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, cbar=True, linewidths=0.5, linecolor='black', annot_kws={"size": 8})

    # 设置坐标轴标签
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0, ha='right')
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)

    # 添加标题和坐标轴标签
    plt.title('Bandgap Analysis')
    plt.xlabel('Elements')

    # 添加图例，设置背景为白色和调整字体大小
    handles = [
        Patch(color='red', label=f'Metal (0.0): {color_counts["Metal"]}'),
        Patch(color=cmap(25), label=f'Semi-metal (<0.1): {color_counts["Semi-metal"]}'),
        Patch(color=cmap(200), label=f'Semi-conductor (>=0.1): {color_counts["Semi-conductor"]}')
    ]
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize='large', frameon=True, facecolor='white', framealpha=1)

    # 在带有“half”的位置添加标记（例如圆圈）
    for (i, j), val in np.ndenumerate(half_mask):
        if val:
            plt.gca().add_patch(plt.Circle((j + 0.5, i + 0.5), 0.2, color='black', fill=False, lw=2))

    # 在数值为0.0的位置添加红色标记
    for (i, j), val in np.ndenumerate(df.values):
        if val == 0.0:
            plt.gca().add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=0.1))

    # 显示图表
    plt.show()

# 主程序，调用函数并传入用户输入的文件路径
if __name__ == "__main__":
    file_path = 'E:\\桌面\\bandgap.xlsx'  # 这里输入文件路径
    plot_bandgap_heatmap(file_path)
    
    
    
##模块四、D带中心热力图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def create_heatmap(xlsx_filename, sheet_name, output_filename=None):
    """
    创建热力图的主函数。

    参数:
    xlsx_filename: str, Excel文件的路径。
    sheet_name: str, 工作表的名称。
    output_filename: str, 可选，输出图片的文件名。
    """
    try:
        # 使用pandas读取指定工作表
        df = pd.read_excel(xlsx_filename, sheet_name=sheet_name, index_col=0)

        # 创建热力图，设置数值的字体大小和保留小数位数
        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5,
                              xticklabels=df.columns, yticklabels=df.index,
                              annot_kws={'size': 8})  # 设置热力图中数值的字体大小

        # 添加标题
        plt.title('Heatmap of d-band Center', fontsize=16, fontweight='bold')

        # 设置x轴和y轴标签的旋转角度和字体大小
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)

        # 如果指定了输出文件名，则保存图片
        if output_filename:
            plt.savefig(output_filename)

        # 显示热力图
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

# 主程序，调用函数并传入用户输入的文件路径
if __name__ == "__main__":

# 使用示例
xlsx_filename = r'E:\桌面\collected_summaries\D带中心整理.xlsx'
sheet_name = 'Bigger'
create_heatmap(xlsx_filename, sheet_name)





##模块五、能带和dos图绘制
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from matplotlib.gridspec import GridSpec

def replace_doscar_data(current_folder, reference_folder):
    current_doscar_path = os.path.join(current_folder, 'DOSCAR')
    reference_doscar_path = os.path.join(reference_folder, 'DOSCAR')

    with open(reference_doscar_path, 'r') as reference_file:
        for _ in range(5):
            next(reference_file)
        sixth_line = next(reference_file).strip().split()
        if len(sixth_line) >= 4:
            data_to_replace = sixth_line[3]
        else:
            print(f"参考文件 {reference_doscar_path} 中第六行第四列数据不完整。")
            return

    if os.path.exists(current_doscar_path):
        with open(current_doscar_path, 'r') as current_file:
            lines = current_file.readlines()

        if len(lines) >= 6:
            current_data = lines[5].strip().split()
            if len(current_data) >= 4:
                current_data[3] = data_to_replace
                lines[5] = ' '.join(current_data) + '\n'
            else:
                print(f"当前文件夹 {current_doscar_path} 中 DOSCAR 第六行数据不完整。")
                return
        else:
            print(f"当前文件夹 {current_doscar_path} 中 DOSCAR 文件行数不足。")
            return

        with open(current_doscar_path, 'w') as current_file:
            current_file.writelines(lines)
    else:
        print(f"找不到路径 {current_doscar_path} 下的 DOSCAR 文件。")

def run_vaspkit():
    os.system("vaspkit 211")
    os.system('echo "211" | vaspkit')

def load_pdos_data(file_path, spin):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    name = pd.read_csv(file_path, sep='\s+', header=None, nrows=1).values.tolist()[0]
    df = pd.read_csv(file_path, sep='\s+', header=None, names=name)[1:]
    df = df.apply(pd.to_numeric, errors='coerce')
    df['#Energy'] = df['#Energy'].astype(float)
    cond = df[(df['#Energy'] > -2) & (df['#Energy'] < 2)]

    return cond['#Energy'], cond['tot']

def plot_band_structure(band_data_file, dos_data_folder):
    data = np.loadtxt(band_data_file, comments='#')

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    k = data[:, 0]
    num_bands = int((len(data[0]) - 1) / 2)

    k_min = np.min(k)
    k_max = np.max(k)
    ignore_indices = np.where((k == k_min) | (k == k_max))[0]

    colors = ['red', 'blue']

    for i in range(num_bands):
        energy_up = data[:, 2 * i + 1]
        energy_down = data[:, 2 * i + 2]

        energy_up[ignore_indices] = np.nan
        energy_down[ignore_indices] = np.nan

        ax1.plot(k, energy_up, linestyle='-', color=colors[0], linewidth=5, label='Spin-Up' if i == 0 else "")
        ax1.plot(k, energy_down, linestyle='-', color=colors[1], linewidth=1.5, label='Spin-Down' if i == 0 else "")

    sorted_data = np.unique(k)
    second_smallest = sorted_data[2]
    second_largest = sorted_data[-2]

    ax1.set_xlabel('Wave Vector')
    ax1.set_ylabel('Energy (eV)')
    ax1.set_title('Band Structure')
    ax1.grid(True)
    ax1.set_ylim(-2, 2)
    ax1.set_xlim(second_smallest, second_largest)
    ax1.legend(fontsize=15)

    ax1.text(0, -2.3, 'Γ', fontsize=12, ha='center')
    ax1.text(np.max(k), -2.3, 'Y', fontsize=12, ha='center')

    ax1.set_xticks([])

    folder_path = os.getcwd()
    element_combo = folder_path.split("/")[-1]
    file_name = folder_path.split("/")[-2] + "/" + folder_path.split("/")[-1]
    DOS_file = os.path.join("/home/xuff/zy/dos/01DRAWING", file_name)

    replace_doscar_data(folder_path, DOS_file)

    num = re.findall('[A-Z][^A-Z]*', element_combo)
    M1, M2 = num[0], num[1]

    elements = [M1, M2, 'C', 'N', 'O', 'H']
    y_data_up = []
    y_data_dw = []

    for element in elements:
        x, y_up = load_pdos_data(f'PDOS_{element}_UP.dat', os.path.join(folder_path, 'PDOS_{element}_UP.dat'))
        _, y_dw = load_pdos_data(f'PDOS_{element}_DW.dat', os.path.join(folder_path, 'PDOS_{element}_DW.dat'))
        y_data_up.append(y_up)
        y_data_dw.append(y_dw)

    x, y_1_up = load_pdos_data(f'PDOS_{M1}_UP.dat', os.path.join(folder_path, 'PDOS_{M1}_UP.dat'))
    _, y_1_dw = load_pdos_data(f'PDOS_{M1}_DW.dat', os.path.join(folder_path, 'PDOS_{M1}_DW.dat'))
    x, y_2_up = load_pdos_data(f'PDOS_{M2}_UP.dat', os.path.join(folder_path, 'PDOS_{M2}_UP.dat'))
    _, y_2_dw = load_pdos_data(f'PDOS_{M2}_DW.dat', os.path.join(folder_path, 'PDOS_{M2}_DW.dat'))

    y_data_up = [pd.to_numeric(y, errors='coerce').fillna(0) for y in y_data_up]
    y_data_dw = [pd.to_numeric(y, errors='coerce').fillna(0) for y in y_data_dw]

    y_1_up = pd.to_numeric(y_1_up, errors='coerce').fillna(0)
    y_1_dw = pd.to_numeric(y_1_dw, errors='coerce').fillna(0)
    y_2_up = pd.to_numeric(y_2_up, errors='coerce').fillna(0)
    y_2_dw = pd.to_numeric(y_2_dw, errors='coerce').fillna(0)

    total_y_up = sum(y_data_up)
    total_y_dw = sum(y_data_dw)

    ax2.fill_betweenx(x, total_y_up, where=(total_y_up >= 0), color='grey', label='Total', alpha=0.3)
    ax2.fill_betweenx(x, total_y_dw, where=(total_y_dw <= 0), color='grey', alpha=0.3)

    ax2.plot(y_1_up, x, color='deeppink', label=f'{M1}', linewidth=1)
    ax2.plot(y_1_dw, x, color='deeppink', linewidth=1)

    if M1 == M2:
        ax2.plot(y_2_up, x, color='blue', linewidth=1)
    else:
        ax2.plot(y_2_up, x, color='blue', label=f'{M2}', linewidth=1)
        ax2.plot(y_2_dw, x, color='blue', linewidth=1)

    ax2.set_xlabel('DOS')
    ax2.set_title('Density of States (DOS)')
    ax2.grid(True)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-2, 2)
    ax2.legend(fontsize=15)

    ax2.yaxis.set_tick_params(labelleft=False)
    ax2.yaxis.set_ticks_position('none')

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{element_combo}.png")

def main(current_folder, reference_folder, band_data_file, dos_data_folder):
    run_vaspkit()
    plot_band_structure(band_data_file, dos_data_folder)

if __name__ == "__main__":
    current_folder = os.getcwd()
    reference_folder='/home/xuff/zy/dos/01DRAWING/05Mn/16MnRu'
    # 设置路径
    #reference_folder = "/path/to/reference/folder"  # 替换为参考文件夹路径
    band_data_file = current_folder+"/BAND.dat"  # 替换为BAND.dat文件的路径
    dos_data_folder = current_folder+"/DOSCAR"  # 替换为DOS数据文件夹的路径

    main(current_folder, reference_folder, band_data_file, dos_data_folder)
