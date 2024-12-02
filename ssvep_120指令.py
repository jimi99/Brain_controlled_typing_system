# %% load modules
import string
import math
import numpy as np
from numpy.core.fromnumeric import var

from psychopy import (core, data, visual, monitors, event)  # core核心功能 data数据处理 visual视觉刺激 monitor显示器设置 enent事件处理
from ex_base import NeuroScanPort  # 端口控制

# %%
'''1.gaussian函数：计算并返回高斯分布的概率密度函数在给定点x处的值'''

def gaussian(x, sigma, miu):
    left_part = 1. / (math.sqrt(2 * math.pi) * sigma)
    up_denom = -(2 * sigma ** 2)
    up_numer = np.power((x - miu), 2.)
    return left_part * np.exp(up_numer / up_denom)

# load in code infomation
global code_series, n_codes  # code_series：编码序列；n_codes：全局变量

event.clearEvents()  # event模块在PsychoPy中通常用于处理键盘、鼠标等输入事件
event.globalKeys.add(key='escape', func=core.quit)  # 按esc键执行ending the experiment

# lina修改1(这里要用师姐的代码，串口)
# port_available = True
# port_address = 0x4FF8
#port = NeuroScanPort(port_address=port_address)

code_series = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1],
               [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0],
               [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0],
               [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0],
               [0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1],
               [1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0],
               [1, 0, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0],
               [1, 1, 0, 1, 0, 0], [1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 1],
               [1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0]]
code_series = np.array(code_series).T  # 先转化为array数组，在.T转置，转置对二位数组来说就是交换行和列，
n_codes = code_series.shape[0]  # n_codes=一个方块需要几位01编码

win = visual.Window([1920, 1080], color=(-1, -1, -1), fullscr=False, monitor='testmonitor',
                    units='pix', screen=0, waitBlanking=False, allowGUI=True)
# fullscr=True全屏模式 monitor='testmonitor'指定了使用的显示器 units='pix'像素单位
# screen=0主屏幕  allowGUI=True是否允许GUI元素（如按钮、滑块等）的显示
win.mouseVisible = False  # 隐藏鼠标

# 该core是刺激参数的配置
# %% config code-VEP stimuli
#n_elements = 40  # number of the objects
n_elements = 120  # number of the objects
stim_sizes = np.zeros((n_elements, 2))  # size array | unit: pix  (32,2)  刺激对象的大小
stim_pos = np.zeros((n_elements, 2))  # position array                   位置
stim_oris = np.zeros((n_elements,))  # orientation array (default 0)    方向
stim_sfs = np.ones((n_elements,))  # spatial frequency array (default 1)空间频率
stim_phases = np.zeros((n_elements,))  # phase array                  相位
stim_opacities = np.ones((n_elements,))  # opacity array (default 1)    透明度，默认不透明
stim_contrs = np.ones((n_elements,))  # contrast array (default 1)   对比度

square_len = 60  # 单个方块的尺寸(60,60)
square_size = np.array([square_len, square_len])  # 单个方块的尺寸(60,60)
stim_sizes[:] = square_size  # 初始化，120行每行都是(60,60)

win_size = np.array([1920, 1080])  # 窗口的尺寸是1920和1080像素
#rows, columns = 5, 8  # 将屏幕划分为5行8列的区域
rows, columns = 8, 15  # 将屏幕划分为8行15列的区域
distribution = np.array([columns, rows])
# print(distribution)

'''计算并设置刺激位置思路：
首先，计算第一行第一列中心位置origin_pos；
接着，检查计算出的中心位置是否足够大，以容纳一个方块。如果不够大，则抛出异常；
然后，遍历每个区域，计算并设置每个刺激对象的位置
[1920/8/2, 1080/4/2] = [120, 135] 
调整坐标原点：
将刺激位置的原点从屏幕的左上角移动到屏幕的中心。这是通过从每个刺激的位置中减去屏幕尺寸的一半来实现的；
翻转y轴坐标，以便将屏幕坐标系的原点从左下角（许多图形库和框架的默认设置）移动到左上角，同时保持x轴方向不变。
[1920/15/2, 1080/8/2] = [64, 108] 
'''
# divide the whole screen into rows*columns blocks, and pick the center of each block as the position
origin_pos = np.array([win_size[0] / columns, win_size[1] / rows]) / 2  # (1920/8, 1080/4)/2  找到第一行第一列区域的中心位置
if (origin_pos[0] < (square_len / 2)) or (origin_pos[1] < (square_len / 2)):  # 如果划分的这片区域比目标方块还小
    raise Exception('Too much blocks or too big the single square!')  # 要么就是划分的区域太多了，要么就是单个方块太大了。
else:
    for i in range(distribution[0]):  # loop in columns = 15
        for j in range(distribution[1]):  # loop in rows = 8
            stim_pos[i * distribution[1] + j] = origin_pos + [i, j] * origin_pos * 2

# 把坐标原点调整为屏幕中间
stim_pos -= win_size / 2  # from Quadrant 1 to Quadrant 3    stim_pos-[win_size/2] 左移下移
stim_pos[:, 1] *= -1  # invert the y-axis    y轴翻转，把右下方向为正调整为右上方向为正

''''核心部分：正弦正负相位交替出现编码
思路：
1.初始化实验参数：设置刷新率（refresh_rate），每秒更新屏幕的次数。定义各种时间参数。
计算并设置单位帧（unit_frame），即每种颜色状态持续的帧数。
2.准备刺激颜色：
创建两种颜色序列（color_0和color_1），每种颜色序列包含一系列的颜色值。
使用高斯权重（gener_weight函数）调整颜色序列的对比度，并生成调整后的颜色序列（color_0_gau和color_1_gau）。
3.编码刺激：
根据code_series（一个包含0和1的二维数组，代表编码序列）生成完整的颜色帧序列（color_frame）。
每列代表一个编码序列，每行代表一个时间点的颜色值。
使用visual.ElementArrayStim为每个时间点的每个元素创建刺激对象，这些对象的颜色由color_frame决定。

'''
# 下面修改了很大部分。。。。lina
# %% config time template (using frames)
#refresh_rate = np.floor(win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20))  # 窗口的实际帧率
refresh_rate = 60   #硬件上调成的240Hz

# 帧率是指每秒显示的帧数（FPS），而屏幕刷新率是指屏幕每秒重绘其图像的次数。在理想情况下，两者应该匹配
# display_time, index_time, lag_time, blink_time = 0.5, 0.5, 0.5, 1   # index_time提示0.5   lag_time延迟时间0.5  眨眼=1s

display_time, index_time, lag_time, blink_time = 0.5, 0.5, 0.5, 0  # index_time提示0.5   lag_time延迟时间0.5
flash_time = 0.3  # flash: 闪烁时间200ms
# unit_frame = int(flash_time * refresh_rate)  # unit_frame:计算在闪烁周期内应该有多少帧
unit_frame = 72 # 闪烁刺激

# 存储不同帧的颜色值：每个数组都有unit_frame行和3列，分别对应RGB颜色通道。
color_0 = np.zeros((unit_frame, 3))  # color_0和color_1代表两种不同的颜色或颜色模式
color_1 = np.zeros((unit_frame, 3))
color_0_gau = np.zeros((unit_frame, 3))  # color_0_gau和color_1_gau存储经过高斯模糊后的颜色值
color_1_gau = np.zeros((unit_frame, 3))

color_00= np.zeros((unit_frame, 3))  # color_0和color_1代表两种不同的颜色或颜色模式
color_11 = np.zeros((unit_frame, 3))
color_00_gau = np.zeros((unit_frame, 3))  # color_0_gau和color_1_gau存储经过高斯模糊后的颜色值
color_11_gau = np.zeros((unit_frame, 3))

color_000 = np.zeros((unit_frame, 3))  # color_0和color_1代表两种不同的颜色或颜色模式
color_111 = np.zeros((unit_frame, 3))
color_000_gau = np.zeros((unit_frame, 3))  # color_0_gau和color_1_gau存储经过高斯模糊后的颜色值
color_111_gau = np.zeros((unit_frame, 3))

for i in range(int(unit_frame)):  # 这个循环遍历每个帧，并根据i%6的结果为color_0数组分配不同的颜色值
    # 这意味着每个闪烁周期（由unit_frame定义）被分为6个不同的颜色阶段。
    if i % 6 == 0:  # 1st frame (in each period)
        color_0[i, :] = np.array((0, 0, 0))
    elif i % 6 == 1:  # 2nd frame (in each period)
        color_0[i, :] = np.array((0.87226, 0.87226, 0.87226))
    elif i % 6 == 2:  # 3rd frame (in each period)
        color_0[i, :] = np.array((0.85313, 0.85313, 0.85313))
    elif i % 6 == 3:  # 4th frame (in each period)
        color_0[i, :] = np.array((-0.03784, -0.03784, -0.03784))
    elif i % 6 == 4:  # 5th frame (in each period)
        color_0[i, :] = np.array((-0.89015, -0.89015, -0.89015))
    elif i % 6 == 5:  # 6th frame (in each period)
        color_0[i, :] = np.array((-0.83278, -0.83278, -0.83278))

for j in range(int(unit_frame)):  # 这个循环遍历每个帧，并根据i%6的结果为color_00数组分配不同的颜色值
    # 这意味着每个闪烁周期（由unit_frame定义）被分为8个不同的颜色阶段。
    if j % 8 == 0:  # 1st frame (in each period)
        color_00[j, :] = np.array((0, 0, 0))
    elif j % 8 == 1:  # 2nd frame (in each period)
        color_00[j, :] = np.array((0.42979, 0.42979, 0.42979))
    elif j % 8 == 2:  # 3rd frame (in each period)
        color_00[j, :] = np.array((0.97181, 0.97181, 0.97181))
    elif j % 8 == 3:  # 4th frame (in each period)
        color_00[j, :] = np.array((0.79576, 0.79576, 0.79576))
    elif j % 8 == 4:  # 5th frame (in each period)
        color_00[j, :] = np.array((0.03172, 0.03172, 0.03172))
    elif j % 8 == 5:  # 6th frame (in each period)
        color_00[j, :] = np.array((-0.75574, -0.75574,-0.75574))
    elif j % 8 == 6:  # 7th frame (in each period)
        color_00[j, :] = np.array((-0.98480, -0.98480,-0.98480))
    elif j % 8 == 7:  # 8th frame (in each period)
        color_00[j, :] = np.array((-0.48619, -0.48619,-0.48619))

for k in range(int(unit_frame)):  # 这个循环遍历每个帧，并根据i%10的结果为color_000数组分配不同的颜色值
    # 这意味着每个闪烁周期（由unit_frame定义）被分为10个不同的颜色阶段。
    if k % 10 == 0:  # 1st frame (in each period)
        color_000[k, :] = np.array((0, 0, 0))
    elif k % 10== 1:  # 2nd frame (in each period)
        color_000[k, :] = np.array((0.58778, 0.58778, 0.58778))
    elif k % 10== 2:  # 3td frame (in each period)
        color_000[k, :] = np.array((0.95105, 0.95105, 0.95105))
    elif k % 10== 3:  # 4td frame (in each period)
        color_000[k, :] = np.array((0.95105, 0.95105, 0.95105))
    elif k % 10== 4:  # 5td frame (in each period)
        color_000[k, :] = np.array((0.58778, 0.58778, 0.58778))
    elif k % 10== 5:  # 6td frame (in each period)
        color_000[k, :] = np.array((0, 0, 0))
    elif k % 10== 6:  # 7td frame (in each period)
        color_000[k, :] = np.array((-0.58778,-0.58778,-0.58778))
    elif k % 10== 7:  # 8td frame (in each period)
        color_000[k, :] = np.array((-0.95105,-0.95105,-0.95105))
    elif k % 10== 8:  # 9td frame (in each period)
        color_000[k, :] = np.array((-0.95105,-0.95105,-0.95105))
    elif k % 10== 9:  # 10td frame (in each period)
        color_000[k, :] = np.array((-0.58778,-0.58778,-0.58778))


'''权重生成函数gener_weight：
函数生成一个权重数组，用于调整刺激的重要性或影响程度。
x是一个等差数列，从0到unit_frame-1，共unit_frame个点。
am_weight是根据高斯分布计算的权重数组，其标准差为20，均值为unit_frame/2-1（即中心位置）。
contrast参数用于调整权重数组的幅度。
'''

def gener_weight(contrast):  # comtrast：对比度
    x = np.linspace(0, unit_frame - 1, unit_frame)  # 等差数组
    am_weight = 50 * contrast * gaussian(x, 20, unit_frame / 2 - 1)  # am_weight:权重数组
    # 标准差设置为20，均值设置为unit_frame/2-1（即中心位置，因为索引从0开始）
    return am_weight

condition = [1, 0.8, 0.6, 0.4]  # condition列表中选择第一个（最大对比度）作为输入，生成权重数组am_weight
am_weight = gener_weight(condition[0])  # 从这里改最大对比度

color_0_gau[:, 0] = color_0[:, 0] * am_weight / 0.878
color_0_gau[:, 1] = color_0_gau[:, 0]  # color_0_gau的第一个通道通过乘以am_weight并除以0.878进行调整
color_0_gau[:, 2] = color_0_gau[:, 0]  # 将其值复制到其他两个通道
color_1_gau = -1 * color_0_gau  # color_1_gau是color_0_gau的负值，用于生成对比色或相反的颜色效果
color_0 = color_0 / 0.878  # 对color_0进行全局缩放
color_1 = -1 * color_0  # 计算其相反色作为color_1

color_00_gau[:, 0] = color_00[:, 0] * am_weight / 0.878
color_00_gau[:, 1] = color_00_gau[:, 0]
color_00_gau[:, 2] = color_00_gau[:, 0]
color_11_gau = -1 * color_00_gau
color_00 = color_00 / 0.878
color_11 = -1 * color_00

color_000_gau[:, 0] = color_000[:, 0] * am_weight / 0.878
color_000_gau[:, 1] = color_000_gau[:, 0]
color_000_gau[:, 2] = color_000_gau[:, 0]
color_111_gau = -1 * color_000_gau
color_000 = color_000 / 0.878
color_111 = -1 * color_000

len_code = 6  # 编码长度
code_frames = int(len_code * unit_frame)  # 编码帧数
# ？？不是48吗？？6*72帧=432帧=1.8s
color_frame = np.zeros((code_frames, n_elements, 3))  # (432, 120, 3)
n = 0


cvep_stimuli = []  # 初始化一个空列表，用于存储视觉刺激对象

# 创建方块刺激
for i in range(code_frames):  # 对288帧进行操作,变量主要是方块的color？288帧
    # visual.ElementArrayStim：创建刺激对象
    cvep_stimuli.append(visual.ElementArrayStim(win=win, units='pix', nElements=n_elements,
                                                sizes=stim_sizes, xys=stim_pos, colors=color_frame[i, ...],
                                                opacities=stim_opacities,
                                                oris=stim_oris, sfs=stim_sfs, contrs=stim_contrs, phases=stim_phases,
                                                elementTex=np.ones((128, 128)),
                                                elementMask=None, texRes=48))

# config text stimuli  配置文本刺激
# 可打印字符
printable_chars = ''.join([
    string.ascii_uppercase,
    string.ascii_lowercase,
    '1234567890',
    '~!@#$%^&*()_+-={}[]|\\:;"\'<>,.?/，。'
])

# 控制键
control_keys = ['Ctrl', 'Alt', 'Shift', 'Enter', 'Space', 'Esc', 'Home', 'End', 'PgUp', 'PgDn',
                'Insert', 'Delete', 'Tab', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
                'F11', 'F12']

# 初始化文本刺激列表
text_stimuli = []

# 遍历可打印字符和控制键，并为它们分别创建文本刺激
# 注意：您需要确保stim_pos的长度与printable_chars的长度加上control_keys的长度相等
pos_index = 0

# 添加可打印字符的刺激
for char in printable_chars:
    if pos_index < len(stim_pos):
        text_stimuli.append(visual.TextStim(win=win, text=char, font='Arial', pos=stim_pos[pos_index], color=(0, 0, 0),
                                            colorSpace='rgb',
                                            units='pix', height=square_len / 2, bold=True, name=char, autoLog=False))
        pos_index += 1

    # 添加控制键的刺激
for key in control_keys:
    if pos_index < len(stim_pos):
        # 对于控制键，您可能想要添加一些额外的文本或格式化，以便在视觉上区分它们
        # 这里我们简单地将它们括在方括号中
        text_stimuli.append(
            visual.TextStim(win=win, text=f'[{key}]', font='Arial', pos=stim_pos[pos_index], color=(0, 0, 0),
                            colorSpace='rgb',
                            units='pix', height=square_len / 3, bold=True, name=key, autoLog=False))
        pos_index += 1

# 绘制所有刺激
# for text_stimulus in text_stimuli:
#     text_stimulus.draw()

# 更新窗口显示
win.flip()

# config index stimuli: downward triangle   倒三角提示符，提示看哪一个方块
index_stimuli = visual.TextStim(win=win, text='\u2BC6', font='Arial', color=(1., 1., 0.), colorSpace='rgb',
                                units='pix', height=square_len, bold=True, name='\u2BC6', autoLog=False)

# config experiment parameters 配置实验参数
# 创建一个包含n_elements个字典的列表，每个字典都有一个'id'键
cvep_conditions = [{'id': i} for i in range(n_elements)]  # 循环条件
cvep_nrep = 1  # 循环次数，意味着每个条件只执行一次
# trials = data.TrialHandler(cvep_conditions, cvep_nrep, name='code-vep', method='random')  # 使用data.TrialHandler类来管理这些条件
trials = data.TrialHandler(cvep_conditions, cvep_nrep, name='code-vep', method='sequential')  # 按顺序执行

paradigm_clock = core.Clock()  # 代码初始化了两个计时器：paradigm_clock用于跟踪实验的全局时间
routine_timer = core.CountdownTimer()  # routine_timer用于跟踪特定实验例程的时间

t = 0
paradigm_clock.reset(0)

# lina修改5：全局的routine_timer.add改为addTime，共4处
# %% start routine
# display speller
routine_timer.reset(0)  # 重置时间
routine_timer.addTime(display_time)  # 设置倒计时开始的时间，已知display_time为0.5s
while routine_timer.getTime() > 0:  # 在该循环中，只要routine_timer的时间大于0，就遍历text_stimuli列表中的每个文本刺激对象
    for text_stimulus in text_stimuli:  # 调用它们的.draw()方法将其绘制到窗口上，并通过win.flip()更新屏幕显示。
        text_stimulus.draw()
    win.flip()

# begin to flash                       #trials是通过data.TrialHandler管理的，包含了实验的所有条件或试次
for trial in trials:
    # initialise index position
    id = int(trial['id'])
    index_stimuli.setPos(stim_pos[id] + np.array([0, square_len / 2]))

    # Phase 1: speller & index   第一阶段：拼写器与索引。
    # 在每个试次中，首先设置并重置routine_timer为index_time（索引显示时间）。
    # 然后，进入一个循环，在该循环中，绘制文本刺激，还绘制index_stimuli（倒三角提示符）
    routine_timer.reset(0)
    routine_timer.addTime(index_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        index_stimuli.draw()
        win.flip()

    # Phase 2: eye shifting   视觉转移
    # 等待一段时间（lag_time），用于允许被试的视线从索引刺激转移到接下来的刺激上。
    routine_timer.reset(0)
    routine_timer.addTime(lag_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()

    # lina修改3
    # Phase 3: code-VEP flashing
    frame_counter = 0
    for i in range(code_frames):
        # frame_counter += 1
        # # 每72帧记录一个标签
        # if frame_counter % 72 == 0:
        #     win.callOnFlip(port.sendLabel, id + 1)
        cvep_stimuli[i].draw()
        win.flip()

    # Phase 4: blink眨眼，类似第二阶段
    routine_timer.reset(0)
    routine_timer.addTime(blink_time)
    while routine_timer.getTime() > 0:
        for text_stimulus in text_stimuli:
            text_stimulus.draw()
        win.flip()

t = paradigm_clock.getTime()
win.close()  # 关闭图形窗口
core.quit()  # 退出PsychoPy的核心模块core
print('实验总时间：',t)
