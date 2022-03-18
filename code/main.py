import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq

move_planner = []


class AstarPoint(object):
    '''
    Astar点类
    ========== 输入 ==========
    self_x,self_y : int, 点的坐标
    g: float, 点的g值
    f_value: float, 点的f值
    ParentPosition: tuple,父节点的坐标
    '''

    def __init__(self, self_x, self_y, g, f_value, ParentPosition=None):
        '''

        '''
        self.SelfPosition = (self_x, self_y)
        self.ParentPosition = ParentPosition
        self.f_value = f_value
        self.g = g

    def __hash__(self):
        return self.SelfPosition

    def __eq__(self, other):
        return self.SelfPosition == other.SelfPosition

    def __repr__(self):
        if self.ParentPosition != None:
            return "AstarPoint(%d, %d, %d, %d" % (self.SelfPosition[0], self.SelfPosition[1], self.g, self.f_value) + \
                   ", " + str(self.ParentPosition[0]) + ', ' + str(self.ParentPosition[1]) + ")"
        else:
            return "AstarPoint(%d, %d, %d, %d" % (self.SelfPosition[0], self.SelfPosition[1], self.g, self.f_value) + \
                   ", " + str(None) + ', ' + str(None) + ")"

    def __str__(self):
        if self.ParentPosition == None:
            return "AstartPoint类 | 坐标为:       " + str(self.SelfPosition) + "\t\t f_value : " + str(self.f_value) + \
                   "\n\t      | 当前节点无父节点"
        else:
            return "AstartPoint类 | 坐标为:       " + str(self.SelfPosition) + "\t\t f_value : " + str(self.f_value) + \
                   "\n\t      | 父节点坐标为: " + str(self.ParentPosition)


class CostMap(object):
    '''地图类'''

    def __init__(self, init_map):
        '''
        ========== 输入 ==========
            init_map : np.ndarray, 初始化的CostMap,通常为np.zeros,可为二维,三维或者更高位
        '''
        self.map = init_map

    def obstacle_setter(self, start_x, end_x, start_y, end_y, is_clean=False):
        '''
        obstacle_adder用于添加障碍物
        ========== 输入 ==========
        start_x, start_y : 起点的坐标
        end_x, end_y :终点的坐标
        is_clean : Ture表示清除所有障碍物
        '''
        if not is_clean:
            self.map[start_x:end_x, start_y:end_y] = np.inf
        else:
            self.map[:, :] = 0


# 可视化函数
def ShowMap(CostMap, StartPoint=None, EndPoint=None, pandas=True, Path=None):
    '''
    ShowMap用于以表格形式或者以栅格图形式展示地图
    =========== 输入 ==========
    CostMap : np.ndarray
    StartPoint, EndPoint : tuple, 默认为None
    Path : list
    =========== 输出 ==========
    return None
    '''
    if pandas:
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        shape = CostMap.shape
        CostMap = CostMap.T[::-1, :]
        return pd.DataFrame(CostMap, index=np.arange(shape[1] - 1, -1, -1), columns=np.arange(0, shape[0], 1))
    else:
        ColorPlan = list("rgbcwk")
        squareSpace = 0.33333
        squareAera = 3000

        # 创建地图
        Figure, Axes = plt.subplots(1)
        Figure.set_size_inches(CostMap.shape)

        # 绘制图形
        Obstacle = np.where(CostMap == np.inf)
        Suspects = np.where((CostMap > 0) & (CostMap < np.inf))

        # 添加障碍物,起点,终点和路径
        for (x, y) in zip(Obstacle[0], Obstacle[1]):
            Axes.scatter(x + 0.5, y + 0.5, marker='s', c=ColorPlan[-1], s=squareAera)
        for (x, y) in zip(Suspects[0], Suspects[1]):
            Axes.scatter(x + 0.5, y + 0.5, marker='s', c=ColorPlan[2], s=squareAera)
            style = dict(size=13, color='black')
            Axes.text(x, y, str(CostMap[x, y]), **style)

        if Path != None:
            for (x, y) in Path:
                Axes.scatter(x + 0.5, y + 0.5, marker='s', c=ColorPlan[3], s=squareAera)
            Axes.scatter([], [], marker='s', c=ColorPlan[3], label='Path Point')

        if StartPoint != None:
            Axes.scatter(StartPoint[0] + 0.5, StartPoint[1] + 0.5, marker='s', c=ColorPlan[1], s=squareAera)
            Axes.scatter([], [], marker='s', c=ColorPlan[0], label='Start Point')
        if EndPoint != None:
            Axes.scatter(EndPoint[0] + 0.5, EndPoint[1] + 0.5, marker='s', c=ColorPlan[0], s=squareAera)
            Axes.scatter([], [], marker='s', c=ColorPlan[1], label='End Point')

        # 设置图例
        Axes.scatter([], [], marker='s', c=ColorPlan[-1], label='Obstacle')
        Axes.scatter([], [], marker='s', c=ColorPlan[2], label='Suspect Point')

        # 图案设置
        Axes.set_xlim(0, CostMap.shape[0], 1)
        Axes.set_ylim(0, CostMap.shape[1], 1)
        Axes.set_xticks(np.arange(0, CostMap.shape[0] + 1, 1))
        Axes.set_yticks(np.arange(0, CostMap.shape[1] + 1, 1))
        Axes.grid(True)
        Axes.legend(bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0., fontsize='xx-large')
        plt.show()


# 获得最小f值的函数
def GetMinimunFValueIndex(OpenList):
    FValueList = []
    FValueHeap = []
    for point in OpenList:
        heapq.heappush(FValueHeap, point.f_value)
        FValueList.append(point.f_value)
    return FValueList.index(heapq.nsmallest(1, FValueHeap)[0])


def A_star_path_planning(cost_map, start_point, goal_point):
    """
    A star 路径规划
    ========== 输入 ==========
    cost_map    : 代价图
    start_point : tuple，表示起点坐标，例如 (1, 1) 或者 (1, 1, 1)
    goal_point  : tuple，表示终点坐标，例如 (19, 19) 或者 (19, 19, 19)

    ========== 输出 ==========
    path : 点list，例如 [(1, 1, 1), (2, 2, 2), ..., (19, 19, 19)]
    move_planner: list ,输出移动方向
    """

    OpenList = [
        AstarPoint(start_point[0], start_point[1], 0,
                   np.sqrt(np.sum((start_point[0] - goal_point[0]) ** 2 + (start_point[1] - goal_point[1]) ** 2)))
    ]
    CloseList = []
    Path = []
    i = 0
    isDone = False
    k = 1
    while (len(OpenList) > 0) and (not isDone):

        # 获得最小f值的点
        index = GetMinimunFValueIndex(OpenList)  # 获得f值最小的点的索引

        # OpenList中弹出f值最小的点,加入CloseList中
        tempPoint = OpenList.pop(index)
        CloseList.append(tempPoint)

        # 将周围的点加入OpenList中
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:

                x_next = tempPoint.SelfPosition[0] + i
                y_next = tempPoint.SelfPosition[1] + j

                # 首先排除一些不可能的情况
                if i == 0 and j == 0:  # 1. 点是本身
                    continue
                if (x_next < 0 or x_next >= cost_map.shape[0] or y_next < 0 or y_next >= cost_map.shape[
                    1]):  # 2. 点在地图外
                    continue
                if cost_map[x_next, y_next] == np.inf:  # 3. 点是障碍物
                    continue

                if (AstarPoint(x_next, y_next, 0, 0) in CloseList):  # 4. 点已经在OpenList或者在CloseList中
                    continue

                if (AstarPoint(x_next, y_next, 0, 0) not in OpenList):
                    g_next = tempPoint.g + 1
                    f_value_next = g_next + np.sqrt(
                        np.sum((x_next - goal_point[0]) ** 2 + (y_next - goal_point[1]) ** 2))

                    parent_x = x_next - i
                    parent_y = y_next - j

                    OpenList.append(
                        AstarPoint(x_next, y_next, g_next, f_value_next, ParentPosition=(parent_x, parent_y))
                    )

                    AstarMap.map[x_next, y_next] = g_next
                    k += 1
                else:
                    # 如果当前点已经在OpenList中,且G更小,更新父节点
                    if (tempPoint.g > OpenList[OpenList.index(AstarPoint(x_next, y_next, 0, 0))].g):
                        parent_x = x_next
                        parent_y = x_next

        # 如果终点已经在CloseList中
        if AstarPoint(goal_point[0], goal_point[1], 0, 0) in CloseList:
            isDone = True

    # 回溯路径
    son_index = CloseList.index(AstarPoint(goal_point[0], goal_point[1], 0, 0))
    while start_point not in Path:
        try:
            Path.append(CloseList[son_index].SelfPosition)
            parent_x, parent_y = CloseList[son_index].ParentPosition
            son_index = CloseList.index(AstarPoint(parent_x, parent_y, 0, 0))
        except TypeError:
            print("已回溯到起点")
            return Path


# 初始化地图和起点终点
shape = tuple([int(x) for x in input("请输入地图大小(x_length,y_length): ").split(',')])
startPoint = tuple([int(x) for x in input('请输入起点坐标(x,y): ').split(',') if (int(x) < shape[0] and int(x) < shape[1])])
endPoint = tuple([int(x) for x in input('请输入终点坐标(x,y): ').split(',') if (int(x) < shape[0] and int(x) < shape[1])])

AstarMap = CostMap(np.zeros(shape))

# 添加障碍物
AstarMap.obstacle_setter(None, None, None, None, is_clean=True)
AstarMap.obstacle_setter(0, 9, 20, 40)
AstarMap.obstacle_setter(10, 19, 10, 12)
AstarMap.obstacle_setter(17, 18, 6, 10)
AstarMap.obstacle_setter(41, 50, 0, 9)
AstarMap.obstacle_setter(15, 25, 20, 25)
AstarMap.obstacle_setter(9, 41, 32, 40)
AstarMap.obstacle_setter(31, 41, 18, 20)
ShowMap(AstarMap.map, StartPoint=startPoint, EndPoint=endPoint, pandas=False)

# 路径规划
Path = A_star_path_planning(AstarMap.map, startPoint, endPoint)
print(Path)
# 可视化
ShowMap(AstarMap.map, StartPoint=startPoint, EndPoint=endPoint, pandas=False, Path=Path)
