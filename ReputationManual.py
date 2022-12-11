import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters  # 去除panda的警告

register_matplotlib_converters()


class Reputation:
    Reputation = list()  # 对于每个新节点,信誉值都需要从0累积,而对于恶意节点来说,其很可能在实际停车之前直接进行恶意行为,记录每一次行为后的信誉值
    Success = list()  # 节点信誉值中成功行为的改变记录
    SuccessInfo = list()  # 节点每次进行成功行为的信息,如第一次进行停车行为,停车了60分钟,则FailTime=60
    Fail = list()  # 节点信誉值中失败行为的改变记录
    FailInfo = list()  # 节点每次进行失败行为的信息,如第一次进行恶意行为,预约了20分钟,则FailTime=20
    NoCF = 0  # 节点进行失败行为的次数
    Time = list()  # 记录每一次行为的发生时间

    def Change(self):  # 每一次行为后,信誉值的动态变化
        success = 0
        fail = 0
        self.Success = [0]
        for i in range(len(self.SuccessInfo)):  # 成功行为信誉值的累加
            success += self.SuccessInfo[i][3]
            self.Success.append(success)

        self.Fail = [0]
        for i in range(len(self.FailInfo)):  # 失败行为信誉值的累加
            if self.FailInfo[i][4] <= 100:  # 惩罚系数,在不同情况下进行恶意行为,严重程度也不一样
                punish = 2
            else:
                punish = 1
            fail -= (self.FailInfo[i][0] + 1) * punish * self.FailInfo[i][3]
            self.Fail.append(fail)
        self.Reputation.append(round(success + fail, 2))

    def SuccessBehavior(self):  # 模拟成功行为
        starttime = datetime.datetime.strptime(input(), "%Y-%m-%d %H:%M:%S")  # 格式如2017-10-01 12:12:12
        endtime = datetime.datetime.strptime(input(), "%Y-%m-%d %H:%M:%S")
        seconds = (endtime - starttime).seconds
        interval = round(int(seconds) / 60, 2)
        self.SuccessInfo.append([len(self.SuccessInfo) + 1, str(starttime), str(endtime), interval])
        self.FailInfo.append([0, str(endtime), str(endtime), 0, 0])
        self.Time.append(str(endtime))

    def FailBehavior(self):  # 模拟失败行为
        starttime = datetime.datetime.strptime(input(), "%Y-%m-%d %H:%M:%S")  # 格式如2017-10-01 12:12:12
        endtime = datetime.datetime.strptime(input(), "%Y-%m-%d %H:%M:%S")
        seconds = (endtime - starttime).seconds
        interval = round(int(seconds) / 60, 2)
        NoRPS = int(input())
        self.SuccessInfo.append([0, str(endtime), str(endtime), 0])
        self.FailInfo.append([len(self.FailInfo) + 1, str(starttime), str(endtime), interval, NoRPS])
        self.Time.append(str(endtime))

    def Draw(self):
        fig = plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 20
        plt.tick_params(top=True, right=True)
        plt.tick_params(axis='x', direction="in", pad=10, length=10)
        plt.tick_params(axis='y', direction="in", pad=10, length=10)
        plt.xlim(pd.to_datetime("2021/10/10 09:00"), pd.to_datetime("2021/10/11 00:00"))
        plt.ylim(-5000, 1000)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid('+')

        successtime = ["2021/10/10 00:00"]
        failtime = ["2021/10/10 00:00"]
        for info in self.SuccessInfo:
            successtime.append(info[2])
        for info in self.FailInfo:
            failtime.append(info[2])

        plt.plot(pd.to_datetime(self.Time), self.Reputation, label="Reputation Value", linewidth=3, color="steelblue",
                 markersize=15, marker='o',
                 markerfacecolor="none")
        plt.plot(pd.to_datetime(successtime), self.Success, label="Reputation Value (success)", linewidth=3,
                 color="green", markersize=15, marker='d',
                 markerfacecolor="none")
        plt.plot(pd.to_datetime(failtime), self.Fail, label="Reputation Value (fail)", linewidth=3, color="darkorange",
                 markersize=15, marker='x',
                 markerfacecolor="none")
        plt.legend(loc="upper left")
        fig.autofmt_xdate(rotation=45)
        plt.show()


if __name__ == '__main__':
    reputate = Reputation()  # 实例化对象
    while True:
        print(reputate.SuccessInfo)
        print(reputate.FailInfo)
        print(reputate.Reputation)
        Behavior = input()
        if Behavior == "S":  # 模拟成功行为
            print("Start to simulate successful behavior:")
            reputate.SuccessBehavior()
            reputate.Change()  # 信誉值动态变化
            print("Simulate is ending.")
        elif Behavior == "F":  # 模拟失败行为
            print("Start to simulate failure behavior:")
            reputate.FailBehavior()
            reputate.Change()  # 信誉值动态变化
            print("Simulate is ending.")
        elif Behavior == "D":  # 画图
            reputate.Draw()
        else:  # 结束模拟
            break
