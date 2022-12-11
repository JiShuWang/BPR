import datetime
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.ndimage import gaussian_filter1d
from pandas.plotting import register_matplotlib_converters  # 去除pandas的警告

register_matplotlib_converters()


class Reputation:
    Initial = 100
    Reputation = [Initial]  # 对于恶意节点来说,其很可能在实际停车之前直接进行恶意行为,记录每一次行为后的信誉值
    Success = list()  # 节点信誉值中成功部分的改变记录
    SuccessInfo = list()  # 节点每次进行成功行为的信息,如第一次进行停车行为,停车了60分钟,则FailTime=60
    Fail = list()  # 节点信誉值中失败部分的改变记录
    FailInfo = list()  # 节点每次进行失败行为的信息,如第一次进行恶意行为,预约了20分钟,则FailTime=20
    ContinuousFailure = list()  # 节点每次行为后其连续失败的次数
    NoCF = 0  # 节点进行失败行为的次数
    Time = ['2014-9-10 00:00:00']  # 记录每一次行为的发生时间
    InitialTime = datetime.datetime.strptime("2014-09-10 00:00", "%Y-%m-%d %H:%M")
    NowTime = datetime.datetime.strptime("2014-09-10 00:00", "%Y-%m-%d %H:%M")
    EndTime = datetime.datetime.strptime("2014-09-10 00:00", "%Y-%m-%d %H:%M")
    NodeType = ["Honest"]
    NowNodeType = "Honest"
    Capatity = [0]  # 停车位的剩余数量

    ContinuousFailureLimit = 5
    Strict = [0]  # 记录节点每次行为后被限制使用预约的时间(秒)

    def Change(self, type):  # 每一次行为后,信誉值的动态变化
        print(self.NodeType)
        success = 0
        fail = 0
        self.Success = [0]
        for i in range(len(self.SuccessInfo)):  # 成功行为信誉值的累加
            if self.NodeType[i] == "Malicious":
                success = 0
                self.Success.append(0)
            else:
                success += self.SuccessInfo[i][3]
                self.Success.append(success)

        self.Fail = [0]
        for i in range(len(self.FailInfo)):  # 失败行为信誉值的累加
            if self.FailInfo[i][4] <= 125:  # 惩罚系数,在不同情况下进行恶意行为,严重程度也不一样
                punish = 5
            elif 125 < self.FailInfo[i][4] <= 625:
                punish = 2
            else:
                punish = 1
            if self.NodeType[i] == "Malicious":
                fail += (self.FailInfo[i][0]) * punish * self.FailInfo[i][3]
            else:
                fail += punish * self.FailInfo[i][3]
            self.Fail.append(fail)
        if self.NowNodeType == "Malicious":
            success = 0
        self.Reputation.append(round(self.Initial + success - fail, 2))
        if self.Reputation[-1] < 0:
            self.NodeType[-1] = "Malicious"
            self.NowNodeType = "Malicious"

        if self.NodeType[-1] == "Malicious":  # 信誉值为负数,需要被限制预约
            interval = datetime.timedelta(seconds=round(abs(self.Reputation[-1]) / 20, 2))
            self.NowTime += interval
            self.EndTime = self.NowTime
            self.Strict.append(round(abs(self.Reputation[-1]) / 20, 2))
        else:
            self.Strict.append(0)

    def SuccessBehavior(self):  # 模拟成功行为
        nowtime = self.NowTime
        mins = random.randint(60, 180)  # 停车时间从1小时到3小时
        interval = datetime.timedelta(minutes=mins)
        endtime = nowtime + interval
        self.NowTime = endtime
        self.EndTime = endtime
        self.SuccessInfo.append([len(self.SuccessInfo) + 1, str(nowtime), str(endtime), mins])
        self.FailInfo.append([0, str(endtime), str(endtime), 0, 0])
        self.Time.append(str(endtime))
        self.ContinuousFailure.append(0)
        self.NowNodeType = "Honest"
        self.NodeType.append("Honest")
        self.Capatity.append(0)

    def FailBehavior(self):  # 模拟失败行为
        self.NoCF += 1
        nowtime = self.NowTime
        mins = random.randint(1, 60)  # 预约状态持续时间从1分钟到30分钟
        interval = datetime.timedelta(minutes=mins)
        endtime = nowtime + interval
        if (12 <= int(str(nowtime.hour)) <= 14) or (17 <= int(str(nowtime.hour)) <= 21):  # 高峰期,停车位较少
            NoRPS = random.randint(1, 125)
        else:
            NoRPS = random.randint(126, 1250)
        self.Capatity.append(NoRPS)
        self.NowTime = endtime
        self.EndTime = endtime
        self.SuccessInfo.append([0, str(endtime), str(endtime), 0])
        self.FailInfo.append([self.NoCF, str(nowtime), str(endtime), mins, NoRPS])
        self.Time.append(str(endtime))
        if len(self.ContinuousFailure) == 0:
            self.ContinuousFailure.append(1)
        else:
            self.ContinuousFailure.append(self.ContinuousFailure[-1] + 1)
        if self.Reputation[-1] < 0 or self.ContinuousFailure[
            -1] >= self.ContinuousFailureLimit:
            self.NowNodeType = "Malicious"
            self.NodeType.append("Malicious")
        else:
            self.NowNodeType = "Honest"
            self.NodeType.append("Honest")

    def Draw(self, type):
        print(self.Strict)
        print(self.FailInfo)
        print(self.Fail)
        print(self.Reputation)
        successtime = ["2014-09-10 00:00"]
        failtime = ["2014-09-10 00:00"]
        for info in self.SuccessInfo:
            successtime.append(info[2])
        for info in self.FailInfo:
            failtime.append(info[2])

        fig = plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 20
        plt.tick_params(top=True, right=True)
        plt.tick_params(axis='x', direction="in", pad=10, length=10)
        plt.tick_params(axis='y', direction="in", pad=10, length=10)
        # plt.xlim(pd.to_datetime(self.InitialTime), pd.to_datetime(self.EndTime))
        # plt.ylim(min(min(self.Reputation), min(self.Fail)) - 1000,
        #          max(max(self.Reputation), max(self.Fail), max(self.Strict)) + 1000)
        # plt.xlabel("Time")
        plt.ylabel("Value")
        # plt.grid('.')

        # plt.plot(pd.to_datetime(self.Time), self.Reputation, label="Reputation value", linewidth=3, color="steelblue",
        #          markersize=10, marker='o',
        #          markerfacecolor="none", zorder=3, clip_on=False)
        # plt.plot(pd.to_datetime(successtime), self.Success, label="Reputation value (success)", linewidth=3,
        #          color="green", markersize=10, marker='d',
        #          markerfacecolor="none", zorder=3, clip_on=False)
        # plt.plot(pd.to_datetime(failtime), self.Fail, label="Reputation value (fail)", linewidth=3, color="darkorange",
        #          markersize=10, marker='x',
        #          markerfacecolor="none", zorder=4, clip_on=False)
        # plt.plot(pd.to_datetime(self.Time), self.Strict,
        #          label="Unable to use the reservation (seconds)", linewidth=3,
        #          color="red",
        #          markersize=15, marker='h',
        #          markerfacecolor="none", zorder=5, clip_on=False)
        # if type == "T":
        #     faillimit = 0
        #     successlimit = 0
        #     strictlimit = 0
        #     if self.NowNodeType == "Honest":
        #         for x, y, z, h in zip(pd.to_datetime(self.Time), self.Capatity, self.Fail, self.FailInfo):
        #             if z > faillimit:
        #                 faillimit = z
        #                 plt.text(x + datetime.timedelta(hours=0.2), z + 100,
        #                          str(z) + '\nCapacity:' + str(y) + "\nDuration:" + str(
        #                              self.FailInfo[self.FailInfo.index(h) - 1][3]),
        #                          ha="left",
        #                          va="bottom",
        #                          fontsize=15,
        #                          bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="darkorange"), zorder=4)
        #                 plt.scatter(x, z, marker='x', s=300, color="darkorange", clip_on=False, zorder=3)
        plt.xlabel("Number of Reservations")
        index = [i for i in range(len(self.Strict))]
        plt.xlim(-1, max(index) + 1)
        plt.ylim(min(min(self.Reputation), min(self.Fail)) - 2000,
                 max(max(self.Reputation), max(self.Fail), max(self.Strict)) + 2000)
        plt.xticks([i for i in range(0, 22, 2)])
        plt.bar([i - 0.25 for i in index], self.Reputation, width=0.5, color="steelblue", label="Reputation value",
                zorder=5)
        plt.bar([i + 0.25 for i in index], self.Fail, width=0.5, color="darkorange", label="Reputation value (fail)",
                zorder=5)
        plt.plot(index, self.Strict, color="red", label="Unable to use the reserve (seconds)", markersize=15,
                 marker='h', markerfacecolor="none", zorder=11, clip_on=False, linewidth=1)
        # plt.bar([i + 0.825 for i in index], self.Strict, width=0.33, color="red",
        #         label="Unable to use the reservation (seconds)")
        index += [len(index)]
        for x, y, z, h in zip([i + 0.625 for i in index], self.Fail, self.Strict, self.Reputation):
            if y >= 0:
                if len(str(y)) == 1:
                    plt.text(x - 0.4, 1200, y, va="center", color="darkorange", fontsize=15, zorder=10)
                elif len(str(y)) == 2:
                    plt.text(x - 0.5, 1200, y, va="center", color="darkorange", fontsize=15, zorder=10)
                elif len(str(y)) == 3:
                    plt.text(x - 0.6, 1200, y, va="center", color="darkorange", fontsize=15, zorder=10)
                elif len(str(y)) == 4:
                    plt.text(x - 0.68, y + 500, y, va="center", color="darkorange", fontsize=15, zorder=10)
                elif len(str(y)) == 5:
                    plt.text(x - 0.75, y + 500, y, va="center", color="darkorange", fontsize=15, zorder=10)
            if h >= 0:
                if len(str(h)) == 1:
                    plt.text(x - 0.8, 1200, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
                elif len(str(h)) == 2:
                    plt.text(x - 0.95, 1200, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
                elif len(str(h)) == 3:
                    plt.text(x - 1.1, 1200, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
            else:
                if len(str(h)) == 1:
                    plt.text(x - 1.1, (h) - 600, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
                elif len(str(h)) == 2:
                    plt.text(x - 1, (h) - 600, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
                elif len(str(h)) == 3:
                    plt.text(x - 1.1, (h) - 600, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
                elif len(str(h)) == 4:
                    plt.text(x - 1.15, (h) - 600, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
                elif len(str(h)) == 5:
                    plt.text(x - 1.25, (h) - 600, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
                elif len(str(h)) == 6:
                    plt.text(x - 1.35, (h) - 600, str(h), va="center", color="steelblue", fontsize=15,
                             zorder=10)
            if z >= 0:
                if len(str(z)) == 1:
                    plt.text(x - 0.75, z + 700,
                             str(round(float(z), 2)), color="red", va="center",
                             fontsize=10, bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="red"), zorder=6)
                elif len(str(z)) == 2:
                    plt.text(x - 0.85, z + 700,
                             str(round(float(z), 2)), color="red", va="center",
                             fontsize=10, bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="red"), zorder=6)
                elif len(str(z)) == 3:
                    plt.text(x - 0.75, z + 700,
                             str(round(float(z), 2)), color="red", va="center",
                             fontsize=10, bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="red"), zorder=6)
                elif len(str(z)) == 4:
                    plt.text(x - 0.8, z + 700,
                             str(round(float(z), 2)), color="red", va="center",
                             fontsize=10, bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="red"), zorder=6)
                elif len(str(z)) == 5:
                    plt.text(x - 0.85, z + 700,
                             str(round(float(z), 2)), color="red", va="center",
                             fontsize=10, bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="red"), zorder=6)
                elif len(str(z)) == 6:
                    plt.text(x - 0.9, z + 700,
                             str(round(float(z), 2)), color="red", va="center",
                             fontsize=10, bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="red"), zorder=6)

        plt.legend(loc="upper left")
        # fig.autofmt_xdate(rotation=45)
        plt.show()


if __name__ == '__main__':
    reputate = Reputation()  # 实例化对象
    # behavior = ['F' for i in range(20)] + ['T']  # Malicious Node
    behavior = ['S' for i in range(5)] + ['F' for i in range(15)] + ['T']  # Suddenly
    i = 0
    while i < len(behavior):
        Behavior = behavior[i]
        if Behavior == "S":  # 模拟成功行为
            print("Start to simulate successful behavior:")
            reputate.SuccessBehavior()
            reputate.Change('S')  # 信誉值动态变化
            print("Simulate is ending.")
        elif Behavior == "F":  # 模拟失败行为
            print("Start to simulate failure behavior:")
            reputate.FailBehavior()
            reputate.Change('F')  # 信誉值动态变化
            print("Simulate is ending.")
        elif Behavior == "D":  # 画图
            reputate.Draw('D')
        elif Behavior == "T":  # 画图的同时多画一个文本框
            reputate.Draw('T')
        i += 1
