import requests
import json
import pandas as pd
from requests.auth import HTTPBasicAuth

def sign_in():

    # 1. 初始化session并完成认证（重点：正确处理认证响应）
    sess = requests.Session()
    # 替换为你的真实账号密码
    sess.auth = HTTPBasicAuth('xwf_8_820@163.com', '8876081xie,.')

    # 执行认证并检查结果
    auth_response = sess.post('https://api.worldquantbrain.com/authentication')
    print(auth_response.status_code)
    print(auth_response.json())
    return sess

sess = sign_in()
# 2. 定义和网页完全一致的searchScope（你确认的正确参数）
searchScope = {
    'instrumentType': 'EQUITY',  # 和网页一致
    'region': 'USA',  # 和网页一致
    'delay': 1,  # 和网页一致
    'universe': 'TOP3000'  # 和网页一致
}


# 3. 修复URL拼接问题的核心函数
def get_datafields(
    s,
    searchScope,
    dataset_id: str ='',
    search: str = ''
):
    import pandas as pd
    instrument_type = searchScope['instrumentType']
    region = searchScope['region']
    delay = searchScope['delay']
    universe = searchScope['universe']
    if len(search) == 0:
        url_template = "https://api.worldquantbrain.com/data-fields?"+ \
                       f"&instrumentType={instrument_type}" + \
                       f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
                        "&offset={x}"
        count = s.get(url_template.format(x=0)).json()['count']
    else:
        url_template = "https://api.worldquantbrain.com/data-fields?" +\
        f"&instrumentType={instrument_type}" +\
        f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
        f"&search={search}" +\
        "&offset={x}"
        count = 100

    datafields_list = []
    for x in range(0, count, 50):
        datafields = s.get(url_template.format(x=x))
        resp_data = datafields.json()
        results = resp_data.get('results', [])  # 👈 安全写法
        datafields_list.append(results)
    datafields_list_flat = [item for sublist in datafields_list for item in sublist]

    datafields_df = pd.DataFrame(datafields_list_flat)
    return datafields_df

# 4. 调用函数（dataset_id和网页一致）
fields = get_datafields(
    s=sess,
    searchScope=searchScope,
    dataset_id='ai_news_scores'
)

# 打印结果预览
if not fields.empty:
    print("\n=== 数据预览 ===")
    print(fields[['id']].head())
else:
    print("\n❌ 未获取到任何数据，请检查认证/参数/权限")

datafields = fields[fields['type'] == "matrix"]
datafields.head()

# ===================== 1. 定义核心参数（仅保留目标循环变量） =====================
# 基本面字段列表（从DataFrame读取）
datafields_list_fields = fields['id'].values

group_functions = ['group_zscore','group_rank']
ts_functions = ['ts_zscore','ts_rank']
group_dimensions = ['subindustry','industry','sector','market']
# 多窗口搜索
ts_windows = [10, 20, 25, 60]

# ===================== 2. 初始化 Alpha 列表 =====================
alpha_list = []
seen_expressions = set()

# ===================== 3. 嵌套循环 =====================
for group_dim in group_dimensions:
    for group_func in group_functions:
        for ts_func in ts_functions:
            for window in ts_windows:
                for datafield in datafields_list_fields:

                    # ---------- A. 时间序列信号 ----------
                    # 将原始字段转换为历史相对强弱信号
                    alpha_expression = (
                        f"a={ts_func}({datafield},{window});"
                        f"market_cap_rank=rank(cap);"
                        f"market_cap_bucket=bucket("
                        f"market_cap_rank,range='0.1,1,0.1');"
                        f"a1={group_func}(a,market_cap_bucket);"
                        f"a2={group_func}(a1,{group_dim});"
                        f"a2"
                    )

                    # 去重
                    if alpha_expression not in seen_expressions:
                        seen_expressions.add(alpha_expression)

                        simulation_data = {
                            'type': 'REGULAR',
                            'settings': {
                                'instrumentType': 'EQUITY',
                                'region': 'GLB',
                                'universe': 'MINVOL1M',
                                'delay': 1,
                                'decay': 5,
                                'neutralization': 'SUBINDUSTRY',
                                'truncation': 0.08,
                                'pasteurization': 'ON',
                                'unitHandling': 'VERIFY',
                                'nanHandling': 'ON',
                                'language': 'FASTEXPR',
                                'visualization': False,
                            },
                            'regular': alpha_expression
                        }

                        alpha_list.append(simulation_data)
# ===================== 4. 输出生成结果 =====================
total_alpha = len(datafields_list_fields) * len(group_functions) * len(ts_functions) * len(group_dimensions)
print(f'\n总共生成了 {len(alpha_list)} 个Alpha待模拟（理论值：{total_alpha}）')


from time import sleep
import logging

alpha_fail_attempt_tolerance=30 #每个alpha 允许的最大失败尝试次数


for i in range(0,len(alpha_list)):
    keep_trying = True
    failure_count = 0
    print(i)

    while keep_trying:
        try:
            # 发试发送模拟请求
            sim_resp = sess.post(
            'https://api.worldquantbrain.com/simulations',
            json=alpha_list[i])
            # 将 alpha(一个JSON)发送至服务器

            sim_progress_url=sim_resp.headers['Location'] #从响应头中欢取位置
            logging.info(f'Alpha location is: (sim_progress_url]')
            print(f'Alpha location is: {sim_progress_url}')
            # sim_progress_resp = sess.get(sim_progress_url)
            # alpha_id = sim_progress_resp.json()["alpha"]
            # print(alpha_id)
            keep_trying = False #成功获取位置,退出 while 循环

        except Exception as e:
            #处理梁常:记录错误,休眼并增加失败次数计数
            # logging.error(f"No Location, sleep 15 and retry, error message: {str(e)}")
            # print("No Location, sleep 10 and retry")
            sleep(30)
            failure_count += 1

            # 检查失敗次教足否达到溶忍限度
            if failure_count >= alpha_fail_attempt_tolerance:
                sess=sign_in() #重新认证会话
                failure_count=0 #重言失败次数计数
                # 记录并打印消息,指示将移动到下一个 alpha
                logging.error(f"No location for too many times, move to next alpha {alpha_list[i]['regular']}")
                print(f"No location for too many times, move to next alpha {alpha_list[i]['regular']}")
                break #退出 while 循环,移动别 for 循环中的下一个 alpha

