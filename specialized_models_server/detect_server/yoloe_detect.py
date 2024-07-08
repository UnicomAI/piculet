import sys

# 要添加的路径
new_path = 'PaddleDetection-release-2.7'

# 将新路径追加到 sys.path 列表中
sys.path.append(new_path)


import paddle
from deploy.python.infer import Detector
import pickle
import cv2 
import os

    
class Yoloe:
    def __init__(self):
        with open('FLAGS_yoloe.pkl', 'rb') as file:
            FLAGS = pickle.load(file)
        print(FLAGS)
        FLAGS.model_dir='PaddleDetection-release-2.7/output_inference/ppyoloe_plus_crn_x_60e_objects365/'
        self.detector = Detector(
            FLAGS.model_dir,
            device=FLAGS.device,
            run_mode=FLAGS.run_mode,
            batch_size=FLAGS.batch_size,
            trt_min_shape=FLAGS.trt_min_shape,
            trt_max_shape=FLAGS.trt_max_shape,
            trt_opt_shape=FLAGS.trt_opt_shape,
            trt_calib_mode=FLAGS.trt_calib_mode,
            cpu_threads=FLAGS.cpu_threads,
            enable_mkldnn=FLAGS.enable_mkldnn,
            enable_mkldnn_bfloat16=FLAGS.enable_mkldnn_bfloat16,
            threshold=FLAGS.threshold,
            output_dir=FLAGS.output_dir,
            use_fd_format=FLAGS.use_fd_format)
        self.FLAGS = FLAGS 
        self.labels = self.detector.pred_config.labels
        self.labels_zh = ['人', '运动鞋', '椅子', '其他鞋类', '帽子', '汽车', '灯', '眼镜', '瓶子', '书桌', '杯子', '街灯', '橱柜/架子', '手袋/公文包', '手镯', '盘子', '图片/画框', '头盔', '书', '手套', '储物盒', '小船', '皮鞋', '花', '长椅', '盆栽', '碗/盆', '旗帜', '枕头', '靴子', '花瓶', '麦克风', '项链', '戒指', 'SUV', '酒杯', '腰带', '显示器/电视', '背包', '伞', '交通灯', '扬声器', '手表', '领带', '垃圾桶', '拖鞋', '自行车', '凳子', '桶', '面包车', '沙发', '凉鞋', '篮子', '鼓', '笔/铅笔', '公共汽车', '野鸟', '高跟鞋', '摩托车', '吉他', '地毯', '手机', '面包', '相机', '罐装食品', '卡车', '交通锥', '钹', '救生圈', '毛巾', '绒毛玩具', '蜡烛', '帆船', '笔记本电脑', '遮阳篷', '床', '水龙头', '帐篷', '马', '镜子', '电源插座', '水槽', '苹果', '空调', '刀', '曲棍球棒', '桨', '皮卡车', '叉子', '交通标志', '气球', '三脚架', '狗', '勺子', '时钟', '锅', '牛', '蛋糕', '餐桌', '绵羊', '衣架', '黑板/白板', '餐巾', '其他鱼类', '橙子/柑橘', '化妆用品', '键盘', '西红柿', '灯笼', '机械车辆', '风扇', '绿色蔬菜', '香蕉', '棒球手套', '飞机', '鼠标', '火车', '南瓜', '足球', '滑雪板', '行李', '床头柜', '茶壶', '电话', '手推车', '头戴耳机', '跑车', '停车标志', '甜点', '摩托车', '婴儿车', '起重机', '遥控器', '冰箱', '烤箱', '柠檬', '鸭子', '棒球棒', '监控摄像头', '猫', '大罐', '西兰花', '钢琴', '比萨', '大象', '滑板', '冲浪板', '枪', '滑雪鞋', '煤气灶', '甜甜圈', '领结', '胡萝卜', '马桶', '风筝', '草莓', '其他球类', '铲子', '辣椒', '电脑箱', '卫生纸', '清洁用品', '筷子', '微波炉', '鸽子', '棒球', '切菜板', '咖啡桌', '边桌', '剪刀', '马克笔', '派', '梯子', '雪板', '饼干', '散热器', '消防栓', '篮球', '斑马', '葡萄', '长颈鹿', '土豆', '香肠', '三轮车', '小提琴', '鸡蛋', '灭火器', '糖果', '消防车', '台球', '转换器', '浴缸', '轮椅', '高尔夫球杆', '公文包', '黄瓜', '雪茄/香烟', '画笔', '梨', '重型卡车', '汉堡包', '抽油烟机', '延长线', '夹子', '网球拍', '文件夹', '美式橄榄球', '耳机', '口罩', '水壶', '网球', '船', '秋千', '咖啡机', '滑梯', '马车', '洋葱', '青豆', '投影仪', '飞盘', '洗衣机/烘干机', '鸡肉', '打印机', '西瓜', '萨克斯管', '纸巾', '牙刷', '冰淇淋', '热气球', '大提琴', '薯条', '秤', '奖杯', '卷心菜', '热狗', '搅拌机', '桃子', '米饭', '钱包/手袋', '排球', '鹿', '鹅', '胶带', '平板电脑', '化妆品', '小号', '菠萝', '高尔夫球', '救护车', '停车计费器', '芒果', '钥匙', '栏杆', '钓鱼竿', '奖牌', '长笛', '刷子', '企鹅', '扩音器', '玉米', '生菜', '大蒜', '天鹅', '直升机', '青葱', '三明治', '坚果', '速限标志', '电磁炉', '扫帚', '镗鼓', '李子', '人力车', '金鱼', '猕猴桃', '路由器/调制解调器', '扑克牌', '烤面包机', '虾', '寿司', '奶酪', '记事纸', '樱桃', '钳子', '光盘', '意大利面', '锤子', '拍', '鳄梨', '甜瓜', '烧瓶', '蘑菇', '螺丝刀', '肥皂', '竖笛', '熊', '茄子', '黑板擦', '椰子', '卷尺/尺子', '猪', '淋浴头', '地球仪', '薯片', '牛排', '人行道标志', '订书机', '骆驼', '一级方程式', '石榴', '洗碗机', '螃蟹', '电动滑板', '肉丸', '电饭煲', '大号', '计算器', '木瓜', '羚羊', '鹦鹉', '海豹', '蝴蝶', '哑铃', '驴', '狮子', '小便池', '海豚', '电钻', '吹风机', '蛋挞', '水母', '跑步机', '打火机', '柚子', '游戏板', '拖把', '萝卜', '包子', '靶子', '法国', '春卷', '猴子', '兔子', '铅笔盒', '牦牛', '红包菜', '双筒望远镜', '芦笋', '杠铃', '扇贝', '面条', '梳子', '饺子', '牡蛎', '乒乓球拍', '化妆刷/眼线笔', '电锯', '橡皮擦', '龙虾', '榴莲', '秋葵', '口红', '化妆镜', '冰壶', '乒乓球']
        print(self.labels)

    def detect(self,image):
        result=self.detector.predict_image(
                [image],
                self.FLAGS.run_benchmark,
                repeats=1,
                visual=False,
                save_results=False)
        bboxes = result['boxes']
        content = []
        # print(bboxes)
        for ii in range(bboxes.shape[0]):
            if bboxes[ii,1]>0.5:
                label = self.labels[int(bboxes[ii,0])]
                content.append(label)

        # print(content)
        return content