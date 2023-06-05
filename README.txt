文件夹内容：
-reference：参考文献集
-project：项目文件
	-demo
		-data：yolo的hyp等部分库文件，用来测试手势的训练集
		-engine：
		-hand_data_iter：手部关键点检测相关文件
		-image_handpose：手部关键点检测的测试图片
		-loss
		-models:yolo的模型文件
		-runs：手势识别的优化结果
			-HaGRID-result：测试集的结果
			-yolov5s：yolov5s测试的结果
				-weights：模型文件，所采用的是best.pt
			-yolov5s_640：和yolov5s文件相同，只是调用时做了备份
		-utils
		-demo.py:程序文件源码
		-train.py：手势识别的训练代码
		-ReXNetV1-size-256-loss-wing_loss102-20211104.pth：关键点回归的模型
		-train_handpoint.py：手关键点回归的训练代码
		-inference.py:手部关键点检测推理的测试
	-HaGRID：手势识别所用的训练集，再原基础上做了简化，来自:https://github.com/hukenovs/hagrid
	-handpose_datasets_v1：手部关键点回归的数据集，来自:http://www.rovit.ua.es/dataset/mhpdataset/#download
	-demo.MP4:示范视频

项目功能简介：
	在手势识别和手部关键点识别的基础上，利用已获得的数据来进行点读操作的模拟。实现了两种方式的框选，以及文字识别，翻译点读的功能

测试代码使用方法：
	直接运行demo.py即可，请确保摄像头和cuda可用。
	针对以下几种手势设计了操作：拳头：返回主菜单；主菜单下，1：对角线框选模式；2：四角框选模式（3：本计划做划线识别，未完成）；进入各模式后做出任意手势，会识别出手部的关节，两指并拢表示点击，为避免误触，会在右下方读条1.5s，读条结束后视为一次点击操作。通过点击操作完成框选后，大拇指向上给出翻译+点读的指令，大拇指向下完成不翻译直接点读的指令。

推理测试使用方法：
	运行inference.py进行手部关键点检测的推理，注意需要修改相关的模型和测试集的地址
	收拾检测因为直接用推理的文件改写成了主函数，所以没有单独准备推理测试

通过tensorboard可视化查看手势检测的训练结果：
	进入训练数据的文件夹：-yolov5s或-yolov5s_640，在终端执行以下命令：tensorboard --logdir ./ 打开终端中的链接查看数据
