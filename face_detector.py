#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时人脸检测程序
使用 InsightFace 和 OpenCV 实现摄像头实时人脸检测
"""

import cv2
import numpy as np
import threading
import time
import os
import chromadb
import torch
import torch.nn.functional as F
from pathlib import Path
from queue import Queue
from insightface.app import FaceAnalysis


class MiniFASNet(torch.nn.Module):
    """MiniFASNet 活体检测模型"""
    
    def __init__(self):
        super(MiniFASNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.fc = torch.nn.Linear(64 * 20 * 20, 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RealtimeFaceDetector:
    """实时人脸检测器类"""
    
    def __init__(self, camera_id=0, det_size=(320, 320), skip_frames=10, face_db_path=None, enable_liveness=True):
        """
        初始化人脸检测器
        
        Args:
            camera_id: 摄像头ID，默认为0（第一个摄像头）
            det_size: 检测图像的尺寸，默认为(320, 320)，越小速度越快
            skip_frames: 跳帧检测，每N帧检测一次，默认为10（约100ms采样一次）
            face_db_path: 人脸库文件夹路径，存放已知人脸图片
            enable_liveness: 是否启用活体检测
        """
        self.camera_id = camera_id
        self.det_size = det_size
        self.skip_frames = skip_frames
        self.enable_liveness = enable_liveness
        
        # 人脸库相关
        self.face_db_path = face_db_path
        self.face_threshold = 0.4  # 人脸相似度阈值，越小越严格
        self.liveness_threshold = 0.7  # 活体检测阈值
        
        # 初始化活体检测模型
        self.liveness_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if enable_liveness:
            self._init_liveness_model()
        
        # 初始化 ChromaDB 向量数据库
        self.chroma_client = None
        self.face_collection = None
        if face_db_path:
            self._init_vector_db()
        
        # 异步检测相关
        self.frame_queue = Queue(maxsize=1)  # 待检测帧队列
        self.result_queue = Queue(maxsize=1)  # 检测结果队列
        self.detection_thread = None
        self.stop_detection = False
        self.latest_faces = []  # 最新检测结果
        
        # 初始化 InsightFace（使用更小的模型）
        print("正在初始化 InsightFace 模型...")
        self.app = FaceAnalysis(
            name='buffalo_sc',  # 使用 buffalo_sc 小模型（比 buffalo_l 快很多）
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        print("模型初始化完成！")
        
        # 加载人脸库
        if face_db_path:
            self.load_face_database(face_db_path)
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        
        # 设置摄像头参数（降低分辨率提升性能）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def _init_liveness_model(self):
        """初始化活体检测模型"""
        print("正在初始化活体检测模型...")
        self.liveness_model = MiniFASNet().to(self.device)
        self.liveness_model.eval()
        
        # 尝试加载预训练权重（如果存在）
        model_path = Path("./models/minifasnet.pth")
        if model_path.exists():
            try:
                self.liveness_model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("已加载预训练活体检测模型")
            except Exception as e:
                print(f"警告: 无法加载预训练模型，使用随机初始化: {e}")
        else:
            print("警告: 未找到预训练模型，活体检测可能不准确")
            print(f"请将模型文件放置在: {model_path}")
    
    def check_liveness(self, face_img):
        """
        检测人脸是否为活体
        
        Args:
            face_img: 人脸区域图像 (BGR格式)
            
        Returns:
            tuple: (is_live: bool, score: float)
        """
        if not self.enable_liveness or self.liveness_model is None:
            return True, 1.0
        
        try:
            # 预处理图像
            img = cv2.resize(face_img, (80, 80))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                output = self.liveness_model(img)
                prob = F.softmax(output, dim=1)
                live_score = prob[0][1].item()  # 活体概率
            
            is_live = live_score > self.liveness_threshold
            return is_live, live_score
            
        except Exception as e:
            print(f"活体检测错误: {e}")
            return True, 0.0  # 出错时默认为活体
    
    def _init_vector_db(self):
        """初始化 ChromaDB 向量数据库"""
        print("正在初始化向量数据库...")
        
        # 创建或连接到 ChromaDB
        db_storage_path = Path(self.face_db_path) / ".vector_db"
        db_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=str(db_storage_path))
        
        # 获取或创建人脸特征集合
        try:
            self.face_collection = self.chroma_client.get_collection(
                name="face_embeddings"
            )
            print(f"已加载现有人脸库，共 {self.face_collection.count()} 条记录")
        except:
            # 集合不存在，创建新集合
            self.face_collection = self.chroma_client.create_collection(
                name="face_embeddings",
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            print("创建新的人脸库")
    
    def load_face_database(self, db_path):
        """
        从指定文件夹加载人脸库到向量数据库
        
        Args:
            db_path: 人脸库文件夹路径，支持 jpg, jpeg, png 格式
        """
        print(f"\n正在加载人脸库: {db_path}")
        db_path = Path(db_path)
        
        if not db_path.exists():
            print(f"警告: 人脸库路径不存在，将创建: {db_path}")
            db_path.mkdir(parents=True, exist_ok=True)
            return
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in db_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("警告: 人脸库为空，请添加人脸图片")
            print(f"图片命名格式: 姓名.jpg (例如: 张三.jpg, John.png)")
            return
        
        # 检查哪些图片已经在数据库中
        existing_ids = set()
        if self.face_collection.count() > 0:
            existing_data = self.face_collection.get()
            existing_ids = set(existing_data['ids'])
        
        added_count = 0
        skipped_count = 0
        
        # 加载每个人脸图片
        for img_file in image_files:
            try:
                # 使用文件路径作为唯一ID
                file_id = str(img_file.name)
                
                # 如果已存在，跳过
                if file_id in existing_ids:
                    print(f"  → 已存在: {img_file.stem}")
                    skipped_count += 1
                    continue
                
                # 读取图片
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"  ✗ 无法读取: {img_file.name}")
                    continue
                
                # 检测人脸并提取特征
                faces = self.app.get(img)
                
                if len(faces) == 0:
                    print(f"  ✗ 未检测到人脸: {img_file.name}")
                    continue
                elif len(faces) > 1:
                    print(f"  ⚠ 检测到多张人脸，使用第一张: {img_file.name}")
                
                # 获取人脸特征向量
                face = faces[0]
                embedding = face.normed_embedding.tolist()  # 转为列表存储
                
                # 提取姓名（文件名去掉扩展名）
                name = img_file.stem
                
                # 添加到向量数据库
                self.face_collection.add(
                    embeddings=[embedding],
                    ids=[file_id],
                    metadatas=[{"name": name, "file": img_file.name}]
                )
                
                print(f"  ✓ 已加载: {name}")
                added_count += 1
                
            except Exception as e:
                print(f"  ✗ 加载失败 {img_file.name}: {e}")
        
        total_count = self.face_collection.count()
        print(f"\n人脸库加载完成: 新增 {added_count} 人，跳过 {skipped_count} 人，总计 {total_count} 人\n")
    
    def recognize_face(self, face_embedding):
        """
        使用向量数据库识别人脸，返回最匹配的姓名
        
        Args:
            face_embedding: 待识别人脸的特征向量
            
        Returns:
            tuple: (姓名, 相似度分数) 或 (None, 0) 如果未识别
        """
        if not self.face_collection or self.face_collection.count() == 0:
            return None, 0
        
        # 在向量数据库中查询最相似的人脸
        results = self.face_collection.query(
            query_embeddings=[face_embedding.tolist()],
            n_results=1  # 返回最相似的1个结果
        )
        
        if not results['ids'] or len(results['ids'][0]) == 0:
            return None, 0
        
        # 获取最佳匹配结果
        best_metadata = results['metadatas'][0][0]
        best_distance = results['distances'][0][0]
        
        # ChromaDB 使用余弦距离，距离越小越相似
        # 余弦距离范围 [0, 2]，0表示完全相同
        # 转换为相似度：similarity = 1 - distance/2
        similarity = 1 - (best_distance / 2)
        
        # 如果相似度超过阈值，返回识别结果
        if similarity > (1 - self.face_threshold):
            return best_metadata['name'], similarity
        
        return None, similarity
        
    def draw_face_info(self, img, face, frame):
        """
        在图像上绘制人脸信息
        
        Args:
            img: 原始图像
            face: InsightFace 检测到的人脸对象
            frame: 完整帧图像，用于提取人脸区域进行活体检测
        """
        # 获取边界框
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # 确保边界框在图像范围内
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 活体检测
        is_live = True
        live_score = 1.0
        if self.enable_liveness and x2 > x1 and y2 > y1:
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                is_live, live_score = self.check_liveness(face_region)
        
        # 只有活体才进行人脸识别
        recognized_name = None
        similarity = 0
        if is_live and hasattr(face, 'normed_embedding') and self.face_collection:
            recognized_name, similarity = self.recognize_face(face.normed_embedding)
        
        # 根据活体检测和识别结果选择颜色
        if not is_live:
            box_color = (0, 0, 255)  # 红色 - 非活体
            label_bg_color = (0, 0, 200)
        elif recognized_name:
            box_color = (0, 255, 0)  # 绿色 - 活体且已识别
            label_bg_color = (0, 200, 0)
        else:
            box_color = (0, 165, 255)  # 橙色 - 活体但未识别
            label_bg_color = (0, 140, 200)
        if recognized_name:
            box_color = (0, 255, 0)  # 绿色 - 活体且已识别
            label_bg_color = (0, 200, 0)
        else:
            box_color = (0, 165, 255)  # 橙色 - 活体但未识别
            label_bg_color = (0, 140, 200)
        
        # 绘制人脸框
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        
        # 获取置信度
        score = face.det_score
        
        # 获取关键点（5个面部特征点）
        if hasattr(face, 'kps') and face.kps is not None:
            kps = face.kps.astype(int)
            # 绘制关键点（蓝色圆点）
            for kp in kps:
                cv2.circle(img, tuple(kp), 2, (255, 0, 0), -1)
        
        # 显示信息
        text_y = y1 - 10
        if text_y < 80:
            text_y = y2 + 25
        
        # 显示活体检测状态
        if self.enable_liveness:
            if not is_live:
                liveness_text = f"FAKE! ({live_score:.2f})"
                (text_w, text_h), _ = cv2.getTextSize(liveness_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x1, text_y - text_h - 10), (x1 + text_w + 10, text_y), (0, 0, 200), -1)
                cv2.putText(img, liveness_text, (x1 + 5, text_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                text_y -= (text_h + 15)
            else:
                liveness_text = f"Live ({live_score:.2f})"
                cv2.putText(img, liveness_text, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                text_y -= 25
        
        # 如果是活体且识别到人脸，显示姓名
        if is_live and recognized_name:
            # 绘制姓名背景
            name_text = f"{recognized_name} ({similarity:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, text_y - text_h - 10), (x1 + text_w + 10, text_y), label_bg_color, -1)
            
            # 绘制姓名文字
            cv2.putText(img, name_text, (x1 + 5, text_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            text_y -= (text_h + 15)
        elif is_live:
            # 活体但未识别
            unknown_text = "Unknown"
            cv2.putText(img, unknown_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            text_y -= 25
            
        # 显示置信度
        confidence_text = f"Det: {score:.2f}"
        cv2.putText(img, confidence_text, (x1, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    def detection_worker(self):
        """异步检测线程，持续从队列获取帧并检测"""
        while not self.stop_detection:
            try:
                # 获取待检测的帧（超时1秒）
                frame = self.frame_queue.get(timeout=1.0)
                
                # 执行人脸检测
                faces = self.app.get(frame)
                
                # 将结果放入队列（如果队列满了，丢弃旧结果）
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                self.result_queue.put(faces)
                
            except:
                continue
    
    def run(self):
        """运行实时人脸检测"""
        print("\n开始实时人脸检测...")
        print("按 'q' 键退出程序")
        print("按 's' 键保存当前帧")
        print("使用异步检测和跳帧优化，提升流畅度\n")
        
        # 启动异步检测线程
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()
        
        frame_count = 0
        fps_start_time = time.time()
        fps_display = 0
        
        while True:
            # 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # 计算FPS（每30帧计算一次）
            if frame_count % 30 == 0:
                fps_display = int(30 / (time.time() - fps_start_time))
                fps_start_time = time.time()
            
            # 跳帧检测：每skip_frames帧才检测一次
            if frame_count % self.skip_frames == 0:
                # 将帧放入检测队列（如果队列满了，替换旧帧）
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except:
                    pass
            
            # 尝试获取最新的检测结果
            try:
                self.latest_faces = self.result_queue.get_nowait()
            except:
                pass  # 没有新结果，使用上一次的结果
            
            # 绘制检测到的人脸信息
            for face in self.latest_faces:
                self.draw_face_info(frame, face, frame)
            
            # 显示统计信息
            db_count = self.face_collection.count() if self.face_collection else 0
            liveness_status = "ON" if self.enable_liveness else "OFF"
            info_text = f"Faces: {len(self.latest_faces)} | FPS: {fps_display} | DB: {db_count} | Liveness: {liveness_status}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 显示提示信息
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示图像
            cv2.imshow('Real-time Face Detection', frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n退出程序...")
                break
            elif key == ord('s'):
                # 保存当前帧
                filename = f'face_detection_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"已保存图像: {filename}")
        
        # 释放资源
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.stop_detection = True
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        self.cap.release()
        cv2.destroyAllWindows()
        print("资源已释放")


def main():
    """主函数"""
    try:
        # 创建检测器实例
        detector = RealtimeFaceDetector(
            camera_id=0,  # 使用第一个摄像头
            det_size=(320, 320),  # 使用较小的检测尺寸，提升速度
            skip_frames=3,  # 每3帧检测一次（约100ms采样）
            face_db_path="./face_database",  # 人脸库路径
            enable_liveness=True  # 启用活体检测
        )
        
        # 运行检测
        detector.run()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
