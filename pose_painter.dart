import 'package:flutter/material.dart';

class PosePainter extends CustomPainter {
  final List<dynamic> landmarks; // 接收後端傳來的座標列表

  PosePainter(this.landmarks);

  // 定義要連接的骨架點 (參考 MediaPipe 標準連線)
  final List<List<int>> connections = [
    [11, 12], // 肩膀
    [11, 13], [13, 15], // 左手
    [12, 14], [14, 16], // 右手
    [11, 23], [12, 24], // 軀幹
    [23, 24], // 髖部
    [23, 25], [25, 27], // 左腳
    [24, 26], [26, 28], // 右腳
    [27, 29], [29, 31], // 左腳掌
    [28, 30], [30, 32], // 右腳掌
  ];

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.isEmpty) return;

    final paintLine = Paint()
      ..color = Colors.green // 線的顏色
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;

    final paintPoint = Paint()
      ..color = Colors.red // 點的顏色
      ..strokeWidth = 5.0
      ..style = PaintingStyle.fill;

    // 1. 畫點 (Keypoints)
    // 我們要把 0~1 的座標轉換成螢幕的實際寬高 (size.width, size.height)
    for (var point in landmarks) {
      double x = (1 - point['x']) * size.width; // 加上 1 - ... 來水平翻轉
      double y = point['y'] * size.height;
      
      // 如果這個點的信心度太低(例如被遮住)，可以選擇不畫
      if (point['v'] > 0.5) {
        canvas.drawCircle(Offset(x, y), 4, paintPoint);
      }
    }

    // 2. 畫線 (Skeleton)
    for (var pair in connections) {
      int idx1 = pair[0];
      int idx2 = pair[1];

      // 確保座標存在
      if (idx1 < landmarks.length && idx2 < landmarks.length) {
        var p1 = landmarks[idx1];
        var p2 = landmarks[idx2];

        // 檢查兩點是否可見
        if (p1['v'] > 0.5 && p2['v'] > 0.5) {
            // 🔥 修改點 2：線的座標也要跟著翻轉
            double x1 = (1 - p1['x']) * size.width;
            double y1 = p1['y'] * size.height;
            double x2 = (1 - p2['x']) * size.width;
            double y2 = p2['y'] * size.height;

            canvas.drawLine(Offset(x1, y1), Offset(x2, y2), paintLine);
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true; // 每次數據更新都要重畫
  }
}