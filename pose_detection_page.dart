import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
// import 'package:http/http.dart' as http; // 不需要 HTTP 了
import 'pose_painter.dart';
import 'package:web_socket_channel/web_socket_channel.dart'; // 引入 WebSocket 套件

List<CameraDescription> cameras = [];

class PoseDetectionCameraPage extends StatefulWidget {
  final String mode;
  const PoseDetectionCameraPage({super.key, required this.mode});

  @override
  State<PoseDetectionCameraPage> createState() =>
      _PoseDetectionCameraPageState();
}

class _PoseDetectionCameraPageState extends State<PoseDetectionCameraPage> {
  CameraController? _controller;
  bool _isCameraInitialized = false;
  String _errorMessage = '';

  // --- WebSocket 相關變數 ---
  WebSocketChannel? _channel;
  static const String serverIp = '192.168.1.127'; // 你的 IP
  final String wsUrl = 'ws://$serverIp:8000/ws_predict'; // WebSocket 網址
  Timer? _timer; // 控制拍照頻率的計時器

  // --- 狀態變數 ---
  String _apiResult = '等待連線...';
  Color _statusColor = Colors.white;
  List<dynamic> _landmarks = []; // 骨架座標

  @override
  void initState() {
    super.initState();
    _connectWebSocket(); // 1. 先連線 WebSocket
    _initializeCamera(); // 2. 再啟動相機
  }

  // 🔥 1. 建立 WebSocket 連線並監聽回傳資料
  void _connectWebSocket() {
    try {
      print("嘗試連線到: $wsUrl");
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));

      // 監聽後端回傳的 JSON
      _channel!.stream.listen((message) {
        try {
          var data = jsonDecode(message);

          if (mounted) {
            setState(() {
              // 更新辨識結果文字
              String label = data['label_zh'] ?? "分析中";
              double conf = data['confidence'] ?? 0.0;
              _apiResult = "$label\n信心度: ${(conf * 100).toStringAsFixed(0)}%";
              _statusColor = conf > 0.7 ? Colors.greenAccent : Colors.amber;

              // 更新紅點座標 (PosePainter 用)
              if (data['landmarks'] != null) {
                _landmarks = data['landmarks'];
              }
            });
          }
        } catch (e) {
          print("解析 JSON 錯誤: $e");
        }
      }, onError: (error) {
        print("WebSocket 錯誤: $error");
        if (mounted) setState(() => _apiResult = "連線錯誤");
      }, onDone: () {
        print("WebSocket 連線關閉");
      });
    } catch (e) {
      print("連線失敗: $e");
      if (mounted) setState(() => _apiResult = "無法連線主機");
    }
  }

  Future<void> _initializeCamera() async {
    try {
      if (cameras.isEmpty) {
        cameras = await availableCameras();
      }
      if (cameras.isEmpty) {
        setState(() => _errorMessage = '找不到相機');
        return;
      }

      final selectedCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _controller = CameraController(
        selectedCamera,
        ResolutionPreset.medium, // 降低解析度有助於加快傳輸速度 (medium 或 low)
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();
      if (!mounted) return;
      setState(() => _isCameraInitialized = true);

      _startAutoCapture(); // 相機初始化完畢，開始計時器抓圖
    } catch (e) {
      if (mounted) setState(() => _errorMessage = '相機啟動錯誤: $e');
    }
  }

  // 🔥 新增一個變數來控制是否正在處理中
  bool _isProcessing = false;

  // 🔥 2. 自動抓圖邏輯 (使用 Timer 替代原本的遞迴)
  void _startAutoCapture() {
    // 每 60ms 執行一次 (約 15-16 FPS)
    _timer = Timer.periodic(const Duration(milliseconds: 150), (timer) async {
      // 安全檢查：相機未初始化或正在拍照時不執行
      if (!_isCameraInitialized ||
          _controller == null ||
          _controller!.value.isTakingPicture) return;

      // 🔥 2. 關鍵修正：如果上一張還在忙，這一次就直接「跳過」，不要讓任務堆積！
      if (_isProcessing) return;

      _isProcessing = true; // 🔒 上鎖

      try {
        // 拍照
        final XFile imageFile = await _controller!.takePicture();
        final bytes = await imageFile.readAsBytes();

        // 🔥 直接把圖片 Bytes 丟進 WebSocket 管線
        if (_channel != null && _channel!.closeCode == null) {
          _channel!.sink.add(bytes);
        }

        // 刪除暫存檔 (避免手機儲存空間爆炸)
        await File(imageFile.path).delete();
      } catch (e) {
        print("抓圖或傳送失敗: $e");
        // 這裡可以選擇不處理錯誤，因為即時串流掉一兩幀沒關係
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel(); // 停止計時器
    _channel?.sink.close(); // 關閉 WebSocket 連線
    _controller?.dispose(); // 釋放相機資源
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    // 判斷是否連線中 (用來改變 UI 燈號顏色)
    bool isConnected = _channel != null && _channel!.closeCode == null;

    return Scaffold(
      backgroundColor: Colors.black,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: Text(widget.mode),
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. 相機預覽層
          if (_isCameraInitialized && _controller != null)
            SizedBox.expand(
              child: FittedBox(
                fit: BoxFit.contain,
                child: SizedBox(
                  width: _controller!.value.previewSize?.height ?? size.width,
                  height: _controller!.value.previewSize?.width ?? size.height,
                  child: CameraPreview(_controller!),
                ),
              ),
            ),

          // 2. 骨架繪製層 (PosePainter)
          if (_isCameraInitialized && _controller != null)
            SizedBox.expand(
              child: FittedBox(
                fit: BoxFit.contain,
                child: SizedBox(
                  width: _controller!.value.previewSize?.height ?? size.width,
                  height: _controller!.value.previewSize?.width ?? size.height,
                  child: CustomPaint(
                    painter: PosePainter(_landmarks),
                  ),
                ),
              ),
            ),

          // 3. 錯誤訊息層
          if (!_isCameraInitialized)
            Center(
              child: _errorMessage.isNotEmpty
                  ? Text(_errorMessage,
                      style: const TextStyle(color: Colors.red))
                  : const CircularProgressIndicator(),
            ),

          // 4. 右上角狀態指示燈
          Positioned(
            top: 60,
            right: 20,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                children: [
                  // 綠色代表連線中，灰色代表斷線
                  Icon(Icons.circle,
                      color: isConnected ? Colors.green : Colors.grey,
                      size: 12),
                  const SizedBox(width: 8),
                  Text(isConnected ? "WS 連線中" : "離線",
                      style: const TextStyle(color: Colors.white)),
                ],
              ),
            ),
          ),

          // 5. 底部結果顯示層
          Positioned(
            bottom: 40,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.6),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: Colors.white24)),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    _apiResult,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                        color: _statusColor,
                        fontSize: 32,
                        fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
