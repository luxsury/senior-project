import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'websocket_service.dart'; 
import 'widgets.dart'; 
import 'pose_painter.dart'; 


List<CameraDescription> cameras = [];

class PoseDetectionCameraPage extends StatefulWidget {
  final String mode;
  const PoseDetectionCameraPage({super.key, required this.mode});

  @override
  State<PoseDetectionCameraPage> createState() =>
      _PoseDetectionCameraPageState();
}

class _PoseDetectionCameraPageState extends State<PoseDetectionCameraPage> {
  // --- 相機與狀態 ---
  CameraController? _controller;
  bool _isCameraInitialized = false;
  bool _isProcessing = false; // 防止重複發送
  String _errorMessage = '';
  Timer? _timer;

  // --- 服務 (Service) ---
  // 這裡使用剛剛寫好的 WebSocketService
  final WebSocketService _wsService = WebSocketService(serverIp: '192.168.1.127');

  // --- UI 顯示變數 ---
  String _displayLabel = '等待連線...';
  Color _displayColor = Colors.white;
  List<dynamic> _landmarks = []; // 給 PosePainter 畫圖用

  @override
  void initState() {
    super.initState();
    _initSystem();
  }

  Future<void> _initSystem() async {
    // 1. 先啟動 WebSocket 連線
    _wsService.connect().listen(
      (data) {
        // 收到資料後的處理邏輯
        if (mounted) {
          setState(() {
            String label = data['label_zh'] ?? "分析中";
            double conf = data['confidence'] ?? 0.0;

            _displayLabel = "$label\n信心度: ${(conf * 100).toStringAsFixed(0)}%";
            _displayColor = conf > 0.7 ? Colors.greenAccent : Colors.amber;

            if (data['landmarks'] != null) {
              _landmarks = data['landmarks'];
            }
          });
        }
      },
      onError: (error) {
        print("偵測 錯誤: $error");
        if (mounted) setState(() => _displayLabel = "連線錯誤");
      },
    );

    // 2. 再啟動相機
    await _initializeCamera();
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

      final frontCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _controller = CameraController(
        frontCamera,
        ResolutionPreset.medium, // 傳輸速度平衡點
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();
      await _controller!.setFocusMode(FocusMode.locked); // 鎖定對焦防止閃爍

      if (!mounted) return;
      setState(() => _isCameraInitialized = true);

      _startAutoCapture(); // 開始抓圖循環
    } catch (e) {
      if (mounted) setState(() => _errorMessage = '相機錯誤: $e');
    }
  }

  void _startAutoCapture() {
    // 設定每 150~200ms 抓一次圖
    _timer = Timer.periodic(const Duration(milliseconds: 200), (_) async {
      if (!_isCameraInitialized || _controller == null || _isProcessing) return;

      _isProcessing = true; // 上鎖

      try {
        final image = await _controller!.takePicture();
        final bytes = await image.readAsBytes();

        // 透過 Service 傳送，這裡不用管網路細節了
        _wsService.sendImage(bytes);

        await File(image.path).delete(); // 刪除暫存
      } catch (e) {
        print("抓圖傳輸異常: $e");
      } finally {
        if (mounted) _isProcessing = false; // 解鎖
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _wsService.disconnect(); // 斷線
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: Colors.black,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: Text(widget.mode),
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: BackButton(color: Colors.white),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. 相機畫面
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

          // 2. 骨架繪製 (PosePainter)
          if (_isCameraInitialized)
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

          // 3. 錯誤訊息
          if (!_isCameraInitialized)
            Center(
                child: Text(_errorMessage,
                    style: const TextStyle(color: Colors.red))),

          // 4. 使用分裝好的狀態燈 Widget
          ConnectionStatusIndicator(isConnected: _wsService.isConnected),

          // 5. 使用分裝好的結果顯示 Widget
          PoseResultOverlay(
            resultText: _displayLabel,
            statusColor: _displayColor,
          ),
        ],
      ),
    );
  }
}