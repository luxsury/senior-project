import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'websocket_service.dart'; 
import 'widgets.dart'; 
import 'pose_painter.dart'; 

// ❌ [已移除] import 'package:audioplayers/audioplayers.dart'; 

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
  bool _isProcessing = false; 
  String _errorMessage = '';
  Timer? _timer;

  // --- 服務 (Service) ---
  final WebSocketService _wsService = WebSocketService(serverIp: '192.168.1.127');
  
  // ❌ [已移除] 音效播放器變數 (_audioPlayer, _isPlayingSound)

  // --- UI 顯示變數 ---
  String _displayLabel = '等待連線...';
  Color _displayColor = Colors.white;
  List<dynamic> _landmarks = []; 
  
  // 角度與警告狀態變數
  int _currentAngle = 0;   // 接收 Python 傳來的角度
  bool _isWarning = false; // 接收 Python 傳來的警告 (紅屏)

  @override
  void initState() {
    super.initState();
    _initSystem();
  }

  Future<void> _initSystem() async {
    // 1. 先啟動 WebSocket 連線
    _wsService.connect().listen(
      (data) {
        if (mounted) {
          setState(() {
            String label = data['label_zh'] ?? "分析中";
            double conf = data['confidence'] ?? 0.0;

            _displayLabel = "$label\n信心度: ${(conf * 100).toStringAsFixed(0)}%";
            _displayColor = conf > 0.7 ? Colors.greenAccent : Colors.amber;

            if (data['landmarks'] != null) {
              _landmarks = data['landmarks'];
            }

            // ==========================================
            // 接收角度與警告訊號
            // ==========================================
            _currentAngle = data['angle'] ?? 0;
            _isWarning = data['warning'] ?? false;
            
            // ❌ [已移除] 播放音效的邏輯判斷
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

  // ❌ [已移除] _playWarningSound() 函式

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
        ResolutionPreset.medium, 
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _controller!.initialize();
      await _controller!.setFocusMode(FocusMode.locked); 

      if (!mounted) return;
      setState(() => _isCameraInitialized = true);

      _startAutoCapture(); 
    } catch (e) {
      if (mounted) setState(() => _errorMessage = '相機錯誤: $e');
    }
  }

  void _startAutoCapture() {
    _timer = Timer.periodic(const Duration(milliseconds: 200), (_) async {
      if (!_isCameraInitialized || _controller == null || _isProcessing) return;

      _isProcessing = true; 

      try {
        final image = await _controller!.takePicture();
        final bytes = await image.readAsBytes();
        _wsService.sendImage(bytes);
        await File(image.path).delete(); 
      } catch (e) {
        print("抓圖傳輸異常: $e");
      } finally {
        if (mounted) _isProcessing = false; 
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _wsService.disconnect(); 
    _controller?.dispose();
    // ❌ [已移除] _audioPlayer.dispose();
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

          // 2. 骨架繪製
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

          // ==========================================================
          // 3. 紅色警告遮罩 (保留)
          // ==========================================================
          // 當 _isWarning 為 true 時顯示半透明紅色，否則全透明
          Positioned.fill(
            child: IgnorePointer( // 讓點擊可以穿透，不會擋住按鈕
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 300), // 淡入淡出效果
                color: _isWarning 
                    ? Colors.red.withOpacity(0.4) 
                    : Colors.transparent,
              ),
            ),
          ),

          // 4. 錯誤訊息
          if (!_isCameraInitialized)
            Center(
                child: Text(_errorMessage,
                    style: const TextStyle(color:Colors.red))),

          // 5. 狀態燈
          ConnectionStatusIndicator(isConnected: _wsService.isConnected),

          // 6. 結果顯示 (文字標籤)
          PoseResultOverlay(
            resultText: _displayLabel,
            statusColor: _displayColor,
          ),
          
          // ==========================================================
          // 7. 角度數值顯示 (保留)
          // ==========================================================
          Positioned(
            top: 100, 
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                "角度: $_currentAngle°",
                style: TextStyle(
                  color: _isWarning ? Colors.redAccent : Colors.white, // 警告時文字變紅
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}