import 'dart:convert';
import 'dart:typed_data';
import 'package:web_socket_channel/web_socket_channel.dart';

class WebSocketService {
  // 單例模式 (可選，但這裡我們用簡單的物件建立)
  WebSocketChannel? _channel;
  final String serverIp;
  final String port;

  WebSocketService({required this.serverIp, this.port = '8000'});

  // 1. 建立連線，並回傳一個 Stream 讓外部監聽數據
  Stream<dynamic> connect() {
    final wsUrl = 'ws://$serverIp:$port/ws_predict';
    print("正在連線到 WebSocket: $wsUrl");

    try {
      _channel = WebSocketChannel.connect(Uri.parse(wsUrl));
      
      // 將收到的 Raw String 轉成 JSON Map 丟出去
      return _channel!.stream.map((message) {
        return jsonDecode(message);
      });
    } catch (e) {
      print("WebSocket 連線失敗: $e");
      // 回傳一個空的 Stream 避免崩潰
      return const Stream.empty();
    }
  }

  // 2. 傳送圖片 Bytes
  void sendImage(Uint8List bytes) {
    if (_channel != null && _channel!.closeCode == null) {
      _channel!.sink.add(bytes);
    }
  }

  // 3. 關閉連線
  void disconnect() {
    if (_channel != null) {
      _channel!.sink.close();
      print("WebSocket 連線已關閉");
    }
  }

  // 取得目前連線狀態 (true = 連線中)
  bool get isConnected => _channel != null && _channel!.closeCode == null;
}