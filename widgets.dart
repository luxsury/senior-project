import 'package:flutter/material.dart';

// --- 元件 1: 右上角狀態燈 ---
class ConnectionStatusIndicator extends StatelessWidget {
  final bool isConnected;

  const ConnectionStatusIndicator({super.key, required this.isConnected});

  @override
  Widget build(BuildContext context) {
    return Positioned(
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
            Icon(Icons.circle,
                color: isConnected ? Colors.greenAccent : Colors.grey, 
                size: 12),
            const SizedBox(width: 8),
            Text(isConnected ? "WS 連線中" : "離線",
                style: const TextStyle(color: Colors.white)),
          ],
        ),
      ),
    );
  }
}

// --- 元件 2: 底部結果顯示框 ---
class PoseResultOverlay extends StatelessWidget {
  final String resultText;
  final Color statusColor;

  const PoseResultOverlay({
    super.key,
    required this.resultText,
    required this.statusColor,
  });

  @override
  Widget build(BuildContext context) {
    return Positioned(
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
              resultText,
              textAlign: TextAlign.center,
              style: TextStyle(
                  color: statusColor,
                  fontSize: 32,
                  fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}