// ==========================================
// 3. 動作詳情頁 (ExerciseDetailScreen) - 保持不變
// ==========================================
import 'package:flutter/material.dart';

class ExerciseDetailScreen extends StatelessWidget {
  final String title;
  final String iconPath;
  final String description;

  const ExerciseDetailScreen({
    super.key,
    required this.title,
    required this.iconPath,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
        backgroundColor: Colors.transparent,
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: double.infinity,
              height: 250,
              color: Colors.black,
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Image.asset(
                      iconPath,
                      width: 120,
                      height: 120,
                      color: Colors.white70,
                      errorBuilder: (context, error, stackTrace) => const Icon(
                          Icons.directions_run,
                          size: 100,
                          color: Colors.white24),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      "動作示意圖",
                      style: TextStyle(color: Colors.white38),
                    )
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Image.asset(iconPath,
                          width: 32,
                          height: 32,
                          color: Colors.blueAccent,
                          errorBuilder: (context, error, stackTrace) =>
                              const SizedBox()),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          title,
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    "動作說明",
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.blueAccent,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.grey[900],
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white12),
                    ),
                    child: Text(
                      description,
                      style: const TextStyle(
                        fontSize: 16,
                        color: Colors.white70,
                        height: 1.6,
                      ),
                    ),
                  ),
                  const SizedBox(height: 30),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
