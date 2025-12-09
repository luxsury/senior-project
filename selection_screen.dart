// lib/screens/selection_screen.dart
import 'package:flutter/material.dart';
import 'exercise_detail_screen.dart'; // 引入詳情頁
import 'pose_detection_page.dart';    // 引入相機頁

// ==========================================
// 2. 選單頁 (SelectionScreen) - 保持不變
// ==========================================
class SelectionScreen extends StatelessWidget {
  const SelectionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('選擇模式'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildActionCard(
                context,
                title: '熱身',
                subtitle: 'Warm-up',
                color: Colors.orangeAccent,
                subIcons: [
                  _buildClickableIcon(
                    context,
                    assetPath: 'assets/icons/hamstring_sweep.png',
                    title: '向下划船 (Hamstring Sweep)',
                    description:
                        '雙腳與肩同寬，一步向前伸出，腳跟著地，腳尖勾起。雙手像划船一樣向下畫圓，感受大腿後側的伸展。',
                  ),
                  _buildClickableIcon(
                    context,
                    assetPath: 'assets/icons/high_kicks.png',
                    title: '踢腿 (High Kicks)',
                    description: '保持上半身直立，將一腿向前踢高，嘗試用對側手觸碰腳尖。這能有效動態伸展腿後肌群。',
                  ),
                  _buildClickableIcon(
                    context,
                    assetPath: 'assets/icons/knee_hugs.png',
                    title: '抬膝抱腿 (Knee Hugs)',
                    description: '單腳站立，將另一腳膝蓋抬高並用雙手抱向胸口，感受臀部肌肉的伸展。保持背部挺直。',
                  ),
                ],
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) =>
                            const PoseDetectionCameraPage(mode: '熱身模式')),
                  );
                },
              ),
              const SizedBox(height: 30),
              _buildActionCard(
                context,
                title: '收操',
                subtitle: 'Cool-down',
                color: Colors.tealAccent,
                subIcons: [
                  _buildClickableIcon(
                    context,
                    assetPath: 'assets/icons/kneeling_quad_stretch.png',
                    title: '跪姿股四頭肌伸展',
                    description: '單膝跪地，另一手抓住後腳腳踝往臀部拉近。這個動作能深度放鬆大腿前側肌肉。',
                  ),
                  _buildClickableIcon(
                    context,
                    assetPath: 'assets/icons/hamstring_stretch.png',
                    title: '腿後肌伸展',
                    description: '坐姿或站姿，單腳伸直，上半身向前傾，直到感覺大腿後側有拉伸感。維持15-30秒。',
                  ),
                ],
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) =>
                            const PoseDetectionCameraPage(mode: '收操模式')),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildClickableIcon(
    BuildContext context, {
    required String assetPath,
    required String title,
    required String description,
  }) {
    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ExerciseDetailScreen(
              title: title,
              iconPath: assetPath,
              description: description,
            ),
          ),
        );
      },
      child: Container(
        padding: const EdgeInsets.all(4),
        child: Image.asset(
          assetPath,
          width: 48,
          height: 48,
          color: Colors.white70,
          errorBuilder: (context, error, stackTrace) =>
              const Icon(Icons.broken_image, color: Colors.white24),
        ),
      ),
    );
  }

  Widget _buildActionCard(
    BuildContext context, {
    required String title,
    required String subtitle,
    required Color color,
    required VoidCallback onTap,
    List<Widget>? subIcons,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        width: double.infinity,
        height: subIcons != null ? 200 : 160,
        decoration: BoxDecoration(
          color: Colors.grey[900],
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(0.5), width: 1),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(0.2),
              blurRadius: 15,
              offset: const Offset(0, 5),
            )
          ],
        ),
        child: Row(
          children: [
            Expanded(
              child: Padding(
                padding: const EdgeInsets.only(left: 20.0, right: 8.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    Text(
                      subtitle,
                      style: const TextStyle(
                        fontSize: 16,
                        color: Colors.white54,
                        letterSpacing: 1.2,
                      ),
                    ),
                    if (subIcons != null) ...[
                      const SizedBox(height: 16),
                      Row(
                        children: [
                          Column(
                            children: [
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(
                                  color: color.withOpacity(0.2),
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: Text(
                                  "含${subIcons.length}個動作",
                                  style: TextStyle(fontSize: 10, color: color),
                                ),
                              ),
                              const SizedBox(height: 4),
                              const Text(
                                "(點擊圖示查看)",
                                style: TextStyle(
                                    fontSize: 8, color: Colors.white38),
                              ),
                            ],
                          ),
                          const SizedBox(width: 8),
                          ...subIcons.map((widget) => Padding(
                                padding: const EdgeInsets.only(right: 12.0),
                                child: widget,
                              )),
                        ],
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}