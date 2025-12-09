// lib/screens/welcome_screen.dart
import 'package:flutter/material.dart';
import 'selection_screen.dart'; // 引入選單頁

// ==========================================
// 1. 首頁 (WelcomeScreen) - 保持不變
// ==========================================
class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});

  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _opacityAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _opacityAnimation = Tween<double>(begin: 0.2, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _navigateToSelection() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => const SelectionScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: _navigateToSelection,
      child: Scaffold(
        body: Container(
          width: double.infinity,
          height: double.infinity,
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Color(0xFF2C3E50),
                Color(0xFF1F2D3A),
                Color(0xFF121212),
              ],
            ),
          ),
          child: SafeArea(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Spacer(flex: 2),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24.0),
                  child: Column(
                    children: const [
                      Icon(Icons.directions_run, size: 80, color: Colors.white),
                      SizedBox(height: 20),
                      Text(
                        '跑步前後',
                        style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.w300,
                            color: Colors.white70,
                            letterSpacing: 2.0),
                      ),
                      Text(
                        '熱身與伸展指導',
                        style: TextStyle(
                            fontSize: 36,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                            letterSpacing: 1.5),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
                const Spacer(flex: 3),
                FadeTransition(
                  opacity: _opacityAnimation,
                  child: Column(
                    children: const [
                      Padding(
                        padding: EdgeInsets.symmetric(vertical: 10.0),
                        child: Text(
                          'Start',
                          style: TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                            letterSpacing: 8.0,
                            height: 2.0,
                          ),
                          overflow: TextOverflow.visible,
                          softWrap: false,
                        ),
                      ),
                      Text('點擊螢幕開始',
                          style:
                              TextStyle(fontSize: 14, color: Colors.white38)),
                    ],
                  ),
                ),
                const SizedBox(height: 50),
              ],
            ),
          ),
        ),
      ),
    );
  }
}