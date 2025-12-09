import 'dart:async'; 
import 'dart:io'; // 雖然這裡沒用到，但保留無妨
import 'package:flutter/material.dart';
import 'package:camera/camera.dart'; 

// 引入拆分後的檔案
import 'screens/welcome_screen.dart';
import 'screens/pose_detection_page.dart'; // 為了使用 cameras 變數

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    // 預先取得相機列表 (cameras 變數現在位於 pose_detection_page.dart)
    cameras = await availableCameras();
  } on CameraException catch (e) {
    print('Error: $e.code\nError Message: $e.message');
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '跑步熱身指導',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        primarySwatch: Colors.blue,
        scaffoldBackgroundColor: const Color(0xFF121212),
        useMaterial3: true,
      ),
      home: const WelcomeScreen(),
    );
  }
}