import 'package:flutter/services.dart';

/// 피아노 건반이 나오는 화면(튜토리얼/악보 재생 등) 전용 가로모드 잠금 헬퍼.
/// 앱 기본은 세로(main.dart)이므로, 이 화면들은 initState에서 lockLandscape(),
/// dispose에서 lockPortrait()를 호출해 진입/이탈 시점에만 방향을 전환한다.
Future<void> lockLandscape() => SystemChrome.setPreferredOrientations([
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);

Future<void> lockPortrait() => SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
