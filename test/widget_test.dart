import 'package:flutter_test/flutter_test.dart';

import 'package:transnote/main.dart';

void main() {
  testWidgets('앱 셸에 3개 탭이 뜨는지 확인', (WidgetTester tester) async {
    await tester.pumpWidget(const TransNoteApp());

    expect(find.textContaining('튜토리얼'), findsWidgets);
    expect(find.text('악보'), findsWidgets);
    expect(find.text('연습'), findsWidgets);
  });
}
