// webpage/js/piano.js 포팅 — 88건반(A0~C8), 흰건반 대문자 / 검은건반 숫자(1~5).
import 'dart:async';
import 'package:flutter/material.dart';
import '../data/samples.dart';

/// 피아노를 화면 하단에 붙는 "악기 독"처럼 보이게 감싸는 카드.
/// 둥근 위쪽 모서리 + 그림자로 콘텐츠 영역과 분리하고, 홈 인디케이터가 있는
/// 기기에서도 건반이 잘리지 않도록 하단 SafeArea를 챙긴다.
class PianoDock extends StatelessWidget {
  final Widget child;
  const PianoDock({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerLow,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 12,
            offset: const Offset(0, -3),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(10, 14, 10, 10),
          child: child,
        ),
      ),
    );
  }
}

List<String> _generate88Keys() {
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  final all = <String>[];
  for (var oct = 0; oct <= 8; oct++) {
    for (final n in names) {
      all.add('$n$oct');
    }
  }
  return all.sublist(all.indexOf('A0'), all.indexOf('C8') + 1);
}

final List<String> kPianoKeys = _generate88Keys();

bool _isBlack(String pitch) => pitch.contains('#');

/// 정답/오답 피드백을 외부(연습·악보 화면)에서 트리거하기 위한 컨트롤러.
class PianoController extends ChangeNotifier {
  String? _flashPitch;
  bool _flashOk = true;
  Timer? _timer;

  void flashCorrect(String pitch) => _flash(pitch, true);
  void flashWrong(String pitch) => _flash(pitch, false);

  void _flash(String pitch, bool ok) {
    _timer?.cancel();
    _flashPitch = pitch;
    _flashOk = ok;
    notifyListeners();
    _timer = Timer(const Duration(milliseconds: 380), () {
      _flashPitch = null;
      notifyListeners();
    });
  }

  String? get flashPitch => _flashPitch;
  bool get flashOk => _flashOk;

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

class PianoWidget extends StatefulWidget {
  final void Function(String pitch)? onKeyPress;
  final String? expectedPitch;
  final Set<String> expectedPitches; // 화음일 때 여러 개 동시 표시
  final PianoController? controller;
  final String initialCenterPitch;
  final double whiteKeyWidth;

  const PianoWidget({
    super.key,
    this.onKeyPress,
    this.expectedPitch,
    this.expectedPitches = const {},
    this.controller,
    this.initialCenterPitch = 'C4',
    this.whiteKeyWidth = 34,
  });

  @override
  State<PianoWidget> createState() => _PianoWidgetState();
}

class _PianoWidgetState extends State<PianoWidget> {
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToCenter());
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToCenter() {
    if (!_scrollController.hasClients) return;
    final idx = kPianoKeys.indexOf(widget.initialCenterPitch);
    if (idx < 0) return;
    var whiteCount = 0;
    for (var i = 0; i < idx; i++) {
      if (!_isBlack(kPianoKeys[i])) whiteCount++;
    }
    final targetX = whiteCount * widget.whiteKeyWidth - 120;
    _scrollController.jumpTo(targetX.clamp(0, _scrollController.position.maxScrollExtent));
  }

  @override
  Widget build(BuildContext context) {
    final whiteW = widget.whiteKeyWidth;
    final blackW = whiteW * 0.62;
    const whiteH = 130.0;
    final blackH = whiteH * 0.6;

    final whiteKeys = <Widget>[];
    final blackKeys = <Widget>[];
    var whiteCount = 0;

    for (final pitch in kPianoKeys) {
      final isBlack = _isBlack(pitch);
      final isExpected = widget.expectedPitch == pitch || widget.expectedPitches.contains(pitch);

      if (!isBlack) {
        final x = whiteCount * whiteW;
        whiteKeys.add(Positioned(
          left: x,
          top: 0,
          width: whiteW,
          height: whiteH,
          child: _PianoKey(
            key: ValueKey(pitch),
            pitch: pitch,
            isBlack: false,
            width: whiteW,
            height: whiteH,
            isExpected: isExpected,
            controller: widget.controller,
            onTap: () => widget.onKeyPress?.call(pitch),
          ),
        ));
        whiteCount++;
      } else {
        final centerX = whiteCount * whiteW;
        blackKeys.add(Positioned(
          left: centerX - blackW / 2,
          top: 0,
          width: blackW,
          height: blackH,
          child: _PianoKey(
            key: ValueKey(pitch),
            pitch: pitch,
            isBlack: true,
            width: blackW,
            height: blackH,
            isExpected: isExpected,
            controller: widget.controller,
            onTap: () => widget.onKeyPress?.call(pitch),
          ),
        ));
      }
    }

    final totalWidth = whiteCount * whiteW;

    return RepaintBoundary(
      child: SizedBox(
        height: whiteH,
        child: SingleChildScrollView(
          controller: _scrollController,
          scrollDirection: Axis.horizontal,
          child: SizedBox(
            width: totalWidth,
            height: whiteH,
            child: Stack(children: [...whiteKeys, ...blackKeys]),
          ),
        ),
      ),
    );
  }
}

/// 정답/오답 플래시는 [controller]가 바뀔 때만 이 위젯 스스로 다시 그린다 — 건반을
/// 하나 누를 때마다 88개 건반 전체가 다시 빌드되던 것을 막기 위해 부모(PianoWidget)가
/// controller를 직접 듣지 않고 각 키가 자기 몫만 AnimatedBuilder로 구독한다.
class _PianoKey extends StatelessWidget {
  final String pitch;
  final bool isBlack;
  final double width;
  final double height;
  final bool isExpected;
  final PianoController? controller;
  final VoidCallback onTap;

  const _PianoKey({
    super.key,
    required this.pitch,
    required this.isBlack,
    required this.width,
    required this.height,
    required this.isExpected,
    required this.controller,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final ctrl = controller;
    if (ctrl == null) {
      return _KeyVisual(
        pitch: pitch,
        isBlack: isBlack,
        isExpected: isExpected,
        isFlash: false,
        flashOk: true,
        onTap: onTap,
      );
    }
    return AnimatedBuilder(
      animation: ctrl,
      builder: (context, _) {
        final isFlash = ctrl.flashPitch == pitch;
        return _KeyVisual(
          pitch: pitch,
          isBlack: isBlack,
          isExpected: isExpected,
          isFlash: isFlash,
          flashOk: ctrl.flashOk,
          onTap: onTap,
        );
      },
    );
  }
}

class _KeyVisual extends StatelessWidget {
  final String pitch;
  final bool isBlack;
  final bool isExpected;
  final bool isFlash;
  final bool flashOk;
  final VoidCallback onTap;

  const _KeyVisual({
    required this.pitch,
    required this.isBlack,
    required this.isExpected,
    required this.isFlash,
    required this.flashOk,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    Color fill = isBlack ? const Color(0xFF1C1C1E) : Colors.white;
    Color border = isBlack ? Colors.black : const Color(0xFFBFC7D1);
    double borderWidth = 1;

    if (isFlash) {
      fill = flashOk ? const Color(0xFF3DBE64) : const Color(0xFFE5484D);
      border = fill;
      borderWidth = 2;
    } else if (isExpected) {
      border = const Color(0xFF0076CE);
      borderWidth = 3;
    }

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 120),
        margin: const EdgeInsets.symmetric(horizontal: 1),
        decoration: BoxDecoration(
          color: fill,
          border: Border.all(color: border, width: borderWidth),
          borderRadius: const BorderRadius.vertical(bottom: Radius.circular(4)),
          boxShadow: isBlack
              ? const [BoxShadow(color: Colors.black38, blurRadius: 2, offset: Offset(0, 1))]
              : null,
        ),
        alignment: Alignment.bottomCenter,
        padding: const EdgeInsets.only(bottom: 8),
        child: Text(
          formatNoteName(pitch),
          style: TextStyle(
            color: isFlash
                ? Colors.white
                : isBlack
                    ? Colors.white
                    : Colors.black87,
            fontWeight: FontWeight.w700,
            fontSize: isBlack ? 10 : 12,
          ),
        ),
      ),
    );
  }
}
