import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../theme/glory_theme.dart';
import '../theme/glory_page_route.dart';
import 'collection_screen.dart';
import 'score_screen.dart';
import 'tutorial_screen.dart';

/// 하단 탭 "홈" -- Mume_Modified_UI_Kit_PNG 05번(home_suggested) 참고. 원본 킷은 스트리밍
/// 음악 앱(최근 재생/아티스트/최다 재생)이라 우리 데이터 모델과 맞지 않는 섹션은 걷어내고,
/// 실제로 있는 데이터(튜토리얼 진입, 샘플 악보, 스캔한 악보)로 같은 "섹션 + 가로 카드"
/// 레이아웃 패턴만 가져왔다.
class HomeTab extends StatelessWidget {
  final List<ScannedEntry> entries;
  final ValueChanged<ScannedEntry> onOpenEntry;
  const HomeTab({super.key, required this.entries, required this.onOpenEntry});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
        children: [
          Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(shape: BoxShape.circle, gradient: gloryGradient),
                child: const Icon(Icons.music_note_rounded, color: Colors.white, size: 18),
              ),
              const SizedBox(width: 10),
              Text('악보의 대중화', style: TextStyle(color: gloryInk, fontSize: 20, fontWeight: FontWeight.w800)),
            ],
          ),
          const SizedBox(height: 24),
          Material(
            color: Colors.transparent,
            child: InkWell(
              borderRadius: BorderRadius.circular(20),
              onTap: () => Navigator.of(context).push(gloryPageRoute(builder: (_) => const TutorialScreen())),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(gradient: gloryGradient, borderRadius: BorderRadius.circular(20)),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('튜토리얼로 시작하기',
                              style: TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w800)),
                          const SizedBox(height: 6),
                          Text('커스텀 악보 읽는 법을 규칙별로 배워요',
                              style: TextStyle(color: Colors.white.withValues(alpha: .85), fontSize: 12.5)),
                        ],
                      ),
                    ),
                    const Icon(Icons.school_rounded, color: Colors.white, size: 36),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(height: 28),
          Row(
            children: [
              Text('예시 악보', style: TextStyle(color: gloryInk, fontSize: 15, fontWeight: FontWeight.w800)),
              const Spacer(),
              GestureDetector(
                onTap: () => Navigator.of(context).push(gloryPageRoute(builder: (_) => const ScoreScreen())),
                child: Text('전체보기', style: TextStyle(color: gloryAccent, fontSize: 12.5, fontWeight: FontWeight.w700)),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 118,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: samples.length,
              separatorBuilder: (_, _) => const SizedBox(width: 12),
              itemBuilder: (context, i) {
                final s = samples[i];
                return _SampleCard(
                  sample: s,
                  onTap: () => Navigator.of(context).push(gloryPageRoute(builder: (_) => ScoreScreen(initial: s))),
                );
              },
            ),
          ),
          const SizedBox(height: 28),
          Text('최근 스캔', style: TextStyle(color: gloryInk, fontSize: 15, fontWeight: FontWeight.w800)),
          const SizedBox(height: 12),
          if (entries.isEmpty)
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(color: glorySurface, borderRadius: BorderRadius.circular(16)),
              child: Text('촬영 탭에서 악보를 스캔하면 여기에 모여요.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: gloryMuted, fontSize: 12.5)),
            )
          else
            ...entries.take(3).map((e) => EntryTile(entry: e, onTap: () => onOpenEntry(e))),
        ],
      ),
    );
  }
}

class _SampleCard extends StatelessWidget {
  final SampleScore sample;
  final VoidCallback onTap;
  const _SampleCard({required this.sample, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: glorySurface,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Container(
          width: 130,
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(sample.emoji, style: const TextStyle(fontSize: 24)),
              const SizedBox(height: 8),
              Text(sample.title,
                  style: TextStyle(color: gloryInk, fontSize: 13, fontWeight: FontWeight.w700),
                  maxLines: 1, overflow: TextOverflow.ellipsis),
              const SizedBox(height: 2),
              Text('${sample.notes.length}음 · ${sample.tempo}BPM', style: TextStyle(color: gloryMuted, fontSize: 11)),
            ],
          ),
        ),
      ),
    );
  }
}
