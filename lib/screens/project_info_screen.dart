import 'package:flutter/material.dart';

/// webpage/landing.html의 "팀 소개" 섹션 카피를 그대로 옮긴 프로젝트 소개 화면.
class ProjectInfoScreen extends StatelessWidget {
  const ProjectInfoScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final primary = Theme.of(context).colorScheme.primary;
    return Scaffold(
      appBar: AppBar(title: const Text('프로젝트 소개')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '음악과 AI로\n새로운 문을 열다',
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800),
              ),
              const SizedBox(height: 16),
              Text(
                '저희 팀은 음악 교육의 접근성을 높이기 위해 Optical Music Recognition 기술과 '
                '직관적인 UI/UX를 결합했습니다.',
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(height: 1.5),
              ),
              const SizedBox(height: 12),
              Text(
                '오선보를 읽지 못해 악기 연주를 포기한 모든 분들을 위해, 악보를 누구나 이해할 수 '
                '있는 시각 언어로 재해석했습니다.',
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(height: 1.5),
              ),
              const SizedBox(height: 12),
              Text(
                'Flutter 기반 크로스플랫폼 앱으로 개발 중이며, 웹 데모를 통해 핵심 기능을 먼저 '
                '경험해보실 수 있습니다.',
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(height: 1.5),
              ),
              const SizedBox(height: 32),
              Text(
                '함께하는 사람들',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 16),
              GridView.count(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                crossAxisCount: 2,
                mainAxisSpacing: 12,
                crossAxisSpacing: 12,
                childAspectRatio: 1.3,
                children: const [
                  _TeamCard(emoji: '👨‍💻', name: '개발 / 기획', role: 'OMR 파이프라인\nFlutter 앱'),
                  _TeamCard(emoji: '🎨', name: 'UI / UX', role: '표기법 설계\n인터랙션 디자인'),
                  _TeamCard(
                    emoji: '➕',
                    name: '모집 중',
                    role: 'iOS 개발자\n음악 교육 전문가',
                    placeholder: true,
                  ),
                  _TeamCard(
                    emoji: '➕',
                    name: '모집 중',
                    role: '마케팅 / 기획\n음악 콘텐츠',
                    placeholder: true,
                  ),
                ],
              ),
              const SizedBox(height: 32),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: primary.withValues(alpha: 0.08),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('♩', style: TextStyle(fontSize: 22, color: primary)),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'AI 기반 악보 인식(OMR)으로 오선보를 직관적인 커스텀 악보로 변환. '
                        '누구나 10분 안에 악보를 읽고 연주할 수 있게 합니다.',
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _TeamCard extends StatelessWidget {
  final String emoji;
  final String name;
  final String role;
  final bool placeholder;

  const _TeamCard({
    required this.emoji,
    required this.name,
    required this.role,
    this.placeholder = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: placeholder
            ? Theme.of(context).colorScheme.surfaceContainerLow
            : Theme.of(context).colorScheme.primaryContainer.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(14),
        border: placeholder
            ? Border.all(
                color: Theme.of(context).colorScheme.outlineVariant,
                style: BorderStyle.solid,
              )
            : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(emoji, style: const TextStyle(fontSize: 26)),
          const SizedBox(height: 8),
          Text(name, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 14)),
          const SizedBox(height: 4),
          Text(
            role,
            style: TextStyle(fontSize: 12, color: Theme.of(context).colorScheme.onSurfaceVariant),
          ),
        ],
      ),
    );
  }
}
