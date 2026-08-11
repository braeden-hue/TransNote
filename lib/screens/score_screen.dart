import 'package:flutter/material.dart';
import '../data/samples.dart';
import '../services/audio_service.dart';
import '../widgets/notation_widget.dart';
import '../widgets/piano_widget.dart';

class ScoreScreen extends StatefulWidget {
  const ScoreScreen({super.key});

  @override
  State<ScoreScreen> createState() => _ScoreScreenState();
}

class _ScoreScreenState extends State<ScoreScreen> {
  Sample? _selected;
  final _pianoController = PianoController();

  @override
  void dispose() {
    _pianoController.dispose();
    super.dispose();
  }

  void _playNote(String pitch) {
    audioService.playNote(pitch);
    _pianoController.flashCorrect(pitch);
  }

  @override
  Widget build(BuildContext context) {
    if (_selected == null) {
      return _SampleGrid(onSelect: (s) => setState(() => _selected = s));
    }
    final sample = _selected!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
          child: Row(
            children: [
              IconButton(
                icon: const Icon(Icons.arrow_back),
                onPressed: () => setState(() => _selected = null),
              ),
              Text('${sample.emoji} ${sample.title}',
                  style: Theme.of(context).textTheme.titleLarge),
            ],
          ),
        ),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: sample.staves != null
                ? GrandStaffWidget(
                    staves: sample.staves!,
                    onNoteTap: (clef, i, note) {
                      if (note.isRest) return;
                      _playNote(note.pitch);
                    },
                  )
                : NotationWidget(
                    notes: sample.notes,
                    onNoteTap: (i, note) {
                      if (note.isRest) return;
                      _playNote(note.pitch);
                    },
                  ),
          ),
        ),
        PianoDock(
          child: PianoWidget(
            controller: _pianoController,
            onKeyPress: _playNote,
          ),
        ),
      ],
    );
  }
}

class _SampleGrid extends StatelessWidget {
  final void Function(Sample) onSelect;
  const _SampleGrid({required this.onSelect});

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      padding: const EdgeInsets.all(16),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        mainAxisSpacing: 12,
        crossAxisSpacing: 12,
        childAspectRatio: 1.1,
      ),
      itemCount: samples.length,
      itemBuilder: (context, i) {
        final s = samples[i];
        return Card(
          clipBehavior: Clip.antiAlias,
          child: InkWell(
            onTap: () => onSelect(s),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(s.emoji, style: const TextStyle(fontSize: 40)),
                const SizedBox(height: 8),
                Text(s.title, style: const TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text('${s.timeSignature[0]}/${s.timeSignature[1]} · ♩=${s.tempo}',
                    style: TextStyle(color: Colors.grey[600], fontSize: 12)),
              ],
            ),
          ),
        );
      },
    );
  }
}
