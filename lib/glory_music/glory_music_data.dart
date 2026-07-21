import 'package:flutter/material.dart';

class GloryTrack {
  final String title;
  final Duration duration;

  const GloryTrack(this.title, this.duration);
}

class GloryAlbum {
  final String title;
  final String artist;
  final String year;
  final List<Color> gradient;
  final List<GloryTrack> tracks;
  final int currentTrackIndex;

  const GloryAlbum({
    required this.title,
    required this.artist,
    required this.year,
    required this.gradient,
    required this.tracks,
    required this.currentTrackIndex,
  });
}

final galleryAlbums = <GloryAlbum>[
  GloryAlbum(
    title: "In Jesus' Name: A Legacy of Worship & Faith",
    artist: 'Darlene Zschech',
    year: '2015',
    gradient: const [Color(0xFF2B2130), Color(0xFF120D14)],
    currentTrackIndex: 4,
    tracks: const [
      GloryTrack('Hallelujah', Duration(minutes: 4, seconds: 12)),
      GloryTrack('You are Love', Duration(minutes: 3, seconds: 47)),
      GloryTrack("In Jesus' Name (Live)", Duration(minutes: 5, seconds: 2)),
      GloryTrack('Blessed (Live)', Duration(minutes: 4, seconds: 30)),
      GloryTrack('Jesus Lover of My Soul (Live)', Duration(minutes: 3, seconds: 29)),
      GloryTrack("The Potter's Hand (Live)", Duration(minutes: 6, seconds: 8)),
    ],
  ),
  GloryAlbum(
    title: 'Knocking Down My Idols',
    artist: 'Bryann Trejo',
    year: '2019',
    gradient: const [Color(0xFF4A3324), Color(0xFF1A1310)],
    currentTrackIndex: 4,
    tracks: const [
      GloryTrack('His Name (feat. Monica Hill Trejo)', Duration(minutes: 3, seconds: 55)),
      GloryTrack('Healed (feat. Monica Hill Trejo)', Duration(minutes: 4, seconds: 4)),
      GloryTrack('Spirit Filled', Duration(minutes: 3, seconds: 20)),
      GloryTrack('Hold On (feat. Antwoine Hill, 5ive)', Duration(minutes: 4, seconds: 41)),
      GloryTrack('Watered Roots (feat. Young Adults)', Duration(minutes: 3, seconds: 38)),
      GloryTrack('In His Presence (feat. Arize & Skrip)', Duration(minutes: 5, seconds: 15)),
    ],
  ),
];
