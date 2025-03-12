import argparse
import json
import librosa
import numpy as np
import subprocess

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Overlap:
    video_files: tuple
    duration: float
    start_time: datetime
    end_time: datetime


@dataclass
class VideoFile:
    path: Path
    duration: float
    framerate: float
    width: int
    height: int
    start_time: datetime
    end_time: datetime

    def overlaps(self, other: 'VideoFile') -> Overlap | None:
        if self.start_time > other.end_time or self.end_time < other.start_time:
            return None

        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        duration = (overlap_end - overlap_start).total_seconds()

        return Overlap(video_files=(self, other),
                       duration=duration,
                       start_time=overlap_start,
                       end_time=overlap_end)


def path_to_video_file(video_path: Path) -> VideoFile:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v",
        "-show_entries", "format=duration",
        "-show_entries", "stream=r_frame_rate,width,height",
        "-of", "json",
        video_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ChildProcessError('Failed ffprobe command')

    ffprobe_data = json.loads(result.stdout)

    stream = ffprobe_data['streams'][0]
    framerate_str = stream['r_frame_rate']
    num, den = framerate_str.split('/', maxsplit=1)
    framerate = float(num) / float(den)

    duration = float(ffprobe_data['format']['duration'])
    timestamp_str = video_path.stem.split('__', maxsplit=1)[1]
    end_time = datetime.strptime(timestamp_str, "%Y-%m-%d__%H-%M-%S")
    start_time = end_time - timedelta(seconds=duration)

    return VideoFile(path=video_path,
                     duration=duration,
                     framerate=framerate,
                     width=int(stream['width']),
                     height=int(stream['height']),
                     start_time=start_time,
                     end_time=end_time)


def get_video_files_from_folder(folder: Path) -> list[VideoFile]:
    video_files = []
    for entry in folder.iterdir():
        if entry.is_file() and entry.suffix in ['.mp4', '.mkv']:
            video_files.append(path_to_video_file(entry))

    return video_files


def get_overlapping_pairs(video_files_0: list[VideoFile],
                          video_files_1: list[VideoFile]) -> list[Overlap]:
    overlaps = []
    for file_0 in video_files_0:
        for file_1 in video_files_1:
            overlap = file_0.overlaps(file_1)
            if overlap is not None:
                overlaps.append(overlap)

    return overlaps


def extract_audio(video_path: Path, audio_path: Path):
    cmd = ['ffmpeg', '-i', f'{video_path}', "-q:a", "0", "-map", "a", "-y", audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_audio(audio_path: Path) -> tuple:
    y, sr = librosa.load(audio_path)
    return y, sr


def find_audio_offset(sample_rate, audio0, audio1) -> float:
    correlation = np.correlate(audio0, audio1, mode='full')
    best_offset = np.argmax(correlation) - (len(audio1) - 1)
    return best_offset / sample_rate


def sync_overlap_by_audio(overlap: Overlap):
    """Modifies overlap in place when syncing the video clips using audio correlation."""
    TMP_FOLDER = Path('./tmp')
    TMP_FOLDER.mkdir(exist_ok=True, parents=False)

    audio_paths = [TMP_FOLDER / f'audio_{i}.wav' for i, _ in enumerate(overlap.video_files)]
    extract_audio(overlap.video_files[0].path, audio_paths[0])
    extract_audio(overlap.video_files[1].path, audio_paths[1])

    audio0, sr0 = load_audio(audio_paths[0])
    audio1, sr1 = load_audio(audio_paths[1])

    if sr0 != sr1:
        raise ValueError('Audio sample rates do not match')

    offset = find_audio_offset(sr0, audio0, audio1)
    print(f'Offset between files: {overlap.video_files} -> \n{offset:.3f}s')

    # Adjust overlap from audio offset.

    TMP_FOLDER.unlink()


def reencode_overlap(overlap: Overlap):
    sync_overlap_by_audio(overlap)


def main(args):
    video_files_0 = get_video_files_from_folder(args.folders[0])
    video_files_1 = get_video_files_from_folder(args.folders[1])

    overlaps = get_overlapping_pairs(video_files_0, video_files_1)

    [reencode_overlap(o) for o in overlaps]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Replay Splitter', description='Creates a splitscreen view of two videos')
    parser.add_argument('folders', type=Path, nargs=2)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--out', type=Path)

    main(parser.parse_args())
