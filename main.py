import argparse
import json
import numpy as np
import scipy
import shutil
import subprocess

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class VideoFile:
    path: Path
    duration: float
    framerate: float
    width: int
    height: int
    start_time: datetime
    end_time: datetime
    trim_start: float = 0.0
    trim_end: float = 0.0

    def overlaps(self, other: 'VideoFile') -> tuple['VideoFile', 'VideoFile'] | None:
        if self.start_time > other.end_time or self.end_time < other.start_time:
            return None

        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        duration = (overlap_end - overlap_start).total_seconds()

        return (self, other)


def path_to_video_file(video_path: Path) -> VideoFile:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v",
        "-show_entries", "format=duration",
        "-show_entries", "stream=r_frame_rate,width,height",
        "-of", "json",
        str(video_path),
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
                          video_files_1: list[VideoFile]) -> list[tuple[VideoFile, VideoFile]]:
    overlaps = []
    for file_0 in video_files_0:
        for file_1 in video_files_1:
            overlap = file_0.overlaps(file_1)
            if overlap is not None:
                overlaps.append(overlap)

    return overlaps


def detect_number_audio_streams(video_path: Path) -> int:
    cmd = ['ffprobe',
           '-v', 'error',
           '-select_streams', 'a',
           '-show_entries', 'stream=index',
           '-of', 'csv=p=0',
           str(video_path)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

    return int(result.stdout.strip()[-1])


def extract_audio(video_path: Path, audio_path: Path):
    num_audio_streams = detect_number_audio_streams(video_path)

    cmd = ['ffmpeg', '-i', str(video_path),
           '-filter_complex', f'[0:a]amerge=inputs={num_audio_streams}[aout]',
           '-map', '[aout]',
           '-ar', '48000',
           '-ac', '1',
           '-loglevel', 'quiet',
           '-hide_banner',
           '-stats',
           '-y', str(audio_path)]
    print(f'Found {num_audio_streams} audio stream(s). Extracting to file via command: \n{" ".join(cmd)}')
    subprocess.run(cmd, text=True, check=True)


def load_audio(audio_path: Path) -> tuple:
    print(f'Loading audio from: {audio_path}')
    sr, audio = scipy.io.wavfile.read(audio_path)
    return sr, audio


def find_audio_offset(sample_rate, audio0, audio1) -> float:
    """Calculates the sync offset between two audio files.
    Return value > 0: Trim this amount from the start of audio0.
    Return value < 0: Trim this amount from the start of audio1.
    """
    audio0 = audio0.astype(np.float32)
    audio1 = audio1.astype(np.float32)

    correlation = scipy.signal.correlate(audio0, audio1)
    best_offset = np.argmax(correlation) - (len(audio1) - 1)
    return best_offset / sample_rate


def sync_videos_by_audio(vid0: VideoFile, vid1: VideoFile):
    """Modifies overlap in place when syncing the video clips using audio correlation."""
    TMP_FOLDER = Path('./tmp')
    TMP_FOLDER.mkdir(exist_ok=True, parents=False)

    audio_0_path = TMP_FOLDER / 'audio_0.wav'
    audio_1_path = TMP_FOLDER / 'audio_1.wav'
    extract_audio(vid0.path, audio_0_path)
    extract_audio(vid1.path, audio_1_path)

    sr0, audio0 = load_audio(audio_0_path)
    sr1, audio1 = load_audio(audio_1_path)

    if sr0 != sr1:
        raise ValueError('Audio sample rates do not match')

    offset = find_audio_offset(sr0, audio0, audio1)
    print(f'Offset between files: {offset:.3f}s')

    # Adjust overlap from audio offset.
    if offset > 0:
        vid0.trim_start = offset
        duration_after_trim = vid0.duration - offset
        duration_out = min(duration_after_trim, vid1.duration)
    else:
        vid1.trim_start = offset
        duration_after_trim = vid1.duration - offset
        duration_out = min(duration_after_trim, vid0.duration)

    vid0.trim_end = min(vid0.duration, vid0.trim_start + duration_out)
    vid1.trim_end = min(vid1.duration, vid1.trim_start + duration_out)

    shutil.rmtree(TMP_FOLDER, ignore_errors=True)


def reencode_overlapping(vid0: VideoFile, vid1: VideoFile, out_folder: Path):
    """Re-encode videos into a split-screen view."""
    out_folder.mkdir(exist_ok=True, parents=True)

    sync_videos_by_audio(vid0, vid1)

    timestamp = (vid0.start_time + timedelta(seconds=vid0.trim_start)).strftime('%Y-%m-%d__%H-%M-%S')
    out_name = f'SplitView__{timestamp}.mkv'
    cmd = ['ffmpeg',
           '-i', str(vid0.path), '-i', str(vid1.path),
           '-filter_complex',
           f'[0:v]scale=1280:720,crop=640:720:320:0,trim=start={vid0.trim_start}:end={vid0.trim_end},'
           'setpts=PTS-STARTPTS[v0];'
           f'[1:v]scale=1280:720,crop=640:720:320:0,trim=start={vid1.trim_start}:end={vid1.trim_end},'
           'setpts=PTS-STARTPTS[v1];'
           '[v0][v1]hstack=inputs=2[vout]',
           '-map', '[vout]',
           '-c:v', 'libx264',
           '-strict', 'experimental',
           '-vsync', 'vfr',
           '-loglevel', 'quiet',
           '-hide_banner',
           '-stats',
           '-y', str(out_folder / out_name)]

    subprocess.run(cmd, text=True, check=True)


def main(args):
    video_files_0 = get_video_files_from_folder(args.folders[0])
    video_files_1 = get_video_files_from_folder(args.folders[1])

    overlaps = get_overlapping_pairs(video_files_0, video_files_1)

    [reencode_overlapping(*o, args.out) for o in overlaps[-1:]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Replay Splitter', description='Creates a splitscreen view of two videos')
    parser.add_argument('folders', type=Path, nargs=2)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check', action='store_true')

    main(parser.parse_args())
