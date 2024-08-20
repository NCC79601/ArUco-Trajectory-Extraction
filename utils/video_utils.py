import ffmpeg
from datetime import datetime, timedelta

def get_start_time(video_path):
    """Extract video start time from metadata."""
    probe = ffmpeg.probe(video_path)
    start_time_str = probe["streams"][0]["tags"].get("creation_time")
    return datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")


def get_duration(video_path):
    """Extract video duration in seconds."""
    probe = ffmpeg.probe(video_path)
    return float(probe["format"]["duration"])


def crop_video(video_path, start_offset, duration, output_path):
    """
    Crop video from start_offset for the specified duration.
    
    - without audio track.
    """
    ffmpeg.input(video_path, ss=start_offset, t=duration)\
          .output(output_path,an=None,c='copy').run()


def align_videos(video1, video2, output1, output2):
    """Align two videos based on their start and end times."""
    start1, start2 = get_start_time(video1), get_start_time(video2)
    dur1, dur2 = get_duration(video1), get_duration(video2)

    # Determine aligned start and end times
    aligned_start = max(start1, start2)
    aligned_end = min(start1 + timedelta(seconds=dur1), start2 + timedelta(seconds=dur2))
    aligned_duration = (aligned_end - aligned_start).total_seconds()

    # Calculate offsets for cropping
    offset1 = (aligned_start - start1).total_seconds()
    offset2 = (aligned_start - start2).total_seconds()

    # Crop both videos
    crop_video(video1, offset1, aligned_duration, output1)
    crop_video(video2, offset2, aligned_duration, output2)


if __name__ == "__main__":

    # Example usage
    video1 = "GX010027.mp4"
    video2 = "GX010385.mp4"
    output1 = f"aligned_{video1}"
    output2 = f"aligned_{video2}"

    align_videos(video1, video2, output1, output2)