import base64
from io import BytesIO

import audioread
import av
import librosa
import numpy as np


SAMPLE_RATE=16000
def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def _load_audio_with_av(path, sr=SAMPLE_RATE, offset=0.0, duration=None):
    container = av.open(str(path))
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        raise ValueError(f"No audio stream found in {path}")
    frames = list(container.decode(stream))
    container.close()
    if not frames:
        return np.zeros(0, dtype=np.float32)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
    chunks = []
    for frame in frames:
        for resampled in resampler.resample(frame):
            chunk = resampled.to_ndarray()
            if chunk.size > 0:
                chunks.append(chunk)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32)
    start = int(sr * offset)
    end = None if duration is None else start + int(sr * duration)
    return audio[start:end]


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                if isinstance(data, str):
                    audios.append(
                        _load_audio_with_av(
                            data,
                            sr=SAMPLE_RATE,
                            offset=audio_start,
                            duration=(audio_end - audio_start) if audio_end is not None else None,
                        )
                    )
                else:
                    audios.append(
                        librosa.load(
                            data,
                            sr=SAMPLE_RATE,
                            offset=audio_start,
                            duration=(audio_end - audio_start) if audio_end is not None else None,
                        )[0]
                    )
    if len(audios) == 0:
        audios = None
    return audios
