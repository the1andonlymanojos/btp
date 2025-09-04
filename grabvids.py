import yt_dlp
from pathlib import Path

video_urls = [
    "https://www.youtube.com/watch?v=AO45Z-PELWY",
    "https://www.youtube.com/watch?v=pwCNV51rvMU",
    "https://www.youtube.com/watch?v=XsZHCDPU86E",
    "https://www.youtube.com/watch?v=qmuuzYkbKX0",
    "https://www.youtube.com/watch?v=RoFxn1tnyG4",
    "https://www.youtube.com/watch?v=FCU_dDgX-Dc",
    "https://www.youtube.com/watch?v=YQfA-qg2DVw",
]

output_dir = Path("vids")
output_dir.mkdir(exist_ok=True)

ydl_opts = {
    "outtmpl": str(output_dir / "%(title).100s.%(ext)s"),
    "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/mp4",
    "merge_output_format": "mp4",
    "quiet": False,
    "noplaylist": True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for url in video_urls:
        print(f"Downloading {url}")
        try:
            ydl.download([url])
        except Exception as e:
            print(f"Failed to download {url}: {e}")
