"""
Animation Recorder Module.
Records simulation animations to video files.
"""
import os
import tempfile
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RecordingSettings:
    """Animation recording settings."""
    fps: int = 30
    width: int = 800
    height: int = 600
    format: str = 'gif'  # 'gif' or 'mp4'
    quality: int = 85
    loop: bool = True


class AnimationRecorder:
    """
    Records canvas animations to video files.
    
    Uses PIL for frame capture and ffmpeg/imageio for video encoding.
    """
    
    def __init__(self, settings: RecordingSettings = None):
        self.settings = settings or RecordingSettings()
        self.frames: List = []
        self.recording = False
        self.temp_dir = None
        
    def start_recording(self):
        """Start recording frames."""
        self.frames = []
        self.recording = True
        self.temp_dir = tempfile.mkdtemp(prefix='eq_sim_')
        
    def stop_recording(self):
        """Stop recording."""
        self.recording = False
        
    def capture_frame(self, canvas):
        """
        Capture a frame from tkinter canvas.
        
        Args:
            canvas: Tkinter Canvas widget
        """
        if not self.recording:
            return
            
        try:
            from PIL import Image, ImageGrab
            
            # Get canvas position on screen
            x = canvas.winfo_rootx()
            y = canvas.winfo_rooty()
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            
            # Capture screenshot
            img = ImageGrab.grab((x, y, x + w, y + h))
            
            # Resize if needed
            if img.width != self.settings.width or img.height != self.settings.height:
                img = img.resize((self.settings.width, self.settings.height), Image.LANCZOS)
                
            self.frames.append(img)
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            
    def capture_from_array(self, rgb_array):
        """
        Capture frame from numpy array.
        
        Args:
            rgb_array: (H, W, 3) uint8 array
        """
        if not self.recording:
            return
            
        try:
            from PIL import Image
            img = Image.fromarray(rgb_array)
            self.frames.append(img)
        except Exception as e:
            print(f"Array capture error: {e}")
            
    def save_gif(self, output_path: str) -> str:
        """Save recorded frames as GIF."""
        if not self.frames:
            raise ValueError("No frames recorded")
            
        duration = int(1000 / self.settings.fps)
        
        self.frames[0].save(
            output_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration,
            loop=0 if self.settings.loop else 1,
            optimize=True
        )
        
        return output_path
        
    def save_mp4(self, output_path: str) -> str:
        """
        Save recorded frames as MP4.
        
        Requires ffmpeg.
        """
        import subprocess
        import shutil
        
        if not self.frames:
            raise ValueError("No frames recorded")
            
        # Save frames to temp directory
        for i, frame in enumerate(self.frames):
            frame_path = os.path.join(self.temp_dir, f"frame_{i:05d}.png")
            frame.save(frame_path)
            
        # Check for ffmpeg
        if shutil.which('ffmpeg'):
            pattern = os.path.join(self.temp_dir, "frame_%05d.png")
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(self.settings.fps),
                '-i', pattern,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', str(30 - self.settings.quality // 4),
                output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            # Fallback to GIF
            gif_path = output_path.replace('.mp4', '.gif')
            return self.save_gif(gif_path)
            
        return output_path
        
    def save(self, output_path: str) -> str:
        """Save to appropriate format based on extension."""
        if output_path.lower().endswith('.gif'):
            return self.save_gif(output_path)
        else:
            return self.save_mp4(output_path)
            
    def clear(self):
        """Clear recorded frames."""
        self.frames = []
        
        # Clean up temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    @property
    def frame_count(self) -> int:
        return len(self.frames)
        
    @property
    def duration_seconds(self) -> float:
        return len(self.frames) / self.settings.fps if self.settings.fps > 0 else 0


class SimulationPlayer:
    """
    Plays back recorded simulation with controls.
    """
    
    def __init__(
        self,
        time_array,
        data_array,
        render_callback: Callable,
        fps: int = 30
    ):
        """
        Args:
            time_array: Time values
            data_array: Data for each time step
            render_callback: Function(data, time) to render frame
            fps: Playback FPS
        """
        self.time = time_array
        self.data = data_array
        self.render = render_callback
        self.fps = fps
        
        self.current_frame = 0
        self.playing = False
        self.speed = 1.0
        self.loop = True
        
        self._job = None
        self._root = None
        
    def set_root(self, root):
        """Set tkinter root for scheduling."""
        self._root = root
        
    @property
    def total_frames(self) -> int:
        return len(self.time)
        
    @property
    def current_time(self) -> float:
        if self.current_frame < len(self.time):
            return self.time[self.current_frame]
        return 0
        
    @property
    def progress(self) -> float:
        return self.current_frame / max(1, self.total_frames - 1)
        
    def play(self):
        """Start playback."""
        self.playing = True
        self._schedule_next()
        
    def pause(self):
        """Pause playback."""
        self.playing = False
        if self._job and self._root:
            self._root.after_cancel(self._job)
            self._job = None
            
    def stop(self):
        """Stop and reset to start."""
        self.pause()
        self.current_frame = 0
        self._render_current()
        
    def step_forward(self, n: int = 1):
        """Step forward n frames."""
        self.current_frame = min(self.current_frame + n, self.total_frames - 1)
        self._render_current()
        
    def step_backward(self, n: int = 1):
        """Step backward n frames."""
        self.current_frame = max(self.current_frame - n, 0)
        self._render_current()
        
    def seek(self, progress: float):
        """Seek to position (0-1)."""
        self.current_frame = int(progress * (self.total_frames - 1))
        self._render_current()
        
    def set_speed(self, speed: float):
        """Set playback speed multiplier."""
        self.speed = max(0.1, min(10, speed))
        
    def _schedule_next(self):
        """Schedule next frame."""
        if not self.playing or not self._root:
            return
            
        delay = int(1000 / (self.fps * self.speed))
        self._job = self._root.after(delay, self._advance)
        
    def _advance(self):
        """Advance to next frame."""
        if not self.playing:
            return
            
        self.current_frame += 1
        
        if self.current_frame >= self.total_frames:
            if self.loop:
                self.current_frame = 0
            else:
                self.pause()
                return
                
        self._render_current()
        self._schedule_next()
        
    def _render_current(self):
        """Render current frame."""
        if self.current_frame < len(self.data):
            self.render(self.data[self.current_frame], self.current_time)


def create_recording_dialog(parent, recorder: AnimationRecorder):
    """Create recording control dialog."""
    import tkinter as tk
    from tkinter import ttk, filedialog
    
    dialog = tk.Toplevel(parent)
    dialog.title("Animation Recording")
    dialog.geometry("300x200")
    dialog.transient(parent)
    
    # Status
    status_var = tk.StringVar(value="Ready")
    ttk.Label(dialog, textvariable=status_var, font=('Arial', 12)).pack(pady=10)
    
    # Frame count
    frame_var = tk.StringVar(value="Frames: 0")
    ttk.Label(dialog, textvariable=frame_var).pack()
    
    def update_status():
        frame_var.set(f"Frames: {recorder.frame_count}")
        if recorder.recording:
            dialog.after(100, update_status)
    
    # Buttons
    btn_frame = ttk.Frame(dialog)
    btn_frame.pack(pady=20)
    
    def start():
        recorder.start_recording()
        status_var.set("🔴 Recording...")
        update_status()
        
    def stop():
        recorder.stop_recording()
        status_var.set(f"Stopped ({recorder.frame_count} frames)")
        
    def save():
        path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF", "*.gif"), ("MP4", "*.mp4")]
        )
        if path:
            try:
                recorder.save(path)
                status_var.set(f"Saved: {os.path.basename(path)}")
            except Exception as e:
                status_var.set(f"Error: {e}")
    
    ttk.Button(btn_frame, text="▶ Start", command=start).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="⏹ Stop", command=stop).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="💾 Save", command=save).pack(side=tk.LEFT, padx=5)
    
    return dialog
