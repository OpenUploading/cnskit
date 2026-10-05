from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PixelStreamWorld:
    signalling_url: str
    ue_project: str
    bridge_host: str = "127.0.0.1"
    fly_bridge_port: int = 8899
    mouse_bridge_port: int = 8898

    def launch_args(self) -> tuple[str, ...]:
        return (
            f"-PixelStreaming2SignallingURL={self.signalling_url}",
            "-RenderOffscreen",
            "-AudioMixer",
        )
