import numpy as np
import logging
from typing import Optional, Tuple
import io

logger = logging.getLogger(__name__)

class VADHandler:
    """Voice Activity Detection handler using energy-based detection"""
    
    def __init__(self,
                 energy_threshold: float = 0.02,  # Balanced threshold for reliable detection
                 silence_duration: float = 2.0,   # Reasonable wait time for complete speech
                 min_speech_duration: float = 0.8, # Minimum duration to avoid very short sounds
                 sample_rate: int = 16000):
        """
        Initialize VAD handler
        
        Args:
            energy_threshold: Minimum energy level to consider as speech
            silence_duration: Duration of silence (in seconds) to consider speech ended
            min_speech_duration: Minimum duration (in seconds) for valid speech
            sample_rate: Audio sample rate
        """
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.sample_rate = sample_rate

        # State tracking
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = []

        # Enhanced noise filtering
        self.noise_floor = 0.0
        self.noise_samples = []
        self.noise_adaptation_rate = 0.1
        self.speech_quality_threshold = 0.02  # Lowered threshold to be less strict
        
    def calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate the energy (RMS) of an audio chunk"""
        if len(audio_chunk) == 0:
            return 0.0
        return np.sqrt(np.mean(audio_chunk ** 2))

    def update_noise_floor(self, energy: float):
        """Adaptively update noise floor estimation"""
        if not self.is_speaking:
            self.noise_samples.append(energy)
            # Keep only recent samples for noise estimation
            if len(self.noise_samples) > 50:
                self.noise_samples.pop(0)

            # Update noise floor as moving average
            if self.noise_samples:
                new_noise_floor = np.mean(self.noise_samples)
                self.noise_floor = (1 - self.noise_adaptation_rate) * self.noise_floor + \
                                 self.noise_adaptation_rate * new_noise_floor

    def calculate_speech_quality(self, audio_segment: np.ndarray) -> float:
        """Calculate speech quality to filter out noise and partial words"""
        if len(audio_segment) == 0:
            return 0.0

        # Calculate various quality metrics
        energy = np.mean(audio_segment ** 2)

        # Check for consistent energy (not just noise spikes)
        energy_variance = np.var(audio_segment ** 2)
        energy_consistency = energy / (energy_variance + 1e-8)

        # Simple speech quality based on energy patterns
        # Real speech has more consistent energy than random noise
        if energy > self.noise_floor * 3:  # Must be significantly above noise floor
            quality_score = min(energy * energy_consistency, 1.0)
        else:
            quality_score = 0.0

        return quality_score
    
    def process_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process an audio chunk and determine if speech is detected
        
        Args:
            audio_chunk: Audio data as numpy array
            timestamp: Current timestamp in seconds
            
        Returns:
            Tuple of (speech_detected, complete_speech_segment)
            speech_detected: True if currently detecting speech
            complete_speech_segment: Audio data if a complete speech segment is ready
        """
        energy = self.calculate_energy(audio_chunk)

        # Update noise floor estimation
        self.update_noise_floor(energy)

        # Enhanced speech detection with noise floor consideration
        dynamic_threshold = max(self.energy_threshold, self.noise_floor * 2.5)  # Less aggressive multiplier
        has_speech = energy > dynamic_threshold

        # Debug logging for voice detection issues
        if energy > self.energy_threshold * 0.5:  # Log when there's some audio activity
            logger.debug(f"Audio activity - Energy: {energy:.4f}, Threshold: {dynamic_threshold:.4f}, "
                        f"Noise floor: {self.noise_floor:.4f}, Has speech: {has_speech}")
        
        if has_speech:
            if not self.is_speaking:
                # Start of speech detected
                self.is_speaking = True
                self.speech_start_time = timestamp
                self.audio_buffer = [audio_chunk]
                logger.debug(f"Speech started at {timestamp:.2f}s, energy: {energy:.4f}")
            else:
                # Continue speech
                self.audio_buffer.append(audio_chunk)
            
            self.last_speech_time = timestamp
            
        else:
            if self.is_speaking:
                # Check if silence duration exceeded
                silence_duration = timestamp - self.last_speech_time
                
                if silence_duration >= self.silence_duration:
                    # End of speech detected
                    speech_duration = timestamp - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration:
                        # Check speech quality before accepting
                        complete_audio = np.concatenate(self.audio_buffer)
                        speech_quality = self.calculate_speech_quality(complete_audio)

                        if speech_quality >= self.speech_quality_threshold:
                            # Valid speech segment with good quality
                            logger.info(f"Speech ended. Duration: {speech_duration:.2f}s, "
                                      f"Samples: {len(complete_audio)}, Quality: {speech_quality:.3f}")

                            # Reset state
                            self.is_speaking = False
                            self.speech_start_time = None
                            self.last_speech_time = None
                            self.audio_buffer = []

                            return False, complete_audio
                        else:
                            # Poor quality speech (likely noise or partial words)
                            logger.debug(f"Speech rejected due to poor quality: {speech_quality:.3f} < {self.speech_quality_threshold}")
                            # Reset state and continue
                            self.is_speaking = False
                            self.speech_start_time = None
                            self.last_speech_time = None
                            self.audio_buffer = []
                    else:
                        # Too short, discard
                        logger.debug(f"Speech too short ({speech_duration:.2f}s), discarding")
                        self.is_speaking = False
                        self.speech_start_time = None
                        self.last_speech_time = None
                        self.audio_buffer = []
                else:
                    # Still in potential speech, add silence to buffer
                    self.audio_buffer.append(audio_chunk)
        
        return self.is_speaking, None
    
    def reset(self):
        """Reset the VAD state"""
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = []
        
    def set_sensitivity(self, sensitivity: float):
        """
        Adjust sensitivity (0.0 to 1.0)
        Higher values = more sensitive (detects quieter sounds)
        Lower values = less sensitive (only loud sounds)
        """
        # Map sensitivity to energy threshold - balanced for reliable detection
        # sensitivity 0.0 -> threshold 0.08 (less sensitive, need clear speech)
        # sensitivity 1.0 -> threshold 0.015 (more sensitive)
        self.energy_threshold = 0.08 * (1.0 - sensitivity) + 0.015 * sensitivity

        # Also adjust speech quality threshold based on sensitivity
        # Lower sensitivity = higher quality requirement
        self.speech_quality_threshold = 0.05 * (1.0 - sensitivity) + 0.01 * sensitivity

        logger.info(f"VAD sensitivity set to {sensitivity:.2f}, "
                   f"energy threshold: {self.energy_threshold:.4f}, "
                   f"quality threshold: {self.speech_quality_threshold:.4f}")


class AudioProcessor:
    """Helper class for audio processing and chunking"""
    
    def __init__(self, chunk_duration: float = 0.1, sample_rate: int = 16000):
        """
        Initialize audio processor
        
        Args:
            chunk_duration: Duration of each audio chunk in seconds
            sample_rate: Audio sample rate
        """
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.buffer = np.array([])

        # Simple noise gate parameters - less aggressive to avoid filtering speech
        self.noise_gate_threshold = 0.005  # Lower threshold to preserve quiet speech
        self.noise_gate_ratio = 0.3        # Less aggressive reduction (0.3 = reduce to 30%)

    def apply_noise_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simple noise gate to reduce background noise"""
        # Calculate RMS energy for each sample window
        window_size = 160  # 10ms at 16kHz
        gated_audio = audio_data.copy()

        for i in range(0, len(audio_data) - window_size, window_size):
            window = audio_data[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))

            if rms < self.noise_gate_threshold:
                # Apply noise reduction
                gated_audio[i:i + window_size] *= self.noise_gate_ratio

        return gated_audio

    def add_audio(self, audio_data: np.ndarray) -> list:
        """
        Add audio data and return complete chunks
        
        Args:
            audio_data: New audio data to add
            
        Returns:
            List of complete audio chunks
        """
        # Apply noise gate to incoming audio
        filtered_audio = self.apply_noise_gate(audio_data)

        # Add filtered data to buffer
        self.buffer = np.concatenate([self.buffer, filtered_audio])

        # Extract complete chunks
        chunks = []
        while len(self.buffer) >= self.chunk_size:
            chunk = self.buffer[:self.chunk_size]
            chunks.append(chunk)
            self.buffer = self.buffer[self.chunk_size:]

        return chunks
    
    def get_remaining(self) -> Optional[np.ndarray]:
        """Get any remaining audio data in buffer"""
        if len(self.buffer) > 0:
            remaining = self.buffer.copy()
            self.buffer = np.array([])
            return remaining
        return None
