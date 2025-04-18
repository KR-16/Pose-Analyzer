import pyttsx3
import queue
import threading
from ..config import EXERCISE_CONFIG

class VoiceFeedback:
    def __init__(self):
        self.engine = self._init_engine()
        self.message_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.start()
    
    def _init_engine(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        return engine
    
    def _process_queue(self):
        while self.running or not self.message_queue.empty():
            try:
                msg = self.message_queue.get(timeout=1)
                self.engine.say(msg)
                self.engine.runAndWait()
                self.message_queue.task_done()
            except queue.Empty:
                continue
    
    def speak(self, message, priority=False):
        if priority:
            with self.message_queue.mutex:
                self.message_queue.queue.clear()
        self.message_queue.put(message)
    
    def give_exercise_feedback(self, exercise_type, feedback_key, priority=False):
        """Give exercise-specific feedback"""
        if exercise_type in EXERCISE_CONFIG:
            message = EXERCISE_CONFIG[exercise_type]['voice_cues'].get(feedback_key, "")
            if message:
                self.speak(message, priority)
    
    def stop(self):
        self.running = False
        self.thread.join()