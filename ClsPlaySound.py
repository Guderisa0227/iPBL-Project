import pygame
import threading
import time

class ClsPlaySound:
    #kp start
    def __init__(self):
        pygame.init()
        self.sound_queue = []
        self.last_played = 0
        self.min_interval = 1 # milliseconds
    #kp end
    def playsound(self,filename):
        #pygame.init()
        sound=pygame.mixer.Sound(filename)
        sound.play()
    #kp start
    def play_with_delay(self, filename, delay=0):
        def delayed_play():
            time.sleep(delay)
            current_time = time.time()
            
            if current_time - self.last_played >= self.min_interval:
                sound = pygame.mixer.Sound(filename)
                sound.play()
                self.last_played = current_time
                
        sound_thread = threading.Thread(target=delayed_play)
        sound_thread.start()
    #kp end