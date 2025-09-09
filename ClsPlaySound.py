import pygame

class ClsPlaySound:
    def playsound(filename):
        pygame.init()
        sound=pygame.mixer.Sound(filename)
        sound.play()