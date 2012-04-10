#!/usr/bin/python
# -*- coding: utf-8 -*-
#Programmed by Kaan Akşit

import sys,os,time,pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


class pencere:
    def __init__(self,genislik,yukseklik):
        yerelzaman = time.asctime(time.localtime(time.time()))
        print "\x1b\x5b1;33;32m" + yerelzaman + ', Kaan Akşit, 2011' + '\x1b[0m'
        print "\x1b\x5b1;31;40m" + 'Pygame version: '+ "\x1b[0m" + str(pygame.version.ver)
        pygame.init()
        self.yukseklik = yukseklik +40
        self.genislik  = genislik
        self.ekran     = pygame.display.set_mode((self.genislik, self.yukseklik),HWSURFACE|OPENGL|DOUBLEBUF)
        print "\x1b\x5b1;31;40m" + 'Video info:'+' '+ "\x1b[0m",
        print "\x1b\x5b1;33;20m", pygame.display.Info()
        self.arkaplan            = pygame.Surface(self.ekran.get_size())
        self.arkakutu            = self.arkaplan.get_rect()
        self.arkakutu.height    -= 40
        self.arkaplan.set_colorkey((250, 250, 250))
        pygame.display.set_caption('Odak')
        glEnable(GL_DEPTH_TEST)   
        glShadeModel(GL_FLAT)
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)        
        glLight(GL_LIGHT0, GL_POSITION,  (0, 1, 1, 0)) 
        glMaterial(GL_FRONT, GL_AMBIENT, (0.1, 0.1, 0.1, 1.0))    
        glMaterial(GL_FRONT, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    def goster(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLight(GL_LIGHT0, GL_POSITION,  (0, 1.5, 1, 0))
        map.render()
        pygame.display.flip()
        return

def ana():
    pencere1 = pencere(640,480) 
    while True:
        olaylar = pygame.event.get()
        for e in olaylar:
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                sys.exit()
                return  
        pass
    return True

if __name__ == '__main__':
    sys.exit(ana())