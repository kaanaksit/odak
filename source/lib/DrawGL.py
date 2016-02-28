#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

__author__  = ('Kaan Akşit')
__version__ = '0.1'

# Class for OpenGL drawing.
class draw():
    def __init__(self):
        self.list  = []
        self.cache = []
        self.sex   = 0.
        self.sey   = 0.
        self.sez   = 0.
        self.ex    = 3.
        self.ey    = 3.
        self.ez    = -3.
        self.cx    = 0.
        self.cy    = 0.
        self.cz    = 0.
        self.wx    = 100.
        self.wy    = 100.
        self.wz    = 100.
        self.step  = 10.
    def axis(self,length):
        # Definition to draw an arrow with a cone at the top.
        glPushMatrix()
        glBegin(GL_LINES)
        glVertex3d(0,0,0)
        glVertex3d(0,0,length)
        glEnd()
        glTranslated(0,0,length)
        glutWireCone(0.04,0.2, 12, 9)
        glPopMatrix()
        return True
    def threeAxis(self,length):
        # Definition to draw lines for three axes.
        glPushMatrix()
        # Z-axis
        glColor3f(1.0,0.0,0.0)
        self.axis(length)
        # X-axis
        glRotated(90,0,1.0,0)
        glColor3f(0.0,1.0,0.0)
        self.axis(length)
        # Y-axis
        glRotated(-90,1.0,0,0)
        glColor3f(0.0,0.0,1.0)
        self.axis(length)
        glPopMatrix()
        return True
    def DrawTriangle(self,point0,point1,point2,color=[0.,0.,1.]):
        # Definition to draw a triangle.
        self.DrawLine(point0,point1,color)
        self.DrawLine(point1,point2,color)
        self.DrawLine(point2,point0,color)
        return True
    def DrawLine(self,p0,p1,color=[0.,0.,1.]):
        # Definition to draw a line.
        glBegin(GL_LINES)
        glColor3f(color[0],color[1],color[2])
        glVertex3f(p0[0], p0[1], p0[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glEnd()
        return True
    def DrawVector(self,vec,distance,color=[0.,0.,1.]):
        # Definition to draw a vector.
        p0 = vec[0].reshape(3)
        p1 = p0 + vec[1].reshape(3)*distance
        self.DrawLine(p0,p1,color=color)
        return True
    def Draw3Axes(self,length=500):
        # Definition to draw axes coordinates.
        glClear(GL_COLOR_BUFFER_BIT)
        self.threeAxis(length)
        return True
    def SizeOfView(self):
        # Determine the viewing window.
        glOrtho(self.cx-self.wx,self.cx+self.wx,self.cy-self.wy,self.cy+self.wy,self.cz-self.wz,self.cz+self.wz)
        return True
    def LookFrom(self):
        gluLookAt(self.sex, self.sey, self.sez,
                  self.ey, self.ez, self.ex,
                  0.0, 1.0, 0.0)
        return True
    def displayFun(self):
        # Display callback definition.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.SizeOfView()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self.LookFrom()
        self.Draw3Axes()
        self.list = list(self.cache)
        for item in self.list:
            if item[0] == 'vector':
               self.DrawVector(item[1],item[2],color=item[3])
        glFlush()
        return True
    def update(self):
        # Definition to update screen.
        glutPostRedisplay()
        return True
    def add(self,item):
        # Definition to add item to draw list.
        self.cache.append(item)
        self.update()
        return True
    def keyboard(self,key,x,y):
        if key == 'w':
           self.ex += self.step
        elif key == 's':
           self.ex -= self.step
        elif key == 'a':
           self.ey += self.step
        elif key == 'd':
           self.ey -= self.step
        elif key == 'f':
           self.ez += self.step
        elif key == 'c':
           self.ez -= self.step
        elif key == 'y':
           self.cy -= self.step
        elif key == 'h':
           self.cy += self.step
        elif key == 'g':
           self.cx += self.step
        elif key == 'j':
           self.cx -= self.step
        elif key == 'n':
           self.cz += self.step
        elif key == 'm':
           self.cz -= self.step
        self.update()
        return True
    def CreateWindow(self,res=[640,480],name='Odak'):
        glutInit()
        glutInitWindowSize(res[0],res[1])
        glutCreateWindow(name)
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glClearColor(0.0,0.0,0.0,0.0)
        # Call back functions.
        glutDisplayFunc(self.displayFun)
        glutKeyboardFunc(self.keyboard)
        glutMainLoop()
        return

if __name__ == '__main__':
    pass
