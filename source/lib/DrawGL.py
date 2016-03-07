#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy,time
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

__author__  = ('Kaan Akşit')
__version__ = '0.1'

# Class for OpenGL drawing.
class draw():
    def __init__(self,q=None):
        # Queue for multiprocessing.
        self.q               = q
        self.delay0          = 1
        # 3D related.
        self.list            = []
        self.cache           = []
        self.g_fViewDistance = 200.
        self.g_nearPlane     = 1.
        self.g_farPlane      = 6000.
        self.action          = ''
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
    def DrawTriangle(self,p0,p1,p2,color=[1.,1.,1.]):
        # Definition to draw a triangle.
        type = GL_LINE_LOOP
        glBegin(type)
        glColor3f(color[0],color[1],color[2])
        glVertex3f(p0[0], p0[1], p0[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
        glEnd()
        return True
    def DrawLine(self,p0,p1,color=[0.,0.,1.],type=GL_LINES):
        # Definition to draw a line.
        glBegin(type)
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
    def displayFun(self):
        # Display callback definition.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, -self.g_fViewDistance, 0, 0, 0, -.1, 0, 0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.zoom, float(self.g_Width)/float(self.g_Height), self.g_nearPlane, self.g_farPlane)
        glMatrixMode(GL_MODELVIEW)
        self.PolarView()
        self.scene()
        glutSwapBuffers()
        return True
    def scene(self):
        self.Draw3Axes()
        self.list = list(self.cache)
        for item in self.list:
            if item[0] == 'vector':
               self.DrawVector(item[1],item[2],color=item[3])
            elif item[0] == 'triangle':
               self.DrawTriangle(item[1],item[2],item[3])
        return True
    def UpdateScreen(self,dt):
        # Definition to update screen.
        glutPostRedisplay()
        if self.update == True:
            glutTimerFunc(self.delay,self.UpdateScreen,0)
        return True
    def ClearScreen(self):
        # Definition to clear the screen.
        self.list  = []
        self.cache = []
        self.UpdateScreen(0)
        return True
    def add(self,item):
        # Definition to add item to draw list.
        #print '%s: New item %s is added.' % (time.ctime(),item[0])
        if item == 'clear':
            print item
            return self.ClearScreen()
        self.cache.append(item)
        if self.update == False:
            self.UpdateScreen(0)
        return True
    def init(self):
        glEnable(GL_NORMALIZE)
        glLightfv(GL_LIGHT0,GL_POSITION,[ .0, 10.0, 10., 0. ] )
        glLightfv(GL_LIGHT0,GL_AMBIENT,[ .0, .0, .0, 1.0 ]);
        glLightfv(GL_LIGHT0,GL_DIFFUSE,[ 1.0, 1.0, 1.0, 1.0 ]);
        glLightfv(GL_LIGHT0,GL_SPECULAR,[ 1.0, 1.0, 1.0, 1.0 ]);
        glEnable(GL_LIGHT0)
#        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glShadeModel(GL_SMOOTH)
        self.ResetView()
    def ResetView(self):
        # Taken from http://carloluchessa.blogspot.com/2012/09/simple-viewer-in-pyopengl.html
        self.zoom    = 5.
        self.xRotate = 10.
        self.yRotate = 0.
        self.zRotate = 0.
        self.xTrans  = 1200.
        self.yTrans  = 0.
        self.zTrans  = 0.
        self.UpdateScreen(0)
        return True
    def mouse(self,button,state,x,y):
        # Taken from http://carloluchessa.blogspot.com/2012/09/simple-viewer-in-pyopengl.html
        if (button==GLUT_LEFT_BUTTON):
            if (glutGetModifiers() == GLUT_ACTIVE_SHIFT):
                self.action = "MOVE_EYE_2"
            else:
                self.action = "MOVE_EYE"
        elif (button==GLUT_MIDDLE_BUTTON):
            self.action = "TRANS"
        elif (button==GLUT_RIGHT_BUTTON):
            self.action = "ZOOM"
        self.xStart = x
        self.yStart = y
        return
    def keyboard(self,key, x, y):
        if(key=='r'): self.ResetView()
        if(key=='q'): exit(0)
        self.UpdateScreen(0)
    def PolarView(self):
        # Taken from http://carloluchessa.blogspot.com/2012/09/simple-viewer-in-pyopengl.html
        glTranslatef( self.yTrans/100., 0.0, 0.0 )
        glTranslatef(  0.0, -self.xTrans/100., 0.0)
        glTranslatef(  0.0, 0.0, -self.zTrans/100.)
        glRotatef( -self.zRotate, 0.0, 0.0, 1.0)
        glRotatef( -self.xRotate, 1.0, 0.0, 0.0)
        glRotatef( -self.yRotate, .0, 1.0, 0.0)
        return True
    def motion(self,x,y):
        # Taken from http://carloluchessa.blogspot.com/2012/09/simple-viewer-in-pyopengl.html
        if (self.action=="MOVE_EYE"):
            self.xRotate += x - self.xStart
            self.yRotate -= y - self.yStart
        elif (self.action=="MOVE_EYE_2"):
            self.zRotate += y - self.yStart
        elif (self.action=="TRANS"):
            self.xTrans += x - self.xStart
            self.yTrans += y - self.yStart
        elif (self.action=="ZOOM"):
            self.zoom -= y - self.yStart
            if self.zoom > 150.:
                self.zoom = 150.
            elif self.zoom < 1.1:
                self.zoom = 1.1
        self.xStart = x
        self.yStart = y
        self.UpdateScreen(0)
        return True
    def reshape(self,width, height):
        g_Width  = width
        g_Height = height
        glViewport(0, 0, self.g_Width, self.g_Height)
        return True
    def printHelp(self):
         print """\n\n
                -------------------------------------------------------------------\n
                Left Mousebutton       - move eye position (+ Shift for third axis)\n
                Middle Mousebutton     - translate the scene\n
                Right Mousebutton      - move up / down to zoom in / out\n
                Key                - reset viewpoint\n
                Key                - exit the program\n
                -------------------------------------------------------------------\n
                \n"""
    def CreateWindow(self,res=[640,480],delay=100,name='Odak',update=True):
        self.delay    = delay
        self.update   = update
        self.g_Width  = res[0]
        self.g_Height = res[1]

        glutInit()
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize(self.g_Width,self.g_Height)
        glutInitWindowPosition (0 + 4, self.g_Height / 4)
        glutCreateWindow(name)

        self.init()

        # Call back functions.
        glutReshapeFunc(self.reshape)
        glutDisplayFunc(self.displayFun)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)
        glutKeyboardFunc(self.keyboard)

#        if self.update == True:
#            glutTimerFunc(self.delay,self.UpdateScreen,0)
        if self.q != None:
            glutTimerFunc(self.delay0,self.QueueUpdate,0)

        glutMainLoop()
        return True
    def QueueUpdate(self,dt):
        # Definition for adding incoming items from other processes.
        if self.q.empty() == False:
            item = self.q.get()
            self.add(item)
        glutTimerFunc(self.delay0,self.QueueUpdate,0)
        return True

if __name__ == '__main__':
    pass
