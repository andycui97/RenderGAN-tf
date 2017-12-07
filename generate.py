import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import pickle

DEG2RAD = 3.14159/180

def Tag(radius, code, roll, pitch, yaw):
    glLoadIdentity()
    gluPerspective(90, 1, 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)
    glRotated( roll,0,0,1)
    glRotated( pitch,0,1,0)
    glRotated( yaw,1,0,0)

    glColor3f(1,1,1)        
    glBegin(GL_TRIANGLE_FAN)
    for i in range(360):
        degInRad = i*DEG2RAD
        glVertex3f(np.sin(degInRad)*radius,np.cos(degInRad)*radius,0)
    glEnd()

    for s in range(12):
        if code[s] == '1':
            glColor3f(1,1,1)
        else:
            glColor3f(0,0,0)        
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0,0,0)
        for i in range(s*30, (s+1)*30+1):
            degInRad = i*DEG2RAD
            glVertex3f(np.sin(degInRad)*radius*.8,np.cos(degInRad)*radius*.8,0)
        glEnd()

    glColor3f(1,1,1)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(0,0,0)
    for i in range(-90, 91):
        degInRad = i*DEG2RAD
        glVertex3f(np.sin(degInRad)*radius*.4,np.cos(degInRad)*radius*.4,0)
    glEnd()
    glColor3f(0,0,0)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(0,0,0)
    for i in range(90, 271):
        degInRad = i*DEG2RAD
        glVertex3f(np.sin(degInRad)*radius*.4,np.cos(degInRad)*radius*.4,0)
    glEnd()
    

def main():
    num = 0
    pygame.init()
    display = (800,800)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        roll, pitch, yaw = np.random.uniform(360),np.random.uniform(45),np.random.uniform(45)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        tag = ''
        for i in range(12):
            tag += str(np.random.randint(2))
        Tag(3, tag , roll, pitch, yaw)
        pygame.display.flip()
        pygame.image.save(pygame.display.get_surface(), "data/"+str(num)+".png")
        params = map(float,list(tag))
        params.extend([roll, pitch, yaw])
        pickle.dump(params,open("data/"+str(num)+".data", "wb+"))
        num += 1



main()    

