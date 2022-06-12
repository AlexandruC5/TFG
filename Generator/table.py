# Esto esta hecho para generar una base de una mesa
# con formas basicas usando la libreria VPython.
from matplotlib.pyplot import axis
from vpython import *

# Parte de arriba de la mesa
top = box(pos=vector(0, 0, 0),
          axis=vector(0, 0, 0),
          size=vector(18, 0.29, 10),
          up=vector(0, 0, 0),
          color=color.blue)


# Patas de la mesa

pata1 = cylinder(pos=vector(8.5, 0, -4.5), axis=vector(0, -5, 0), radius=0.5)
pata2 = cylinder(pos=vector(-8.5, 0, 4.5), axis=vector(0, -5, 0), radius=0.5)
pata3 = cylinder(pos=vector(8.5, 0, 4.5), axis=vector(0, -5, 0), radius=0.5)
pata4 = cylinder(pos=vector(-8.5, 0, -4.5),axis=vector(0, -5, 0), radius=0.5)
 