# Esto esta hecho para generar una base de una mesa
# con formas basicas usando la libreria VPython. 
from vpython import box, vector, cylinder, color


#Parte de arriba de la mesa
top = box(pos=vector(-1,4,4), 
        axis=vector(-1.5,0,-1.5),
        size=vector(18,0.29,10),
        up=vector(5,25,5),
        color = color.blue)


#Patas de la mesa
leg1 = cylinder(pos=(-3,-3,0),
                axis=vector(0,7,0), radius=0.45,
                color = color.green )
leg2 = cylinder(pos=vector(3.5,-5,3),
                axis=vector(0,9,0), radius=0.45,
                color = color.green )
leg3 = cylinder(pos=vector(10.5,-3.9,2),
                axis=vector(0,9,0), radius=0.45,
                color = color.green )
leg4 = cylinder(pos=vector(-11.5,-6.5,-2.2),
                axis=vector(0,11,0), radius=0.45,
                color = color.green )