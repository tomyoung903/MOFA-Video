import numpy as np
import matplotlib.pyplot as plt

''' Create a matrix of random values, of shape (20,400)
I used random integer values here between 0 and 255 
but you can do the same for decimal pixel intensities '''

mat = np.random.randint(0,255,400*20).reshape(400,20)
# Call imshow:
plt.imshow(mat, cmap='gray')
# save the image
plt.savefig('random_matrix.png')
