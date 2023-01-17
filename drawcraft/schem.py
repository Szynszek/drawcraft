import mcschematic
import numpy as np




def createScheamtic(dir, m, name):

    # mirror and rotate matrix 90 degrees
    matrix = np.fliplr(m)
    matrix = np.rot90(m, 3)

    # Create a new schematic object
    schem = mcschematic.MCSchematic()
    # Iterate through the matrix
    for x, row in enumerate(matrix):
        for y, block in enumerate(row):
            schem.setBlock((x, y, 0), "minecraft:" + block)

    # Save the schematic
    schem.save(dir, name, mcschematic.Version.JE_1_19)
