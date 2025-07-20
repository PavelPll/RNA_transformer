import subprocess
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Plot RNA secondary structure
# part of ViennaRNA package
# installation https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/install.html

def plot_rna_structure(sequence, structure, output_image="rna_structure.png"):

    # Define the PostScript output file from RNAplot
    ps_file = "rna.eps"  # RNAplot generates this file by default
    original_cwd = os.getcwd()
    print(original_cwd)

    try:
        os.chdir(original_cwd)
        # Run RNAplot with sequence and structure via stdin
        print("Running RNAplot...")
        result = subprocess.run(
            ["RNAplot"],
            input=f"{sequence}\n{structure}\n",
            text=True,
            capture_output=True,
            check=True
        )
        print("RNAplot executed successfully.")

        # Check if the PostScript file was created
        if not os.path.exists(ps_file):
            raise FileNotFoundError(f"Expected PostScript file {ps_file} not found.")

        # Convert the PostScript file to PNG using magick
        # install imagemagick
        # https://imagemagick.org/script/download.php#windows
        # C:\Program Files\ImageMagick-7.1.1-Q16-HDRI
        subprocess.run(["magick", ps_file, output_image], check=True)
        print(f"RNA structure image saved as {output_image}")

        # Display the image
        img = Image.open(output_image)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title("RNA Secondary Structure")
        plt.show()

    except subprocess.CalledProcessError as e:
        print(f"Error running RNAplot or ImageMagick: {e}")
        print(f"Standard Error Output: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
    except FileNotFoundError as e:
        print(e)
    finally:
        # Clean up temporary PostScript file
        if os.path.exists(ps_file):
            os.remove(ps_file)


def plot_circular_structure(sequence, structure):
    """
    Plot RNA secondary structure as a circular diagram.

    Parameters:
        sequence (str): The RNA sequence.
        structure (str): Dot-bracket notation of the secondary structure.
    """
    # Create a circular layout
    angles = np.linspace(0, 2 * np.pi, len(sequence), endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Plot the bases
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c="green", s=200, label="Bases")
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi * 1.1, yi * 1.1, sequence[i], ha="center", va="center")

    # Plot base pair connections
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            plt.plot([x[i], x[j]], [y[i], y[j]], c="red", lw=1)

    # Final touches
    plt.axis("equal")
    plt.axis("off")
    #plt.title("RNA Secondary Structure (Circular Representation)")
    plt.legend()
    plt.show()