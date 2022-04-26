import matplotlib.pyplot as plt


def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()
