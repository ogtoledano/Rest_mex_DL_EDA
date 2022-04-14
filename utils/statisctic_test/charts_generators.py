import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

if __name__ == "__main__":
    labels= [1,2,3,4,5]
    series = [547, 730, 2121, 5878, 20936]
    fig, ax = plt.subplots()
    rect=ax.bar(labels,series,width=0.5)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Examples')
    autolabel(rect,ax)
    plt.savefig('polarity.eps', format='eps')
    plt.show()

    labels = ["Hotel", "Restaurant", "Attractive"]
    series = [16565, 8450, 5197]
    fig, ax = plt.subplots()
    rect = ax.bar(labels, series, width=0.5)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Examples')
    autolabel(rect, ax)
    plt.savefig('attraction.eps', format='eps')
    plt.show()

