import math, matplotlib
import numpy as np
from scipy.misc import imresize
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _npcirclerender(image, cx, cy, radius, color, transparency=0.0): #render circles
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    image[cy-radius:cy+radius, cx-radius:cx+radius][index] = (
        image[cy-radius:cy+radius, cx-radius:cx+radius][index].astype('float32') * transparency +
        np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')

def chck_pnt(xCurr, yCurr, xMin, yMin, xMax, yMax):
    return xMin < xCurr < xMax and yMin < yCurr < yMax

def renderJoints(img, pos): #render Joints
    marker_size = 8
    minx = 2 * marker_size
    miny = 2 * marker_size
    maxx = img.shape[1] - 2 * marker_size
    maxy = img.shape[0] - 2 * marker_size
    Countjoints = pos.shape[0]

    imgrender = img.copy()
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 245, 255], [255, 131, 250], [255, 255, 0],
              [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

    for p_idx in range(Countjoints):
        xCurr = pos[p_idx, 0]
        yCurr = pos[p_idx, 1]
        if chck_pnt(xCurr, yCurr, minx, miny, maxx, maxy):
            _npcirclerender(imgrender,
                            xCurr, yCurr,
                            marker_size,
                            colors[p_idx],
                            transparency=0.5)
    return imgrender

def heatmapRender(conf, image, sourcemap, pos, cmap="jet"):
    interp = "trilinear"
    allJoints = conf.allJoints
    allJointsNames = conf.allJointsNames
    subPlot_width = 4
    subPlot_height = math.ceil((len(allJoints) + 1) / subPlot_width)
    f, axarr = plt.subplots(subPlot_height, subPlot_width)
    for pidx, part in enumerate(allJoints):
        plot_j = (pidx + 1) // subPlot_width
        plot_i = (pidx + 1) % subPlot_width
        sourcemap_part = np.sum(sourcemap[:, :, part], axis=2)
        sourcemap_part = imresize(sourcemap_part, 8.0, interp='bicubic')
        sourcemap_part = np.lib.pad(sourcemap_part, ((4, 0), (4, 0)), 'minimum')
        plotCurrentState = axarr[plot_j, plot_i]
        plotCurrentState.set_title(allJointsNames[pidx])
        plotCurrentState.axis('off')
        plotCurrentState.imshow(image, interpolation=interp)
        plotCurrentState.imshow(sourcemap_part, cmap=cmap, alpha=0.5, interpolation=interp)
    plotCurrentState = axarr[0, 0]
    plotCurrentState.set_title("Pose")
    plotCurrentState.axis('off')
    plotCurrentState.imshow(renderJoints(image, pos))
    plt.show()

def arrowRender(conf, image, pos, arrows):
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(image)
    ax.set_title("image")

    bx = fig.add_subplot(2, 2, 2)
    plt.imshow(image)
    bx.set_title("Predicted diff")

    colorChan=['r', 'g', 'b', 'c', 'm', 'y', 'k']
    pairsJoint = [(6, 5), (6, 11), (6, 8), (6, 15), (6, 0)]
    colorPoints = []
    for id, pairsJoint in enumerate(pairsJoint):
        end_joint_side = ("r " if pairsJoint[0] % 2 == 0 else "l ") if pairsJoint[1] != 0 else ""
        end_joint_name = end_joint_side + conf.allJointsNames[int(math.ceil(pairsJoint[1] / 2))]
        start = arrows[pairsJoint][0]
        end = arrows[pairsJoint][1]
        bx.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=3, head_length=3, fc=colorChan[id], ec=colorChan[id], label=end_joint_name)
        colorPoint = mpatches.Patch(color=colorChan[id], label=end_joint_name)
        colorPoints.append(colorPoint)

    plt.legend(handles=colorPoints, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def waitButton():
    plt.waitforbuttonpress(timeout=1)
    