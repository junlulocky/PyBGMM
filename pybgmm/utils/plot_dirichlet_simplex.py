import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
from scipy.stats import gamma as gamma_dist
import random
import scipy.integrate as integrate

from ..dist.Dirichlet import Dirichlet
from ..dist.gDirichlet import gDirichlet

corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])


refiner = tri.UniformTriRefiner(triangle)
trimesh = refiner.refine_triangulation(subdiv=4)

# plt.figure(figsize=(8, 4))
# for (i, mesh) in enumerate((triangle, trimesh)):
#     plt.subplot(1, 2, i+ 1)
#     plt.triplot(mesh)
#     plt.axis('off')
#     plt.axis('equal')

midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 \
             for i in range(3)]
def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75 \
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


def draw_pdf_contours(dist, nlevels=200, subdiv=8, **kwargs):

    import math

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')

    ## save plot
    # save_path = os.path.dirname(__file__) + '/gmplots'
    # if value:
    #     plt.savefig(save_path + "/dirichlet_" + str(value) + ".pdf")
    #     plt.savefig(save_path + "/dirichlet_" + str(value) + ".png")
    plt.show()

def plot_points(X, barycentric=True, border=True, **kwargs):
    '''Plots a set of points in the simplex.
    Arguments:
        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.
        `barycentric` (bool): Indicates if `X` is in barycentric coords.
        `border` (bool): If True, the simplex border is drawn.
        kwargs: Keyword args passed on to `plt.plot`.
    '''
    if barycentric is True:
        X = X.dot(corners)
    plt.plot(X[:, 0], X[:, 1], 'k.', ms=1, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.hold(1)
        plt.triplot(triangle, linewidth=1)



if __name__ == '__main__':
    ## example
    save_path = os.path.dirname(__file__) + '/res_gdir/res_simplex'

    isGDirichlet = True

    if isGDirichlet:
        a=0.01
        b=0.01
        dist = gDirichlet(a=a, b=b, K=3)
        # draw_pdf_contours(dist)
        plot_points(dist.rvs(50000))

        plt.savefig(save_path + "/gdirichlet({},{}).pdf".format(a,b))
        plt.savefig(save_path + "/gdirichlet({},{}).png".format(a,b))

    else:
        value = [0.01,0.01,0.01]
        dist = Dirichlet(value)
        plot_points(dist.rvs(50000))
        # draw_pdf_contours(dist)
        plt.savefig(save_path + "/dirichlet({}).pdf".format(str(value)))
        plt.savefig(save_path + "/dirichlet({}).png".format(str(value)))

