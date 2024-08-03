
import numpy as np
import matplotlib.path as mpltPath
from scipy.spatial import Voronoi, Delaunay
#from scipy import spatial
#from scipy import spatial.voronoi
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib notebook

def intOfFOverT(f, N, T):
    x1 = T[0,0]
    x2 = T[1,0]
    x3 = T[2,0]
    y1 = T[0,1]
    y2 = T[1,1]
    y3 = T[2,1]
    xyw = TriGaussPoints(N)
    A = abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2
    NP = np.size(xyw[:,0])
    I = np.array([0])
    for j in range(NP):
        x = x1*(1-xyw[j,0]-xyw[j,1])+x2*xyw[j,0]+x3*xyw[j,1]
        y = y1*(1-xyw[j,0]-xyw[j,1])+y2*xyw[j,0]+y3*xyw[j,1]
        I = I + f(x,y)*xyw[j,2]

    I = A*I
    return I

def TriGaussPoints(n):
    if n==1:
        xw = np.array([[0.33333333333333, 0.33333333333333, 1.00000000000000]])
    elif n==2:
        xw = np.array([[0.16666666666667, 0.16666666666667, 0.33333333333333],\
            [0.16666666666667, 0.66666666666667, 0.33333333333333],\
            [0.66666666666667, 0.16666666666667, 0.33333333333333]])
    elif n==3:
        xw = np.array([[0.33333333333333, 0.33333333333333, -0.56250000000000],\
            [0.20000000000000, 0.20000000000000, 0.52083333333333],\
            [0.20000000000000, 0.60000000000000, 0.52083333333333],\
            [0.60000000000000, 0.20000000000000, 0.52083333333333]])
    elif n==4:
        xw = np.array([[0.44594849091597, 0.44594849091597, 0.22338158967801],\
            [0.44594849091597, 0.10810301816807, 0.22338158967801],\
            [0.10810301816807, 0.44594849091597, 0.22338158967801],\
            [0.09157621350977, 0.09157621350977, 0.10995174365532],\
            [0.09157621350977, 0.81684757298046, 0.10995174365532],\
            [0.81684757298046, 0.09157621350977, 0.10995174365532]])
    elif n==5:
        xw = np.array([[0.33333333333333, 0.33333333333333, 0.22500000000000],\
            [0.47014206410511, 0.47014206410511, 0.13239415278851],\
            [0.47014206410511, 0.05971587178977, 0.13239415278851],\
            [0.05971587178977, 0.47014206410511, 0.13239415278851],\
            [0.10128650732346, 0.10128650732346, 0.12593918054483],\
            [0.10128650732346, 0.79742698535309, 0.12593918054483],\
            [0.79742698535309, 0.10128650732346, 0.12593918054483]])
    elif n==6:
        xw = np.array([[0.24928674517091, 0.24928674517091, 0.11678627572638],\
            [0.24928674517091, 0.50142650965818, 0.11678627572638],\
            [0.50142650965818, 0.24928674517091, 0.11678627572638],\
            [0.06308901449150, 0.06308901449150, 0.05084490637021],\
            [0.06308901449150, 0.87382197101700, 0.05084490637021],\
            [0.87382197101700, 0.06308901449150, 0.05084490637021],\
            [0.31035245103378, 0.63650249912140, 0.08285107561837],\
            [0.63650249912140, 0.05314504984482, 0.08285107561837],\
            [0.05314504984482, 0.31035245103378, 0.08285107561837],\
            [0.63650249912140, 0.31035245103378, 0.08285107561837],\
            [0.31035245103378, 0.05314504984482, 0.08285107561837],\
            [0.05314504984482, 0.63650249912140, 0.08285107561837]])
    elif n==7:
        xw = np.array([[0.33333333333333, 0.33333333333333, -0.14957004446768],\
            [0.26034596607904, 0.26034596607904, 0.17561525743321],\
            [0.26034596607904, 0.47930806784192, 0.17561525743321],\
            [0.47930806784192, 0.26034596607904, 0.17561525743321],\
            [0.06513010290222, 0.06513010290222, 0.05334723560884],\
            [0.06513010290222, 0.86973979419557, 0.05334723560884],\
            [0.86973979419557, 0.06513010290222, 0.05334723560884],\
            [0.31286549600487, 0.63844418856981, 0.07711376089026],\
            [0.63844418856981, 0.04869031542532, 0.07711376089026],\
            [0.04869031542532, 0.31286549600487, 0.07711376089026],\
            [0.63844418856981, 0.31286549600487, 0.07711376089026],\
            [0.31286549600487, 0.04869031542532, 0.07711376089026],\
            [0.04869031542532, 0.63844418856981, 0.07711376089026]])
    elif n==8:
        xw = np.array([[0.33333333333333, 0.33333333333333, 0.14431560767779],\
            [0.45929258829272, 0.45929258829272, 0.09509163426728],\
            [0.45929258829272, 0.08141482341455, 0.09509163426728],\
            [0.08141482341455, 0.45929258829272, 0.09509163426728],\
            [0.17056930775176, 0.17056930775176, 0.10321737053472],\
            [0.17056930775176, 0.65886138449648, 0.10321737053472],\
            [0.65886138449648, 0.17056930775176, 0.10321737053472],\
            [0.05054722831703, 0.05054722831703, 0.03245849762320],\
            [0.05054722831703, 0.89890554336594, 0.03245849762320],\
            [0.89890554336594, 0.05054722831703, 0.03245849762320],\
            [0.26311282963464, 0.72849239295540, 0.02723031417443],\
            [0.72849239295540, 0.00839477740996, 0.02723031417443],\
            [0.00839477740996, 0.26311282963464, 0.02723031417443],\
            [0.72849239295540, 0.26311282963464, 0.02723031417443],\
            [0.26311282963464, 0.00839477740996, 0.02723031417443],\
            [0.00839477740996, 0.72849239295540, 0.02723031417443]])
    return xw

def getRandomPointsInPolygon(n,domain):
    # n points that we want generated
    # domain is a closed polygon, dimensions are 2xnVertices

    xMin = np.min(domain[:, 0])
    xMax = np.max(domain[:, 0])
    yMin = np.min(domain[:, 1])
    yMax = np.max(domain[:, 1])
    xy = []
    # Check this
    while(len(xy) < n):
        x = xMin+np.random.rand()*(xMax-xMin)
        y = yMin+np.random.rand()*(yMax-yMin)
        poly = mpltPath.Path(domain)
        if(poly.contains_points([[x,y]])):
            xy.append([x, y])

    xy = np.array(xy)
    xy = xy.T
    return xy

"""Robot Data class"""
class RobotData:
    def __init__(self,n,environment):
        # Instantiate RobotData object
        #
        # SYNTAX:
        #   r = RobotData(n_robots, environment)
        #
        # INPUTS:
        #   n_robots = (1 x 1 integer)
        #         Number of robots in the team
        #   environment = (2 x numVertices double)
        #         Matrix with the coordinates of each vertex in the
        #         environment
        #
        # NOTES:
        #----------------------------------------------------------------
        self.n = n  # number of robots
        xMin = np.min(environment[:, 0])
        xMax = np.max(environment[:, 0])
        yMin = np.min(environment[:, 1])
        yMax = np.max(environment[:, 1])
        points = 1e3*xMax*np.ones([self.n, 2])
        path = mpltPath.Path(environment)
        for i in range(self.n):
          point = np.reshape(points[i,:], [np.size(points[i,:]),1]).T
          while not (path.contains_points(point)):
            point[0,0] = 0.5*(xMax-xMin)*np.random.rand()+(0.25*(xMax-xMin))+xMin
            point[0,1] = 0.5*(yMax-yMin)*np.random.rand()+(0.25*(yMax-yMin))+yMin
            points[i,:] = point
        self.x = points[:,0]
        self.y = points[:,1]
        self.xy = np.array([self.x.T, self.y.T])
        self.theta = -np.pi + 2*np.pi *np.random.rand(n,1)
        self.satFlag = False;    # 1x1 logical indicating if velocity saturates
        self.maxV = 0;           # maximum velocity (if saturation is active)




    def updateControl(self, control, deltaT):
        # Update position of single integrators according to the
        # specified control
        #
        # SYNTAX:
        #   obj.updateControl(controlInput, deltaT)
        #
        # INPUTS:
        #   controlInput - (2 x obj.n double)
        #         Control input for the agents
        #   deltaT - (1 x 1 integer)
        #         Time elapsed
        #
        # NOTES:
        #----------------------------------------------------------------
        if self.satFlag:
            controlNorm = np.sqrt(np.sum(np.multiply(control,control), axis = 0))
            control[:,controlNorm>self.maxV] = np.divide(control[:,controlNorm>self.maxV], controlNorm[controlNorm>self.maxV]) * self.maxV


        for i in range(self.n):
            self.xy[:, i] = self.xy[:, i] + control[:, i] * deltaT

        self.x = self.xy[0, :].T
        self.y = self.xy[1, :].T


    def setSaturation(self, satFlag = False, maxV = 0):
        # Activate or deactivate velocity saturation for the robots
        #
        # SYNTAX:
        #   obj.setSaturation(satFlag, maxV)
        #
        # INPUTS:
        #   satFlag - (1 x 1 logical)
        #         Activate/Deactivate the saturation
        #   maxV - (1 x 1 double)
        #         Maximum velocity at which each robot can move in the
        #         plane
        #
        # NOTES:
        #----------------------------------------------------------------
        if satFlag:
            if maxV > 0:
                self.satFlag = True
                self.maxV = maxV
            elif maxV < 0:
                print('Incorrect maximum velocity. Saturation not set');
                self.satFlag = False
                self.maxV = 0
            else:
                self.satFlag = False
                self.maxV = maxV
                print('A maximum velocity needs to be specified')


    def setXY(self, x, y, theta=np.array([0])):
        # Specify positions for the robots in the 2D plane
        #
        # SYNTAX:
        #   obj.setXY(x, y)
        #
        # INPUTS:
        #   x - (obj.n x 1 double)
        #         x coordinates
        #   y - (obj.n x 1 double)
        #         y coordinates
        #
        # NOTES:
        #----------------------------------------------------------------
        self.x = x
        self.y = y
        self.xy = np.array([self.x.T, self.y.T])
        if np.size(theta) != 1:
            self.theta = theta


    def updateUnicycle(self, vw, tElapsed):
        #print(np.shape(vw))
        for i in range(self.n):
            #print(vw[0,i])
            #print(self.theta[i].squeeze())
            self.xy[:,i] = self.xy[:,i] + np.array([vw[0,i]*np.cos(self.theta[i].squeeze()), vw[0,i]*np.sin(self.theta[i].squeeze())])*tElapsed
            self.theta[i] = self.theta[i] + vw[1,i]*tElapsed
            self.theta[i] = np.arctan2(np.sin(self.theta[i]),np.cos(self.theta[i]))

        self.x = self.xy[0,:].T
        self.y = self.xy[1,:].T

"""Coverage Class"""

class Coverage:


    # -----------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------
    def __init__(self, density, environment, robot, delta_line = 100, epsilon = 1e-3):
    #function self = Coverage(density, environment, robot)

      # Instantiate a Coverage selfect
      #
      # SYNTAX:
      #   coverage = Coverage(density, environment, robot)
      #
      # INPUTS:
      #   density - (1 x 1 function handle)
      #         The function to cover over the domain
      #   environment - (numVertices x 2 double)
      #         Matrix with the coordinates of each vertex in the
      #         environment
      #   robot - (1 x 1 RobotData selfect)
      #         selfect containing the information about the robots
      #         positions in the team
      #
      # NOTES:
      #----------------------------------------------------------------
      self.density = density
      self.environment = environment
      [self.VoronoiCM,self.VoronoiCells] = self.coverageControl(robot)
      self.epsilon = epsilon  #perturbation for the density in dG_dt
      self.delta_line = delta_line #number of divisions for the line integral in dG_dx



    def coverageControl(self, robot, densities = [], mode = 'central'):
      # Get control law
      #
      # SYNTAX:
      #   [control, VoronoiCellCenterOfMass, VoronoiCellData] = self.coverageControl(robot)
      #
      # INPUTS:
      #   robot - (1 x 1 RobotData selfect)
      #         Information about the position of the robot
      #
      # OUTPUTS:
      #   control - (2 x numRobots double)
      #         Single integrator dynamics for each of the robots
      #   VoronoiCellCenterofMass - (2 x numRobots double)
      #         Location of the center of mass of the Voronoi cell
      #         associated with each of the robots
      #   VoronoiBoundaries - (2 x numRobots double)
      #         Location of the vertices of the Voronoi Cell (with the
      #         first vertex repeated at the end for plotting)
      #
      #----------------------------------------------------------------
        if mode == 'central':
            for i in range(robot.n):
                densities.append(self.density)
        else:
            pass


        P = np.append(robot.xy, self.__mirrorRobotsAboutEnvironmentBoundary(robot.xy), axis = 1)
        ''' TODO '''
        vor = Voronoi(P.T)
        self.VoronoiGraph = vor
        # Set V equal to the coordinates of the Voronoi vertices
        V = vor.vertices
        # Set C equal to the Indices of the Voronoi vertices forming each Voronoi region
        C = vor.regions          # Voronoi vertices and cells
        ''' END TODO '''


        # creating bounded large scalar based on environment
        envmax = np.max(np.abs(self.environment[:]))

        C.remove([])
        CenterOfMass = np.zeros([2, robot.n])        # centroid

        A = np.zeros([robot.n,1])
        h = np.zeros([robot.n,1])

        VoronoiCells = []
        xMin = np.min(self.environment[0, :])
        xMax = np.max(self.environment[0, :])
        yMin = np.min(self.environment[1, :])
        yMax = np.max(self.environment[1, :])

        # need to find the voronoi cells inside the domain, assigning to
        # corresponding robot
        domain_idx = np.zeros(robot.n)
        for i in range(len(C)):
          x_voro = V[C[i],0]
          y_voro = V[C[i],1]
          max_x = np.round(np.max(x_voro),4)
          min_x = np.round(np.min(x_voro),4)
          max_y = np.round(np.max(y_voro),4)
          min_y = np.round(np.min(y_voro),4)


          # ignoring voronoi cells outside of domain
          if max_x > xMax or min_x < xMin:
            continue
          elif max_y > yMax or min_y < yMin:
            continue
          else:
            # checking which robot is in specified voronoi cell
            for k in range(robot.n):
              poly = mpltPath.Path(V[C[i],:])
              if(poly.contains_points([[robot.x[k],robot.y[k]]])):
                # ordering regions by robot
                domain_idx[k] = i


        # each robot goes to center of its corresponding cell
        assert np.size(domain_idx) == robot.n, "need as many voronoi cells as robots"
        for j in range(robot.n):
            # index of the cell that is within the domain
            i = int(domain_idx[j])
            VoronoiCells.append(np.array([V[C[i],0], V[C[i],1]]).T)
            # calculating mass and center of mass of voronois cells


            [Gi, Ai] = self.__centroid(VoronoiCells[j], densities[j])
            CenterOfMass[:,j] = Gi.squeeze()
            A[j] = Ai
            h[j] = self.__calculateCost(VoronoiCells[j], robot.xy[:, j], densities[j])

        neighbors = {}
        for i in range(len(domain_idx)):
            nlist = []
            cell_set = set(C[int(domain_idx[i])])
            for j in range(len(domain_idx)):
                if i == j:
                    continue
                else:
                    other_set = set(C[int(domain_idx[j])])
                    # checking if voronoi cells have more than 1 shared vertex
                    if len(cell_set.intersection(other_set)) > 1:
                        nlist.append(j)
            neighbors[i] = nlist


        self.neighbors = neighbors # dictionary keys in order of robots 1-N
        self.VoronoiCells = VoronoiCells
        self.VoronoiCM = CenterOfMass
        self.VoronoiMass = A
        self.cost = h

        return [CenterOfMass, VoronoiCells]



    def setDensityFunction(self,density):
        self.density = density


    def getVoronoiCells(self):
        return(self.VoronoiCells)


    def getVoronoiCMs(self):
        return(self.VoronoiCM)


    def getCost(self):
        return(self.cost)



    #methods (Access = private)
    def __centroid(self, P, density):

        phiA = lambda x, y: density(x,y)
        phiSx = lambda x, y: x*density(x,y)
        phiSy = lambda x, y: y*density(x,y)

        trngltn = Delaunay(P)

        A = 0
        S = 0
        for i in range(np.size(trngltn.simplices[:,0])):
          ''' TODO '''
          # Let A be equivalent to the mass (sum of m)
          A = A + intOfFOverT(phiA, 8 , P[trngltn.simplices[i,:],:])
          # Let S be equivalent to the mass time distance (sum of m*r)
          S = S + np.array([intOfFOverT(phiSx, 8, P[trngltn.simplices[i,:],:]),\
                            intOfFOverT(phiSy, 8, P[trngltn.simplices[i,:],:])])
        # Calculate the center of mass
        G = S/A
        ''' END TODO '''

        return [G, A]



    def __calculateCost(self, P, rXY, density):
        phiH = lambda x,y: (((rXY[0]-x)**2+(rXY[1]-y)**2)*density(x,y))
        trngltn = Delaunay(P)
        H = 0
        for i in range(np.size(trngltn.simplices[:,0])):
          H = H + intOfFOverT(phiH, 8, P[trngltn.simplices[i,:],:])
        return H


    def __mirrorRobotsAboutEnvironmentBoundary(self, p):
        mirroredRobots = np.zeros([2,np.size(p[0,:])*(np.size(self.environment[0,:])-1)])
        for i in range(np.size(p[0,:])):
            point = p[:,i]
            for j in range(np.size(self.environment[0,:])-1):
                pointWrtSide = (point - self.environment[:,j])
                side = self.environment[:,j+1] - self.environment[:,j]
                lengthOfPProjectedOntoL = pointWrtSide.T @ side / np.linalg.norm(side)**2
                projectedVector = self.environment[:,j] + lengthOfPProjectedOntoL * side
                mirroredRobots[:,(i)*(np.size(self.environment[0,:])-1)+j] = point - 2 * (point - projectedVector)
        return mirroredRobots

"""*Visualization* Class"""


class Visualization:
  def __init__(self, covCtrl, density):
    # regression points
    xLine = np.arange(covCtrl.xMin, covCtrl.xMax, covCtrl.dx)
    yLine = np.arange(covCtrl.yMin, covCtrl.yMax, covCtrl.dy)
    [self.XMesh,self.YMesh] = np.meshgrid(xLine,yLine)
    # densities
    self.meshDensity = density(self.XMesh, self.YMesh)      #estimation
    self.vmax = np.max(self.meshDensity)
    self.meshSurrogate = 1e-10*np.ones(np.shape(self.meshDensity))
    self.meshSigma = np.zeros(np.shape(self.meshDensity))

    self.xMin = covCtrl.xMin
    self.yMin = covCtrl.yMin
    self.xMax = covCtrl.xMax
    self.yMax = covCtrl.yMax

    self.voroPlot = None
    #####  Initial Visualization Coverage
    plt.ion()
    # visualization
    figHandle = plt.figure(figsize=(4.5,8))
    #figHandle.canvas.set_window_title('Coverage Control')
    figHandle.facecolor = 'white'

    # left to right: robots executing coverage, uncertainty over domain,
    # difference between the real phi and the surrogate
    self.covAxes = figHandle.add_subplot(2, 1, 1)
    self.covAxes.set_title('Coverage')
    dens_cbar = self.covAxes.pcolormesh(self.XMesh, self.YMesh, self.meshSurrogate, shading = 'auto', cmap = 'summer', vmin = 0, vmax = self.vmax)
    self.dens_cb = plt.colorbar(dens_cbar, ax = self.covAxes)
    self.covAxes.plot(-2,-2,marker = 'x', c='red') # have some initial drawing
    self.covAxes.scatter(covCtrl.robot.x, covCtrl.robot.y, c='black')

    self.covAxes.set_yticklabels([])
    self.covAxes.set_xticklabels([])
    self.covAxes.set_xlim([self.xMin, self.xMax])
    self.covAxes.set_ylim([self.yMin, self.yMax])

    self.densitiesAxes = figHandle.add_subplot(2, 1, 2,  projection='3d')
    trueDens = self.densitiesAxes.plot_surface(self.XMesh, self.YMesh, self.meshDensity, cmap= 'summer',linewidth=0, antialiased=False)
    self.densitiesAxes.plot_wireframe(self.XMesh, self.YMesh, self.meshSurrogate, rcount = 30, ccount = 30)
    self.densitiesAxes.set_title('Estimation')
    m = cm.ScalarMappable(cmap= trueDens.cmap, norm = trueDens.norm)
    m.set_array(self.meshDensity)
    plt.colorbar(m, ax=self.densitiesAxes)
    self.densitiesAxes.set_xticklabels([])
    self.densitiesAxes.set_yticklabels([])
    self.densitiesAxes.set_zticklabels([])
    plt.draw()

    self.itr = 1
    plt.savefig('cov00'+str(self.itr)+'.png')

  def updateCov(self,covCtrl, coverage, goalx, goaly):

    self.covAxes.clear()
    #self.dens_cb.remove()
    voronoi_plot_2d(coverage.VoronoiGraph, ax = self.covAxes, show_vertices = False, show_points = False)
    dens_cbar = self.covAxes.pcolormesh(self.XMesh, self.YMesh, self.meshSurrogate, cmap = 'summer', shading = 'auto', vmin = 0, vmax = self.vmax)
    #self.dens_cb = plt.colorbar(dens_cbar, ax = self.covAxes)

    self.covAxes.scatter(covCtrl.robot.x, covCtrl.robot.y, c='black')
    self.covAxes.scatter(goalx,goaly,marker = 'x', c='red')

    self.covAxes.set_title('Coverage')
    self.covAxes.set_yticklabels([])
    self.covAxes.set_xticklabels([])
    self.covAxes.set_xlim([self.xMin,self.xMax])
    self.covAxes.set_ylim([self.yMin,self.yMax])
    plt.draw()

    plt.pause(0.0001)
    #print(t-time.time())
    self.itr += 1
    if self.itr < 10:
      plt.savefig('cov00'+str(self.itr)+'.png')
    elif self.itr < 100:
      plt.savefig('cov0'+str(self.itr)+'.png')
    else:
      plt.savefig('cov'+str(self.itr)+'.png')

  def updateDensity(self):
    self.densitiesAxes.clear()
    self.densitiesAxes.set_title('Estimation')
    self.densitiesAxes.plot_wireframe(self.XMesh, self.YMesh, self.meshSurrogate, rcount = 30, ccount = 30)
    self.densitiesAxes.plot_surface(self.XMesh, self.YMesh, self.meshDensity, cmap='summer',linewidth=0.1, antialiased=False)
    self.densitiesAxes.set_xticklabels([])
    self.densitiesAxes.set_yticklabels([])
    self.densitiesAxes.set_zticklabels([])
    plt.draw()
    plt.pause(0.001)


  def updateVoro(self,coverage):
    self.covAxes.clear()
    self.dens_cb.remove()
    self.voroPlot =  voronoi_plot_2d(coverage.VoronoiGraph, ax = self.covAxes, show_vertices = False, show_points = False)
    dens_cbar = self.covAxes.pcolormesh(self.XMesh, self.YMesh, self.meshSurrogate, cmap = 'summer', shading = 'auto', vmin = 0, vmax = 45)
    self.dens_cb = plt.colorbar(dens_cbar, ax = self.covAxes)

"""Main Algorithm"""

def NLC(CovCont):
  density = CovCont.densities[0]
  visGraph = Visualization(CovCont, density)

  coverage = Coverage(density, CovCont.domain_closed.T, CovCont.robot)

  xCov = []
  xCov.append(CovCont.robot.xy.T)

  # Initialization
  iter = np.arange(1,CovCont.nIter+1)

  x = CovCont.robot.xy.T                                          # initial robot locations

  for iter in range(CovCont.nIter):
    x = np.copy(x)
    # update density in visualizations
    for i in range(np.size(visGraph.XMesh[:,0])):
      for j in range(np.size(visGraph.XMesh[0,:])):
        visGraph.meshSurrogate[i,j] = visGraph.meshDensity[i,j]

    visGraph.updateDensity()

    error_cm = 1
    control = np.zeros([2, CovCont.N])

    while error_cm > CovCont.tol:
      ''' TODO'''
      [CM, VoronoiCellData] = coverage.coverageControl(CovCont.robot)


      goal = CM
      # x position of the goal
      goalx = CM[0,:]
      # y position of the goal
      goaly = CM[1,:]

      # Hint, check out CovCont.K and CovCont.robot.xy
      control = CovCont.K*(CM-CovCont.robot.xy)
      ''' END TODO '''

      if CovCont.satFlag:
        controlNorm = np.sqrt(np.sum(np.multiply(control,control), axis = 0))
        control[:,controlNorm>CovCont.maxV] = np.divide(control[:,controlNorm>CovCont.maxV], controlNorm[controlNorm>CovCont.maxV]) * CovCont.maxV
      visGraph.updateCov(CovCont, coverage, goalx, goaly)
      CovCont.robot.updateControl(control, 0.05)

      
      error_cm = np.linalg.norm(goal-CovCont.robot.xy,ord=2)

    print(iter)
    print(CovCont.robot.xy)

"""Run this to execute"""

class CoverageControl:
  def __init__(self, params):
    self.xMin = params['xMin']
    self.xMax = params['xMax']
    self.yMin = params['yMin']
    self.yMax = params['yMax']
    self.dx = params['dx']
    self.dy= params['dy']

    ''' SETTING SAT FLAG '''
    self.satFlag = True
    self.maxV = params['maxV']
    self.safety = params['safety']

    domain = [[self.xMin, self.yMin], [self.xMin, self.yMax], [self.xMax, self.yMax], [self.xMax, self.yMin]]
    domain_closed = domain
    domain_closed.append(domain_closed[0])
    self.domain = np.array(domain)
    self.domain_closed = np.array(domain_closed)

    # Robots
    self.N = params['N']

    # Density
    ''' TODO Modify the Density! '''
    np.random.seed(1)
    npeaks = 9
    xRange = self.xMax - self.xMin
    yRange = self.yMax - self.yMin
    xd = self.xMin + xRange * np.random.rand(npeaks, 1)
    yd = self.yMin + yRange * np.random.rand(npeaks, 1)
    density = lambda x,y: \
    20*np.exp((-np.power((x-xd[0]),2) - np.power((y-yd[0]),2))/0.04) + \
    20*np.exp((-np.power((x-xd[1]),2) - np.power((y-yd[1]),2))/0.04) + \
    20*np.exp((-np.power((x-xd[2]),2) - np.power((y-yd[2]),2))/0.04) + \
    20*np.exp((-np.power((x-xd[3]),2) - np.power((y-yd[3]),2))/0.04) + \
    20*np.exp((-np.power((x-xd[4]),2) - np.power((y-yd[4]),2))/0.04) + \
    20*np.exp((-np.power((x-xd[5]),2) - np.power((y-yd[5]),2))/0.04) + \
    20*np.exp((-np.power((x-xd[6]),2) - np.power((y-yd[6]),2))/0.04) + \
    20*np.exp((-np.power((x-xd[7]),2) - np.power((y-yd[7]),2))/0.04) + \
    20*np.exp((-np.power((x-xd[8]),2) - np.power((y-yd[8]),2))/0.04)
    ''' END TODO '''

    # Controller
    self.dt = params['dt']
    self.K = params['K']

    # loop variables
    self.nIter = params['nIter']
    self.tol = params['tol']
    self.densities = [density]


  def roboSetup(self):
    np.random.seed(6)
    ''' TODO '''
    # self.N is the number of robots
    # self.domain_closed is the environment boundaries
    robot_initial_poses =  getRandomPointsInPolygon(self.N,self.domain_closed)# Hint: check out the helper functions at the top!
    ''' END TODO '''
    self.robot = RobotData(self.N, self.domain_closed)
    self.robot.setXY(robot_initial_poses[0,:].T, robot_initial_poses[1,:].T)
    self.robot.setSaturation(satFlag = True, maxV = self.maxV) # maybe make this an input
    self.roboXY = np.ones([3, self.N])


  def safeControl(self, vw):
    N = self.N


# parameters of the problem
if __name__ == '__main__':

  params = {}
  params['xMin'] = -1.778/2 # x 140 in
  params['xMax'] = 1.778/2
  params['yMin'] = -2.159/2 # y 170 in
  params['yMax'] = 2.159/2
  params['dx'] = 0.05
  params['dy'] = 0.05
  params['N'] = 8
  params['maxV'] = 0.2
  params['safety'] = 0.3
  params['K'] = 20

  # Controller
  params['dt'] = 0.01
  # loop variables
  nIter = 1
  params['nIter'] = nIter
  params['tol'] = 1e-3


  cc = CoverageControl(params)
  cc.roboSetup()

  NLC(cc)

"""Run this to visualize the simulation"""

from PIL import Image
import glob

# Create the frames
frames = []
imgs = glob.glob("*.png")
for i in sorted(imgs):
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('png_to_gif.gif', format='GIF',append_images=frames[1:],save_all=True,duration=300, loop=0)
from IPython.display import Image, display
display(Image(filename='png_to_gif.gif'))
