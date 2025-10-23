################################################################################
Function_flag           = 'pred_mode' 
modelname               = 'IFENN_A1'
filename                = 'SNT1'
meshname                = filename + '.msh'
picklename              = 'V3d'
CNNname                 = 'Sym17' 
loading_type            = 'Tension' # Tension, Shear
staggered_type          = 'OP' # OP, MP
loadstep_type           = 'LT1'
gaussian_flag           = 'on'
IFENN_flag              = 'on'          # "on": IFENN will be activated
                                        # "off": FEM-only
IFENN_switch_criterion  = "dmax_based"   # "inc_based": IFENN is activated at the beginning of a predefined increment
                                        # "dmax_based": IFENN is activated when phase-field reaches a predefined maximum value
IFENN_switch_inc        = 781           # 781 for SNT-'MP5000'

IFENN_switch            = 'off'         # Always leave as 'off', it will become 'on' if the appropriate condition is met
print("Modelname: ", modelname)

################################################################################
# Hyperparameters for FEM/IFENN an7alysis
length          = 1                     # length of domain sides
Lx              = length                # length of x-side 
Ly              = length                # length of y-side 
Gc              = 2.7                   # critical energy release rate
lc              = 0.03                  # characteristic length
elem_order      = 1                     # order of finite elements
quad_order      = 2

# Hyperparameters for PICNN training
Npoints         = 400                   # no. pixels at each direction (coincides with no. GPs at this direction)
h               = length / Npoints      # length of pixel
gaussian_kernel = 5                    # Gaussian filter kernel
gaussian_sigma  = 2                   # Gaussian filter sigma
seeding_number  = 1                     # fix seeding

################################################################################
if Function_flag   == "pred_mode":
    from dolfinx import mesh, fem, plot, io, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    from dolfinx.io import gmshio
    from petsc4py.PETSc import ScalarType, InsertMode, ScatterMode, Viewer, KSP
    from petsc4py import PETSc
    import ufl
    import basix
    from mpi4py import MPI
    from scipy.interpolate import griddata
    
    comm = MPI.COMM_WORLD
    comm_rank = MPI.COMM_WORLD.rank

import numpy as np
import os
import torch
import torch.nn.functional as F
import scipy.io
import torch.nn as nn
import torch.optim as optim
import numpy.random as npr
import random
from scipy.ndimage import gaussian_filter
import time

################################################################################
npr.seed(seeding_number); torch.manual_seed(seeding_number); random.seed(seeding_number)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device used is {device}')

def piecewise_time_discret2(transition_time, time_increments):
    # Assuming that total pseudotime = 1
    # This function creates a list of time instants. It has two time intervals [0,transition_time] and [transition_time,1.0]
    # The first is split into time_increments[0] segments, while the second one into time_increments[1]
    total_time    = 1.0
    tm_incr0      = time_increments[0]
    tm_incr1      = time_increments[1]
    Dt_initial    = transition_time/tm_incr0
    Dt_final      = (total_time-transition_time)/tm_incr1
    time_instants = [ Dt_initial*index for index in range(1,tm_incr0+1) ] + [ Dt_initial*tm_incr0+Dt_final*index for index in range(1,tm_incr1+1) ]
    return time_instants

##################################################################################################
class SymmetricConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.a = nn.Parameter(torch.randn(out_channels, in_channels)*0.1)  
        self.b = nn.Parameter(torch.randn(out_channels, in_channels)*0.1)  
        self.c = nn.Parameter(torch.randn(out_channels, in_channels)*0.1)  
        self.d = nn.Parameter(torch.randn(out_channels, in_channels)*0.1)  
        self.e = nn.Parameter(torch.randn(out_channels, in_channels)*0.1)  
        self.f = nn.Parameter(torch.randn(out_channels, in_channels)*0.1)  

    def forward(self, x):
        a = self.a[:, :, None, None]
        b = self.b[:, :, None, None]
        c = self.c[:, :, None, None]
        d = self.d[:, :, None, None]
        e = self.e[:, :, None, None]
        f = self.f[:, :, None, None]

        kernel = torch.cat([
            torch.cat([a, b, c, b, a], dim=3),
            torch.cat([b, d, e, d, b], dim=3),
            torch.cat([c, e, f, e, c], dim=3),
            torch.cat([b, d, e, d, b], dim=3),
            torch.cat([a, b, c, b, a], dim=3)
        ], dim=2)
        return F.conv2d(x, kernel, padding=2)
        
##################################################################################################
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = SymmetricConv2d(in_channels=1, out_channels=24)
        self.conv2 = SymmetricConv2d(in_channels=24, out_channels=24)
        self.conv3 = SymmetricConv2d(in_channels=24, out_channels=24)
        self.conv4 = SymmetricConv2d(in_channels=24, out_channels=1)
        self.sigmoid_activation = nn.Sigmoid()
        self.tanh_activation = nn.Tanh()
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.tanh_activation(self.conv1(x))
        x = self.tanh_activation(self.conv2(x))
        x = self.tanh_activation(self.conv3(x))
        x = self.sigmoid_activation(self.conv4(x))
        return x

def gaussian_smoothing(phi_pred, kernel_size=gaussian_kernel, sigma=gaussian_sigma):
    return torch.tensor(gaussian_filter(phi_pred, sigma=sigma))
    
# ------------------------------------------------------------------------------------------------
# Define the 3x3 Laplacian filter 
kernel = torch.tensor([[[[0.25, 0.5, 0.25],
                         [0.50, 0.0, 0.50],
                         [0.25, 0.5, 0.25]]]], dtype=torch.float32).to(device)  
print(kernel)

################################################################################
if Function_flag == "train_mode":

    # Specify CNN training parameters
    Adam_train_epochs   = 5000
    Adam_save_epochs    = 1000
    Adam_learning_rate  = 0.001
    loss_vec            = []

    # Load data for training
    Htrain = np.array(list(scipy.io.loadmat('M18_M27Mc_Ratio6_PixelValuesFEM_D3').items()), dtype = object)[:,1] 
    Htrain = Htrain[3]
    Htrain = torch.tensor(Htrain[0:2,:,:,:], requires_grad=False, dtype=torch.float32).to(device) # 
    
    print(Htrain.shape)
    print(torch.min(Htrain))
    print(torch.max(Htrain))

    CNN_model = CNNnet().to(device)
    optimizer = optim.Adam(CNN_model.parameters(), lr=Adam_learning_rate)

    print("Training starts!")

    for epoch in range(Adam_train_epochs):
        optimizer.zero_grad()
        phi = CNN_model(Htrain)
        phi_laplacian = F.conv2d(phi, kernel, padding=1) - 3*phi
        Residual_PDE_GP = Gc / lc * phi  - Gc * lc / h**2 * ( phi_laplacian )  - 2 * (1 - phi) * Htrain
        loss = torch.linalg.vector_norm(Residual_PDE_GP, ord = 2) 
        loss_vec.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % Adam_save_epochs == 0:
            print("------------------------")
            print(f"Epoch [{epoch+1}/{Adam_train_epochs}], Loss: {loss.item():.4f}")
            print("max pred: ", torch.max(phi))
            print("mean pred: ", torch.mean(phi))
            print("min pred: ", torch.min(phi))

            phi_smooth = gaussian_smoothing(phi.detach().cpu().numpy()).to(device)

            scipy.io.savemat(CNNname + '_' + modelname + '_' + filename + '_' + picklename + 'pred_Adam' + str(epoch+1) + '.mat', {'phi_pixelpred':phi.squeeze().detach().cpu().numpy(),
                                                                                                                'phi_laplacian_pixelpred':phi_laplacian.squeeze().detach().cpu().numpy(),
                                                                                                                'phismooth_pixelpred':phi_smooth.squeeze().detach().cpu().numpy()})
    scipy.io.savemat(CNNname + '_' + modelname + '_' + filename + '_' + picklename + 'pred.mat',
                     {'phi_pixelpred':phi.squeeze().detach().cpu().numpy(),
                     'phi_laplacian_pixelpred':phi_laplacian.squeeze().detach().cpu().numpy(),
                     'phismooth_pixelpred':phi_smooth.squeeze().detach().cpu().numpy(),
                     'loss_vec':loss_vec})

    total_params = sum(p.numel() for p in CNN_model.parameters())
    print(f"Number of parameters: {total_params}")
    torch.save(CNN_model.state_dict(), CNNname + '_M19_' + filename + '_' + picklename)

elif Function_flag == "pred_mode":

    ################################################################################
    CNN_model = CNNnet()
    CNN_model.load_state_dict(torch.load(CNNname + '_M19_M27Mc_Ratio6_' + picklename, map_location = device))

    index_pixel = np.array(list(scipy.io.loadmat('index_pixel_' + filename + '.mat').items()), dtype = object)[:,1] 
    index_pixel = index_pixel[3]
    index_pixel = torch.tensor(index_pixel, requires_grad=False, dtype=torch.long).to(device) 
    
    ################################################################################
    # ------------------------------------------------------------------------------
    # Load mesh file
    domain, _, _ = gmshio.read_from_msh(meshname, comm, gdim=2)

    # ------------------------------------------------------------------------------
    ## Problem Parameters
    rho     = 1
    lambda_ = 121154    # 1st Lame constant
    mu      = 80770     # 2nd Lame constant (shear modulus)
    
    if loading_type == "Tension":
        u_max   = 700e-5
        print("u_max: ", u_max)
    elif loading_type == "Shear":
        u_max   = 1800e-5
        print("u_max: ", u_max)
    else:
        print("Check your loading type!")    
    c1      = Gc*(lc)
    c2      = Gc/(lc)
    c3      = -2

    # ------------------------------------------------------------------------------
    # Check for other elastic constants
    Kbulk  = lambda_ + 2 * mu / 3
    nu = lambda_ / (2 * (lambda_ + mu))
    E  = (mu * (3*lambda_ + 2*mu)) / (lambda_ + mu)

    ################################################################################
    # ------------------------------------------------------------------------------
    ## Function Spaces
    V  = fem.functionspace(domain, ("CG", elem_order, (domain.geometry.dim,)))
    Vd = fem.functionspace(domain, ("CG", elem_order)) 
    WW = fem.functionspace(domain, ('DG', 0))

    QUAD_DEG = quad_order
    META_DATA = {"quadrature_degree":QUAD_DEG}
    dx = ufl.dx(metadata=META_DATA)
    ds = ufl.Measure("ds", domain = domain, metadata=META_DATA)

    quadrature_points, wts = basix.make_quadrature(basix.CellType.quadrilateral, QUAD_DEG)
    print(quadrature_points)
    print(wts)

    Qelem = basix.ufl.quadrature_element(basix.CellType.quadrilateral,degree=QUAD_DEG)
    QQ    = fem.functionspace(domain, Qelem)

    Lelem = basix.ufl.quadrature_element(basix.CellType.quadrilateral,degree=QUAD_DEG) # Space to host laplacian 
    LL    = fem.functionspace(domain, Lelem) 

    # ------------------------------------------------------------------------------
    ## Boundary Conditions
    def clamped_boundary(x):
        return np.isclose(x[1], -Ly/2)

    def clamped_boundary_left(x):
        return np.isclose(x[0], -Lx/2)

    def clamped_boundary_right(x):
        return np.isclose(x[0], Lx/2)

    def load_boundary(x):
        return np.isclose(x[1], Ly/2)

    def imposed_disp(t):
        return u_max*t

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    total_facets            = mesh.exterior_facet_indices(domain.topology)
    boundary_facets         = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
    boundary_facets_left    = mesh.locate_entities_boundary(domain, fdim, clamped_boundary_left)
    boundary_facets_right   = mesh.locate_entities_boundary(domain, fdim, clamped_boundary_right)
    load_facets             = mesh.locate_entities_boundary(domain, fdim, load_boundary)
    
    boundary_facets_left_dofs_y     = fem.locate_dofs_topological(V.sub(1), fdim, boundary_facets_left)
    boundary_facets_right_dofs_y    = fem.locate_dofs_topological(V.sub(1), fdim, boundary_facets_right)
    load_facets_dofs_y              = fem.locate_dofs_topological(V.sub(0), fdim, load_facets)
    
    # Clamped bottom edge
    u_D_bottom  = np.array([0, 0], dtype=default_scalar_type)
    bc_u_bottom = fem.dirichletbc(u_D_bottom, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

    # Traction BCs
    Traction = fem.Constant(domain, default_scalar_type((0, 0)))    

    # ------------------------------------------------------------------------------
    ## Variational Formulation
    def McBracketsPlus(x):
        return (x+abs(x))/2.0

    def McBracketsMinus(x):
        return (x-abs(x))/2.0

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def g(d):
        return (1-d)**2

    def g_diff(d):
        return -2*(1-d)

    def psi(u):
        return 0.5 * lambda_ * (ufl.tr(epsilon(u)))**2 + mu * ufl.inner(epsilon(u),epsilon(u))

    def eig_max(A):
        cr_root=ufl.tr(A)**2-4.0*ufl.det(A)
        lambda1 = (ufl.tr(A) + ufl.sqrt(cr_root))/2.0
        return lambda1

    def eigvec_max(A):
        a = A[0,0]
        b = A[0,1]
        c = A[1,1]

        cr_root=ufl.tr(A)**2-4.0*ufl.det(A)

        if (b!=0):
            v1 = -((-a+c-ufl.sqrt(cr_root))/(2*b))
            v2 = 1.0

        return ufl.conditional( ufl.ne(b, 0),ufl.as_vector((v1,v2)),ufl.conditional(ufl.ge(a, c),ufl.as_vector((1,0)),ufl.as_vector((0,1))))

    def eig_min(A):
        cr_root=ufl.tr(A)**2-4.0*ufl.det(A)
        lambda2 = (ufl.tr(A) - ufl.sqrt(cr_root))/2
        return lambda2

    def eigvec_min(A):
        a = A[0,0]
        b = A[0,1]
        c = A[1,1]

        cr_root=ufl.tr(A)**2-4.0*ufl.det(A)

        if (b!=0):
            v1 = -((-a+c+ufl.sqrt(cr_root))/(2*b))
            v2 = 1.0
        return ufl.conditional( ufl.ne(b, 0),ufl.as_vector((v1,v2)),ufl.conditional(ufl.ge(a, c),ufl.as_vector((1,0)),ufl.as_vector((0,1))))

    def psi_plus(u):
        strain_plus = eig_max(epsilon(u))*ufl.outer(eigvec_max(epsilon(u)),eigvec_max(epsilon(u)))/ufl.inner(eigvec_max(epsilon(u)),eigvec_max(epsilon(u))) + eig_min(epsilon(u))*ufl.outer(eigvec_min(epsilon(u)),eigvec_min(epsilon(u)))/ufl.inner(eigvec_min(epsilon(u)),eigvec_min(epsilon(u)))
        return (lambda_/2.0)*McBracketsPlus(ufl.tr(epsilon(u)))**2 +  mu * ufl.inner(strain_plus,strain_plus)

    def sigma_elastic(u):
        return (lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u))

    def vonMises(u,d):
        s = g(d) * sigma_elastic(u) - 1. / 3 * ufl.tr(g(d) * sigma_elastic(u)) * ufl.Identity(len(u))
        von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))
        return von_Mises

    # ------------------------------------------------------------------------------
    ## Displacement trial and test functions
    u   = fem.Function(V)
    w_u = ufl.TestFunction(V)

    # ------------------------------------------------------------------------------
    ## Phase-field trial and test functions
    d   = fem.Function(Vd)              
    w_d = ufl.TestFunction(Vd)
    dp  = fem.Function(Vd) 

    # ------------------------------------------------------------------------------
    # Displacement Initialization
    zeroD   = fem.Constant(domain, default_scalar_type([0, 0]))
    u_expr = fem.Expression(zeroD, V.element.interpolation_points())
    uh     = fem.Function(V)
    uh.interpolate(u_expr)
    uh.name = "Displacements"

    # ------------------------------------------------------------------------------
    # Phase-field Initialization
    zero   = fem.Constant(domain, default_scalar_type(0))
    d_expr = fem.Expression(zero, Vd.element.interpolation_points())
    dh     = fem.Function(Vd)
    dh.interpolate(d_expr)
    dh.name = "PhaseField"

    # von Mises Stress Initialization
    von_Mises = vonMises(uh,dh)
    von_mises_fs   = fem.functionspace(domain, ("DG", 0))
    von_mises_expr = fem.Expression(von_Mises, von_mises_fs.element.interpolation_points())
    von_mises      = fem.Function(von_mises_fs)
    von_mises.interpolate(von_mises_expr) 
    von_mises.name = "vonMises"

    # ------------------------------------------------------------------------------
    # Initialize History Variable
    H0   = fem.Constant(domain, default_scalar_type(0))
    H_expr = fem.Expression(H0, quadrature_points) # QQ.element.interpolation_points()

    map_c = domain.topology.index_map(domain.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    e_eval = H_expr.eval(domain,cells)

    Hist = fem.Function(QQ)
    with Hist.vector.localForm() as Hist_local:
        Hist_local.setBlockSize(Hist.function_space.dofmap.bs)
        Hist_local.setValuesBlocked(QQ.dofmap.list, e_eval, addv=InsertMode.INSERT)

    ## Phase-field at Gauss points
    d_GP_0      = fem.Constant(domain, default_scalar_type(0))
    d_GP_expr   = fem.Expression(d_GP_0, quadrature_points) 
    e_eval_d_GP = d_GP_expr.eval(domain,cells)
    d_GP        = fem.Function(QQ)
    d_GP.name = "d_GP"
    with d_GP.vector.localForm() as d_GP_local:
        d_GP_local.setBlockSize(d_GP.function_space.dofmap.bs)
        d_GP_local.setValuesBlocked(QQ.dofmap.list, e_eval_d_GP, addv=InsertMode.INSERT)

    #vtk = io.VTKFile(domain.comm, "Results.pvd", "w")
    #vtk.write_function([uh, dh], 0)

    ################################################################################
    # ------------------------------------------------------------------------------
    # Problem solution
    T               = 1
    step            = 0
    stag_iter       = 0
    dmax            = 0
    if staggered_type == 'OP':
        TotalStagIter  = 1
    elif staggered_type == 'MP':
        TotalStagIter  = 5000
    else:
        print("Check your staggered_type!")
    ConvergenceTol = 1e-5

    list_nodal_u_values          = []
    list_nodal_d_values          = []
    list_GP_Hist_values          = []
    list_GP_d_values             = []
    list_residual_u_values       = []
    list_residual_d_values       = []
    switch_flag                  = 0
    FEM_displacement_timer_vec   = []
    FEM_phasefield_timer_vec     = []
    IFENN_displacement_timer_vec = []
    IFENN_phasefield_timer_vec   = []
    
    
    if loading_type == "Tension":
        if loadstep_type == "LT1":
            startstepsize  = (1.0e-5)/(u_max)
            finalstepsize  = (1.0-500.0*startstepsize)/1000.0
            stepsize       = startstepsize
            timeinstants = [ startstepsize*index for index in range(1,501) ] + [ startstepsize*500+finalstepsize*index for index in range(1,1001) ]
        elif loadstep_type == "LT2":
            startstepsize  = (1e-5)/(u_max)
            stepsize       = startstepsize
            timeinstants = [ startstepsize*index for index in range(1,701) ]
        elif loadstep_type == "LT3":
            startstepsize  = (2.0e-5)/(u_max)
            stepsize       = startstepsize
            timeinstants = [ startstepsize*index for index in range(1,351) ]
        elif loadstep_type == "LT4":
            startstepsize  = (7.0e-5)/(u_max)
            stepsize       = startstepsize
            timeinstants = [ startstepsize*index for index in range(1,101) ]
        else:
            print("Check your loadstep_type!")
            
    elif loading_type == "Shear":
        if loadstep_type == "LT1":
            timeinstants = piecewise_time_discret2(4.0/6.0, [500,1000])
        elif loadstep_type == "LT2":
            startstepsize  = (1.4e-5)/(u_max)
            stepsize       = startstepsize
            timeinstants = [ startstepsize*index for index in range(1,1501) ]
        elif loadstep_type == "LT3":
            startstepsize  = (4.0e-5)/(u_max)
            stepsize       = startstepsize
            timeinstants = [ startstepsize*index for index in range(1,376) ]
        elif loadstep_type == "LT4":
            startstepsize  = (1.5e-4)/(u_max)
            stepsize       = startstepsize
            timeinstants = [ startstepsize*index for index in range(1,101) ]
        else:
            print("Check your loadstep_type!")
    
    Totalsteps   = len(timeinstants)            
    print("Total timesteps:", Totalsteps)

    #############################################################################################
    # Numerical solver: increment loop
    while step < Totalsteps: 
        
        print("step: ", step)
        t = timeinstants[step]
        stag_iter = 0
        list_residual_u     = []
        list_residual_d     = []
        list_stag_iter      = []

        if comm_rank == 0:
            if step % 50 == 0:
                print("-----------------------------------")
                print('Step = %d' %step, 't = %f' %t)

        # --------------------------------
        # Numerical solver: iteration loop
        while stag_iter < TotalStagIter:
            
            # -----------------------------------------------------   
            # Solve for displacement using FEM & keep track of time 
            FEM_displacement_start = time.time()
            
            f = fem.Constant(domain, default_scalar_type((0, 0))) 
            F_u = g(d_GP) * ufl.inner(sigma_elastic(u), epsilon(w_u)) * dx - ufl.dot(f, w_u) * dx - ufl.dot(Traction, w_u) * ds
            if loading_type == "Tension":
                u_imposed  = np.array([0, imposed_disp(t)], dtype=default_scalar_type)  
            elif loading_type == "Shear":
                u_imposed  = np.array([imposed_disp(t), 0], dtype=default_scalar_type) 
            else:
                print("Check your loading type!")    
            bc_imposed = fem.dirichletbc(u_imposed, fem.locate_dofs_topological(V, fdim, load_facets), V)

            problem_u = NonlinearProblem(F_u, u, bcs=[bc_u_bottom, bc_imposed])
            solver_u = NewtonSolver(domain.comm, problem_u)
            solver_u.solve(u)
            
            FEM_displacement_timer = time.time() - FEM_displacement_start 
            FEM_displacement_timer_vec.append(FEM_displacement_timer)

            # ---------------------------        
            # Check H against the history variable     
            uh.x.array[:]   = u.x.array[:]
            Hvar            = ufl.max_value(psi_plus(uh), Hist)                
            HvarPP_expr     = fem.Expression(Hvar, quadrature_points)
            e_evalPP        = HvarPP_expr.eval(domain,cells)

            # ----------------------
            # Assemble into Function
            HistPP = fem.Function(QQ)
            with HistPP.vector.localForm() as HistPP_local:
                HistPP_local.setBlockSize(HistPP.function_space.dofmap.bs)
                HistPP_local.setValuesBlocked(QQ.dofmap.list, e_evalPP, addv=InsertMode.INSERT)
            GP_Hist_values_tensor = torch.tensor(np.array(HistPP.x.array), requires_grad=False)

            # ------------------------------------
            # Check if IFENN needs to be activated
            if IFENN_flag == 'on' and IFENN_switch == 'off':
                if IFENN_switch_criterion == 'dmax_based':
                    if dmax > 0.99: 
                        IFENN_switch = 'on' 
                        print("IFENN is activated: dmax-based")
                elif IFENN_switch_criterion == 'inc_based':
                    if step == IFENN_switch_inc:
                        IFENN_switch = 'on'
                        print("IFENN is activated: inc-based")
            
            ################################################### PHASEFIELD ###################################################
            # ----------------------------------------------------------------------------------------------------------------      
            if IFENN_switch == 'off':
                
                # Solve for phasefield using FEM & keep track of time 
                FEM_phasefield_start = time.time()

                # ------------------------------------------------------------
                F_d = c1 * ufl.dot(ufl.grad(d), ufl.grad(w_d)) * dx + (c2 + 2.0 * Hvar) * d * w_d * dx + c3 * Hvar * w_d * dx
                problem_d = NonlinearProblem(F_d, d, bcs = [] )
                solver_d  = NewtonSolver(domain.comm, problem_d)
                solver_d.solve(d)
                # ------------------------------------------------------------

                FEM_phasefield_timer = time.time() - FEM_phasefield_start 
                FEM_phasefield_timer_vec.append(FEM_phasefield_timer)

                dmin = domain.comm.allreduce(np.min(d.x.array), op=MPI.MIN)
                dmax = domain.comm.allreduce(np.max(d.x.array), op=MPI.MAX) 

                dh.x.array[:] = d.x.array[:]
                d_GP_expr = fem.Expression(d, quadrature_points) 
                e_eval_d_GP = d_GP_expr.eval(domain,cells)
                with d_GP.vector.localForm() as d_GP_local:
                        d_GP_local.setBlockSize(d_GP.function_space.dofmap.bs)
                        d_GP_local.setValuesBlocked(QQ.dofmap.list, e_eval_d_GP, addv=InsertMode.INSERT)

            elif IFENN_switch == 'on':
                
                # Solve for phasefield using IFENN (PICNN) & keep track of time 
                IFENN_phasefield_start = time.time()

                # ------------------------------------------------------------
                gridH = torch.tensor(torch.take(GP_Hist_values_tensor, index_pixel - 1),dtype=torch.float32).unsqueeze(0)   # Convert the GP-values to pixel-values
                gridH = np.minimum(gridH, 5e4)                                                                              # Normalize the H pixel-values before passing them to CNN
                phi = CNN_model(gridH)                                                                                      # Make the CNN prediction at pixels
                if gaussian_flag == 'on':                                                                                   # Apply Gaussian smoothing filter
                    phi_smooth = gaussian_smoothing(phi.cpu().detach().numpy())
                    phi_cpu_flat = phi_smooth.flatten()
                else:
                    phi_cpu = phi.cpu().detach().numpy()
                    phi_cpu_flat = phi_cpu.flatten()

                _, b         = torch.sort(index_pixel.reshape(-1))                                                          # Move the pixel values to GPs
                phi_GP       = phi_cpu_flat[b]                                                                              # Flatten the values
                d_GP.x.array[:] = np.maximum(d_GP.x.array[:], phi_GP)                                                       # Ensure phasefield is only increasing
                # ------------------------------------------------------------

                IFENN_phasefield_timer = time.time() - IFENN_phasefield_start 
                IFENN_phasefield_timer_vec.append(IFENN_phasefield_timer)

                # Compute total residual
                Total_Res = _residual_u
            
            stag_iter += 1
            list_stag_iter.append(stag_iter)
            
            # -----------------------------
            # Compute displacement residual
            problem_u.form(uh.x.petsc_vec)                 
            problem_u.F(uh.x.petsc_vec, solver_u._b)
            _residual_u = solver_u._b.norm(PETSc.NormType.NORM_2)
            list_residual_u.append(_residual_u)
            
            if IFENN_switch == 'off':
                problem_d.form(dh.x.petsc_vec)                 
                problem_d.F(dh.x.petsc_vec, solver_d._b)
                _residual_d = solver_d._b.norm(PETSc.NormType.NORM_2)
                list_residual_d.append(_residual_d)
                
                Total_Res = _residual_u + _residual_d
            
            elif IFENN_switch == 'on':
                Total_Res = _residual_u
            
            list_stag_iter.append(stag_iter)
            
            if (Total_Res < ConvergenceTol): break 

        # -------------------------------------------------------------------------------------------------
        # -------------------------------------- COMMON POSTPROCESS ---------------------------------------
        # -------------------------------------------------------------------------------------------------

        H_expr      = fem.Expression(Hvar, quadrature_points)
        e_eval      = H_expr.eval(domain,cells)

        # Assemble into Function
        Hist = fem.Function(QQ)
        with Hist.vector.localForm() as Hist_local:
            Hist_local.setBlockSize(Hist.function_space.dofmap.bs)
            Hist_local.setValuesBlocked(QQ.dofmap.list, e_eval, addv=InsertMode.INSERT)

        # ------------------------------------------------
        # Energy computation at nodes
        Elastic_Energy_nodal  = fem.assemble_scalar(fem.form(0.5 * g(d_GP) * ufl.inner(sigma_elastic(uh), epsilon(uh)) * dx))  

        w_dp = ufl.TestFunction(Vd)
        F_dproj = dp * w_dp * dx - d_GP * w_dp * dx

        problem_dproj = NonlinearProblem(F_dproj, dp, bcs = [] )
        solver_dproj  = NewtonSolver(domain.comm, problem_dproj)
        solver_dproj.solve(dp)

        Fracture_Energy_nodal = fem.assemble_scalar(fem.form(0.5 * (Gc/lc) * (d_GP*d_GP+lc*lc*ufl.dot(ufl.grad(dp),ufl.grad(dp)))*dx)) 
        Total_Energy_nodal = Elastic_Energy_nodal + Fracture_Energy_nodal 

        # ---------------------------------------------
        # Reaction forces
        if loading_type == "Tension": 
            direction  = 1
        elif loading_type == "Shear":
            direction  = 0
        else:
            print("Check your loading_type!")

        F_u = g(d_GP) * ufl.inner(sigma_elastic(u), epsilon(w_u)) * dx - ufl.dot(f, w_u) * dx - ufl.dot(Traction, w_u) * ds
        u_reaction = fem.Function(V)
        Vcollapsed, _ = (V.sub(direction)).collapse()
        u_r0 = fem.Function(Vcollapsed)
        u_r0.x.array[:] = -1.0

        dof_reaction    = fem.locate_dofs_topological((V.sub(direction), Vcollapsed), fdim, mesh.locate_entities_boundary(domain, fdim, clamped_boundary))    
        bcF             = fem.dirichletbc(u_r0, dof_reaction, V.sub(direction))
        fem.set_bc(u_reaction.vector, [bcF]) 
        aV              = ufl.action(F_u, u_reaction)
        Reaction        = fem.assemble_scalar(fem.form(aV))          
        Reaction_sum    = domain.comm.allreduce(Reaction, op=MPI.SUM)

        nodal_u_values = np.array(uh.x.array)
        GP_Hist_values = np.array(Hist.x.array)
        GP_d_values    = np.array(d_GP.x.array)
        residual_u_values = np.array(list_residual_u)
        residual_d_values = np.array(list_residual_d)
        
        list_nodal_u_values.append(nodal_u_values)
        list_GP_Hist_values.append(GP_Hist_values)
        list_GP_d_values.append(GP_d_values)
        list_residual_u_values.append({'residual_u_values': residual_u_values})    
        list_residual_d_values.append({'residual_d_values': residual_d_values})    

        if comm_rank == 0:
            if step % 50 == 0:
                print("u_imposed:", u_imposed)
                print("Reaction_sum:", Reaction_sum)
                print("stag_iter:", stag_iter)
                print("max GP_d_values", np.max(GP_d_values))
                print("max GP_Hist_values", np.max(GP_Hist_values))

        with open(modelname + '_' + filename + '_' + staggered_type + '_' + loading_type + '_' + loadstep_type + '_Reactions.txt', 'a') as rfile:
                rfile.write("%s %s\n" % (str(imposed_disp(t)), str(Reaction_sum))) 

        with open(modelname + '_' + filename + '_' + staggered_type + '_' + loading_type + '_' + loadstep_type + '_stagiters.txt', 'a') as rfile:
                rfile.write("%s\n" % (np.array(stag_iter)))

        with open(modelname + '_' + filename + '_' + staggered_type + '_' + loading_type + '_' + loadstep_type + '_Energies.txt', 'a') as rfile:
                rfile.write("%s %s %s\n" % (str(Elastic_Energy_nodal), str(Fracture_Energy_nodal), str(Total_Energy_nodal))) 

        step += 1


    # Store coordinates
    coordinates_nodes = V.tabulate_dof_coordinates()
    coordinates_GPs   = QQ.tabulate_dof_coordinates()
    
    #######################################################################################
    scipy.io.savemat(modelname + '_' + filename + '_' + staggered_type + '_' + loading_type + '_' + loadstep_type + '_residuals.mat', 
                    {"residual_u_values": list_residual_u_values,
                     "residual_d_values": list_residual_d_values})
    
    scipy.io.savemat(modelname + '_' + filename + '_' + staggered_type + '_' + loading_type + '_' + loadstep_type + '_dofmap_V.mat', 
                    {"dofmap_V": V.dofmap.list})
    
    scipy.io.savemat(modelname + '_' + filename + '_' + staggered_type + '_' + loading_type + '_' + loadstep_type + '_nodal_data.mat', 
                    {"nodal_u_values": list_nodal_u_values,
                     "nodal_d_values": list_nodal_d_values,
                     "coordinates_nodes": coordinates_nodes})
    
    scipy.io.savemat(modelname + '_' + filename + '_' + staggered_type + '_' + loading_type + '_' + loadstep_type + '_GP_data.mat', 
                    {"GP_Hist_values": list_GP_Hist_values,
                     "GP_d_values": list_GP_d_values,
                     "coordinates_GPs": coordinates_GPs,
                     "FEM_displacement_timer_vec": FEM_displacement_timer_vec,
                     "FEM_phasefield_timer_vec": FEM_phasefield_timer_vec,
                     "IFENN_displacement_timer_vec": IFENN_displacement_timer_vec,
                     "IFENN_phasefield_timer_vec": IFENN_phasefield_timer_vec})  






