#!/usr/bin/env python

import numpy as np; np.set_printoptions(linewidth=np.nan)
from mpi4py import MPI
from petsc4py import PETSc

SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()


def testme(maxgrad, topology):
    '''In fact, the per-rank maximum occurs in z-direction at *non-ghosted* lattice point
    (xmax-1, ymax-1,0) for ranks at the 'left' side of non-periodic '0' (=Z!) dimension
    where xmax, ymax, zmax refer to the maximum values in the *non-ghosted* lattice
    (i.e. sizes-parameter given to the constructor of rankinfo), and counting starts from
    the [1,1,1] of the ghosted lattice (i.e. first interior point); for all other ranks,
    the maximum is in y-direction at (0,ymax-1,zmax-1). I.e. is sizes==[3,4,5] these are
    at [3,4,1] and [1,4,5] of the ghosted lattice

    The value at *non-ghosted* [x,y,z] == (z+zmax*(y+ymax*(x))+rank)**2, so for the whole
    ghosted lattice the value at [x+1,y+1,z+1] == (z+zmax*(y+ymax*(x))+rank)**2.

    Hence the expected per-rank maximum is for ranks at 'left' boundary (recall the
    boundary condition is lattice[boundary]=0):

    max1 = (((0+1)+zmax*((ymax-1)+ymax*(xmax-1))+rank)**2-0)/2

    and the other ranks have (note that the y-direction is periodic, so ymax means 1):

    max2 = (((zmax-1)+zmax*((1)+ymax*(0))+rank)**2-((zmax-1)+zmax*((ymax-2)+ymax*(0))+rank)**2)/2

    '''
    size = topology.Get_size()
    # rank = topology.Get_rank()
    expected = [1568, 1600, 1640, 1680, 1720, 1760, 1800, 1840]
    print(f'{maxgrad} == {expected[size - 1]} ?')
    return maxgrad == expected[size - 1]


def initialise_values(dm, vec_l, vec_g):
    '''Local arrays need to be STENCIL*2 bigger than the 'real world' in each dimension'''
    # sw = dm.getStencilWidth()
    ranges = np.array(dm.getRanges())
    lens = np.squeeze(np.diff(ranges))
    print(f'lens (loc)={lens}')
    array_g = dm.getVecArray(vec_g)
    array_g[...] = ((np.arange(lens.prod()) + RANK)**2).reshape(lens)
    # print(f'array_g=\n{array_g[...]}')
    try:
        grad = np.array(np.gradient(array_g[...]))
        print(f'grad=\n{grad}')
    except:
        pass
    dm.globalToLocal(vec_g, vec_l)
    array_l = dm.getVecArray(vec_l)
    ranges = np.array(dm.getGhostRanges())
    lens = np.squeeze(np.diff(ranges))
    print(f'lens (ghosted)={lens}')
    (xs, xe), (ys, ye), (zs, ze) = ranges
    array_l[xs:xe, ys:ye, zs:ze] = np.zeros(lens.prod()).reshape(lens)
    return


def find_max_grad(dm, vec_l, vec_g):
    dm.globalToLocal(vec_g, vec_l)
    loc_arr = dm.getVecArray(vec_l)[...]
    (xs, xe), (ys, ye), (zs, ze) = dm.getRanges()
    sw = dm.getStencilWidth()
    # sw = 0
    ranges = np.array(dm.getRanges())
    lens = np.squeeze(np.diff(ranges))
    print(f'(xs, xe)={(xs, xe)} (ys, ye)={(ys, ye)} (zs, ze)={(zs, ze)} sw={sw}')
    print(f'lens={lens}')
    # print(xs + sw, xe + sw, ys + sw, ye + sw, zs + sw, ze + sw)
    print(f'(ghosted) loc_arr.shape={loc_arr.shape}')
    grad = np.array(np.gradient(loc_arr))  # gradient along dim (3) : grad0 grad1, grad2
    # print(f'grad=\n{grad}')
    print(f'grad.shape={grad.shape}')
    if 0:
        # unghosted gradients
        sl = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
    elif 1:
        # since we use periodic boundaries, must use the ghosted gradient array
        sl = (slice(None), slice(0, xe - xs), slice(0, ye - ys), slice(0, ze - zs))
    else:
        sl = (slice(None), slice(sw, xe - xs + sw), slice(sw, ye - ys + sw), slice(sw, ze - zs + sw))
    print(f'sl={sl}')
    gradients = grad[sl]
    print(f'gradients.shape={gradients.shape}')
    maxgrad_local = gradients.max()
    maxgrad_global = np.zeros_like(maxgrad_local)
    dm.getComm().tompi4py().Allreduce([maxgrad_local, MPI.DOUBLE],
                                      [maxgrad_global, MPI.DOUBLE],
                                      op=MPI.MAX)
    return maxgrad_local, maxgrad_global


def main():
    dim = 3
    proc_sizes = tuple(MPI.Compute_dims(SIZE, dim))
    # see lists.mcs.anl.gov/pipermail/petsc-dev/2016-April/018903.html
    # manually distribute the unkowns to the processors
    # mpiexec -n 3
    if SIZE > 1:
        if 1:
            ownership_ranges = (
                np.array([3, 4, 5], dtype='i4'),
                np.array([1], dtype='i4'),
                np.array([1], dtype='i4')
            )
        else:
            ownership_ranges = (
                # proc 0
                (3, 4, 5),
                # proc 1
                (1,),
                # proc 2
                (1,)
            )
    else:
        ownership_ranges = ((3,), (4,), (5,))
    sizes = np.array((
        sum(ownership_ranges[0]),
        sum(ownership_ranges[1]),
        sum(ownership_ranges[2])), dtype='i4'
    )
    dm = PETSc.DMDA().create(dim=dim, sizes=sizes, proc_sizes=proc_sizes,
                             ownership_ranges=ownership_ranges,
                             boundary_type=(PETSc.DMDA.BoundaryType.PERIODIC,
                                            PETSc.DMDA.BoundaryType.PERIODIC,
                                            PETSc.DMDA.BoundaryType.GHOSTED),
                             stencil_type=PETSc.DMDA.StencilType.BOX,
                             stencil_width=1, dof=1, comm=PETSc.COMM_WORLD, setup=True)
    assert dm.getComm().Get_rank() == RANK
    assert dm.getComm().Get_size() == SIZE
    print(f'create problem with {sizes.prod()} unknowns split into dim={dim} (sizes={sizes}) split accross {SIZE} procs with distribution {proc_sizes}')
    vec_l = dm.createLocalVector()
    vec_g = dm.createGlobalVector()
    initialise_values(dm, vec_l, vec_g)
    result_l, result_g = find_max_grad(dm, vec_l, vec_g)
    PETSc.Sys.syncPrint(f'Rank {RANK} had max gradient {result_l} while the global was {result_g}.')
    if RANK == 0:
        if testme(result_g, dm.getComm()):
            print('Result is correct.')
        else:
            print('Result is incorrect!')


if (__name__ == '__main__'):
    main()
